"""
directory: src/shellworks/orchestrator/
file:      validation.py
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Structural types for tool call objects
# ---------------------------------------------------------------------------

class ToolCallFunction(Protocol):
    """
    The two fields validate_tool_call reads off tool_call.function.

    Public so that test stubs can annotate their function attribute
    with this type explicitly, which satisfies Pyright's invariance
    check on mutable Protocol attributes.
    """
    name: str
    arguments: str


class ToolCallLike(Protocol):
    """
    Structural type for any object that looks like a tool call.

    validate_tool_call only reads tool_call.function.name and
    tool_call.function.arguments. Using a Protocol here instead of
    the concrete OpenAI SDK type means:

      - validation.py has no dependency on the OpenAI SDK
      - the validator works with any conforming object, including
        lightweight stubs in tests

    The real OpenAI ChatCompletionMessageToolCall satisfies this
    Protocol structurally, so no cast is needed at call sites.

    Test stubs must annotate their function attribute explicitly as
    ToolCallFunction to satisfy Pyright's invariance check on mutable
    attributes:

        class _StubToolCall:
            function: ToolCallFunction
            def __init__(self, name, arguments):
                self.function = _StubFunction(name, arguments)
    """
    function: ToolCallFunction


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when a tool call fails orchestrator validation."""


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_tool_call(
    tool_call: ToolCallLike,
    *,
    allowed_tools: set[str],
    required_args: Mapping[str, type | tuple[type, ...]],
) -> tuple[str, dict[str, Any]]:
    """
    Validate a tool call object from the model response.

    This function enforces a minimal argument contract.
    It is useful infrastructure, but it should not be treated as a
    complete security boundary.

    Current checks (in order):
      1. Tool name is in the caller-supplied allowlist.
      2. Arguments field is valid JSON.
      3. Parsed value is a JSON object (dict), not a list or scalar.
      4. No unexpected keys are present.
      5. All required keys declared by the caller are present.
      6. Each required key holds a value of the declared type.

    This is intentionally minimal validation.
    It does not yet enforce numeric ranges or validate against a formal
    schema. Later lessons can tighten those rules as the system grows.

    Parameters
    ----------
    tool_call : ToolCallLike
        Any object with a .function.name (str) and .function.arguments
        (str) attribute. The real OpenAI ChatCompletionMessageToolCall
        satisfies this structurally. Concrete test stubs work without
        any cast provided their function attribute is annotated as
        ToolCallFunction.
    allowed_tools : set[str]
        Names the orchestrator is willing to dispatch. Anything else is
        rejected before arguments are even parsed.
    required_args : dict[str, type | tuple[type, ...]]
        Maps each required argument name to its expected Python type or
        tuple of accepted types.
        Example: ``{"a": (int, float), "b": (int, float)}``

    Returns
    -------
    tuple[str, dict[str, Any]]
        ``(tool_name, parsed_args)`` if all checks pass.

    Raises
    ------
    ValidationError
        With a descriptive message identifying which check failed.
    """
    tool_name = tool_call.function.name

    # 1. Allowlist
    if tool_name not in allowed_tools:
        raise ValidationError(
            f"Unknown tool name '{tool_name}'. "
            f"Allowed tools: {sorted(allowed_tools)}"
        )

    # 2. JSON parse
    raw_args = tool_call.function.arguments
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"Tool arguments are not valid JSON: {exc}"
        ) from exc

    # 3. Must be a JSON object
    if not isinstance(args, dict):
        raise ValidationError(
            f"Tool arguments must be a JSON object. Got: {type(args).__name__}"
        )

    # 4. Reject unexpected keys
    unexpected_keys = set(args) - set(required_args)
    if unexpected_keys:
        raise ValidationError(
            f"Unexpected argument(s) in tool call: {sorted(unexpected_keys)}"
        )

    # 5 & 6. Required keys and types
    for key, expected_type in required_args.items():
        if key not in args:
            raise ValidationError(
                f"Missing required argument '{key}' in tool call arguments."
            )
        if not isinstance(args[key], expected_type):
            raise ValidationError(
                f"Argument '{key}' must be {expected_type}. "
                f"Got {type(args[key]).__name__}: {args[key]!r}"
            )

    return tool_name, args