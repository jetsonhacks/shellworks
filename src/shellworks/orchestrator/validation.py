from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall

class ValidationError(Exception):
    """Raised when a tool call fails orchestrator validation."""


def validate_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    *,
    allowed_tools: set[str],
    required_args: Mapping[str, type | tuple[type, ...]],
) -> tuple[str, dict[str, Any]]:
    """
    Validate a tool call object from the model response.

    This function enforces a minimal argument contract for Lesson 1.
    It is useful infrastructure, but it should not be treated as a
    complete security boundary.

    Current checks (in order):
      1. Tool name is in the caller-supplied allowlist.
      2. Arguments field is valid JSON.
      3. Parsed value is a JSON object (dict), not a list or scalar.
      4. No unexpected keys are present.
      5. All required keys declared by the caller are present.
      6. Each required key holds a value of the declared type.

    This is intentionally minimal validation for Lesson 1.
    It does not yet enforce numeric ranges or validate against a formal
    schema. Later lessons can tighten those rules as the system grows.

    Parameters
    ----------
    tool_call :
        The tool call object from ``response.choices[0].message.tool_calls[n]``.
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