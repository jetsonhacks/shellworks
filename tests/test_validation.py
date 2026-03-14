"""
tests/test_validation.py

Unit tests for shellworks.orchestrator.validation.

These tests do not require a running LLM server.

Approach
--------
validate_tool_call() accepts any object satisfying the ToolCallLike
Protocol, which requires:

    tool_call.function.name        str
    tool_call.function.arguments   str  (raw JSON)

We define two small concrete stub classes — _StubFunction and
_StubToolCall — whose attributes are declared explicitly.

The key annotation: _StubToolCall.function is declared as
ToolCallFunction (the Protocol type from validation.py). This is
required to satisfy Pyright's invariance check on mutable Protocol
attributes. Without the explicit annotation, Pyright infers the type
as _StubFunction and rejects the assignment because mutable attributes
must match exactly, not just structurally.
"""

from __future__ import annotations

import json

import pytest

from shellworks.orchestrator.validation import (
    ToolCallFunction,
    ValidationError,
    validate_tool_call,
)


# ---------------------------------------------------------------------------
# Minimal stubs that satisfy ToolCallLike
# ---------------------------------------------------------------------------

class _StubFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    function: ToolCallFunction  # explicit annotation — required for Pyright invariance check

    def __init__(self, name: str, arguments: str) -> None:
        self.function = _StubFunction(name, arguments)
        self.id = "test-id-001"


def _make_tool_call(name: str, arguments: str) -> _StubToolCall:
    """Build a minimal tool call stand-in."""
    return _StubToolCall(name, arguments)


def _args(**kwargs: object) -> str:
    return json.dumps(kwargs)


# Contract used across most tests.
ALLOWED = {"add_numbers"}
REQUIRED = {"a": (int, float), "b": (int, float)}


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

class TestValidateToolCallSuccess:
    def test_integer_args(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=3, b=4))
        name, args = validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)
        assert name == "add_numbers"
        assert args == {"a": 3, "b": 4}

    def test_float_args(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=1.5, b=-2.5))
        name, args = validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)
        assert args["a"] == 1.5
        assert args["b"] == -2.5

    def test_negative_operand_subtraction(self) -> None:
        """Subtraction is addition with a negative second operand."""
        tc = _make_tool_call("add_numbers", _args(a=100, b=-37))
        _, args = validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)
        assert args == {"a": 100, "b": -37}


# ---------------------------------------------------------------------------
# Check 1: Allowlist
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_unknown_tool_name(self) -> None:
        tc = _make_tool_call("multiply_numbers", _args(a=3, b=4))
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_empty_allowlist_rejects_everything(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=1, b=2))
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc, allowed_tools=set(), required_args=REQUIRED)

    def test_tool_name_case_sensitive(self) -> None:
        tc = _make_tool_call("Add_Numbers", _args(a=1, b=2))
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)


# ---------------------------------------------------------------------------
# Check 2: JSON parse
# ---------------------------------------------------------------------------

class TestJsonParse:
    def test_malformed_json(self) -> None:
        tc = _make_tool_call("add_numbers", "{a: 1, b: 2}")
        with pytest.raises(ValidationError, match="not valid JSON"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_empty_string_arguments(self) -> None:
        tc = _make_tool_call("add_numbers", "")
        with pytest.raises(ValidationError, match="not valid JSON"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_truncated_json(self) -> None:
        tc = _make_tool_call("add_numbers", '{"a": 1,')
        with pytest.raises(ValidationError, match="not valid JSON"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)


# ---------------------------------------------------------------------------
# Check 3: Must be a JSON object
# ---------------------------------------------------------------------------

class TestMustBeJsonObject:
    def test_json_array_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", "[1, 2]")
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_json_scalar_string_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", '"hello"')
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_json_null_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", "null")
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_json_number_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", "42")
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)


# ---------------------------------------------------------------------------
# Check 4: No unexpected keys
# ---------------------------------------------------------------------------

class TestNoUnexpectedKeys:
    def test_extra_key_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=1, b=2, c=3))
        with pytest.raises(ValidationError, match="Unexpected argument"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_multiple_extra_keys_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=1, b=2, x=0, y=0))
        with pytest.raises(ValidationError, match="Unexpected argument"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)


# ---------------------------------------------------------------------------
# Check 5: Required keys present
# ---------------------------------------------------------------------------

class TestRequiredKeysPresent:
    def test_missing_first_arg(self) -> None:
        tc = _make_tool_call("add_numbers", _args(b=2))
        with pytest.raises(ValidationError, match="Missing required argument 'a'"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_missing_second_arg(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=1))
        with pytest.raises(ValidationError, match="Missing required argument 'b'"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_empty_object_missing_both(self) -> None:
        tc = _make_tool_call("add_numbers", "{}")
        with pytest.raises(ValidationError, match="Missing required argument"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)


# ---------------------------------------------------------------------------
# Check 6: Type validation
# ---------------------------------------------------------------------------

class TestTypeValidation:
    def test_string_value_for_a_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a="three", b=2))
        with pytest.raises(ValidationError, match="Argument 'a' must be"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_string_value_for_b_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=1, b="four"))
        with pytest.raises(ValidationError, match="Argument 'b' must be"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_null_value_rejected(self) -> None:
        tc = _make_tool_call("add_numbers", _args(a=None, b=2))
        with pytest.raises(ValidationError, match="Argument 'a' must be"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)

    def test_boolean_documents_current_behavior(self) -> None:
        # JSON booleans are Python bool, which is a subclass of int.
        # True and False pass isinstance(x, int), so they pass type
        # validation today. Document this explicitly rather than relying
        # on it silently. If a future lesson tightens this, update here.
        tc = _make_tool_call("add_numbers", _args(a=True, b=2))
        name, args = validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)
        assert args["a"] is True  # behavior documented, not necessarily desired


# ---------------------------------------------------------------------------
# Check ordering: allowlist fires before JSON parse
# ---------------------------------------------------------------------------

class TestCheckOrdering:
    def test_allowlist_before_json_parse(self) -> None:
        """An unknown tool name should be rejected before JSON is even parsed."""
        tc = _make_tool_call("evil_tool", "not-valid-json-at-all")
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc, allowed_tools=ALLOWED, required_args=REQUIRED)