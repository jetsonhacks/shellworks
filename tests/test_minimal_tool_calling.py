"""
directory: tests/
file:      test_minimal_tool_calling.py

Unit tests for orchestrator validation and tool dispatch.

These tests do NOT require a running vLLM server. They test the deterministic
parts of the system — validation, dispatch, and error handling — by working
directly with the orchestrator functions and the tool itself.

Test cases (from the implementation note):
  1. valid tool call runs correctly
  2. unknown tool name is rejected
  3. malformed JSON is rejected
  4. multiple tool calls are rejected
  5. tool exception becomes ERROR: ...
  (plus extras for completeness)
"""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import the units under test
# ---------------------------------------------------------------------------
from shellworks.orchestrator.minimal_tool_calling import (
    ALLOWED_TOOLS,
    ValidationError,
    _orchestrator_error,
    validate_tool_call,
    run_turn,
)
from shellworks.tools.arithmetic import add_numbers


# ---------------------------------------------------------------------------
# Helpers for building mock tool call objects
# ---------------------------------------------------------------------------

def make_tool_call(name: str, arguments: str | dict, call_id: str = "call_test_001"):
    """Build a minimal mock tool call that matches the OpenAI SDK shape."""
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)

    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


# ---------------------------------------------------------------------------
# 1. Valid tool call runs correctly
# ---------------------------------------------------------------------------

class TestValidateToolCallValid:
    def test_integer_args(self):
        tc = make_tool_call("add_numbers", {"a": 2, "b": 3})
        name, args = validate_tool_call(tc)
        assert name == "add_numbers"
        assert args["a"] == 2
        assert args["b"] == 3

    def test_float_args(self):
        tc = make_tool_call("add_numbers", {"a": 1.5, "b": 2.5})
        name, args = validate_tool_call(tc)
        assert args["a"] == 1.5
        assert args["b"] == 2.5

    def test_negative_args(self):
        tc = make_tool_call("add_numbers", {"a": -10, "b": 5})
        name, args = validate_tool_call(tc)
        assert args["a"] == -10
        assert args["b"] == 5

    def test_zero_args(self):
        tc = make_tool_call("add_numbers", {"a": 0, "b": 0})
        name, args = validate_tool_call(tc)
        assert args["a"] == 0


# ---------------------------------------------------------------------------
# 2. Unknown tool name is rejected
# ---------------------------------------------------------------------------

class TestValidateToolCallUnknownName:
    def test_unknown_name_raises(self):
        tc = make_tool_call("delete_files", {"a": 1, "b": 2})
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc)

    def test_empty_name_raises(self):
        tc = make_tool_call("", {"a": 1, "b": 2})
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc)

    def test_similar_name_raises(self):
        # 'add_number' (missing 's') must not pass
        tc = make_tool_call("add_number", {"a": 1, "b": 2})
        with pytest.raises(ValidationError, match="Unknown tool name"):
            validate_tool_call(tc)


# ---------------------------------------------------------------------------
# 3. Malformed JSON is rejected
# ---------------------------------------------------------------------------

class TestValidateToolCallMalformedJSON:
    def test_invalid_json_raises(self):
        tc = make_tool_call("add_numbers", "{not valid json}")
        with pytest.raises(ValidationError, match="not valid JSON"):
            validate_tool_call(tc)

    def test_json_array_raises(self):
        # Valid JSON but not an object
        tc = make_tool_call("add_numbers", "[1, 2]")
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_tool_call(tc)

    def test_json_null_raises(self):
        tc = make_tool_call("add_numbers", "null")
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_tool_call(tc)

    def test_empty_string_raises(self):
        tc = make_tool_call("add_numbers", "")
        with pytest.raises(ValidationError, match="not valid JSON"):
            validate_tool_call(tc)


# ---------------------------------------------------------------------------
# 4. Multiple tool calls are rejected
# ---------------------------------------------------------------------------

class TestMultipleToolCalls:
    """
    The run_turn function rejects multiple tool calls before validation.
    We test this by providing a mock client that returns two tool calls.
    """

    def _make_response_with_n_tool_calls(self, n: int):
        """Build a mock first_response containing n tool calls."""
        tool_calls = [
            make_tool_call("add_numbers", {"a": i, "b": i}, f"call_{i}")
            for i in range(n)
        ]
        msg = MagicMock()
        msg.content = None
        msg.tool_calls = tool_calls

        choice = MagicMock()
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        return response

    def test_two_tool_calls_print_error(self, capsys):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            self._make_response_with_n_tool_calls(2)
        )

        with patch("shellworks.orchestrator.minimal_tool_calling.build_client", return_value=mock_client):
            run_turn("add 1 and 2 and also 3 and 4")

        captured = capsys.readouterr()
        assert "[orchestrator error]" in captured.err
        assert "2 tool calls" in captured.err

    def test_one_tool_call_does_not_error_for_count(self, capsys):
        """Sanity check: a single tool call does NOT trigger the multi-call error."""
        single = self._make_response_with_n_tool_calls(1)

        # The second pass needs a mock too; we don't care about its output here
        final_msg = MagicMock()
        final_msg.content = "The answer is 5."
        final_msg.tool_calls = []
        final_choice = MagicMock()
        final_choice.message = final_msg
        final_response = MagicMock()
        final_response.choices = [final_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [single, final_response]

        with patch("shellworks.orchestrator.minimal_tool_calling.build_client", return_value=mock_client):
            run_turn("what is 2 plus 3")

        captured = capsys.readouterr()
        # No multi-call error
        assert "2 tool calls" not in captured.err


# ---------------------------------------------------------------------------
# 5. Tool exception becomes ERROR: ...
# ---------------------------------------------------------------------------

class TestToolExecutionException:
    """
    If the tool raises an exception after passing validation, the result
    injected into the conversation should be 'ERROR: <message>'.
    """

    def test_tool_exception_becomes_error_string(self, capsys):
        # Patch add_numbers to raise
        def boom(a, b):
            raise ZeroDivisionError("division by zero")

        # Build a mock first response with a valid single tool call
        tc = make_tool_call("add_numbers", {"a": 0, "b": 0})
        first_msg = MagicMock()
        first_msg.content = None
        first_msg.tool_calls = [tc]
        first_choice = MagicMock()
        first_choice.message = first_msg
        first_response = MagicMock()
        first_response.choices = [first_choice]

        # Second response (model reads the ERROR: result)
        final_msg = MagicMock()
        final_msg.content = "There was an error computing your result."
        final_msg.tool_calls = []
        final_choice = MagicMock()
        final_choice.message = final_msg
        final_response = MagicMock()
        final_response.choices = [final_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [first_response, final_response]

        with patch("shellworks.orchestrator.minimal_tool_calling.build_client", return_value=mock_client):
            with patch.dict("shellworks.orchestrator.minimal_tool_calling.TOOL_DISPATCH", {"add_numbers": boom}):
                run_turn("what is 0 divided by 0")

        # The second call to the model should have received an ERROR: result
        second_call_args = mock_client.chat.completions.create.call_args_list[1]
        messages_sent = second_call_args.kwargs["messages"]
        tool_result_message = next(
            m for m in messages_sent if isinstance(m, dict) and m.get("role") == "tool"
        )
        assert tool_result_message["content"].startswith("ERROR:")


# ---------------------------------------------------------------------------
# Arithmetic tool unit tests
# ---------------------------------------------------------------------------

class TestAddNumbers:
    def test_adds_integers(self):
        assert add_numbers(2, 3) == "5"

    def test_adds_floats(self):
        assert add_numbers(1.5, 2.5) == "4.0"

    def test_adds_negative(self):
        assert add_numbers(-1, 1) == "0"

    def test_returns_string(self):
        result = add_numbers(10, 20)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Allowlist integrity
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_allowlist_contains_only_add_numbers(self):
        assert ALLOWED_TOOLS == {"add_numbers"}

    def test_allowlist_is_a_set(self):
        assert isinstance(ALLOWED_TOOLS, set)


# ---------------------------------------------------------------------------
# Missing required argument
# ---------------------------------------------------------------------------

class TestMissingArguments:
    def test_missing_a(self):
        tc = make_tool_call("add_numbers", {"b": 3})
        with pytest.raises(ValidationError, match="Missing required argument 'a'"):
            validate_tool_call(tc)

    def test_missing_b(self):
        tc = make_tool_call("add_numbers", {"a": 3})
        with pytest.raises(ValidationError, match="Missing required argument 'b'"):
            validate_tool_call(tc)

    def test_missing_both(self):
        tc = make_tool_call("add_numbers", {})
        with pytest.raises(ValidationError, match="Missing required argument"):
            validate_tool_call(tc)


# ---------------------------------------------------------------------------
# Wrong argument types
# ---------------------------------------------------------------------------

class TestWrongArgumentTypes:
    def test_string_a_raises(self):
        tc = make_tool_call("add_numbers", {"a": "two", "b": 3})
        with pytest.raises(ValidationError, match="must be a number"):
            validate_tool_call(tc)

    def test_list_b_raises(self):
        tc = make_tool_call("add_numbers", {"a": 1, "b": [3]})
        with pytest.raises(ValidationError, match="must be a number"):
            validate_tool_call(tc)

    def test_null_b_raises(self):
        tc = make_tool_call("add_numbers", {"a": 1, "b": None})
        with pytest.raises(ValidationError, match="must be a number"):
            validate_tool_call(tc)
