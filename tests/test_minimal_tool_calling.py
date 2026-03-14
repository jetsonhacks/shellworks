"""
directory: tests/
file:      test_minimal_tool_calling.py

Integration-style tests for the run_turn orchestrator.

These tests do NOT require a running LLM server. They test the
deterministic orchestrator behavior by injecting a mock provider
and observing what run_turn does with the responses it receives.

Scope
-----
This file tests run_turn() behavior only — the two-pass flow, the
responses to model outputs, and the error paths.

Validator unit tests (the six checks inside validate_tool_call) live in
test_validation.py and are not duplicated here.

Arithmetic unit tests (add_numbers) live here because they are the
only tool Lesson 1 has and do not require a mock.

Mock strategy
-------------
run_turn() takes a provider as a required argument. Tests pass a
MagicMock provider whose complete() method returns pre-built
response objects. No patching of module-level names is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from shellworks.orchestrator.minimal_tool_calling import (
    ALLOWED_TOOLS,
    run_turn,
    _orchestrator_error,
)
from shellworks.tools.arithmetic import add_numbers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, arguments: dict, call_id: str = "call_test_001") -> MagicMock:
    """Build a minimal mock tool call matching the OpenAI SDK shape."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_response(tool_calls: list | None = None, content: str | None = None) -> MagicMock:
    """Build a mock completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_provider(*responses) -> MagicMock:
    """
    Build a mock provider whose complete() returns the given responses
    in sequence, one per call.
    """
    provider = MagicMock()
    provider.endpoint.name = "mock_endpoint"
    provider.endpoint.base_url = "http://localhost:9999/v1"
    provider.model = "mock-model"
    provider.profile.reasoning_control = "none"
    provider.complete.side_effect = list(responses)
    return provider


# ---------------------------------------------------------------------------
# Arithmetic tool — no mock needed
# ---------------------------------------------------------------------------

class TestAddNumbers:
    def test_adds_integers(self) -> None:
        assert add_numbers(2, 3) == "5"

    def test_adds_floats(self) -> None:
        assert add_numbers(1.5, 2.5) == "4.0"

    def test_adds_negative(self) -> None:
        assert add_numbers(-1, 1) == "0"

    def test_subtraction_via_negative_operand(self) -> None:
        assert add_numbers(10, -3) == "7"

    def test_returns_string(self) -> None:
        assert isinstance(add_numbers(10, 20), str)


# ---------------------------------------------------------------------------
# Allowlist integrity
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_contains_only_add_numbers(self) -> None:
        assert ALLOWED_TOOLS == {"add_numbers"}

    def test_is_a_set(self) -> None:
        assert isinstance(ALLOWED_TOOLS, set)


# ---------------------------------------------------------------------------
# Step 4: Direct answer path — model returns text, no tool call
# ---------------------------------------------------------------------------

class TestDirectAnswerPath:
    def test_direct_text_answer_is_printed(self, capsys: pytest.CaptureFixture) -> None:
        provider = _make_provider(
            _make_response(content="I can only add or subtract numbers.")
        )
        run_turn("what is the capital of France?", provider=provider)
        captured = capsys.readouterr()
        assert "I can only add or subtract numbers." in captured.out

    def test_empty_direct_response_prints_orchestrator_error(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        provider = _make_provider(_make_response(content=""))
        run_turn("hello", provider=provider)
        captured = capsys.readouterr()
        assert "[orchestrator error]" in captured.err

    def test_provider_called_once_for_direct_answer(self) -> None:
        provider = _make_provider(
            _make_response(content="No tool needed.")
        )
        run_turn("hello", provider=provider)
        assert provider.complete.call_count == 1


# ---------------------------------------------------------------------------
# Step 5: Multiple tool calls rejected
# ---------------------------------------------------------------------------

class TestMultipleToolCalls:
    def test_two_tool_calls_print_orchestrator_error(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        two_calls = [
            _make_tool_call("add_numbers", {"a": 1, "b": 2}, "call_0"),
            _make_tool_call("add_numbers", {"a": 3, "b": 4}, "call_1"),
        ]
        provider = _make_provider(_make_response(tool_calls=two_calls))
        run_turn("add 1+2 and also 3+4", provider=provider)
        captured = capsys.readouterr()
        assert "[orchestrator error]" in captured.err
        assert "2 tool calls" in captured.err

    def test_two_tool_calls_make_no_second_provider_call(self) -> None:
        two_calls = [
            _make_tool_call("add_numbers", {"a": 1, "b": 2}, "call_0"),
            _make_tool_call("add_numbers", {"a": 3, "b": 4}, "call_1"),
        ]
        provider = _make_provider(_make_response(tool_calls=two_calls))
        run_turn("add 1+2 and also 3+4", provider=provider)
        assert provider.complete.call_count == 1


# ---------------------------------------------------------------------------
# Steps 6 + 7 + 8 + 9 + 10: Happy path — valid tool call, two passes
# ---------------------------------------------------------------------------

class TestHappyPath:
    def _make_happy_provider(self, final_content: str = "The answer is 5.") -> MagicMock:
        first = _make_response(
            tool_calls=[_make_tool_call("add_numbers", {"a": 2, "b": 3})]
        )
        second = _make_response(content=final_content)
        return _make_provider(first, second)

    def test_final_answer_is_printed(self, capsys: pytest.CaptureFixture) -> None:
        provider = self._make_happy_provider("The answer is 5.")
        run_turn("what is 2 plus 3?", provider=provider)
        assert "The answer is 5." in capsys.readouterr().out

    def test_tool_call_label_printed_to_stdout(self, capsys: pytest.CaptureFixture) -> None:
        provider = self._make_happy_provider()
        run_turn("what is 2 plus 3?", provider=provider)
        assert "[tool call] add_numbers" in capsys.readouterr().out

    def test_provider_called_twice(self) -> None:
        provider = self._make_happy_provider()
        run_turn("what is 2 plus 3?", provider=provider)
        assert provider.complete.call_count == 2

    def test_second_call_includes_tool_result_message(self) -> None:
        provider = self._make_happy_provider()
        run_turn("what is 2 plus 3?", provider=provider)
        second_call_messages = provider.complete.call_args_list[1][0][0]
        roles = [m.get("role") if isinstance(m, dict) else m.role
                 for m in second_call_messages]
        assert "tool" in roles

    def test_tool_result_content_is_correct_sum(self) -> None:
        provider = self._make_happy_provider()
        run_turn("what is 2 plus 3?", provider=provider)
        second_call_messages = provider.complete.call_args_list[1][0][0]
        tool_msg = next(
            m for m in second_call_messages
            if isinstance(m, dict) and m.get("role") == "tool"
        )
        assert tool_msg["content"] == "5"


# ---------------------------------------------------------------------------
# Step 7: Tool execution failure — orchestrator error, no second pass
# ---------------------------------------------------------------------------

class TestToolExecutionFailure:
    """
    When the tool itself raises an exception after passing validation,
    the orchestrator prints an error to stderr and stops. It does NOT
    make a second provider call.
    """

    def _make_provider_with_failing_tool(self) -> tuple[MagicMock, MagicMock]:
        tc = _make_tool_call("add_numbers", {"a": 1, "b": 2})
        first = _make_response(tool_calls=[tc])
        provider = _make_provider(first)
        return provider, tc

    def test_tool_failure_prints_orchestrator_error(
        self, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = self._make_provider_with_failing_tool()

        def boom(a: float, b: float) -> str:
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "shellworks.orchestrator.minimal_tool_calling.TOOL_DISPATCH",
            {"add_numbers": boom},
        )
        run_turn("add 1 and 2", provider=provider)
        captured = capsys.readouterr()
        assert "[orchestrator error]" in captured.err
        assert "Tool execution failed" in captured.err

    def test_tool_failure_makes_no_second_provider_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _ = self._make_provider_with_failing_tool()

        def boom(a: float, b: float) -> str:
            raise RuntimeError("disk full")

        monkeypatch.setattr(
            "shellworks.orchestrator.minimal_tool_calling.TOOL_DISPATCH",
            {"add_numbers": boom},
        )
        run_turn("add 1 and 2", provider=provider)
        assert provider.complete.call_count == 1


# ---------------------------------------------------------------------------
# Connection failure — provider returns None
# ---------------------------------------------------------------------------

class TestConnectionFailure:
    def test_none_first_response_is_handled_gracefully(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        provider = _make_provider(None)
        # Should not raise — connection failure is handled inside run_turn
        run_turn("add 1 and 2", provider=provider)

    def test_none_second_response_is_handled_gracefully(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        first = _make_response(
            tool_calls=[_make_tool_call("add_numbers", {"a": 1, "b": 2})]
        )
        provider = _make_provider(first, None)
        run_turn("add 1 and 2", provider=provider)
        # Should not raise