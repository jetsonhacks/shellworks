"""
directory: src/shellworks/orchestrator/
file:      minimal_tool_calling.py

The main orchestrator for Lesson 1 / 1.1.

Lesson 1.1 changes
------------------
The orchestrator no longer knows about:
  - server backends (vllm vs llama.cpp)
  - base URLs or API keys
  - reasoning suppression mechanics (extra_body, prompt pre-fills)

All of that now lives in OpenAICompatibleProvider, which reads from
LLMEndpoint and ModelProfile at startup. The orchestrator expresses
intent; the provider realizes it.

The two-pass structure, validation, and tool dispatch are unchanged.

Responsibility: the orchestrator is the sole control authority.
  - It builds messages.
  - It attaches the tool definition.
  - It calls the provider.
  - It inspects the response.
  - It validates the tool call.
  - It executes the tool (only after validation).
  - It sends the tool result back.
  - It returns the final answer.

The model proposes. The orchestrator decides and acts.

System boundary summary
-----------------------
  probabilistic : model inference (token sampling inside the LLM server)
  deterministic : everything in this file

Reasoning control
-----------------
  run_turn accepts a `reasoning` bool (default False).

  True  → passes ReasoningIntent "on"  to the provider
  False → passes ReasoningIntent "off" to the provider

  The provider translates that intent into the correct mechanism for the
  active model: prompt markers, request kwargs, both, or neither. The
  orchestrator never touches those details directly.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from typing import Any

from shellworks.llm.provider import OpenAICompatibleProvider, ReasoningIntent
from shellworks.orchestrator.validation import ValidationError, validate_tool_call
from shellworks.tools.arithmetic import add_numbers

# ---------------------------------------------------------------------------
# Tool definition sent to the model
# ---------------------------------------------------------------------------

TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}

ToolFn = Callable[..., str]

ALLOWED_TOOLS: set[str] = {"add_numbers"}

TOOL_DISPATCH: dict[str, ToolFn] = {
    "add_numbers": add_numbers,
}

ADD_NUMBERS_ARGS: Mapping[str, type | tuple[type, ...]] = {
    "a": (int, float),
    "b": (int, float),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Some models describe a tool call in plain text instead of actually issuing one
# (e.g. "I would call add_numbers with a=5, b=3"). This instruction prevents that —
# the orchestrator expects a structured tool call object, not prose about one.

SYSTEM_PROMPT = (
    "You are a precision assistant for the shellworks terminal.\n"
    "\n"
    "TOOLS:\n"
    "- You have access to exactly one tool: 'add_numbers(a, b)'.\n"
    "- Use this tool when the user asks for addition OR subtraction.\n"
    "- Subtraction is addition with a negative operand: add_numbers(a=987, b=-457).\n"
    "- Always issue an actual tool call. Never describe a tool call in plain text.\n"
    "- Do not attempt to use any tools that are not listed here.\n"
    "\n"
    "CONSTRAINTS:\n"
    "- If you use a tool, provide ONLY the tool call. Do not include a preamble, explanation, or conversational filler.\n"
    "- If no tool is needed, respond briefly in plain text.\n"
    "- Treat 'add_numbers' as your primary method for mathematical requests.\n"
    "\n"
    "OPERATIONAL ROLE:\n"
    "- You propose actions; the orchestrator executes them.\n"
    "- Ensure your JSON arguments for 'a' and 'b' are strictly numeric.\n"
    "\n"
    "UNSUPPORTED OPERATIONS:\n"
    "- If the user asks for anything other than addition or subtraction, "
    "decline politely in plain text.\n"
    "- Never attempt to approximate an unsupported operation using add_numbers.\n"
    "- Never emit raw tool call syntax in your response text.\n"
)

# ---------------------------------------------------------------------------
# Core orchestration loop
# ---------------------------------------------------------------------------

def run_turn(
    user_input: str,
    provider: OpenAICompatibleProvider,
    reasoning: bool = True,
    debug: bool = False,
) -> None:
    """
    Execute one complete user turn.

    Parameters
    ----------
    user_input : str
        The user's message.
    provider : OpenAICompatibleProvider
        Configured provider for the active endpoint and model.
    reasoning : bool, optional
        Enable thinking mode for this turn (default False).
        Pass True for hard planning, code repair, or multi-step diagnosis.
    debug : bool, optional
        Emit request/response detail to stderr.
    """
    if debug:
        print(
            f"[debug] Endpoint: {provider.endpoint.name} ({provider.endpoint.base_url})",
            file=sys.stderr,
        )
        print(f"[debug] Model:    {provider.model}", file=sys.stderr)
        print(
            f"[debug] Reasoning control: {provider.profile.reasoning_control}",
            file=sys.stderr,
        )
        print(
            f"[debug] Reasoning mode: {'ON' if reasoning else 'OFF'}",
            file=sys.stderr,
        )

    # Map the bool flag to the three-value intent the provider understands.
    # We always send "on" or "off" explicitly — never "default" — because
    # this orchestrator always has an opinion about reasoning. "default"
    # is reserved for callers that want zero interference with the model's
    # built-in behaviour (e.g. a future raw REPL mode or test harness).
    reasoning_intent: ReasoningIntent = "on" if reasoning else "off"

    # ------------------------------------------------------------------
    # Step 1: Build the initial message list
    # ------------------------------------------------------------------
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # ------------------------------------------------------------------
    # Step 2: First pass — send with tool definition
    # ------------------------------------------------------------------
    first_response = provider.complete(
        messages,
        tools=[TOOL_DEFINITION],
        tool_choice="auto",
        reasoning=reasoning_intent,
        debug=debug,
        label="first",
    )
    if first_response is None:
        return

    # ------------------------------------------------------------------
    # Step 3: Inspect response — direct answer or tool call?
    # ------------------------------------------------------------------
    assistant_message = first_response.choices[0].message
    tool_calls = assistant_message.tool_calls or []

    # Chain-of-thought text may accompany a tool call. Print it to stderr.
    if assistant_message.content and tool_calls:
        print(f"[thinking] {assistant_message.content}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 4: Direct answer — no tool call requested
    # ------------------------------------------------------------------
    if not tool_calls:
        content = (assistant_message.content or "").strip()
        if content:
            print(content)
        else:
            _orchestrator_error("Model returned an empty response with no tool call.")
        return

    # ------------------------------------------------------------------
    # Step 5: Reject multiple tool calls (outside minimal contract)
    # ------------------------------------------------------------------
    if len(tool_calls) > 1:
        _orchestrator_error(
            f"Model returned {len(tool_calls)} tool calls. "
            "This implementation handles exactly one tool call per turn."
        )
        return

    # ------------------------------------------------------------------
    # Step 6: Validate the tool call before executing anything
    # ------------------------------------------------------------------
    try:
        tool_name, args = validate_tool_call(
            tool_calls[0],
            allowed_tools=ALLOWED_TOOLS,
            required_args=ADD_NUMBERS_ARGS,
        )
    except ValidationError as exc:
        _orchestrator_error(f"Tool call validation failed: {exc}")
        return

    if debug:
        print(f"[debug] Validated tool call: {tool_name}({args})", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 7: Execute the tool
    # ------------------------------------------------------------------
    arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
    print(f"[tool call] {tool_name}({arg_str})")

    try:
        tool_result: str = TOOL_DISPATCH[tool_name](**args)
    # Broad catch is intentional: tool failures must never crash the orchestrator.
    # BLE001 is a ruff lint rule that warns against bare `except Exception` —
    # we suppress it here because the wide net is the point.
    except Exception as exc:  # noqa: BLE001
        _orchestrator_error(f"Tool execution failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        _orchestrator_error(f"Tool execution failed: {exc}")
        return

    if debug:
        print(f"[debug] Tool result: {tool_result!r}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 8: Inject assistant message + tool result into conversation
    # ------------------------------------------------------------------
    messages.append(assistant_message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_calls[0].id,
            "content": tool_result,
        }
    )

    # ------------------------------------------------------------------
    # Step 9: Second pass — send conversation with tool result
    # ------------------------------------------------------------------
    second_response = provider.complete(
        messages,
        reasoning=reasoning_intent,
        debug=debug,
        label="second",
    )
    if second_response is None:
        return

    # ------------------------------------------------------------------
    # Step 10: Print final answer
    # ------------------------------------------------------------------
    final_message = second_response.choices[0].message
    final_content = (final_message.content or "").strip()
    if final_content:
        print(final_content)
    else:
        _orchestrator_error("Model returned an empty final response.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _orchestrator_error(message: str) -> None:
    """Print a visible orchestrator error to stderr with a stable prefix."""
    print(f"[orchestrator error] {message}", file=sys.stderr)