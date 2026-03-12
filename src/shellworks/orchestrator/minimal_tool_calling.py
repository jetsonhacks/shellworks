"""
directory: src/shellworks/orchestrator/
file:      minimal_tool_calling.py

The main file for Lesson 1.

Read this file to understand how the whole lesson works. The supporting
modules handle infrastructure so this file stays focused:

  validation.py   — validate_tool_call(), ValidationError
  vllm_client.py  — build_client()
  arithmetic.py   — add_numbers()

Responsibility: the orchestrator is the sole control authority.
  - It builds messages.
  - It attaches the tool definition.
  - It calls vLLM.
  - It inspects the response.
  - It validates the tool call.
  - It executes the tool (only after validation).
  - It sends the tool result back.
  - It returns the final answer.

The model proposes. The orchestrator decides and acts.

System boundary summary
-----------------------
  probabilistic : model inference (token sampling inside vLLM)
  deterministic : everything in this file

Reasoning control
-----------------
  run_turn accepts a `reasoning` bool (default False).

  When False: vLLM pre-fills the assistant turn with <think></think>,
    skipping the reasoning phase at the model level — saving real
    compute, not just hiding tokens.

  When True: the model reasons before responding. Any thinking text
    appears in response content alongside tool_calls and is printed
    to stderr via the [thinking] branch in run_turn.

  The caller decides the mode. No automatic escalation here.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from openai import APIConnectionError

from shellworks.llm.vllm_client import build_client
from shellworks.orchestrator.validation import ValidationError, validate_tool_call
from shellworks.tools.arithmetic import add_numbers

# ---------------------------------------------------------------------------
# Tool definition sent to the model
# ---------------------------------------------------------------------------
# This is the *metadata* the model uses to understand what tools exist.
# It is NOT the function. The function lives in tools/arithmetic.py.

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

# Allowlist — only these tool names may be dispatched.
ALLOWED_TOOLS: set[str] = {"add_numbers"}

# Dispatch table — maps allowed names to their Python implementations.
TOOL_DISPATCH: dict[str, ToolFn] = {
    "add_numbers": add_numbers,
}

# Argument contract for validate_tool_call.
ADD_NUMBERS_ARGS: dict[str, tuple[type, ...]] = {
    "a": (int, float),
    "b": (int, float),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
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
    "UNSUPPORTED OPERATIONS:\n"
    "- If the user asks for anything other than addition or subtraction, "
    "decline politely in plain text.\n"
    "- Never attempt to approximate an unsupported operation using add_numbers.\n"
    "- Never emit raw tool call syntax in your response text.\n"
)

# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(
    client,
    model: str,
    messages: list[dict[str, Any]],
    reasoning: bool,
    debug: bool,
    label: str,
    *,
    include_tools: bool = True,
) -> Any | None:
    """
    Send one request to vLLM and return the response.

    Owns: connection error handling, reasoning kwargs, and debug output.
    Returns None and prints an orchestrator error on connection failure.

    Parameters
    ----------
    label : str
        Short name for debug output, e.g. ``"first"`` or ``"second"``.
    include_tools : bool
        Whether to advertise tools in this pass.
    """
    extra_body = {"chat_template_kwargs": {"enable_thinking": reasoning}}

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "extra_body": extra_body,
    }

    if include_tools:
        request["tools"] = [TOOL_DEFINITION]
        request["tool_choice"] = "auto"

    if debug:
        print(f"[debug] {label} request:", file=sys.stderr)
        print(
            json.dumps(
                {
                    "model": request["model"],
                    "messages": [
                        m.model_dump() if hasattr(m, "model_dump") else m
                        for m in request["messages"]
                    ],
                    "tools": request.get("tools"),
                    "tool_choice": request.get("tool_choice"),
                    "extra_body": request["extra_body"],
                },
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )

    try:
        response = client.chat.completions.create(**request)
    except APIConnectionError as exc:
        _orchestrator_error(
            f"Could not reach the vLLM server ({label} pass).\n  Detail: {exc}"
        )
        return None

    if debug:
        print(f"[debug] {label} response:", file=sys.stderr)
        print(response.model_dump_json(indent=2), file=sys.stderr)

    return response


# ---------------------------------------------------------------------------
# Core orchestration loop
# ---------------------------------------------------------------------------

def run_turn(user_input: str, reasoning: bool = False, debug: bool = False) -> None:
    """
    Execute one complete user turn.

    Parameters
    ----------
    user_input : str
        The user's message.
    reasoning : bool, optional
        Enable Nemotron reasoning mode for this turn (default False).
        Pass True for hard planning, code repair, or multi-step diagnosis.
    debug : bool, optional
        Emit request/response detail to stderr.
    """

    client, model = build_client()

    if debug:
        print(f"[debug] Reasoning mode: {'ON' if reasoning else 'OFF'}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 1: Build the initial message list
    # ------------------------------------------------------------------
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # ------------------------------------------------------------------
    # Step 2: First pass — send request with tool definition to vLLM
    # ------------------------------------------------------------------
    first_response = _call_llm(
        client,
        model,
        messages,
        reasoning,
        debug,
        "first",
        include_tools=True,
    )
    if first_response is None:
        return

    # ------------------------------------------------------------------
    # Step 3: Inspect response — direct answer or tool call?
    # ------------------------------------------------------------------
    assistant_message = first_response.choices[0].message
    tool_calls = assistant_message.tool_calls or []

    # Chain-of-thought text may accompany a tool call. Print it, but
    # treat tool_calls as the controlling output.
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
    except Exception as exc:  # noqa: BLE001
        _orchestrator_error(f"Tool execution failed: {exc}")
        return

    if debug:
        print(f"[debug] Tool result: {tool_result!r}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Step 8: Inject assistant message + tool result into conversation
    # ------------------------------------------------------------------
    # Why append *both* messages?
    #
    # The model does not call Python directly. Instead, the orchestrator
    # maintains a conversation transcript.
    #
    # Before the first request:
    #   system -> sets behavior
    #   user   -> asks the question
    #
    # After the first model pass:
    #   assistant -> proposes a tool call
    #
    # After tool execution:
    #   tool -> returns the tool result associated with that tool_call_id
    #
    # The second model pass sees this updated transcript and can turn the
    # tool result into a final natural-language answer for the user.

    messages.append(assistant_message)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_calls[0].id,
            "content": tool_result,
        }
    )

    # ------------------------------------------------------------------
    # Step 9: Second pass — send conversation with tool result to vLLM
    # ------------------------------------------------------------------
    second_response = _call_llm(
        client,
        model,
        messages,
        reasoning,
        debug,
        "second",
        include_tools=False,
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
