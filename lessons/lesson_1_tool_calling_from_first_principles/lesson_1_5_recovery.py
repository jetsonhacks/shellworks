"""
directory: src/shellworks/orchestrator/
file:      lesson_1_5_recovery.py

Lesson 1.5: Agentic Recovery

Extends the Lesson 1 pattern with a retry loop. When the model proposes
a malformed tool call, the orchestrator feeds the validation error back
as a tool result rather than terminating. The model can then correct its
arguments and try again.

Recovery contract:
  - Errors are returned via the tool role, not the user role.
  - The tool_call_id is preserved so the transcript stays structurally valid.
  - The orchestrator still controls the retry ceiling (max_retries).
  - A direct answer at any point short-circuits the loop immediately.
"""

from __future__ import annotations

import sys
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall

from shellworks.llm.vllm_client import build_client
from shellworks.orchestrator.validation import ValidationError, validate_tool_call
from shellworks.tools.arithmetic import add_numbers

# ---------------------------------------------------------------------------
# Tool definition
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

ALLOWED_TOOLS: set[str] = {"add_numbers"}
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
    "- Use this tool ONLY when the user explicitly asks for addition.\n"
    "- Do not attempt to use any tools that are not listed here.\n"
    "\n"
    "CONSTRAINTS:\n"
    "- If you use a tool, provide ONLY the tool call. Do not include a preamble, explanation, or conversational filler.\n"
    "- If no tool is needed, respond briefly in plain text.\n"
    "\n"
    "RECOVERY:\n"
    "- If the tool returns a validation error, read it carefully.\n"
    "- Correct only the argument(s) identified in the error.\n"
    "- Reissue the tool call with the corrected arguments.\n"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(tool_call_id: str, message: str) -> dict[str, Any]:
    """
    Build a tool-role message that reports a validation error back to the model.

    Using the tool role (with the matching tool_call_id) keeps the transcript
    structurally valid. The model sees a failed tool result rather than an
    out-of-band user interjection, which produces more reliable self-correction.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": f"[validation error] {message}",
    }


def _orchestrator_error(message: str) -> None:
    print(f"[orchestrator error] {message}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_turn(user_input: str, max_retries: int = 2, debug: bool = False) -> None:
    """
    Execute one user turn with agentic recovery on validation failure.

    Parameters
    ----------
    user_input : str
        The user's message.
    max_retries : int
        Maximum number of correction attempts after the first try.
    debug : bool
        Emit request/response detail to stderr.
    """
    client, model = build_client()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for attempt in range(max_retries + 1):
        if debug:
            print(f"[debug] Attempt {attempt + 1} of {max_retries + 1}", file=sys.stderr)

        # ------------------------------------------------------------------
        # Pass 1: Get proposal from model
        # ------------------------------------------------------------------
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[TOOL_DEFINITION],
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        # ------------------------------------------------------------------
        # Direct answer — no tool needed
        # ------------------------------------------------------------------
        if not tool_calls:
            content = (assistant_message.content or "").strip()
            if content:
                print(content)
            else:
                _orchestrator_error("Model returned an empty response with no tool call.")
            return

        # ------------------------------------------------------------------
        # Validation gate
        # ------------------------------------------------------------------
        tool_call: ChatCompletionMessageToolCall = tool_calls[0]

        try:
            tool_name, args = validate_tool_call(
                tool_call,
                allowed_tools=ALLOWED_TOOLS,
                required_args=ADD_NUMBERS_ARGS,
            )
        except ValidationError as exc:
            print(f"[recovery] Attempt {attempt + 1} failed validation: {exc}", file=sys.stderr)

            if attempt == max_retries:
                _orchestrator_error("Max retries reached. Stopping.")
                return

            # Feed the error back via the tool role so the transcript stays valid.
            messages.append(assistant_message)
            messages.append(_error_result(tool_call.id, str(exc)))
            print("[recovery] Error returned to model. Retrying...", file=sys.stderr)
            continue

        # ------------------------------------------------------------------
        # Execute the tool
        # ------------------------------------------------------------------
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        print(f"[tool call] {tool_name}({arg_str})")

        try:
            result: str = add_numbers(**args)
        except Exception as exc:  # noqa: BLE001
            _orchestrator_error(f"Tool execution failed: {exc}")
            return

        if debug:
            print(f"[debug] Tool result: {result!r}", file=sys.stderr)

        # ------------------------------------------------------------------
        # Pass 2: Return result to model for final answer
        # ------------------------------------------------------------------
        messages.append(assistant_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        final_content = (final_response.choices[0].message.content or "").strip()
        if final_content:
            print(final_content)
        else:
            _orchestrator_error("Model returned an empty final response.")
        return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lesson 1.5: Agentic Recovery")
    parser.add_argument("prompt", nargs="+", help="User prompt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parsed = parser.parse_args()

    run_turn(" ".join(parsed.prompt), max_retries=parsed.retries, debug=parsed.debug)