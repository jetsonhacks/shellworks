# shellworks — Lesson 1: Tool Calling from First Principles

This is the smallest complete demonstration of tool calling in a terminal-based
Python program using a local language model served by vLLM.

---

## System boundaries

```
User  ──stdin/stdout──►  Orchestrator (Python)
                              │
                         HTTP (OpenAI-compatible API)
                              │
                         vLLM Server
                              │
                         Local Model
```

Inside the orchestrator process, the tool implementation also lives:

```
Orchestrator process
└── add_numbers(a, b)   ← ordinary Python, called only by the orchestrator
```

The model **proposes** tool calls. The orchestrator **decides and acts**.

---

## Requirements

- Python 3.10+
- A running [vLLM](https://docs.vllm.ai/) server exposing an OpenAI-compatible API
- A chat-capable model loaded by that server (with a chat template that supports tool calling)

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .        # installs the shellworks package in editable mode
```

---

## Starting the vLLM server

```bash
# Example — adjust model path and port to match your setup
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --chat-template <path-to-tool-calling-template.jinja>
```

> **Chat template note**: If the model does not call tools as expected, the
> most likely cause is a chat template that silently drops the `tools` key.
> Use a Jinja2 template that explicitly handles the `tools` parameter.
> See `lessons/lesson_1_tool_calling_from_first_principles/minimal_tool_calling_addendum.md`.

---

## Configuration

Override defaults with environment variables before running:

| Variable        | Default                      | Description           |
|-----------------|------------------------------|-----------------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1`   | vLLM server base URL  |
| `VLLM_API_KEY`  | `not-needed`                 | API key (if required) |
| `VLLM_MODEL`    | `local-model`                | Model name            |

Example:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

---

## Running the demo

```bash
python -m shellworks.main
```

With debug output (shows raw request/response JSON on stderr):

```bash
python -m shellworks.main --debug
```

Example session:

```
shellworks — Lesson 1: Tool Calling from First Principles
Type your request and press Enter. Press Ctrl-C or Ctrl-D to quit.

You: What is 2 plus 3?
The sum of 2 and 3 is 5.

You: What is the capital of France?
The capital of France is Paris.
```

---

## Running the tests

The tests do **not** require a running vLLM server. They exercise the
deterministic orchestrator logic only.

```bash
pytest tests/
```

---

## Project layout

```
shellworks/                                          ← project root
├── README.md                                        ← this file
├── requirements.txt                                 ← runtime + test deps
├── pyproject.toml                                   ← package build config
│
├── lessons/                                         ← lesson documents (read-only reference)
│   └── lesson_1_tool_calling_from_first_principles/
│       ├── minimal_tool_calling_sdd.md
│       ├── minimal_tool_calling_addendum.md
│       └── minimal_tool_calling_implementation_note_shellworks.md
│
├── src/                                             ← all runnable source code lives here
│   └── shellworks/                                  ← the installable Python package
│       ├── __init__.py
│       ├── main.py                                  ← entry point; start here to run the demo
│       │
│       ├── llm/                                     ← vLLM client configuration
│       │   ├── __init__.py
│       │   └── vllm_client.py                       ← builds the OpenAI-compatible client
│       │
│       ├── orchestrator/                            ← control loop; the main lesson file
│       │   ├── __init__.py
│       │   └── minimal_tool_calling.py              ← READ THIS to understand the whole lesson
│       │
│       └── tools/                                   ← tool implementations (pure Python)
│           ├── __init__.py
│           └── arithmetic.py                        ← add_numbers(a, b) — the one lesson tool
│
└── tests/                                           ← pytest unit tests (no vLLM needed)
    └── test_minimal_tool_calling.py
```

### File-by-file directory reference

| File | Directory | Purpose |
|------|-----------|---------|
| `main.py` | `src/shellworks/` | Entry point. Thin wrapper that starts the REPL and calls `run_turn`. |
| `vllm_client.py` | `src/shellworks/llm/` | Reads `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL` from env and constructs the OpenAI client. |
| `minimal_tool_calling.py` | `src/shellworks/orchestrator/` | The full control loop: request construction → vLLM → inspect → validate → execute → inject result → final answer. |
| `arithmetic.py` | `src/shellworks/tools/` | Contains `add_numbers(a, b) → str`. Ordinary Python. Called only by the orchestrator. |
| `test_minimal_tool_calling.py` | `tests/` | 26 unit tests covering validation, dispatch, error handling, and the arithmetic tool. No server required. |
| `requirements.txt` | `shellworks/` (root) | `openai>=1.0` and `pytest>=7.0`. |
| `pyproject.toml` | `shellworks/` (root) | Declares the `shellworks` package under `src/`. |

**The key file to read is `src/shellworks/orchestrator/minimal_tool_calling.py`.**
It contains the full 10-step control loop with inline comments explaining every
responsibility boundary.

---

## What this lesson is NOT

- A generic agent framework
- A multi-tool system
- A multi-provider system
- Production-ready code

It is the **smallest working demonstration** of the architectural principle:
> The model proposes. The orchestrator decides and acts.
