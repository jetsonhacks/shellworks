# shellworks

Shellworks is a minimal local terminal assistant for learning LLM tool calling from first principles.

The project is intentionally small. It is designed to make the core architecture visible: a local model can propose a tool call, but the orchestrator validates the request, executes the tool, and remains in control of the application flow.

## Status

This repository is a work in progress.

The current implementation focuses on **Lesson 1**: a minimal end-to-end tool-calling loop in the terminal.

## Features

- Local terminal interface
- OpenAI-compatible local model server support
- Endpoint-based configuration
- Minimal orchestrator with validation
- Simple deterministic tool execution
- Session-level reasoning mode toggle

## Repository Layout

```text
.
├── configs/        # endpoint and model configuration
├── lessons/        # lesson notes and supporting material
├── scripts/        # helper scripts for local model workflows
├── src/shellworks/ # application source
├── tests/          # test suite
├── env.example     # example environment file
├── pyproject.toml  # project metadata and entry point
└── README.md
```

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- A local OpenAI-compatible model server

## Setup

Clone the repository:

```bash
git clone https://github.com/jetsonhacks/shellworks.git
cd shellworks
```

Create your environment file:

```bash
cp env.example .env
```

Install dependencies:

```bash
uv sync
```

## Configuration

Shellworks uses endpoint configuration files in:

```text
configs/endpoints/
```

The default endpoint is:

```text
local_primary
```

which maps to:

```text
configs/endpoints/local_primary.toml
```

Set environment variables in `.env` as needed. A typical local setup looks like:

```env
SHELLWORKS_ENDPOINT=local_primary
LOCAL_LLM_API_KEY=not-needed
```

Update the endpoint config so it matches your local server:

- `base_url`
- `default_model`
- `api_key_env`

## Run

Start the application with:

```bash
uv run shellworks
```

Run with debug output:

```bash
uv run shellworks --debug
```

## Commands

- `/help`
- `/reasoning on`
- `/reasoning off`
- `/status`
- `/exit`

## Testing

Run the test suite with:

```bash
uv run pytest
```

## Design Goal

Shellworks is intended to be understandable, inspectable, and easy to extend.

The model may propose actions. The orchestrator decides what is allowed.

## License

MIT

## Initial Release
### March, 2026
* Section 1  - LLM tool calling through an orchestrator
* Test on NVIDIA Jetson Orin Nano and NVIDIA Jetson AGX Thor
* Tested with Nemotron 3 Nano 30B A38 NVFP4 on Thor
* Tested with Qwen3.5-4B-GGUF on Jetson Orin Nano
