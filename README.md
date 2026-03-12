# shellworks

Shellworks is the starting point for a minimal terminal assistant that can grow into a more capable local agent environment.

This repository is a **work in progress**. The initial goal is to build a small, understandable foundation: a local assistant in the terminal with clear boundaries, simple tools, and room to evolve over time.

Lesson 1 has now been added. It presents tool calling from first principles through a minimal local terminal assistant. The lesson is designed to make the core architecture visible: the language model can propose a tool call, but the orchestrator remains the control authority that validates inputs, dispatches tools, and manages the conversation state.

The current repository focuses on a deliberately small implementation built around:
- a terminal-based interface
- a minimal orchestrator
- a local OpenAI-compatible vLLM server
- basic tool calling and validation
- clear separation between probabilistic model behavior and deterministic program control

The longer-term goal is to grow this foundation into a more capable local agent environment while keeping the system understandable and inspectable.

More documentation, examples, lessons, and code will be added as the project takes shape.
