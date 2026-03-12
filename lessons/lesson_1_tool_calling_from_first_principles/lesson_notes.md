### Lesson 1

Lesson 1 introduces the basic tool-calling contract in Shellworks. The model can propose using a tool, but the orchestrator is the component that validates the proposal, executes the tool, and decides what happens next. That separation is the key idea of the lesson.

The flow is simple. The user asks a question. The model sees the available tool descriptions and may return a tool call. The orchestrator checks that the tool call is allowed and well-formed, runs the tool in ordinary Python, and then sends the result back to the model so it can produce a final answer. In other words: **the model proposes; the orchestrator decides and acts.**

This lesson also includes a practical note about model configuration. The demo model, **Nemotron 3**, defaults to **reasoning mode**. That behavior can be useful in some settings, but it is not ideal for a first tool-calling demo. Here we turn reasoning off so the interaction stays as clear as possible. We want the model to behave like a clean tool-selection component: read the tool description, propose a tool call, and leave execution to the orchestrator. That choice is about clarity, not about declaring reasoning good or bad. It shows an important first-principles idea: model defaults affect system behavior, so configuration is part of the design.

The arithmetic tool in this lesson is intentionally simple. The point is not the math. The point is to make the handshake between the model and the orchestrator visible. The model handles language and selection. The orchestrator handles validation and execution. That is the foundation for everything that comes later.
