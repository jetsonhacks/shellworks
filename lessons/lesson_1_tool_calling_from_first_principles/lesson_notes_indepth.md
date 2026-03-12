## Lesson 1 Notes

### Expanded version

This lesson introduces the simplest useful pattern in an agentic terminal system: the model can **propose** using a tool, but the orchestrator is the component that **decides, validates, and executes** that tool call.

That boundary matters. The model is good at interpreting language and choosing an action that seems appropriate. But the model does not directly control the machine. In Shellworks, the orchestrator remains in charge. It receives the model output, checks whether the requested tool call is allowed and well-formed, executes the tool in ordinary Python, and then returns the tool result back into the conversation.

This is the core handshake of the lesson:

1. the user asks for something
2. the model sees the available tool descriptions
3. the model proposes a tool call
4. the orchestrator validates that proposal
5. the orchestrator runs the tool
6. the result is sent back to the model for the final response

The important idea is that these are not the same kind of work. The model performs probabilistic inference over text. The orchestrator performs deterministic system logic. Lesson 1 is meant to make that separation visible.

A second point in this lesson is that the behavior of the model is part of the system design. The demo model used here, **Nemotron 3**, defaults to **reasoning mode**. That can be useful in some situations, but it is not the best fit for an introductory tool-calling lesson. In this lesson, we turn reasoning off so the model behaves more like a clean tool-selection component. That keeps the exchange easier to understand: the model reads the tool description, proposes a tool call, and leaves execution to the orchestrator. This is not a claim that reasoning is bad. It is a design choice for clarity. For teaching the basic tool-calling contract, a simpler non-reasoning response makes the boundary between model behavior and orchestrator control much easier to see.

Another thing to notice is that the tool itself is intentionally small. The arithmetic function is not the interesting part. The important part is the path the request takes through the system. In a more advanced lesson, the tool might access files, run shell commands, or query outside services. Here we keep the tool simple so the orchestration pattern stays in focus.

Validation is also part of the lesson. The orchestrator does not execute whatever text the model emits. It checks that the tool name is allowed, that the arguments are valid JSON, and that the expected fields are present with the expected types. This is an early example of a larger Shellworks principle: the model may suggest actions, but the system must enforce policy.

So the main takeaway from Lesson 1 is straightforward: **tool calling is not the model “doing” the work directly.** Tool calling is the model participating in a controlled protocol. The orchestrator remains the trusted component that decides what actually happens on the machine.

