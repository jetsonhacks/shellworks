# Solution Design Document  
## Minimal Tool Calling in the Terminal Using a Local Language Model Served by vLLM

**Document ID:** SDD-LLM-TOOLCALL-001  
**Version:** 1.1  
**Status:** Draft for Review  
**Author:** JetsonHacks in collaboration with OpenAI / Claude.ai  
**Date:** March 10, 2026  

### Revision History

| Version | Date | Summary |
|---|---|---|
| 1.0 | March 10, 2026 | Initial complete SDD draft |
| 1.1 | March 10, 2026 | Added document metadata, runtime dependency guidance, system prompt guidance, `tool_choice` tradeoff note, connection-failure behavior, and sequence diagram |

## 1. Overview

This document specifies the smallest complete system that demonstrates tool calling in a terminal-based Python program using:

- a local language model
- served by vLLM
- through its OpenAI-compatible API
- with a minimal Python orchestrator
- and one simple tool

The design goal is not to build a general agent framework. The goal is to define, from first principles, the minimum architecture required for a real tool-calling loop in which a model may propose a tool call, while deterministic software remains responsible for parsing, validation, execution, and error handling.

This design is intentionally narrow. It exists to make system boundaries, responsibility boundaries, and control boundaries explicit.

Normative language is used as follows:

- **MUST** indicates a required behavior or constraint.
- **SHOULD** indicates a recommended behavior that may be relaxed with justification.
- **MAY** indicates an optional behavior.

For consistency in this document, **MUST** is used for implementation requirements and **SHALL** is used primarily for higher-level system statements, acceptance framing, and declared design constraints.

---

## 2. Problem Statement

In many descriptions of tool calling, several distinct system roles are collapsed into a single statement such as “the AI used a tool.” That wording is not precise enough for implementation, review, or extension.

A real terminal-based tool-calling system contains at least four distinct technical roles:

1. a **model** that produces text or structured output
2. a **serving layer** that exposes that model through an API
3. an **orchestrator** that controls the interaction loop
4. a **tool implementation** that executes machine-side logic

Without those distinctions, it becomes difficult to reason about:

- what is probabilistic
- what is deterministic
- what has execution authority
- what may fail
- what must be validated before machine-side action occurs

The problem this document addresses is therefore:

> How to specify the smallest real terminal-based tool-calling system such that architectural responsibilities are explicit, execution boundaries are clear, and the implementation remains small enough to understand directly.

This document solves that problem by defining a minimal end-to-end design with one tool, one orchestrator, one local model server, and one tool-calling round-trip per user turn.

---

## 3. Design Goals

The system SHALL be designed to satisfy the following goals.

### 3.1 First-principles clarity

The design MUST clearly distinguish between:

- the **model**
- the **vLLM server**
- the **orchestrator**
- the **tool**

The document MUST not rely on language that obscures these boundaries.

### 3.2 Minimal architecture

The design MUST include only the components required to demonstrate a complete tool-calling loop.

The design SHOULD avoid additional frameworks, services, or abstractions unless they are necessary to make the loop functional.

### 3.3 Deterministic control

The orchestrator MUST remain the control authority for:

- request construction
- response inspection
- parsing
- validation
- tool dispatch
- tool execution
- failure handling
- final output to the user

The model MUST NOT be treated as an execution authority.

### 3.4 Local execution

The system SHALL assume:

- a local model
- a vLLM server exposing that model
- a terminal-based Python application as the orchestrator

### 3.5 Educational transparency

The implementation SHOULD keep the control flow explicit and visible.

Hidden control abstractions SHOULD be avoided in the minimal version.

### 3.6 Verifiable minimality

The system MUST be small enough that a reviewer can identify:

- where tool definitions are sent
- where tool calls are parsed
- where validation occurs
- where the tool is executed
- where the final answer is printed

without traversing large framework code or indirect runtime layers.

---

## 4. Non-Goals

The following are explicitly out of scope for this design:

- multiple tools
- multiple model providers
- parallel tool calls
- multi-step planning
- memory systems
- persistence
- long-running jobs
- concurrency
- sandboxing
- permissions
- shell execution
- file-system mutation
- authentication and multi-user control
- retries and backoff policies
- streaming responses
- GUI or browser interfaces
- framework-level agent runtimes
- benchmarking and optimization
- production security hardening

This document specifies a minimal demonstrator, not a production agent platform.

---

## 5. Success Criteria

The implementation SHALL be considered correct with respect to this design if all of the following are true.

### 5.1 Functional success criteria

The system MUST:

1. accept a user request in a terminal
2. construct and send a chat completion request to the vLLM server
3. include exactly one tool definition in that request
4. accept either:
   - a normal assistant response, or
   - a single tool call response
5. validate the requested tool name and arguments before execution
6. execute only the allowed tool
7. send the tool result back into the conversation
8. print a final answer to the terminal

### 5.2 Safety success criteria

The system MUST:

1. reject unknown tool names
2. reject malformed tool arguments
3. avoid executing any tool if validation fails
4. avoid crashing the interaction turn due to an unhandled tool exception

### 5.3 Scope success criteria

The system MUST handle at most:

- one user request at a time
- one tool call per user turn
- one local tool
- one final assistant response per turn

Any design element beyond those boundaries is outside the intended scope of this document.

---

## 6. System Context

The system contains five principal actors.

### 6.1 User

The user is the human interacting with the terminal application. The user provides a natural-language request and receives the final printed answer.

### 6.2 Terminal application / orchestrator

The orchestrator is a local Python program and is the control authority of the system.

The orchestrator MUST:

- read user input from the terminal
- construct chat requests
- include tool definitions in requests to the model server
- inspect model responses for tool calls
- parse and validate tool arguments deterministically
- execute the tool implementation
- send tool results back into the conversation loop
- print the final answer to the terminal

### 6.3 vLLM server

The vLLM server is the serving layer for the local model. It exposes an OpenAI-compatible API surface to the orchestrator.

The vLLM server MUST be treated as a server interface and transport boundary, not as the orchestrator.

### 6.4 Model

The model is the local language model loaded by vLLM.

The model MAY produce either:

- a normal assistant response, or
- a structured tool-call request

The model MUST be treated as a probabilistic generator of candidate output.

The model MUST NOT be treated as directly executing tools.

### 6.5 Tool implementation

The tool implementation is a normal Python function located in the same process as the orchestrator.

For this design, the tool SHALL be:

```text
add_numbers(a, b)
```

The tool implementation MUST be ordinary deterministic code executed only by the orchestrator.

---

## 7. Architecture

### 7.1 Architectural view

The system architecture is intentionally small.

```text
+--------+        stdin/stdout        +------------------------+
|  User  | <------------------------> | Terminal App /         |
|        |                            | Orchestrator (Python)  |
+--------+                            +-----------+------------+
                                                  |
                                                  | HTTP request/response
                                                  | OpenAI-compatible API
                                                  v
                                     +------------+------------+
                                     | vLLM Server             |
                                     | OpenAI-compatible       |
                                     | serving layer           |
                                     +------------+------------+
                                                  |
                                                  | local inference
                                                  v
                                     +------------+------------+
                                     | Local Model             |
                                     | text / structured       |
                                     | output generation       |
                                     +-------------------------+

Inside the orchestrator process:
    +----------------------------------------------+
    | Tool implementation: add_numbers(a, b)       |
    +----------------------------------------------+
```

### 7.2 Architectural constraints

The minimal design SHALL enforce the following constraints:

- the orchestrator and tool SHALL run in the same Python process
- the vLLM server SHALL be external to that process and accessed by API
- the model SHALL be behind the vLLM server, not directly invoked by the orchestrator
- the tool SHALL be invoked only through orchestrator dispatch logic
- no concurrency SHALL be assumed

### 7.3 Responsibility boundaries

The architectural responsibilities SHALL be interpreted as follows:

- the **user** expresses intent
- the **model** proposes text or a tool call
- the **vLLM server** transports requests and responses between orchestrator and model
- the **orchestrator** decides what to send, what to accept, and what to execute
- the **tool** performs deterministic machine-side computation

### 7.4 Sequence view

The following sequence diagram shows the message flow for the minimal successful tool-calling path.

```text
User                Orchestrator              vLLM Server               Model                 Tool
 |                       |                         |                      |                     |
 |-- request ----------->|                         |                      |                     |
 |                       |-- chat request -------> |                      |                     |
 |                       |                         |-- inference request ->|                     |
 |                       |                         |<-- tool call response-|                     |
 |                       |<-- assistant tool call -|                      |                     |
 |                       |-- validate tool call    |                      |                     |
 |                       |-- execute --------------------------------------------------------->|
 |                       |<--------------------------------------------------------------- result|
 |                       |-- chat request + tool result -->|                |                     |
 |                       |                                 |-- inference request ->|              |
 |                       |                                 |<-- final answer ------|              |
 |                       |<-- final assistant response ----|                      |              |
 |<-- final answer ------|                                 |                      |              |
```

This sequence view complements the structural architecture diagram. The structural view shows component placement. The sequence view shows control flow and message order.

---

## 8. Execution Model

### 8.1 Overview

The system SHALL implement the smallest complete end-to-end tool-calling loop.

That loop consists of:

1. receiving a user request
2. sending the request and tool definition to the model server
3. receiving either a normal answer or a tool call
4. validating any tool call
5. executing the tool if valid
6. sending the tool result back to the model
7. printing the final answer

### 8.2 Detailed control loop

#### Step 1: User request

The user enters a request in the terminal.

Example:

```text
What is 2 plus 3?
```

#### Step 2: Request construction

The orchestrator constructs a chat request containing:

- a system message
- a user message
- one tool definition
- a tool choice policy that permits tool selection

The orchestrator MUST be the component that constructs this request.

The orchestrator SHOULD provide a system message that clearly instructs the model to use only the provided tool when appropriate and not invent additional tools. The exact wording MAY vary by model family, but the intent of the system message SHOULD remain stable across implementations.

#### Step 3: Request to vLLM

The orchestrator sends the request to the vLLM server over the OpenAI-compatible API.

The vLLM server forwards the request to the model.

#### Step 4: Model response

The model returns one of two valid outcomes for the minimal design:

1. a normal assistant response with no tool call
2. an assistant message containing exactly one tool call

#### Step 5: Response inspection

The orchestrator inspects the response.

- If no tool call is present, the orchestrator SHALL print the assistant response and terminate the turn.
- If exactly one tool call is present, the orchestrator SHALL continue to validation.
- If more than one tool call is present, the orchestrator SHALL reject the response as outside the minimal contract.

#### Step 6: Validation

The orchestrator SHALL validate:

- the tool name
- the argument encoding
- the argument structure
- the argument types

The tool MUST NOT be executed until validation succeeds.

#### Step 7: Tool execution

If validation succeeds, the orchestrator SHALL execute the local Python tool function.

If execution raises an exception, the orchestrator SHALL catch it and convert it into an explicit tool result message indicating error.

#### Step 8: Tool result injection

The orchestrator SHALL append:

1. the assistant message containing the tool call
2. a tool result message associated with the same tool call identifier

#### Step 9: Final model response

The orchestrator sends the extended conversation back to the vLLM server.

The model then produces a final assistant response that incorporates the tool result.

#### Step 10: Terminal output

The orchestrator prints the final assistant response to the terminal and ends the turn.

---

## 9. Interface Contracts

### 9.1 Contract principle

The system contains two distinct interface types:

- a **message protocol** between orchestrator and vLLM server
- a **function call interface** between orchestrator and tool

These MUST remain conceptually separate.

### 9.2 Request contract to the vLLM server

The orchestrator MUST send a chat completion request containing at least:

- `model`
- `messages`
- `tools`
- `tool_choice`

For the minimal design, `tool_choice` SHOULD default to `"auto"` in order to preserve the natural branching behavior of the system: the model may answer directly or may request the tool.

The design also recognizes a teaching and testing tradeoff:

- `tool_choice="auto"` preserves the architectural truth that tool use is model-selected and therefore probabilistic.
- `tool_choice="required"` may be used in a teaching, demonstration, or test mode when deterministic exercise of the tool path is more important than preserving branching behavior.

If `tool_choice="required"` is used, the implementation remains compatible with this architecture, but the demonstration becomes less representative of the unconstrained branching case.

A minimal conceptual request shape is:

```json
{
  "model": "local-model-name",
  "messages": [
    {
      "role": "system",
      "content": "Use the tool only when needed. Use only the provided tool."
    },
    {
      "role": "user",
      "content": "What is 2 plus 3?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add_numbers",
        "description": "Add two numbers and return the sum.",
        "parameters": {
          "type": "object",
          "properties": {
            "a": { "type": "number" },
            "b": { "type": "number" }
          },
          "required": ["a", "b"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### 9.3 Tool definition contract

The tool definition MUST declare:

- the tool name
- a short description
- a parameter schema
- required fields

For this design, the only allowed tool is:

```text
add_numbers
```

The tool parameters SHALL be:

- `a`: number
- `b`: number

### 9.4 Tool call response contract

If the model requests a tool, the minimal implementation SHALL accept only a response that contains:

- exactly one tool call
- a tool call identifier
- a function name
- a JSON-encoded argument string

A conceptual tool call response is:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "add_numbers",
        "arguments": "{\"a\": 2, \"b\": 3}"
      }
    }
  ]
}
```

The orchestrator MUST treat the `arguments` field as untrusted structured text until parsed and validated.

### 9.5 Tool result contract

After execution, the orchestrator SHALL append a tool result message containing:

- `role="tool"`
- the matching `tool_call_id`
- serialized result content

A conceptual tool result message is:

```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "5"
}
```

The result content MAY be a simple string representation.

### 9.6 Tool implementation contract

The tool implementation contract for this design is:

```python
def add_numbers(a: float, b: float) -> str:
    ...
```

The function SHALL return a value that can be serialized into the tool result message.

---

## 10. Determinism and Control Boundaries

This section defines why the orchestrator exists as a distinct component.

### 10.1 Probabilistic components

The following behaviors are probabilistic:

- whether the model decides to answer directly or request the tool
- the exact wording of assistant responses
- the correctness of the model’s chosen arguments
- the semantic quality of the model’s decision to call the tool

These behaviors depend on model inference and are not guaranteed by ordinary program control.

### 10.2 Deterministic components

The following behaviors MUST be deterministic in the orchestrator:

- reading terminal input
- constructing the request object
- checking whether tool calls are present
- rejecting multiple tool calls
- parsing argument JSON
- validating tool name and types
- dispatching to the correct Python function
- catching exceptions
- constructing the tool result message
- printing final output

### 10.3 Machine-executing boundary

There are two forms of machine activity in this system.

#### Model inference

This is local numerical computation performed by the model under vLLM. It produces tokens or structured token sequences.

#### Tool execution

This is ordinary Python code executed by the orchestrator.

Only the second category has direct machine-side operational effect in this design.

### 10.4 Control authority

The orchestrator SHALL be the sole control authority for machine-side execution.

The model MAY request a tool call.

The model MUST NOT be assumed to have authority to execute the tool simply because it produced a tool-call structure.

That boundary is the central control principle of the system.

---

## 11. Validation and Failure Handling

### 11.1 Allowed tool names

The orchestrator MUST maintain an explicit allowlist of tool names.

For this design, the allowlist SHALL contain exactly one entry:

```text
add_numbers
```

If the model returns any other tool name, the orchestrator MUST reject the request.

### 11.2 Argument validation

The orchestrator MUST validate tool-call arguments before execution.

At minimum, validation SHALL confirm that:

- the arguments field is valid JSON
- the parsed value is an object
- the required keys are present
- the values for `a` and `b` are numeric

No tool execution SHALL occur until these checks pass.

### 11.3 Failure policy

The minimal implementation SHALL use the following failure policy.

#### Malformed tool call

A malformed tool call includes, but is not limited to:

- invalid JSON
- missing required arguments
- wrong argument types
- unknown tool name
- multiple tool calls
- incomplete tool-call structure

For malformed tool calls, the orchestrator SHALL:

1. reject execution
2. stop the turn
3. print a visible orchestrator error to the terminal

#### Tool execution exception

If the tool raises an exception after successful validation, the orchestrator SHALL:

1. catch the exception
2. convert the exception into a tool result string of the form `ERROR: <message>`
3. send that tool result back through the second model pass
4. print the final assistant response produced from that result

This policy keeps malformed control messages and execution-time failures distinct.

### 11.4 Fail-closed principle

If the orchestrator cannot determine that a tool call is valid, it MUST fail closed and avoid execution.

For this design, fail closed means:

- do not dispatch the tool
- do not continue with tool execution
- surface an explicit terminal-visible failure condition

### 11.5 Terminal output policy

For the minimal implementation, the terminal output policy SHOULD be:

- final assistant responses are written to `stdout`
- orchestrator errors are written to `stderr`

A visible orchestrator error SHOULD use a stable prefix such as:

```text
[orchestrator error]
```

This improves observability while keeping the design minimal.

---

## 12. Minimal Deployment and Runtime Assumptions

The design assumes the following runtime environment.

### 12.1 Local terminal application

The orchestrator runs as a local Python program in a terminal session.

### 12.2 vLLM server availability

A vLLM server is already running and reachable through its OpenAI-compatible API endpoint.

If the vLLM server is unreachable, the orchestrator SHALL fail fast, print a terminal-visible error, and terminate the turn without attempting tool dispatch.

### 12.3 Single configured model

Exactly one chat-capable local model is configured for the running process.

This design assumes the server and model configuration support the required tool-calling behavior.

### 12.4 Single-process orchestration and tool execution

The orchestrator and tool implementation run in the same Python process.

### 12.5 No concurrency

The design assumes no concurrent requests, no parallel tool calls, and no background workers.

### 12.6 No persistence across runs

The system does not require:

- databases
- file-based state
- session restoration
- durable logs

### 12.7 Single-turn focus

The design is centered on a single user turn with at most one tool round-trip before final answer generation.

---

## 13. Minimal Implementation Layout

The implementation SHOULD use the smallest project layout that preserves clarity.

A recommended layout is:

```text
tool_demo/
├── demo.py
├── requirements.txt
└── README.md
```

### 13.1 `demo.py`

This file SHOULD contain:

- endpoint and model configuration
- tool definition metadata
- the tool implementation
- validation logic
- the orchestration loop
- the terminal entry point

### 13.2 `requirements.txt`

This file SHOULD contain only the minimal dependency set required to call the OpenAI-compatible API from Python.

For this design, the runtime dependencies SHOULD be stated explicitly as:

- Python 3.x
- `openai>=1.0`

The vLLM server is an external runtime dependency and is assumed to expose an OpenAI-compatible Chat Completions API.

### 13.3 `README.md`

This file SHOULD contain:

- how to start the vLLM server
- how to configure the endpoint and model name
- how to run the terminal demo
- a short explanation of the system boundaries

---

## 14. Future Pressures

Although the design is intentionally minimal, several architectural pressures will appear quickly once the first version works.

### 14.1 More tools

Adding more tools introduces:

- tool selection ambiguity
- larger allowlists
- more complex dispatch logic
- greater need for consistent schema design

### 14.2 Richer schemas

As tool arguments become more complex, the orchestrator will need:

- stronger validation
- optional-field handling
- enumerated constraints
- nested object support

### 14.3 Context growth

Longer conversations increase the size of the message history and eventually require:

- truncation strategy
- summarization
- state management

### 14.4 Better error handling

A more capable system will need explicit policy for:

- retrying malformed tool calls
- structured error reporting
- deciding when to stop versus continue

### 14.5 Verification

As tools become more important, the system may need explicit checks for:

- tool choice correctness
- argument reasonableness
- result plausibility

### 14.6 Security boundaries

Once tools interact with the file system, shell, network, or credentials, the simple boundary in this document is no longer sufficient.

At that point, the architecture would need stronger controls such as:

- path restrictions
- sandboxing
- permissions
- auditing
- isolation of sensitive operations

### 14.7 Server and model abstraction

As the system grows, it may become useful to separate:

- model-facing message construction
- provider or server adaptation
- tool registry
- policy enforcement
- transcript management

This design intentionally does not include those abstractions.

---

## 15. Summary

This document specifies the smallest complete terminal-based tool-calling system built from first principles.

Its main architectural claims are:

- the **model** generates candidate structured output
- the **vLLM server** serves the model over an OpenAI-compatible API
- the **orchestrator** owns deterministic control and execution authority
- the **tool** is ordinary local Python code executed only by the orchestrator

The system is deliberately minimal, but it is complete enough to demonstrate the essential control loop and to serve as a sound baseline for future extension.

The next natural companion artifact is the minimal implementation that conforms exactly to this design.
