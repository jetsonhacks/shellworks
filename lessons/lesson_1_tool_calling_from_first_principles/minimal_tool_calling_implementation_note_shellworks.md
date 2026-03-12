# Lesson 1 Implementation Note  
## Tool Calling from First Principles in `shellworks`

This note is for building the first runnable example for Lesson 1.

The goal is simple:

Build the smallest clear example of tool calling in the terminal that matches the SDD.

Keep it easy to read. Do not try to build the future architecture yet.

---

## What this lesson is building

For Lesson 1, we want a terminal program that does this:

1. the user types a request
2. the orchestrator sends that request and one tool definition to the local vLLM server
3. the model either answers normally or asks for the tool
4. the orchestrator validates the tool call
5. the orchestrator runs the tool
6. the orchestrator sends the tool result back to the model
7. the final answer is printed

That is all.

---

## Where the code should go

Keep the lesson documents where they already are:

```text
lessons/
└── lesson_1_tool_calling_from_first_principles/
    ├── minimal_tool_calling_sdd.md
    ├── minimal_tool_calling_implementation_guide.md
    └── minimal_tool_calling_addendum.md
```

Put the runnable code in `src/shellworks/`:

```text
src/
└── shellworks/
    ├── main.py
    ├── llm/
    │   └── vllm_client.py
    ├── orchestrator/
    │   └── minimal_tool_calling.py
    └── tools/
        └── arithmetic.py
```

Put the tests in:

```text
tests/
└── test_minimal_tool_calling.py
```

---

## What each file should do

### `src/shellworks/main.py`

This should be the entry point for the demo.

For Lesson 1, keep it thin. It should just start the Lesson 1 tool-calling flow, not contain all the logic itself.

### `src/shellworks/orchestrator/minimal_tool_calling.py`

This is the main file for the lesson.

It should contain the full control loop:

- build the messages
- attach the tool definition
- call vLLM
- inspect the response
- validate the tool call
- run the tool
- send the tool result back
- return or print the final answer

This is the file someone should read to understand how the whole lesson works.

### `src/shellworks/tools/arithmetic.py`

This should contain the one lesson tool:

```python
def add_numbers(a: float, b: float) -> str:
    ...
```

Keep it trivial.

### `src/shellworks/llm/vllm_client.py`

This should hold the small amount of client setup needed for the local vLLM server.

Keep it simple. It can hold:

- base URL
- API key
- model name
- OpenAI client construction

No big abstraction layer is needed here.

---

## Rules for Lesson 1

These are the important rules.

### 1. One tool only

Use only one tool for this lesson:

```text
add_numbers
```

### 2. Keep the orchestrator in control

The orchestrator is responsible for:

- parsing
- validation
- tool dispatch
- error handling

The model can suggest a tool call. The orchestrator decides whether to execute it.

### 3. Validate before executing

Do not run the tool unless the tool call is valid.

At minimum, check:

- the tool name is allowed
- the arguments are valid JSON
- the decoded value is an object
- `a` and `b` are present
- `a` and `b` are numbers

### 4. Stop on malformed tool calls

If the tool call is malformed, do not try to recover in a fancy way.

For Lesson 1, fail closed:

- do not run the tool
- stop the turn
- show an orchestrator error

### 5. Do not overbuild

Do not add:

- a generic tool registry
- a provider abstraction layer
- a plugin system
- persistence
- planning
- multiple tools
- concurrency

Those may come later. They are not part of Lesson 1.

---

## A few practical notes

### Debug mode helps

A simple debug flag is a good idea.

When debug mode is on, it is helpful to print:

- the first request
- the first response
- the parsed tool call
- the tool result message
- the final response

This makes it much easier to see what is going on between the orchestrator and vLLM.

### vLLM may fail in boring ways

If the vLLM server is not reachable, fail fast and print an error.

Do not continue the turn.

### Chat-template issues are real

Sometimes the request shape looks correct, but the model still does not call tools the way you expect.

That can be a server/model/chat-template issue, not necessarily an orchestrator bug.

### Assistant content and tool calls can both appear

If the assistant message contains both normal `content` and `tool_calls`, treat the tool call as the thing that controls the next step.

Do not print the assistant content as the final answer yet.

---

## What “done” looks like

Lesson 1 is in good shape when all of these are true:

- a user can type a request in the terminal
- the program sends the request to the local vLLM server
- the model can either answer directly or request the tool
- the orchestrator validates the tool call before running anything
- `add_numbers` runs only after validation succeeds
- the tool result is sent back into the loop
- the final answer is printed
- malformed tool calls do not execute

---

## Suggested tests

Keep the tests small and focused on orchestrator behavior.

Good first tests are:

1. valid tool call runs correctly
2. unknown tool name is rejected
3. malformed JSON is rejected
4. multiple tool calls are rejected
5. tool exception becomes `ERROR: ...`

---

## Final note

The purpose of Lesson 1 is not to build a framework.

The purpose is to build the smallest clear example that shows:

- the model
- the vLLM server
- the orchestrator
- the tool

as separate parts of one working system.

