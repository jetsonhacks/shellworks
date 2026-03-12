### **Implementation Addendum: Integration Nuances for SDD-LLM-TOOLCALL-001**

While the SDD defines the structural boundaries and responsibilities of the system, successful execution against a local vLLM instance often requires addressing these two "invisible" logic hurdles.

#### **1. Handling the "Thought" or "Reasoning" Buffer**

The current response contract in the SDD assumes the `content` field is `null` when a tool call is requested. However, modern models often generate a "Chain of Thought" string before the structured output.

* **The Observation**: The model might return a response where `content` contains text (e.g., "I will add these two numbers for you...") alongside the `tool_calls` array.
* **Orchestrator Responsibility**: The orchestrator must decide whether to discard this text or print it to the terminal as part of the "probabilistic" phase of the loop.
* **Recommended Action**: For a minimal implementation, you should either explicitly ignore non-null `content` when `tool_calls` exists or print it with a "Thinking..." prefix to maintain transparency for the user.

#### **2. The Chat Template "Silent Failure"**

Tool definitions are sent to the vLLM server as structured JSON, but the underlying model only perceives this information if it is correctly formatted into a prompt by a Chat Template.

* **The Issue**: If the vLLM server uses a template that does not support the `tools` key, the tool definitions may be stripped out before the request reaches the model.
* **The Symptom**: The model may respond as if no tool exists, providing a text-only answer like "I cannot perform math".
* **The Fix**: Ensure the vLLM server is launched with a `--chat-template` that explicitly supports tool-calling schemas, such as specific Jinja2 templates for models like Llama 3 or Mistral.

#### **Implementation Pro-Tip: Debugging the Handshake**

To ensure verifiable minimality and clarity, it is highly recommended to log the raw JSON exchanged between the orchestrator and the vLLM server during initial testing. This reveals exactly where the "handshake" fails—whether the tools never reached the model in the first pass or the model's reasoning was misparsed by the orchestrator.
