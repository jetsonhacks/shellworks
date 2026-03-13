# Lesson 1.1 Design Doc: LLM Endpoints and Model Profiles

## Status

Draft

## Purpose

This document defines the configuration and runtime boundary for how Shellworks connects to language model servers and how it understands the models served by those endpoints.

Lesson 1 established the minimal path for tool calling with a local LLM. Lesson 1.1 introduces the next architectural boundary. Shellworks should not treat the LLM as a single opaque box. In practice, it talks to an endpoint through a protocol, selects a model at runtime, and may need model-specific instructions to access features such as reasoning control.

The goal of this design is to separate those concerns cleanly.

## Problem

The early prototype mixes several different kinds of knowledge together:

- where the server lives
- how the server is accessed
- which model is currently being used
- how a specific model exposes optional features
- backend-specific request shaping details

When these concerns are blended together, the architecture becomes muddy. The orchestrator starts to absorb server and model quirks. Configuration becomes tied to one concrete setup. Runtime experiments start to look like persistent truth. The result works for a demo, but it does not define a clean system boundary.

The design problem is not simply how to call a model. The problem is how to represent the stable parts of the system, the changeable parts of the system, and the model-specific knowledge Shellworks needs in order to interact correctly.

## Design Goals

Shellworks should:

- treat the endpoint description as persistent configuration
- treat model behavior as a separate persistent description
- treat the currently selected model as runtime state
- keep server launch and lifecycle outside Shellworks
- keep provider-specific and model-specific feature handling out of the orchestrator
- use one protocol-oriented provider abstraction for OpenAI-compatible endpoints
- keep the first version small, explicit, and easy to debug

## Non-Goals

This design does not attempt to solve:

- local server launch or process management
- model downloading
- fuzzy model matching or alias discovery
- cloud routing or failover
- load balancing across multiple endpoints
- a complete abstraction for every possible provider family

For Lesson 1.1, Shellworks assumes the server already exists and is reachable.

## Core Concepts

### LLMEndpoint

`LLMEndpoint` is a persistent description of a reachable language model endpoint.

It answers:

- what provider type is used
- where the endpoint lives
- how to authenticate to it
- what model should be used by default

This is the stable saved configuration for reaching a server.

It does not represent the current runtime model selection as a hard truth. It stores only the default model preference.

### ModelProfile

`ModelProfile` is a separate persistent description of how Shellworks should interact with a specific model.

It answers:

- what the canonical model name is
- how optional features such as reasoning are controlled
- whether prompt markers are required
- whether request kwargs are required
- any minimal model-specific interaction preferences Shellworks needs

A model profile is not a global in-code registry of all models. It is an independent entity stored on disk and loaded when needed.

### Runtime State

Runtime state is the live selection used during a specific run of Shellworks.

It includes:

- the active endpoint
- the active model

The active model may begin from the endpoint default model, but it can change during a run without mutating the endpoint on disk.

### Provider

The provider abstraction remains protocol-oriented.

For Lesson 1.1, the relevant provider type is:

- `OpenAICompatibleProvider`

The provider knows how to speak the API contract. It should not be the place where persistent endpoint definitions or model descriptions are stored.

## First Principles

Shellworks needs to answer four different questions, and each question belongs to a different part of the architecture.

**Where do I send requests?**
This is the responsibility of `LLMEndpoint`.

**What model am I currently using?**
This is runtime state.

**How do I interact with that model correctly?**
This is the responsibility of `ModelProfile`.

**How do I speak the protocol used by the endpoint?**
This is the responsibility of the provider.

This separation keeps the design simple and legible. Each object answers one kind of question.

## Persistent State vs Runtime State

A major design commitment in this document is the separation of stored state from runtime state.

### Persistent on disk

Persistent state includes:

- endpoint definitions
- model profiles
- endpoint default model preferences

### Runtime only

Runtime state includes:

- the active endpoint selected for the current run
- the active model selected for the current run

The active model is not automatically written back to disk when it changes during a run.

## Persistence Rule

Runtime model switches do not persist automatically.

If a user switches models during a run, that change applies only to the current runtime state.

If a user explicitly requests that a model become the new default, then Shellworks updates the endpoint definition on disk.

This rule separates experimentation from saved preference.

## Model Profile Resolution

For the first version, model profile resolution uses an exact canonical model name match.

This means:

- the active model name is resolved exactly
- the system loads the `ModelProfile` whose canonical model name matches that runtime model name
- no fuzzy matching or alias resolution is performed in Lesson 1.1

This choice is deliberate. It keeps the behavior deterministic, easy to debug, and easy to explain.

## Server Launch Boundary

Server launch and lifecycle are external to Shellworks.

Shellworks does not own:

- starting `llama.cpp`
- starting `vLLM`
- container lifecycle
- local server scripts
- model download workflows

Those concerns belong to external tooling or operational scripts.

This design keeps Lesson 1.1 focused on connection, model selection, and request shaping rather than infrastructure management.

## Minimal Configuration Shape

A simple endpoint file might look like this:

```toml
name = "local_primary"
provider_type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
api_key_env = "LOCAL_LLM_API_KEY"
default_model = "Qwen/Qwen3-4B-GGUF"
```

A simple model profile file might look like this:

```toml
name = "qwen3_4b"
canonical_model = "Qwen/Qwen3-4B-GGUF"
reasoning_control = "hybrid"
reasoning_on_marker = "/think"
reasoning_off_marker = "/no_think"
reasoning_kwargs_path = "extra_body.chat_template_kwargs.enable_thinking"
```

In this example, `hybrid` means Shellworks applies both mechanisms together for explicit reasoning `on` or `off` requests.

Another model profile might look like this:

```toml
name = "nemotron3_nano"
canonical_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
reasoning_control = "request_kwargs"
reasoning_kwargs_path = "extra_body.chat_template_kwargs.enable_thinking"
```

These examples are intentionally small. The goal is to define only the minimum Shellworks needs in order to interact correctly.

## Recommended Filesystem Layout

A simple layout could be:

```text
configs/
  endpoints/
    local_primary.toml
  model_profiles/
    qwen3_4b.toml
    nemotron3_nano.toml
```

In this structure:

- `configs/endpoints/` stores endpoint descriptions
- `configs/model_profiles/` stores model descriptions

Lesson 1.1 does not define a persistent runtime state file. Runtime selection exists in memory during a run. If Shellworks later chooses to persist session state between runs, that should be introduced in a separate design step rather than implied here.

## Internal Types

A minimal Python representation could look like this:

```python
from dataclasses import dataclass

@dataclass
class LLMEndpoint:
    name: str
    provider_type: str
    base_url: str
    api_key_env: str | None
    default_model: str | None = None

@dataclass
class ModelProfile:
    name: str
    canonical_model: str
    reasoning_control: str
    reasoning_on_marker: str | None = None
    reasoning_off_marker: str | None = None
    reasoning_kwargs_path: str | None = None

@dataclass
class RuntimeLLMState:
    active_endpoint_name: str
    active_model: str
```

These types are intentionally minimal. The first implementation should resist adding more fields until there is a demonstrated need.

## Reasoning Control Contract

`ModelProfile` may describe how Shellworks enables, disables, or leaves unchanged a model's reasoning behavior for an individual request.

For Lesson 1.1, the allowed values for `reasoning_control` are:

- `none`
- `prompt_marker`
- `request_kwargs`
- `hybrid`

These values are part of the contract. They are not informal examples.

### `none`

Shellworks does not apply any model-specific reasoning control.

Required fields:

- no additional fields required

Behavior:

- if the orchestrator requests reasoning `on`, `off`, or `default`, the provider does not inject prompt markers and does not set request kwargs for reasoning control
- the request proceeds unchanged with respect to reasoning

### `prompt_marker`

Shellworks controls reasoning by injecting model-specific markers into the outbound prompt.

Required fields:

- `reasoning_on_marker`
- `reasoning_off_marker`

Behavior:

- `reasoning = on` → inject `reasoning_on_marker`
- `reasoning = off` → inject `reasoning_off_marker`
- `reasoning = default` → do not inject either marker

The provider should inject markers into the outbound request copy only. It should not mutate stored conversation state.

### `request_kwargs`

Shellworks controls reasoning by setting a nested request field in the outbound request payload.

Required fields:

- `reasoning_kwargs_path`

Behavior:

- `reasoning = on` → set the nested field at `reasoning_kwargs_path` to `true`
- `reasoning = off` → set the nested field at `reasoning_kwargs_path` to `false`
- `reasoning = default` → do not set the field

`reasoning_kwargs_path` is interpreted as a dot-separated path into the outbound request dictionary.

For example:

- `extra_body.chat_template_kwargs.enable_thinking`

means the provider should ensure the nested dictionaries exist and then set:

```python
request["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True or False
```

### `hybrid`

Shellworks uses both request kwargs and prompt markers.

Required fields:

- `reasoning_on_marker`
- `reasoning_off_marker`
- `reasoning_kwargs_path`

Behavior:

- `reasoning = on` → set request kwargs to `true` and inject `reasoning_on_marker`
- `reasoning = off` → set request kwargs to `false` and inject `reasoning_off_marker`
- `reasoning = default` → apply neither mechanism

For Lesson 1.1, `hybrid` means both mechanisms are applied together for explicit `on` and `off` requests. It does not mean the provider chooses one at runtime.

This rule is intentionally simple and deterministic.

### Provider interpretation

`OpenAICompatibleProvider` is responsible for realizing these control modes based on the loaded `ModelProfile`.

In Lesson 1.1, the provider may implement this with a small set of conditional branches on the `reasoning_control` field. If the number of control styles grows later, those branches may be extracted into a separate strategy layer in a future design.

### Minimal worked example

Suppose runtime state selects:

- endpoint: `local_primary`
- model: `Qwen/Qwen3-4B-GGUF`

Suppose the matching `ModelProfile` is:

```toml
name = "qwen3_4b"
canonical_model = "Qwen/Qwen3-4B-GGUF"
reasoning_control = "hybrid"
reasoning_on_marker = "/think"
reasoning_off_marker = "/no_think"
reasoning_kwargs_path = "extra_body.chat_template_kwargs.enable_thinking"
```

If the orchestrator requests reasoning `off` for a tool-calling turn, the provider should:

1. build the outbound request normally
2. set `extra_body.chat_template_kwargs.enable_thinking = false`
3. inject `/no_think` into the outbound prompt copy
4. send the request without mutating stored conversation state

If the orchestrator requests reasoning `default`, the provider should apply neither the kwargs path nor the prompt marker.

## Runtime Flow

A typical runtime flow should be:

1. Load the chosen `LLMEndpoint`.
2. Resolve the initial active model from runtime override or the endpoint default model.
3. Resolve the `ModelProfile` by exact canonical model name match.
4. Instantiate `OpenAICompatibleProvider` using the endpoint, active model, and model profile.
5. Run the orchestrator using the provider abstraction only.
6. If the user switches models, update runtime state only.
7. If the user explicitly requests a new default, write that change back to the endpoint file.

This keeps the runtime behavior explicit and easy to reason about.

## Provider Responsibility

`OpenAICompatibleProvider` should own:

- request formatting for OpenAI-compatible APIs
- model selection in outgoing requests
- response normalization
- tool call extraction
- realization of model-specific feature controls based on the loaded `ModelProfile`

The provider should not own:

- endpoint persistence
- model profile storage
- launch management
- runtime policy decisions about whether a model switch should persist

The orchestrator should express intent, not mechanism.

For example, the orchestrator may request reasoning on or off for a call. The provider uses the model profile to decide whether that becomes request kwargs, prompt markers, or another model-specific control convention.

## Design Constraints

This design makes a few explicit constraints for the first version.

### Constraint 1: One main provider category

For Lesson 1.1, Shellworks assumes `openai_compatible` is the primary provider type.

### Constraint 2: Exact model resolution only

Model profile resolution uses exact canonical model name matching.

### Constraint 3: Minimal model profile contents

A `ModelProfile` should contain only the minimal information Shellworks needs to interact with the model correctly.

It should not become a dumping ground for:

- launcher information
- benchmark notes
- model download locations
- UI copy
- unrelated documentation

### Constraint 4: Server launch is outside the scope

Shellworks connects to running endpoints. It does not launch them.

## Why This Design Is Better

This design replaces one muddy object with a clean separation of concerns.

Instead of treating endpoint, model, runtime selection, and launch mechanics as one combined deployment object, Shellworks now has:

- a persistent endpoint description
- a persistent model description
- a runtime model selection
- a provider that speaks the protocol

That makes the system easier to teach, easier to implement, and easier to extend.

It also creates a clearer Lesson 1.1:

Shellworks does not just call a model. It connects to an endpoint, selects a model at runtime, resolves how that model should be controlled, and communicates through a protocol adapter.

## Risks and Watch Points

The main risk in the first version is overgrowth.

`ModelProfile` will need discipline. If too many unrelated fields are added, it will lose its purpose.

The second watch point is model naming. Exact canonical model name matching is the right first choice, but some servers may report names differently than expected. If this becomes a real issue later, aliases can be added in a future lesson.

The third watch point is provider growth. In Lesson 1.1, a small number of `reasoning_control` branches inside `OpenAICompatibleProvider` is acceptable. If the number of control styles grows, Shellworks should consider extracting that logic into a separate strategy layer rather than letting provider conditionals expand without bound.

For Lesson 1.1, the simpler rule is the better rule.

## Final Design Commitments

Shellworks adopts the following design commitments for Lesson 1.1:

1. `LLMEndpoint` is a persistent description of a reachable endpoint.
2. `ModelProfile` is a separate persistent description of how to interact with a specific model.
3. Runtime model selection is not persistent unless the user explicitly requests a new default.
4. Model profile resolution uses exact canonical model name match.
5. Server launch and lifecycle are external to Shellworks.
6. `OpenAICompatibleProvider` remains the protocol abstraction for the first version.
7. The first implementation keeps both endpoint and model profile definitions minimal.

## Recommended Lesson 1.1 Outcome

After implementing this design, Shellworks should be able to:

- connect to a saved OpenAI-compatible endpoint
- use the endpoint’s default model for a run
- resolve a separate model profile for that model
- shape requests according to the model profile through the provider
- allow temporary model switches at runtime
- persist a new default model only on explicit user request

