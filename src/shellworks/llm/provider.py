"""
directory: src/shellworks/llm/
file:      provider.py

OpenAICompatibleProvider — speaks the OpenAI-compatible API contract.

Owns:
  - request formatting for OpenAI-compatible APIs
  - model selection in outgoing requests
  - response normalization
  - tool call extraction
  - realization of model-specific reasoning controls based on ModelProfile

Does not own:
  - endpoint persistence
  - model profile storage
  - launch management
  - runtime policy decisions about whether a model switch should persist

The orchestrator expresses intent. The provider realizes it.

Reasoning intent
----------------
The orchestrator passes one of three reasoning intents per request:

  "on"       Enable reasoning for this request.
  "off"      Disable reasoning for this request.
  "default"  Do not apply any reasoning control for this request.

The provider maps that intent to the appropriate mechanism as described
in the loaded ModelProfile.

build_provider()
----------------
Module-level factory that loads an endpoint and model profile from disk
and returns a configured provider. Called once at startup.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from openai import APIConnectionError, OpenAI

from shellworks.llm.endpoint import LLMEndpoint, resolve_endpoint
from shellworks.llm.model_profile import ModelProfile, resolve_model_profile

ReasoningIntent = Literal["on", "off", "default"]

# Default configs directory: resolved relative to this source file, not the
# working directory. This means `uv run shellworks` works from any directory,
# not just the project root.
# Override with the SHELLWORKS_CONFIGS_DIR environment variable.
_DEFAULT_CONFIGS_DIR = Path(__file__).parent.parent.parent.parent / "configs"

class OpenAICompatibleProvider:
    """
    Protocol adapter for OpenAI-compatible language model endpoints.

    Initialized with a concrete endpoint, active model name, and model
    profile. Immutable after construction — runtime model switches
    produce a new provider instance rather than mutating this one.
    """

    def __init__(
        self,
        endpoint: LLMEndpoint,
        model: str,
        profile: ModelProfile,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._profile = profile
        self._client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    @property
    def endpoint(self) -> LLMEndpoint:
        return self._endpoint

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        reasoning: ReasoningIntent = "default",
        debug: bool = False,
        label: str = "request",
    ) -> Any | None:
        """
        Send one completion request to the endpoint.

        Applies reasoning control from the model profile before sending.
        Works on a deep copy of messages — never mutates the caller's state.

        Parameters
        ----------
        messages : list[dict]
            Conversation history to send. Not mutated.
        tools : list[dict] | None
            Tool definitions to advertise. Omit to send no tools.
        tool_choice : str | None
            Tool choice policy (e.g. "auto"). Omit if no tools.
        reasoning : ReasoningIntent
            Reasoning intent for this specific request.
        debug : bool
            Print request/response detail to stderr.
        label : str
            Short label for debug output (e.g. "first", "second").

        Returns
        -------
        The raw OpenAI response object, or None on connection failure.
        Connection failure is reported to stderr via [orchestrator error].
        """
        # Deep copy: reasoning control may modify the outbound messages.
        # The caller's list is never touched.
        outbound_messages = copy.deepcopy(messages)
        request: dict[str, Any] = {
            "model": self._model,
            "messages": outbound_messages,
        }

        self._apply_reasoning_control(outbound_messages, request, reasoning)

        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice

        if debug:
            self._debug_print_request(label, request)

        try:
            response = self._client.chat.completions.create(**request)
        except APIConnectionError as exc:
            print(
                f"[orchestrator error] Could not reach the LLM server ({label} pass).\n"
                f"  Endpoint: {self._endpoint.base_url}\n"
                f"  Detail:   {exc}",
                file=sys.stderr,
            )
            return None

        if debug:
            print(f"[debug] {label} response:", file=sys.stderr)
            print(response.model_dump_json(indent=2), file=sys.stderr)

        return response

    # ------------------------------------------------------------------
    # Reasoning control
    # ------------------------------------------------------------------

    def _apply_reasoning_control(
        self,
        messages: list[dict[str, Any]],
        request: dict[str, Any],
        reasoning: ReasoningIntent,
    ) -> None:
        """
        Apply model-specific reasoning control to the outbound request.

        Called on deep-copied data only. Mutates messages and request
        in-place. The caller's originals are never touched.

        When reasoning is "default", no control is applied regardless of
        the reasoning_control mode.
        """
        control = self._profile.reasoning_control

        if control == "none" or reasoning == "default":
            return

        if control == "prompt_marker":
            self._inject_prompt_marker(messages, reasoning)

        elif control == "request_kwargs":
            self._set_reasoning_kwargs(request, reasoning)

        elif control == "hybrid":
            # Both mechanisms applied together. "hybrid" does not mean
            # the provider chooses one at runtime — both always fire.
            self._inject_prompt_marker(messages, reasoning)
            self._set_reasoning_kwargs(request, reasoning)

    def _inject_prompt_marker(
        self,
        messages: list[dict[str, Any]],
        reasoning: ReasoningIntent,
    ) -> None:
        """
        Prepend the reasoning marker to the system message content.

        If no system message exists, a minimal one is inserted at the
        front of the message list.
        """
        marker = (
            self._profile.reasoning_on_marker
            if reasoning == "on"
            else self._profile.reasoning_off_marker
        )
        if marker is None:
            return

        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = marker + "\n\n" + msg["content"]
                return

        # No system message found — insert a minimal one.
        messages.insert(0, {"role": "system", "content": marker})

    def _set_reasoning_kwargs(
        self,
        request: dict[str, Any],
        reasoning: ReasoningIntent,
    ) -> None:
        """
        Set the boolean value at reasoning_kwargs_path in the request dict.

        reasoning_kwargs_path is a dot-separated path. Intermediate dicts
        are created as needed.

        Example path: "extra_body.chat_template_kwargs.enable_thinking"
        Result: request["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True/False
        """
        path = self._profile.reasoning_kwargs_path
        if path is None:
            return
        _set_nested(request, path, reasoning == "on")

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _debug_print_request(self, label: str, request: dict[str, Any]) -> None:
        print(f"[debug] {label} request:", file=sys.stderr)
        print(
            json.dumps(
                {
                    "model": request["model"],
                    "messages": [
                        m.model_dump() if hasattr(m, "model_dump") else m
                        for m in request["messages"]
                    ],
                    "tools": request.get("tools"),
                    "tool_choice": request.get("tool_choice"),
                    "extra_body": request.get("extra_body"),
                },
                indent=2,
                default=str,
            ),
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Dot-path helper
# ---------------------------------------------------------------------------

def _set_nested(d: dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value at a dot-separated path in a nested dict, creating
    intermediate dicts as needed.

    Example:
        d = {}
        _set_nested(d, "extra_body.chat_template_kwargs.enable_thinking", True)
        # d == {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    """
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_provider(
    endpoint_name: str | None = None,
    model_override: str | None = None,
    configs_dir: Path | None = None,
) -> OpenAICompatibleProvider:
    """
    Load an endpoint and model profile from disk and return a configured provider.

    Called once at startup. The active model begins from the endpoint's
    default_model unless overridden by model_override or the SHELLWORKS_MODEL
    environment variable.

    Parameters
    ----------
    endpoint_name : str | None
        Name of the endpoint to load (without .toml extension).
        Falls back to SHELLWORKS_ENDPOINT env var, then "local_primary".
    model_override : str | None
        Active model to use for this run. Overrides endpoint default.
        Falls back to SHELLWORKS_MODEL env var, then endpoint default_model.
    configs_dir : Path | None
        Path to the configs directory.
        Falls back to SHELLWORKS_CONFIGS_DIR env var, then ./configs.

    Returns
    -------
    OpenAICompatibleProvider
        Ready to accept complete() calls from the orchestrator.

    Raises
    ------
    FileNotFoundError
        If the endpoint config or matching model profile cannot be found.
    ValueError
        If no active model can be determined.
    """
    if configs_dir is None:
        configs_dir = Path(
            os.environ.get("SHELLWORKS_CONFIGS_DIR", str(_DEFAULT_CONFIGS_DIR))
        )

    if endpoint_name is None:
        endpoint_name = os.environ.get("SHELLWORKS_ENDPOINT", "local_primary")

    endpoint = resolve_endpoint(configs_dir, endpoint_name)

    active_model = (
        model_override
        or os.environ.get("SHELLWORKS_MODEL")
        or endpoint.default_model
    )
    if active_model is None:
        raise ValueError(
            f"No active model could be determined for endpoint {endpoint_name!r}.\n"
            f"  Set default_model in the endpoint config, or set SHELLWORKS_MODEL."
        )

    profile = resolve_model_profile(configs_dir, active_model)

    return OpenAICompatibleProvider(
        endpoint=endpoint,
        model=active_model,
        profile=profile,
    )
