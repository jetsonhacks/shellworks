"""
directory: src/shellworks/llm/
file:      vllm_client.py

Minimal client configuration for the local LLM server.

Holds:
  - base URL
  - API key (vLLM accepts any non-empty string when --api-key is not set)
  - model name
  - backend type ("vllm" | "llamacpp")
  - OpenAI client construction

No abstraction layer. No provider switching. Just the wiring.

Environment variables
---------------------
  VLLM_BASE_URL   Base URL of the OpenAI-compatible server
                  default: http://localhost:8000/v1
  VLLM_API_KEY    API key (any non-empty string is fine for local servers)
                  default: not-needed
  VLLM_MODEL      Model name as registered with the server
                  default: local-model
  LLM_BACKEND     Server implementation: "vllm" or "llamacpp"
                  default: vllm
                  Thor  → vllm     (enable_thinking via extra_body)
                  Orin  → llamacpp (enable_thinking via assistant pre-fill)
"""

import os
from openai import OpenAI


def build_client() -> tuple[OpenAI, str, str]:
    """
    Construct and return the local LLM client, model name, and backend type.

    Returns
    -------
    client : OpenAI
        Configured OpenAI-compatible client.
    model : str
        Model name to pass in each request.
    backend : str
        One of "vllm" or "llamacpp". Used by the orchestrator to select
        the correct thinking-suppression strategy.
    """
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "not-needed")
    model = os.environ.get("VLLM_MODEL", "local-model")
    backend = os.environ.get("LLM_BACKEND", "vllm")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model, backend