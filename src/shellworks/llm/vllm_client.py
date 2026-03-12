"""
directory: src/shellworks/llm/
file:      vllm_client.py

Minimal client configuration for the local vLLM server.

Holds:
  - base URL
  - API key (vLLM accepts any non-empty string when --api-key is not set)
  - model name
  - OpenAI client construction

No abstraction layer. No provider switching. Just the wiring.
"""

import os
from openai import OpenAI


def build_client() -> tuple[OpenAI, str]:
    """Construct and return the local vLLM client and configured model name."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("VLLM_API_KEY", "not-needed")
    model = os.environ.get("VLLM_MODEL", "local-model")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model