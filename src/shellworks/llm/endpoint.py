"""
directory: src/shellworks/llm/
file:      endpoint.py

LLMEndpoint — persistent description of a reachable language model endpoint.

Answers:
  - what provider type is used
  - where the endpoint lives
  - how to authenticate to it
  - what model should be used by default

Loaded from a TOML file in configs/endpoints/.
Does not represent runtime model selection. Stores only the default preference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib          # stdlib in Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # backport for 3.9 / 3.10

@dataclass
class LLMEndpoint:
    name: str
    provider_type: str
    base_url: str
    api_key_env: str | None
    default_model: str | None = None

    @property
    def api_key(self) -> str:
        """Resolve the API key from the environment variable named in api_key_env."""
        if self.api_key_env is None:
            return "not-needed"
        return os.environ.get(self.api_key_env, "not-needed")


def load_endpoint(path: Path) -> LLMEndpoint:
    """Load an LLMEndpoint from a TOML file."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse endpoint config {path}:\n  {exc}") from exc
    return LLMEndpoint(
        name=data["name"],
        provider_type=data["provider_type"],
        base_url=data["base_url"],
        api_key_env=data.get("api_key_env"),
        default_model=data.get("default_model"),
    )


def resolve_endpoint(configs_dir: Path, name: str) -> LLMEndpoint:
    """
    Load an endpoint by name from configs/endpoints/{name}.toml.

    Raises FileNotFoundError if the config file does not exist.
    """
    path = configs_dir / "endpoints" / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(
            f"Endpoint config not found: {path}\n"
            f"  Set SHELLWORKS_ENDPOINT to a name matching a file in configs/endpoints/."
        )
    return load_endpoint(path)
