"""
directory: src/shellworks/llm/
file:      model_profile.py

ModelProfile — persistent description of how Shellworks interacts with a specific model.

Answers:
  - what the canonical model name is
  - how optional features such as reasoning are controlled
  - whether prompt markers are required
  - whether request kwargs are required

Loaded from a TOML file in configs/model_profiles/.
Resolved at runtime by exact canonical model name match.

Reasoning control contract
--------------------------
reasoning_control may be one of four values:

  none           No reasoning control is applied.

  prompt_marker  Shellworks injects a model-specific marker into the
                 outbound prompt. Requires: reasoning_on_marker,
                 reasoning_off_marker.

  request_kwargs Shellworks sets a nested boolean field in the outbound
                 request payload. Requires: reasoning_kwargs_path.

  hybrid         Both mechanisms are applied together. Requires all three
                 fields: reasoning_on_marker, reasoning_off_marker,
                 reasoning_kwargs_path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import tomllib          # stdlib in Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # backport for 3.9 / 3.10

ReasoningControl = Literal["none", "prompt_marker", "request_kwargs", "hybrid"]

VALID_REASONING_CONTROLS: frozenset[str] = frozenset(
    {"none", "prompt_marker", "request_kwargs", "hybrid"}
)


@dataclass
class ModelProfile:
    name: str
    canonical_model: str
    reasoning_control: ReasoningControl
    reasoning_on_marker: str | None = None
    reasoning_off_marker: str | None = None
    reasoning_kwargs_path: str | None = None


def load_model_profile(path: Path) -> ModelProfile:
    """Load a ModelProfile from a TOML file."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse model profile {path}:\n  {exc}") from exc

    reasoning_control = data["reasoning_control"]
    if reasoning_control not in VALID_REASONING_CONTROLS:
        raise ValueError(
            f"Invalid reasoning_control value {reasoning_control!r} in {path}. "
            f"Must be one of: {sorted(VALID_REASONING_CONTROLS)}"
        )

    return ModelProfile(
        name=data["name"],
        canonical_model=data["canonical_model"],
        reasoning_control=reasoning_control,
        reasoning_on_marker=data.get("reasoning_on_marker"),
        reasoning_off_marker=data.get("reasoning_off_marker"),
        reasoning_kwargs_path=data.get("reasoning_kwargs_path"),
    )


def resolve_model_profile(configs_dir: Path, canonical_model: str) -> ModelProfile:
    """
    Find and load the ModelProfile whose canonical_model matches exactly.

    Scans all .toml files in configs/model_profiles/ and returns the first
    exact match. Resolution is deterministic: exact name match only, no
    fuzzy matching or alias resolution.

    Raises FileNotFoundError if no match is found.
    """
    profiles_dir = configs_dir / "model_profiles"
    if not profiles_dir.exists():
        raise FileNotFoundError(
            f"Model profiles directory not found: {profiles_dir}"
        )

    for path in sorted(profiles_dir.glob("*.toml")):
        profile = load_model_profile(path)
        if profile.canonical_model == canonical_model:
            return profile

    raise FileNotFoundError(
        f"No model profile found for canonical model: {canonical_model!r}\n"
        f"  Searched in: {profiles_dir}\n"
        f"  Ensure a .toml file exists with canonical_model = {canonical_model!r}"
    )
