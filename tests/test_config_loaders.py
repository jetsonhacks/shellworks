"""
tests/test_config_loaders.py

Unit tests for the TOML config loaders: endpoint.py and model_profile.py.

These tests do not require a running LLM server.
They verify that the loaders correctly parse valid configs and reject
invalid or missing ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shellworks.llm.endpoint import LLMEndpoint, load_endpoint, resolve_endpoint
from shellworks.llm.model_profile import (
    ModelProfile,
    load_model_profile,
    resolve_model_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ENDPOINT_TOML = """
name            = "test_endpoint"
provider_type   = "openai_compatible"
base_url        = "http://127.0.0.1:8000/v1"
api_key_env     = "TEST_API_KEY"
default_model   = "test/model-name"
"""

VALID_PROFILE_TOML_KWARGS = """
name                    = "test_profile"
canonical_model         = "test/model-name"
reasoning_control       = "request_kwargs"
reasoning_kwargs_path   = "extra_body.chat_template_kwargs.enable_thinking"
"""

VALID_PROFILE_TOML_NONE = """
name                    = "test_profile_none"
canonical_model         = "test/no-reasoning-model"
reasoning_control       = "none"
"""

VALID_PROFILE_TOML_MARKER = """
name                    = "test_profile_marker"
canonical_model         = "test/marker-model"
reasoning_control       = "prompt_marker"
reasoning_on_marker     = "/think"
reasoning_off_marker    = "/no_think"
"""


# ---------------------------------------------------------------------------
# Endpoint loading
# ---------------------------------------------------------------------------

class TestLoadEndpoint:
    def test_valid_endpoint(self, tmp_path: Path) -> None:
        f = tmp_path / "test_endpoint.toml"
        f.write_text(VALID_ENDPOINT_TOML)
        ep = load_endpoint(f)
        assert isinstance(ep, LLMEndpoint)
        assert ep.name == "test_endpoint"
        assert ep.base_url == "http://127.0.0.1:8000/v1"
        assert ep.default_model == "test/model-name"

    def test_api_key_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        f = tmp_path / "ep.toml"
        f.write_text(VALID_ENDPOINT_TOML)
        monkeypatch.setenv("TEST_API_KEY", "secret-token")
        ep = load_endpoint(f)
        assert ep.api_key == "secret-token"

    def test_api_key_defaults_when_env_missing(self, tmp_path: Path) -> None:
        f = tmp_path / "ep.toml"
        f.write_text(VALID_ENDPOINT_TOML)
        ep = load_endpoint(f)
        assert ep.api_key == "not-needed"

    def test_invalid_toml_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.toml"
        f.write_text("name = [unclosed")
        with pytest.raises(ValueError, match="Failed to parse endpoint config"):
            load_endpoint(f)

    def test_resolve_endpoint_happy_path(self, tmp_path: Path) -> None:
        endpoints_dir = tmp_path / "endpoints"
        endpoints_dir.mkdir()
        (endpoints_dir / "my_server.toml").write_text(VALID_ENDPOINT_TOML)
        ep = resolve_endpoint(tmp_path, "my_server")
        assert ep.name == "test_endpoint"

    def test_resolve_endpoint_missing_raises(self, tmp_path: Path) -> None:
        (tmp_path / "endpoints").mkdir()
        with pytest.raises(FileNotFoundError):
            resolve_endpoint(tmp_path, "nonexistent")


# ---------------------------------------------------------------------------
# Model profile loading
# ---------------------------------------------------------------------------

class TestLoadModelProfile:
    def test_request_kwargs_profile(self, tmp_path: Path) -> None:
        f = tmp_path / "profile.toml"
        f.write_text(VALID_PROFILE_TOML_KWARGS)
        p = load_model_profile(f)
        assert isinstance(p, ModelProfile)
        assert p.reasoning_control == "request_kwargs"
        assert p.reasoning_kwargs_path == "extra_body.chat_template_kwargs.enable_thinking"
        assert p.reasoning_on_marker is None

    def test_none_reasoning_profile(self, tmp_path: Path) -> None:
        f = tmp_path / "profile.toml"
        f.write_text(VALID_PROFILE_TOML_NONE)
        p = load_model_profile(f)
        assert p.reasoning_control == "none"

    def test_prompt_marker_profile(self, tmp_path: Path) -> None:
        f = tmp_path / "profile.toml"
        f.write_text(VALID_PROFILE_TOML_MARKER)
        p = load_model_profile(f)
        assert p.reasoning_control == "prompt_marker"
        assert p.reasoning_on_marker == "/think"
        assert p.reasoning_off_marker == "/no_think"

    def test_invalid_reasoning_control_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "profile.toml"
        f.write_text(
            'name = "x"\ncanonical_model = "x/y"\nreasoning_control = "magic"\n'
        )
        with pytest.raises(ValueError, match="Invalid reasoning_control"):
            load_model_profile(f)

    def test_invalid_toml_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.toml"
        f.write_text("name = [unclosed")
        with pytest.raises(ValueError, match="Failed to parse model profile"):
            load_model_profile(f)

    def test_resolve_model_profile_exact_match(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "model_profiles"
        profiles_dir.mkdir()
        (profiles_dir / "qwen3.toml").write_text(VALID_PROFILE_TOML_KWARGS)
        p = resolve_model_profile(tmp_path, "test/model-name")
        assert p.canonical_model == "test/model-name"

    def test_resolve_model_profile_no_match_raises(self, tmp_path: Path) -> None:
        profiles_dir = tmp_path / "model_profiles"
        profiles_dir.mkdir()
        (profiles_dir / "qwen3.toml").write_text(VALID_PROFILE_TOML_KWARGS)
        with pytest.raises(FileNotFoundError, match="No model profile found"):
            resolve_model_profile(tmp_path, "nonexistent/model")

    def test_resolve_model_profile_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model profiles directory not found"):
            resolve_model_profile(tmp_path, "any/model")
