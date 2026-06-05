"""Tests for pointing Zak Code at a custom OpenAI-compatible endpoint via config.

These verify the wiring only (no network): a configured ``api_base`` / ``api_key`` must
flow Settings -> LiteLLMProvider -> the request kwargs handed to litellm.
"""

from __future__ import annotations

from zakcode.config import Settings, load_settings
from zakcode.providers.litellm_provider import LiteLLMProvider


def test_settings_api_base_and_key_default_none() -> None:
    s = Settings()
    assert s.api_base is None
    assert s.api_key is None


def test_settings_api_base_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ZAKCODE_API_BASE", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("ZAKCODE_API_KEY", "sk-local-noop")
    s = load_settings()
    assert s.api_base == "http://127.0.0.1:8000/v1"
    assert s.api_key == "sk-local-noop"


def test_provider_reads_endpoint_from_settings() -> None:
    s = Settings(
        default_model="openai/qwen2.5-coder",
        api_base="http://127.0.0.1:8000/v1",
        api_key="sk-local-noop",
    )
    provider = LiteLLMProvider(s)
    assert provider.api_base == "http://127.0.0.1:8000/v1"
    assert provider.api_key == "sk-local-noop"


def test_explicit_kwargs_override_settings() -> None:
    s = Settings(default_model="openai/x", api_base="http://from-settings/v1")
    provider = LiteLLMProvider(s, api_base="http://from-kwarg/v1")
    assert provider.api_base == "http://from-kwarg/v1"


def test_endpoint_flows_into_request_kwargs() -> None:
    """A configured api_base/api_key must appear in the kwargs sent to litellm."""
    s = Settings(
        default_model="openai/qwen2.5-coder",
        api_base="http://127.0.0.1:8000/v1",
        api_key="sk-local-noop",
    )
    provider = LiteLLMProvider(s)
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert kwargs["api_key"] == "sk-local-noop"
    assert kwargs["model"] == "openai/qwen2.5-coder"


# ── C1: Ollama num_ctx lift + matching capability window ──────────────────────


def test_ollama_num_ctx_set_and_clamped() -> None:
    from zakcode.providers.litellm_provider import _OLLAMA_NUM_CTX_CAP

    provider = LiteLLMProvider(Settings(default_model="ollama_chat/qwen2.5:3b"))
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert "num_ctx" in kwargs
    assert 0 < kwargs["num_ctx"] <= _OLLAMA_NUM_CTX_CAP


def test_num_ctx_absent_for_non_ollama() -> None:
    provider = LiteLLMProvider(Settings(default_model="openai/gpt-4o"))
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert "num_ctx" not in kwargs


def test_caller_num_ctx_override_wins() -> None:
    provider = LiteLLMProvider(Settings(default_model="ollama_chat/qwen2.5:3b"))
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None, num_ctx=2048)
    assert kwargs["num_ctx"] == 2048


def test_ollama_capability_window_matches_num_ctx_cap() -> None:
    from zakcode.providers.litellm_provider import _OLLAMA_NUM_CTX_CAP

    # qwen2.5:3b resolves (via the registry :tag fallback) to qwen2.5's 32k window,
    # then clamps so the compactor threshold matches the num_ctx we actually request.
    provider = LiteLLMProvider(Settings(default_model="ollama_chat/qwen2.5:3b"))
    assert provider.capabilities().context_window == _OLLAMA_NUM_CTX_CAP
