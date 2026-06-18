"""Tests for pointing Zak Code at a custom OpenAI-compatible endpoint via config.

These verify the wiring only (no network): a configured ``api_base`` / ``api_key`` must
flow Settings -> LiteLLMProvider -> the request kwargs handed to litellm.
"""

from __future__ import annotations

import pytest

from zakcode.config import Settings, load_settings
from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    _model_uses_generic_endpoint,
)


def test_settings_api_base_and_key_default_none(monkeypatch) -> None:
    # Hermeticity: a dev `.env` reaches Settings two ways — pydantic reads the file, and
    # load_settings' load_dotenv step exports it into os.environ. Clear both so this asserts the
    # true unconfigured defaults (CI has no `.env`; keeps local `poe check` matching CI).
    monkeypatch.delenv("ZAKCODE_API_BASE", raising=False)
    monkeypatch.delenv("ZAKCODE_API_KEY", raising=False)
    s = Settings(_env_file=None)
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


def test_api_key_not_forwarded_for_cloud_model_without_api_base(monkeypatch) -> None:
    # audit3 #6: with no api_base (a bare cloud model), a configured api_key must NOT be
    # forwarded — litellm reads the real key from OPENAI_API_KEY; a stale ZAKCODE_API_KEY
    # shadowing it would break the config-only cloud switch with a spurious AuthError.
    # Hermeticity: clear a dev `.env`'s ZAKCODE_API_BASE (env var + file) so the "no api_base"
    # precondition holds locally too (CI has no `.env`; keeps local `poe check` matching CI).
    monkeypatch.delenv("ZAKCODE_API_BASE", raising=False)
    s = Settings(default_model="openai/gpt-4o", api_key="sk-stale-local-dummy", _env_file=None)
    provider = LiteLLMProvider(s)
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert "api_key" not in kwargs


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


# ── api_base provider-scoping (2026-06-17 groq-failover regression) ───────────
#
# The bug: a configured generic ZAKCODE_API_BASE (the local llama.cpp llama-server)
# was forwarded to EVERY call, including ``groq/...``. That rerouted the cloud call
# to the local server, which failed, and the turn failed over to openai/gpt-4o-mini.
# The fix gates api_base forwarding to OpenAI-compatible models only, so local
# (llama.cpp, ``openai/`` prefix) and Groq cloud (``groq/`` prefix) coexist with a
# single configured base. "local" == llama.cpp llama-server (``openai/`` + api_base);
# ollama (``ollama_chat/``) is a separate named-provider path and never inherits the base.


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # llama.cpp llama-server (the local case the base is configured for) -> forward.
        ("openai/qwen3.5-9b", True),
        ("openai/gpt-4o-mini", True),
        ("openai_like/some-server-model", True),
        ("hosted_vllm/mixtral", True),
        # Bare model name -> generic OpenAI path -> forward.
        ("gpt-4o-mini", True),
        # Named clouds carry their own base URL -> never forward the local base.
        ("groq/openai/gpt-oss-20b", False),
        ("groq/openai/gpt-oss-120b", False),
        ("anthropic/claude-sonnet-4-6", False),
        # ollama is a separate named provider (OLLAMA_API_BASE), not the local llama.cpp base.
        ("ollama_chat/qwen2.5:3b", False),
        ("ollama/llama3.1", False),
        # Defensive: empty model -> generic (matches the bare-name default).
        ("", True),
    ],
)
def test_model_uses_generic_endpoint(model: str, expected: bool) -> None:
    assert _model_uses_generic_endpoint(model) is expected


def test_api_base_not_forwarded_to_groq_model() -> None:
    """Regression: a configured local api_base must NOT reach a ``groq/`` call.

    This is the 2026-06-17 failover bug — the local llama.cpp base was attached to the
    Groq cloud call, broke it, and forced a fallback to openai/gpt-4o-mini.
    """
    s = Settings(
        default_model="groq/openai/gpt-oss-20b",
        api_base="http://127.0.0.1:8080/v1",  # local llama.cpp llama-server
        api_key="sk-local-noop",
    )
    provider = LiteLLMProvider(s)
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert "api_base" not in kwargs
    assert "api_key" not in kwargs  # api_key only rides alongside a forwarded api_base
    assert kwargs["model"] == "groq/openai/gpt-oss-20b"


def test_api_base_forwarded_to_local_llamacpp_model_with_groq_base_configured() -> None:
    """Coexistence: with the SAME configured base, an ``openai/`` (llama.cpp) model gets it.

    Proves local + Groq share one ZAKCODE_API_BASE — the local lane still works after the
    scoping fix, it is just no longer leaked onto cloud calls.
    """
    s = Settings(
        default_model="openai/qwen3.5-9b",  # llama.cpp llama-server
        api_base="http://127.0.0.1:8080/v1",
        api_key="sk-local-noop",
    )
    provider = LiteLLMProvider(s)
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert kwargs["api_base"] == "http://127.0.0.1:8080/v1"
    assert kwargs["api_key"] == "sk-local-noop"
