"""PKG-AUTO (D19/D21): the ``default_model: "auto"`` availability resolver.

Hermetic: every probe is an injected fake (no network ever). Covers the detection
acceptance matrix (local / external / preference order / nothing-viable), the
tools-unreliable capability skip (omni's D21 ruling), probe caching + fresh-probe
on failover, the explicit bypass, the fallback_model override, and the loop's
once-per-turn ``model_failover`` seam on both the buffered and streaming paths.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import zakcode
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
)
from zakcode.providers.registry import get_capabilities
from zakcode.providers.resolve import (
    AvailabilityResolver,
    ModelResolutionError,
    clear_probe_cache,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # Probe cache is process-global; resolution reads settings from the cwd .env,
    # the user config home, and real env vars — isolate ALL of them so a dev box's
    # .env (e.g. a configured ZAKCODE_FALLBACK_MODEL) can never change a verdict.
    clear_probe_cache()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in ("ZAKCODE_FALLBACK_MODEL", "ZAKCODE_DEFAULT_MODEL", "ZAKCODE_AUTO_MODEL_PREFERENCE"):
        monkeypatch.delenv(var, raising=False)
    yield
    clear_probe_cache()


class FakeProbe:
    """Scripted probe: url-substring → (status, body); counts calls."""

    def __init__(self, responses: dict[str, tuple[int, str]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def __call__(self, url: str, headers: dict[str, str]) -> tuple[int, str]:
        self.calls.append(url)
        for fragment, response in self.responses.items():
            if fragment in url:
                status, body = response
                if status < 0:
                    raise ConnectionError("scripted probe failure")
                return status, body
        raise ConnectionError("no scripted response")


_OLLAMA_UP = (200, json.dumps({"models": [{"name": "llama3.1:8b"}]}))
_OLLAMA_EMPTY = (200, json.dumps({"models": []}))
_DOWN = (-1, "")
_OK = (200, "{}")


def _resolver(probe: FakeProbe, preference: list[str] | None = None) -> AvailabilityResolver:
    return AvailabilityResolver(
        ollama_base_url="http://localhost:11434",
        preference=preference or ["groq", "openai", "anthropic"],
        probe=probe,
    )


# ── the detection acceptance matrix ───────────────────────────────────────────


def test_local_ollama_wins_when_up() -> None:
    probe = FakeProbe({"api/tags": _OLLAMA_UP})
    resolved = _resolver(probe).resolve(require_tools=True)
    assert resolved.model == "ollama_chat/llama3.1:8b"
    assert resolved.source == "local"


def test_first_viable_external_when_local_down(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    probe = FakeProbe({"api/tags": _DOWN, "api.groq.com": _OK})
    resolved = _resolver(probe).resolve(require_tools=True)
    assert resolved.model == "groq/openai/gpt-oss-120b"  # tools-reliable groq default
    assert resolved.source == "groq"


def test_preference_order_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    probe = FakeProbe({"api/tags": _DOWN, "api.groq.com": _OK, "api.openai.com": _OK})
    resolved = _resolver(probe, preference=["openai", "groq"]).resolve(require_tools=True)
    assert resolved.source == "openai"
    assert resolved.model == "openai/gpt-4o-mini"


def test_missing_keys_skip_without_probing(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    probe = FakeProbe({"api/tags": (200, json.dumps({"models": [{"name": "x"}]}))})
    resolved = _resolver(probe).resolve(require_tools=True)
    assert resolved.source == "local"
    assert all("api." not in url for url in probe.calls)  # no key → no external probe


def test_nothing_viable_is_loud_with_diagnosis(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    probe = FakeProbe({"api/tags": _DOWN})
    with pytest.raises(ModelResolutionError) as info:
        _resolver(probe).resolve(require_tools=True)
    message = str(info.value)
    assert "local ollama" in message
    assert "GROQ_API_KEY not set" in message
    assert "OPENAI_API_KEY not set" in message
    assert "ZAKCODE_DEFAULT_MODEL" in message  # the fix guidance


def test_rejected_key_is_named(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_bad")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    probe = FakeProbe({"api/tags": _OLLAMA_EMPTY, "api.groq.com": (401, "")})
    with pytest.raises(ModelResolutionError) as info:
        _resolver(probe).resolve(require_tools=True)
    assert "present but rejected" in str(info.value)


# ── tools-unreliable is capability metadata (D21 ruling) ─────────────────────


def test_llama33_on_groq_is_marked_tools_unreliable() -> None:
    caps = get_capabilities("groq/llama-3.3-70b-versatile")
    assert caps.supports_tools is True  # nominal support stays true
    assert caps.tools_unreliable is True  # ...but the resolver must skip it
    assert get_capabilities("groq/openai/gpt-oss-120b").tools_unreliable is False
    assert Capabilities().tools_unreliable is False  # unknown models default reliable


def test_resolver_skips_tools_unreliable_models(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    probe = FakeProbe({"api/tags": _DOWN, "api.groq.com": _OK})
    # With the reliable candidates excluded, only llama-3.3 remains in groq — it is
    # skipped when tools are required...
    with pytest.raises(ModelResolutionError) as info:
        _resolver(probe, preference=["groq"]).resolve(
            require_tools=True,
            exclude={"groq/openai/gpt-oss-120b", "groq/qwen/qwen3-32b"},
        )
    assert "no tools-viable candidate" in str(info.value)
    # ...but viable for a tools-free session: reliability gates only tool use.
    clear_probe_cache()
    resolved = _resolver(probe, preference=["groq"]).resolve(
        require_tools=False,
        exclude={"groq/openai/gpt-oss-120b", "groq/qwen/qwen3-32b"},
    )
    assert resolved.model == "groq/llama-3.3-70b-versatile"


def test_exclude_moves_to_the_next_candidate(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    probe = FakeProbe({"api/tags": _DOWN, "api.groq.com": _OK})
    resolved = _resolver(probe).resolve(require_tools=True, exclude={"groq/openai/gpt-oss-120b"})
    assert resolved.model == "groq/qwen/qwen3-32b"


# ── probe caching: cached for the process, fresh on the failover path ────────


def test_probes_are_cached_within_the_process() -> None:
    probe = FakeProbe({"api/tags": _OLLAMA_UP})
    _resolver(probe).resolve(require_tools=True)
    _resolver(probe).resolve(require_tools=True)  # second resolver, same cache
    assert len(probe.calls) == 1


def test_failover_path_probes_fresh() -> None:
    probe = FakeProbe({"api/tags": _OLLAMA_UP})
    _resolver(probe).resolve(require_tools=True)
    fresh = AvailabilityResolver(
        ollama_base_url="http://localhost:11434",
        preference=["groq"],
        probe=probe,
        use_cache=False,
    )
    fresh.resolve(require_tools=True)
    assert len(probe.calls) == 2  # use_cache=False re-probed


# ── Agent integration: sentinel, bypass, loud failure, failover precedence ───


def _patch_probe(monkeypatch, probe: FakeProbe) -> None:
    import zakcode.providers.resolve as resolve_module

    monkeypatch.setattr(resolve_module, "_http_get", probe)


def test_agent_resolves_auto_at_startup(monkeypatch, tmp_path: Path) -> None:
    probe = FakeProbe({"api/tags": _OLLAMA_UP})
    _patch_probe(monkeypatch, probe)
    agent = zakcode.Agent(default_model="auto", workspace_root=tmp_path)
    assert agent.settings.default_model == "ollama_chat/llama3.1:8b"
    assert agent.session.model == "ollama_chat/llama3.1:8b"  # session records concrete
    assert agent.model_resolution is not None
    assert agent.model_resolution.source == "local"


def test_agent_auto_failure_is_loud(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_probe(monkeypatch, FakeProbe({"api/tags": _DOWN}))
    with pytest.raises(ModelResolutionError):
        zakcode.Agent(default_model="auto", workspace_root=tmp_path)


def test_explicit_model_bypasses_resolution(monkeypatch, tmp_path: Path) -> None:
    probe = FakeProbe({"api/tags": _OLLAMA_UP})
    _patch_probe(monkeypatch, probe)
    agent = zakcode.Agent(default_model="ollama_chat/qwen2.5", workspace_root=tmp_path)
    assert agent.model_resolution is None
    assert probe.calls == []  # explicit model: never probed


def test_fallback_model_is_the_explicit_failover_override(monkeypatch, tmp_path: Path) -> None:
    probe = FakeProbe({"api/tags": _OLLAMA_UP})
    _patch_probe(monkeypatch, probe)
    agent = zakcode.Agent(
        default_model="auto", fallback_model="openai/gpt-4o-mini", workspace_root=tmp_path
    )
    switched = agent._model_failover(RequestFailed("boom"))
    assert switched is not None
    _provider, note = switched
    assert "openai/gpt-4o-mini" in note
    assert "fallback_model" in note
    assert agent._active_model == "openai/gpt-4o-mini"


def test_auto_failover_reresolves_excluding_failed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    probe = FakeProbe({"api/tags": _DOWN, "api.groq.com": _OK})
    _patch_probe(monkeypatch, probe)
    agent = zakcode.Agent(default_model="auto", workspace_root=tmp_path)
    assert agent._active_model == "groq/openai/gpt-oss-120b"
    switched = agent._model_failover(RequestFailed("boom"))
    assert switched is not None
    assert agent._active_model == "groq/qwen/qwen3-32b"  # failed model excluded


def test_explicit_model_without_fallback_never_fails_over(monkeypatch, tmp_path: Path) -> None:
    _patch_probe(monkeypatch, FakeProbe({"api/tags": _OLLAMA_UP}))
    agent = zakcode.Agent(default_model="ollama_chat/qwen2.5", workspace_root=tmp_path)
    assert agent._model_failover(RequestFailed("boom")) is None


# ── the loop's model_failover seam (once per turn, both paths) ────────────────


class FailingProvider(Provider):
    """Raises ``error`` forever (or recovers after ``fail_count`` calls)."""

    def __init__(self, error: Exception | None, text: str = "recovered") -> None:
        self.error = error
        self.text = text
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> LLMResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return LLMResult(text=self.text)

    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        yield StreamTextDelta(text=self.text)
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _make_loop(provider: Provider, failover) -> AgentLoop:
    settings = load_settings(workspace_root=Path.cwd(), provider_max_retries=0)
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd="/tmp/work", model="test/model"),
        settings=settings,
        model_failover=failover,
    )


def test_buffered_failover_recovers_the_turn() -> None:
    good = FailingProvider(None, text="failover-recovered")
    calls: list[str] = []

    def failover(exc):
        calls.append(str(exc))
        return good, "test/model -> test/better (scripted)"

    loop = _make_loop(FailingProvider(RequestFailed("primary down")), failover)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "completed"
    assert result.assistant_messages[0].text == "failover-recovered"
    assert calls == ["primary down"]
    assert loop.provider is good  # the loop now drives the replacement


def test_failover_fires_at_most_once_per_turn() -> None:
    calls: list[str] = []

    def failover(exc):
        calls.append(str(exc))
        return FailingProvider(RequestFailed("also down")), "a -> b"

    loop = _make_loop(FailingProvider(RequestFailed("primary down")), failover)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert "also down" in result.error
    assert len(calls) == 1  # once per turn, then graceful stop


def test_rate_limited_exhaustion_does_not_fail_over() -> None:
    calls: list[str] = []

    def failover(exc):  # pragma: no cover — must never run
        calls.append(str(exc))
        return FailingProvider(None), "a -> b"

    loop = _make_loop(FailingProvider(RateLimited("429", retry_after=0.0)), failover)
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    assert calls == []  # non-rate-limit only (spec)


async def _collect(loop: AgentLoop, text: str) -> list[Any]:
    return [ev async for ev in loop.astream_turn(text)]


def test_streaming_failover_announces_the_switch() -> None:
    good = FailingProvider(None, text="failover-recovered")

    def failover(exc):
        return good, "test/model -> test/better (scripted)"

    loop = _make_loop(FailingProvider(RequestFailed("primary down")), failover)
    events = asyncio.run(_collect(loop, "hi"))
    assert events[-1].stop_reason == "completed"
    statuses = [getattr(ev, "message", "") for ev in events]
    assert any("switching model" in s for s in statuses)
    assert any(getattr(ev, "text", None) == "failover-recovered" for ev in events)


class MidstreamFailProvider(FailingProvider):
    async def astream(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> AsyncIterator[ProviderStreamEvent]:
        yield StreamTextDelta(text="partial")
        raise RequestFailed("mid-stream death")


def test_streaming_midstream_failure_never_fails_over() -> None:
    calls: list[str] = []

    def failover(exc):  # pragma: no cover — must never run
        calls.append(str(exc))
        return FailingProvider(None), "a -> b"

    loop = _make_loop(MidstreamFailProvider(None), failover)
    events = asyncio.run(_collect(loop, "hi"))
    assert events[-1].stop_reason == "provider_error"
    assert calls == []  # events already reached the client: terminal


# ── settings plumbing ─────────────────────────────────────────────────────────


def test_auto_preference_parses_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ZAKCODE_AUTO_MODEL_PREFERENCE", "openai, groq")
    settings = load_settings(workspace_root=tmp_path)
    assert settings.auto_model_preference == ["openai", "groq"]
