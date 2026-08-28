"""ADR-0065: a local pod declares its own context window.

Field 2026-08-28 (coach on zc-03): the route model ``openai/zds-qwen3.8-27b`` is an alias
the static table does not know and litellm has no metadata for, so capabilities fell to the
8,192-token default while the server ran a 131,072 context. Everything keyed on the window
was wrong by 16×: the seam clamp cut every tool result to 6 KB (a 39 KB /boot lost Steps
0–11), and the pre-turn compaction threshold sat at 6.5k tokens. The server had been
announcing the real figure the whole time in ``GET /v1/models`` (``zds.ctx_per_engine``),
as vLLM does in ``max_model_len`` and llama.cpp in ``meta.n_ctx_train``. Hermetic: the
HTTP fetch is monkeypatched; no socket is opened.
"""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.providers import litellm_provider as lp
from zakcode.providers.litellm_provider import LiteLLMProvider, discover_context_window

ZDS = {
    "object": "list",
    "data": [
        {"id": "zds-qwen3.6-35b", "object": "model", "zds": {"ctx_per_engine": 131072}},
        {
            "id": "zds-qwen3.8-27b",
            "object": "model",
            "zds": {"canonical": "zds-qwen3.6-35b", "ctx_per_engine": 131072},
        },
    ],
}
ZDS_FANOUT = {  # rb-8892: ctx_per_engine is the engine total; 3 slots per engine share it
    "data": [
        {
            "id": "zds-qwen3.6-27b",
            "zds": {"ctx_per_engine": 131072, "slots_total": 12, "replicas": 4},
        }
    ]
}
VLLM = {"data": [{"id": "Qwen/Qwen3-32B", "max_model_len": 40960}]}
LLAMA_CPP = {"data": [{"id": "qwen3-8b.gguf", "meta": {"n_ctx_train": 32768, "n_params": 8}}]}


# ── the pure reader ────────────────────────────────────────────────────────────


def test_reads_the_window_the_server_declares() -> None:
    assert discover_context_window(ZDS, "openai/zds-qwen3.8-27b") == 131072  # prefix stripped
    assert discover_context_window(ZDS, "zds-qwen3.6-35b") == 131072
    assert discover_context_window(VLLM, "openai/Qwen/Qwen3-32B") == 40960
    assert discover_context_window(LLAMA_CPP, "qwen3-8b.gguf") == 32768


def test_a_zds_slot_fan_out_divides_the_engine_total() -> None:
    assert discover_context_window(ZDS_FANOUT, "openai/zds-qwen3.6-27b") == 43690
    assert discover_context_window(ZDS, "zds-qwen3.6-35b") == 131072  # one slot per engine


def test_unknown_model_or_bogus_value_reads_as_nothing() -> None:
    assert discover_context_window(ZDS, "openai/gpt-4o") is None
    assert discover_context_window({"data": [{"id": "m", "max_model_len": "big"}]}, "m") is None
    assert discover_context_window({"data": [{"id": "m", "max_model_len": True}]}, "m") is None
    assert discover_context_window({"data": [{"id": "m", "max_model_len": 0}]}, "m") is None
    assert discover_context_window({"data": "nope"}, "m") is None
    assert discover_context_window(None, "m") is None


# ── the provider probes once, fails open, and respects LOCAL_ONLY ───────────────


class _Fetch:
    def __init__(self, payload: Any = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, api_base: str, api_key: str | None, timeout: float) -> Any:
        self.calls.append((api_base, api_key))
        if self.error is not None:
            raise self.error
        return self.payload


POD = "http://pod.test:9090/v1"


def _pod_provider(**kw: Any) -> LiteLLMProvider:
    return LiteLLMProvider(model="openai/zds-qwen3.8-27b", api_base=POD, api_key="sk-pod", **kw)


def test_unknown_pod_model_takes_the_servers_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    provider = _pod_provider()
    assert provider.capabilities().context_window == 131072
    assert provider.capabilities().context_window == 131072
    assert fetch.calls == [(POD, "sk-pod")]  # probed once, then remembered


def test_a_failed_probe_keeps_the_default_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _Fetch(error=OSError("connection refused"))
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    provider = _pod_provider()
    assert provider.capabilities().context_window == 8192
    assert provider.capabilities().context_window == 8192
    assert len(fetch.calls) == 1


def test_a_listing_without_the_model_keeps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(VLLM)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    assert _pod_provider().capabilities().context_window == 8192


def test_a_known_model_is_never_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    provider = LiteLLMProvider(model="openai/gpt-4o", api_base=POD)
    assert provider.capabilities().context_window == 128_000
    assert fetch.calls == []


def test_no_api_base_means_no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    provider = LiteLLMProvider(model="openai/zds-qwen3.8-27b")
    assert provider.capabilities().context_window == 8192
    assert fetch.calls == []


def test_local_only_never_probes_an_unlisted_base(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    metered = _pod_provider(local_only=True, local_api_bases=["http://other.test/v1"])
    assert metered.capabilities().context_window == 8192
    assert fetch.calls == []
    listed = _pod_provider(local_only=True, local_api_bases=[POD])
    assert listed.capabilities().context_window == 131072
    assert fetch.calls == [(POD, "sk-pod")]
