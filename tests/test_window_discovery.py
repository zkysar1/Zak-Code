"""ADR-0066: the context window comes from the model's config entry, never a default.

Field 2026-08-28 (coach on zc-03): the route model ``openai/zds-qwen3.8-27b`` is an alias
the static table does not know and litellm has no metadata for, so capabilities fell to
an 8,192-token stand-in while the server ran a 131,072 context. Everything keyed on the
window was wrong by 16×: the seam clamp cut every tool result to 6 KB (a 39 KB /boot lost
Steps 0–11), and the pre-turn compaction threshold sat at 6.5k tokens. ADR-0065 first made
the server's ``GET /v1/models`` listing the source for such a model; ADR-0066 makes the
model's own config entry the source and the listing a CHECK — and a model nobody knows the
window of refuses to run, with the server's figure in the message. Hermetic: the HTTP
fetch is monkeypatched; no socket is opened.
"""

from __future__ import annotations

from typing import Any

import pytest

from zakcode.providers import litellm_provider as lp
from zakcode.providers.base import UnknownContextWindow, WindowResolution
from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    discover_context_window,
    resolve_context_window,
)

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


# ── the pure reader (unchanged from ADR-0065: the check needs it) ───────────────


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


# ── the resolver: config → registry → refuse; the listing only checks ───────────


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
ALIAS = "openai/zds-qwen3.8-27b"


def test_a_declared_window_is_the_source_and_is_not_verified_unless_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    res = resolve_context_window(ALIAS, 131072, api_base=POD, api_key="sk-pod")
    assert res == WindowResolution(131072, "config", None)
    assert fetch.calls == []  # a known window is not probed on the request path


def test_verify_asks_the_server_and_flags_a_mismatch_but_config_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    res = resolve_context_window(ALIAS, 32768, api_base=POD, api_key="sk-pod", verify=True)
    assert res.window == 32768 and res.source == "config"
    assert res.served == 131072 and res.mismatch is True
    assert "server declares 131,072" in res.describe()
    agree = resolve_context_window(ALIAS, 131072, api_base=POD, verify=True)
    assert agree.mismatch is False and agree.describe() == "131,072 (config), server agrees"


def test_a_registry_model_needs_no_declaration() -> None:
    res = resolve_context_window("openai/gpt-4o")
    assert res.window == 128_000 and res.source == "registry"


def test_an_unknown_model_resolves_to_unknown_with_the_servers_figure_offered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    res = resolve_context_window(ALIAS, None, api_base=POD, api_key="sk-pod")
    assert res == WindowResolution(None, "unknown", 131072)
    assert fetch.calls == [(POD, "sk-pod")]
    message = lp.unknown_window_message(ALIAS, res, POD)
    assert '"context_window": 131072' in message
    assert ALIAS in message and POD in message


def test_an_unknown_model_with_no_listing_says_where_to_look(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp, "_fetch_models", _Fetch(error=OSError("connection refused")))
    res = resolve_context_window(ALIAS, None, api_base=POD)
    assert res.window is None and res.served is None
    assert "model card" in lp.unknown_window_message(ALIAS, res, POD)
    assert "no server was configured" in lp.unknown_window_message(ALIAS, res, None)


def test_a_routing_sentinel_has_no_window_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    assert resolve_context_window("zakpick", api_base=POD, verify=True).source == "sentinel"
    sentinel = LiteLLMProvider(
        model="zakpick", api_base=POD, local_only=True, local_api_bases=[POD]
    )
    assert sentinel.capabilities().context_window is None
    assert fetch.calls == []


def test_local_only_never_asks_an_unlisted_base(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    metered = resolve_context_window(
        ALIAS, None, api_base=POD, local_only=True, local_api_bases=["http://other.test/v1"]
    )
    assert metered.served is None and fetch.calls == []
    listed = resolve_context_window(
        ALIAS, None, api_base=POD, api_key="sk-pod", local_only=True, local_api_bases=[POD]
    )
    assert listed.served == 131072 and fetch.calls == [(POD, "sk-pod")]


def test_a_listing_cache_asks_each_base_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    cache: dict[str, Any] = {}
    for _ in range(3):
        resolve_context_window(ALIAS, 131072, api_base=POD, verify=True, listing_cache=cache)
    assert len(fetch.calls) == 1


# ── the provider: refuses to exist without a window, reports the one it has ─────


def test_a_provider_carries_its_declared_window(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    provider = LiteLLMProvider(model=ALIAS, api_base=POD, api_key="sk-pod", context_window=131072)
    assert provider.capabilities().context_window == 131072
    assert provider.window.source == "config"
    assert fetch.calls == []


def test_a_provider_for_an_unknown_model_refuses_with_the_number_to_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lp, "_fetch_models", _Fetch(ZDS))
    with pytest.raises(UnknownContextWindow, match=r'"context_window": 131072'):
        LiteLLMProvider(model=ALIAS, api_base=POD, api_key="sk-pod")


def test_a_known_model_never_needs_a_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = _Fetch(ZDS)
    monkeypatch.setattr(lp, "_fetch_models", fetch)
    provider = LiteLLMProvider(model="openai/gpt-4o", api_base=POD)
    assert provider.capabilities().context_window == 128_000
    assert provider.window.source == "registry"
    assert fetch.calls == []


def test_settings_context_window_describes_the_default_model() -> None:
    from zakcode.config import Settings

    settings = Settings(default_model=ALIAS, context_window=65536)
    provider = LiteLLMProvider(settings)
    assert provider.capabilities().context_window == 65536
    # An explicit kwarg (a routed model's own entry) wins over the settings' value.
    routed = LiteLLMProvider(settings, model=ALIAS, context_window=131072)
    assert routed.capabilities().context_window == 131072
