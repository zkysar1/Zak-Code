"""The ``local_only`` cost guarantee and the endpoint classification behind it.

``local_only`` is the one switch that must never fail open: it exists so an operator can
run many agents against self-hosted hardware and be CERTAIN none of them quietly failed
over to a metered API. Overspending is irreversible, so these tests pin the refusal, not
just the happy path.

Enforcement is deliberately two-layered (rb-605 — anticipation gates warn, application
gates guarantee), and both layers are pinned here:

* ``Agent._assert_local_only`` — startup, names every offender at once.
* ``LiteLLMProvider._build_kwargs`` — the call itself, catches routes nobody enumerated.
"""

from __future__ import annotations

import pytest

from zakcode.config import Settings
from zakcode.providers.endpoints import (
    LocalOnlyViolation,
    classify_destination,
    is_local_model,
    is_sentinel,
    model_uses_generic_endpoint,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import ZakpickModel

POD = "http://zakpod1:9090/v1"


# ── classification: where does a call actually go? ────────────────────────────


@pytest.mark.parametrize(
    ("model", "api_base", "expected_local"),
    [
        # Self-hosted: a generic-OpenAI model redirected by api_base reaches our own box.
        ("openai/zds-qwen3.8-27b", POD, True),
        ("openai_like/whatever", POD, True),
        ("hosted_vllm/mixtral", POD, True),
        ("bare-model-name", POD, True),
        # Ollama is always local, with or without a configured base.
        ("ollama_chat/llama3.1", None, True),
        ("ollama/llama3.1", POD, True),
        # Generic path with NO base falls through to the vendor's own host — metered.
        ("openai/gpt-4o-mini", None, False),
        ("bare-model-name", None, False),
        # Named clouds ignore api_base by the allowlist, so a base cannot make them local.
        # This is the pairing that matters: the SAME api_base that localizes openai/* must
        # NOT be read as localizing groq/*, or the guarantee would be a lie.
        ("groq/qwen/qwen3-32b", POD, False),
        ("anthropic/claude-sonnet-4-6", POD, False),
    ],
)
def test_classify_destination(model: str, api_base: str | None, expected_local: bool) -> None:
    ok, reason = classify_destination(model, api_base)
    assert ok is expected_local
    assert reason, "a refusal must always carry an operator-actionable reason"
    assert is_local_model(model, api_base) is expected_local


def test_classification_agrees_with_the_router() -> None:
    """The cost predicate and the api_base allowlist must never disagree.

    If a model is classified local BECAUSE api_base redirects it, then the router must
    actually forward that api_base. A guarantee that disagrees with the router is not a
    guarantee — this is why both live in ``providers.endpoints``.
    """
    for model in ("openai/x", "openai_like/x", "hosted_vllm/x", "bare"):
        assert model_uses_generic_endpoint(model) is True
        assert is_local_model(model, POD) is True
    for model in ("groq/x", "anthropic/x"):
        assert model_uses_generic_endpoint(model) is False
        assert is_local_model(model, POD) is False


def test_sentinels_are_refused_not_guessed() -> None:
    """``auto``/``zakpick`` name no destination; classifying one would be a coin flip."""
    assert is_sentinel("auto") and is_sentinel("zakpick") and is_sentinel("ZakPick")
    assert not is_sentinel("openai/gpt-4o")
    for sentinel in ("auto", "zakpick"):
        with pytest.raises(ValueError, match="sentinel"):
            classify_destination(sentinel, POD)


# ── layer 2: the application gate (the actual guarantee) ──────────────────────


def test_metered_call_is_refused_at_the_request_builder() -> None:
    provider = LiteLLMProvider(
        Settings(default_model="groq/qwen/qwen3-32b", local_only=True, _env_file=None)
    )
    with pytest.raises(LocalOnlyViolation, match="metered"):
        provider._build_kwargs([{"role": "user", "content": "hi"}], None)


def test_self_hosted_call_is_allowed() -> None:
    provider = LiteLLMProvider(
        Settings(
            default_model="openai/zds-qwen3.8-27b",
            api_base=POD,
            api_key="sk-noop",
            local_only=True,
            _env_file=None,
        )
    )
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert kwargs["api_base"] == POD


def test_generic_model_without_a_base_is_refused() -> None:
    """The subtle one: ``openai/gpt-4o-mini`` LOOKS like the self-hosted shape but, with no
    api_base, litellm sends it to api.openai.com and bills it."""
    provider = LiteLLMProvider(
        Settings(default_model="openai/gpt-4o-mini", local_only=True, _env_file=None)
    )
    with pytest.raises(LocalOnlyViolation, match="no api_base is configured"):
        provider._build_kwargs([{"role": "user", "content": "hi"}], None)


def test_violation_is_not_a_provider_error() -> None:
    """A LocalOnlyViolation must NOT ride the loop's failover machinery.

    ProviderError is recoverable: the loop answers it by trying a DIFFERENT model. For a
    cost refusal that is the one reaction that could turn a single blocked call into the
    very spend the switch exists to prevent. So the violation is deliberately outside that
    hierarchy and nothing retries it.
    """
    from zakcode.providers.base import ProviderError

    assert not issubclass(LocalOnlyViolation, ProviderError)
    assert issubclass(LocalOnlyViolation, RuntimeError)


# ── layer 1: the startup gate (anticipation) ──────────────────────────────────
#
# _assert_local_only reads only ``self.settings`` and ``self._zakpick``, so it is exercised
# against a stub rather than a fully-constructed Agent (which would build providers and
# probe the network). Same code path, no side effects.


class _Stub:
    def __init__(self, settings: Settings, *, zakpick: bool = False) -> None:
        self.settings = settings
        self._zakpick = zakpick


def _assert(settings: Settings, *, zakpick: bool = False) -> None:
    from zakcode import Agent

    Agent._assert_local_only(_Stub(settings, zakpick=zakpick))


def test_startup_gate_passes_for_a_fully_self_hosted_config() -> None:
    _assert(
        Settings(
            default_model="openai/zds-qwen3.8-27b",
            api_base=POD,
            local_only=True,
            _env_file=None,
        )
    )


def test_startup_gate_catches_a_metered_fallback_model() -> None:
    """The headline case: the primary is local, but a failover would silently bill."""
    with pytest.raises(LocalOnlyViolation) as exc:
        _assert(
            Settings(
                default_model="openai/zds-qwen3.8-27b",
                fallback_model="groq/qwen/qwen3-32b",
                api_base=POD,
                local_only=True,
                _env_file=None,
            )
        )
    assert "fallback_model" in str(exc.value)


def test_startup_gate_catches_unset_zakpick_categories() -> None:
    """The case that makes this gate worth writing.

    An operator sets local_only, switches to zakpick, and points deep_code at the pod —
    a config that LOOKS local. The five categories they did not override still route, and
    they fall through to the built-in Groq/OpenAI defaults. Checking only the configured
    overrides would report this clean, which is the most expensive possible false negative.
    """
    settings = Settings(
        default_model="openai/zds-qwen3.8-27b",  # already resolved from the zakpick sentinel
        api_base=POD,
        local_only=True,
        zakpick_models={"deep_code": ZakpickModel(model="zds-qwen3.8-27b", source="openai")},
        _env_file=None,
    )
    with pytest.raises(LocalOnlyViolation) as exc:
        _assert(settings, zakpick=True)
    message = str(exc.value)
    # Every groq-defaulted category is named, so the operator fixes them in one pass.
    for category in ("quick_code", "summarize", "plan", "classify"):
        assert f"zakpick category '{category}'" in message
    # ...and the one they DID point at the pod is not falsely accused.
    assert "zakpick category 'deep_code'" not in message
    # 'delegate' is ALSO absent, and the reason is worth stating because it looks like a
    # hole and is not one: its default is openai/gpt-4o-mini, which with api_base set is a
    # generic-OpenAI name, so litellm sends it to the POD rather than to OpenAI. No money
    # is spent, so local_only is right to permit it. What the operator gets instead is a
    # request for a model the pod does not host, which the proxy answers from its default
    # engine with x-zds-fallback: true — wrong model, right invoice. That is a ROUTING
    # hazard, not a cost one, and it belongs to the separate "fail loudly on an unknown
    # alias" fix. Pinned so a future reader does not read this omission as a cost leak.
    assert "zakpick category 'delegate'" not in message


def test_startup_gate_is_inert_when_local_only_is_off() -> None:
    _assert(
        Settings(
            default_model="groq/qwen/qwen3-32b",
            fallback_model="anthropic/claude-sonnet-4-6",
            local_only=False,
            _env_file=None,
        )
    )


def test_local_only_defaults_off_and_changes_nothing(monkeypatch) -> None:
    """guard-1562: a fail-closed check must newly refuse only what the operator opted into."""
    monkeypatch.delenv("ZAKCODE_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("ZAKCODE_API_BASE", raising=False)
    assert Settings(_env_file=None).local_only is False
    provider = LiteLLMProvider(Settings(default_model="groq/qwen/qwen3-32b", _env_file=None))
    assert provider.local_only is False
    assert provider._build_kwargs([{"role": "user", "content": "hi"}], None)["model"] == (
        "groq/qwen/qwen3-32b"
    )


# ── extra_body passthrough + per-category thinking ────────────────────────────
#
# Mechanism verified end-to-end on 2026-08-17 before any of this was written:
#   * llama.cpp honours {"chat_template_kwargs": {"enable_thinking": false}} —
#     completion_tokens 36 -> 4 on a trivial prompt, answer unchanged.
#   * litellm keeps extra_body through get_optional_params with drop_params=True.
# A per-request ``reasoning_budget`` was ALSO measured and is IGNORED (64 and 256 both
# produced ~13k reasoning chars vs a 13,130 baseline), which is why thinking is a bool.


def test_extra_body_absent_by_default() -> None:
    """The default request shape must be byte-identical to before this feature."""
    provider = LiteLLMProvider(Settings(default_model="openai/gpt-4o", _env_file=None))
    assert "extra_body" not in provider._build_kwargs([{"role": "user", "content": "hi"}], None)


def test_extra_body_reaches_the_request() -> None:
    provider = LiteLLMProvider(
        Settings(
            default_model="openai/zds-qwen3.8-27b",
            api_base=POD,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            _env_file=None,
        )
    )
    kwargs = provider._build_kwargs([{"role": "user", "content": "hi"}], None)
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_zakpick_thinking_becomes_extra_body() -> None:
    assert ZakpickModel(model="m", source="openai").extra_body == {}
    assert ZakpickModel(model="m", source="openai", thinking=False).extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert ZakpickModel(model="m", source="openai", thinking=True).extra_body == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_same_model_different_thinking_gets_different_providers() -> None:
    """The cache-key case. Two categories may name the SAME model and want opposite
    thinking; a model-only cache key would hand the second one the first one's provider and
    silently apply the wrong setting — the kind of bug that shows up as a latency mystery."""
    from zakcode import Agent

    agent = object.__new__(Agent)
    agent.settings = Settings(default_model="openai/zds-qwen3.8-27b", api_base=POD, _env_file=None)
    agent._provider_injected = False
    agent._provider_cache = {}

    off = agent._provider_for(
        "openai/zds-qwen3.8-27b", extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    on = agent._provider_for(
        "openai/zds-qwen3.8-27b", extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
    assert off is not on
    assert len(agent._provider_cache) == 2


def test_per_category_thinking_merges_over_the_global_extra_body() -> None:
    """A global knob and a per-category one must compose, not clobber."""
    from zakcode import Agent

    agent = object.__new__(Agent)
    agent.settings = Settings(
        default_model="openai/zds-qwen3.8-27b",
        api_base=POD,
        extra_body={"seed": 42},
        _env_file=None,
    )
    agent._provider_injected = False
    agent._provider_cache = {}

    built = agent._build_provider(
        "openai/other-model", extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    inner = built.inner if hasattr(built, "inner") else built
    body = inner.extra_body
    assert body["seed"] == 42, "the global knob must survive"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_per_call_extra_body_merges_over_the_instance_body() -> None:
    """The reasoning-overflow retry (ADR-0056) sends the thinking switch for ONE request;
    it must compose with the instance's body (a category knob, a global seed), not replace
    it, and its absence must leave the request shape unchanged."""
    from zakcode.providers.routing import thinking_extra_body

    provider = LiteLLMProvider(
        Settings(
            default_model="openai/zds-qwen3.8-27b",
            api_base=POD,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}, "seed": 42},
            _env_file=None,
        )
    )
    msgs = [{"role": "user", "content": "hi"}]
    assert provider._build_kwargs(msgs, None)["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "seed": 42,
    }
    merged = provider._build_kwargs(msgs, None, extra_body=thinking_extra_body(False))
    assert merged["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "seed": 42,
    }
    bare = LiteLLMProvider(Settings(default_model="openai/gpt-4o", _env_file=None))
    only = bare._build_kwargs(msgs, None, extra_body=thinking_extra_body(False))
    assert only["extra_body"] == thinking_extra_body(False)


# ---- extra_headers: one config line, N distinguishable terminals ----
def test_extra_headers_expand_hostname_and_pid_per_process():
    """The property that makes fleet attribution work without per-terminal provisioning."""
    import os
    import socket

    from zakcode.providers.litellm_provider import _expand_headers

    out = _expand_headers({"X-ZDS-Instance": "{hostname}-{pid}"})
    assert out["X-ZDS-Instance"] == f"{socket.gethostname()}-{os.getpid()}"


def test_extra_headers_tolerate_unknown_placeholders_and_stray_braces():
    """A header value is arbitrary text, not a format string — str.format would raise here
    and take inference down over a typo."""
    from zakcode.providers.litellm_provider import _expand_headers

    assert _expand_headers({"A": "{nope}"}) == {"A": "{nope}"}
    assert _expand_headers({"A": "a{b"}) == {"A": "a{b"}
    assert _expand_headers(None) == {}


def test_extra_headers_reach_the_provider_call_kwargs():
    """Wiring test: a header configured in Settings must actually be sent."""
    from zakcode.config import Settings
    from zakcode.providers.litellm_provider import LiteLLMProvider

    s = Settings(
        default_model="openai/zds-qwen3.8-27b",
        api_base="http://10.0.0.250:9090/v1",
        extra_headers={"X-ZDS-Instance": "{hostname}-{pid}"},
    )
    p = LiteLLMProvider(s)
    assert "X-ZDS-Instance" in p.extra_headers
    assert "{hostname}" not in p.extra_headers["X-ZDS-Instance"], "must be expanded at build"
