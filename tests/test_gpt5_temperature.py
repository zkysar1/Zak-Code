"""OpenAI gpt-5 reasoning models reject any temperature but the default (1).

WHY THIS FILE EXISTS
The gpt-5 reasoning tier (gpt-5, gpt-5.x, and named variants such as
gpt-5.6-terra / -luna, gpt-5-mini, gpt-5-nano) accepts ONLY temperature=1;
every other value is a hard 400 ("temperature does not support 0.7 with this
model. Only the default (1) value is supported"). Measured live against the
fleet key 2026-09-01: 0.7 -> 400, 1 -> 200, 0 -> 400.

litellm does NOT protect us. Its model map flags the gpt-5.6 tier with
``supports_none_reasoning_effort=True``, which its gpt-5 param-mapper reads as
"supports flexible temperature", so it forwards the configured value unchanged
-> OpenAI 400. This is true of BOTH the deployed litellm 1.86.2 and the latest
1.99.0, so a version bump is not the fix. The result in production: a zakpick
mix pinned to gpt-5.6-terra/-luna 400'd on every routed call and failed over to
the (temperature-locked, so litellm-dropped) gpt-5-mini fallback — the whole
world silently ran on the fallback while billing it as an unpriced model.

The fix normalises at ``LiteLLMProvider._build_kwargs`` — the one chokepoint
every completion path funnels through — AFTER the per-call ``kw`` update, so it
covers both the main loop's configured temperature and the structured path's
forced ``temperature=0`` (structured.py demands 0 for schema determinism, which
these models reject). A non-default value is DROPPED so the backend applies its
own default (1). The gpt-5-chat family (flexible temperature) is excluded, and
a gpt-5-named LOCAL model (custom api_base) is left alone.
"""

from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    _is_openai_gpt5_fixed_temperature_model,
)

MSGS = [{"role": "user", "content": "hi"}]


# ── the predicate ────────────────────────────────────────────────────────────
def test_predicate_matches_gpt5_reasoning_family() -> None:
    for m in (
        "gpt-5",
        "gpt-5.6-terra",
        "openai/gpt-5.6-terra",
        "openai/gpt-5.6-luna",
        "openai/gpt-5-mini",
        "openai/gpt-5-nano",
        "azure/gpt-5.6-terra",
        "openai/gpt-5.6-chat",  # versioned chat = reasoning model (litellm routes it so)
    ):
        assert _is_openai_gpt5_fixed_temperature_model(m), m


def test_predicate_excludes_gpt5_chat_and_other_models() -> None:
    for m in (
        "openai/gpt-5-chat",
        "openai/gpt-5-chat-latest",
        "openai/gpt-4o-mini",
        "groq/qwen/qwen3.6-27b",
        "openai/gpt-oss-20b",
        "ollama_chat/llama3",
    ):
        assert not _is_openai_gpt5_fixed_temperature_model(m), m


# ── the normalization at the build chokepoint ────────────────────────────────
def test_main_loop_temperature_dropped_for_gpt5() -> None:
    # The configured (non-1) temperature must NOT reach a gpt-5 reasoning model.
    p = LiteLLMProvider(model="openai/gpt-5.6-terra", temperature=0.7, context_window=400000)
    kwargs = p._build_kwargs(MSGS, None)
    assert "temperature" not in kwargs


def test_structured_temperature_zero_dropped_for_gpt5() -> None:
    # structured.py forces temperature=0 via **kw; the chokepoint must strip it too.
    p = LiteLLMProvider(model="openai/gpt-5.6-luna", context_window=400000)
    kwargs = p._build_kwargs(MSGS, None, temperature=0.0)
    assert "temperature" not in kwargs


def test_explicit_temperature_one_is_preserved_for_gpt5() -> None:
    # The one accepted value must survive untouched.
    p = LiteLLMProvider(model="openai/gpt-5-mini", temperature=1, context_window=400000)
    kwargs = p._build_kwargs(MSGS, None)
    assert kwargs["temperature"] == 1


def test_non_gpt5_temperature_untouched() -> None:
    # A normal model keeps whatever temperature was configured.
    p = LiteLLMProvider(model="openai/gpt-4o-mini", temperature=0.7, context_window=128000)
    kwargs = p._build_kwargs(MSGS, None)
    assert kwargs["temperature"] == 0.7


def test_local_gpt5_named_model_untouched() -> None:
    # A gpt-5-named self-hosted model reached via a custom api_base has no such
    # constraint — its configured temperature must be forwarded unchanged.
    p = LiteLLMProvider(
        model="openai/gpt-5-local",
        api_base="http://10.0.0.205:9090/v1",
        temperature=0.7,
        context_window=131072,
    )
    kwargs = p._build_kwargs(MSGS, None)
    assert kwargs["temperature"] == 0.7
