"""The thinking switch is rendered per destination (ADR-0181, spelling half).

Zak Code's one internal spelling for "thinking off" is llama.cpp's
``{"chat_template_kwargs": {"enable_thinking": false}}`` (ADR-0056). Measured 2026-09-17 on
a served Mind (``vertex_ai_beta``): Vertex AI does not ignore a body key it does not know —
it refuses the whole request (``400 INVALID_ARGUMENT — Invalid JSON payload received.
Unknown name "chat_template_kwargs": Cannot find field.``), so the reasoning-overflow
retry killed the turn it existed to save. The provider now renders the switch for the
destination at request-build time: kept for a self-hosted OpenAI-compatible server (the
measured case), litellm's first-class ``reasoning_effort`` for a Gemini model, dropped
everywhere else. Hermetic: request assembly only, no network.
"""

from __future__ import annotations

import pytest

from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import thinking_extra_body
from zakcode.providers.thinking import (
    GEMINI_THINKING_OFF_EFFORT,
    THINKING_SWITCH_KEY,
    render_thinking_switch,
    rendered_thinking_wire_names,
)

POD = "http://pod.local:8080/v1"
OFF = thinking_extra_body(False)
ON = thinking_extra_body(True)
MSGS = [{"role": "user", "content": "hi"}]


# ── the pure renderer ─────────────────────────────────────────────────────────


def test_a_self_hosted_openai_compatible_server_keeps_the_measured_spelling() -> None:
    body, kwargs = render_thinking_switch("openai/zds-qwen3.8-27b", POD, {**OFF, "seed": 42})
    assert body == {**OFF, "seed": 42}
    assert kwargs == {}


def test_a_bare_model_name_on_a_custom_base_keeps_it_too() -> None:
    body, kwargs = render_thinking_switch("zds-qwen3.8-27b", POD, dict(OFF))
    assert body == OFF and kwargs == {}


@pytest.mark.parametrize(
    "model",
    ["vertex_ai_beta/gemini-2.5-pro", "vertex_ai/gemini-2.5-flash", "gemini/gemini-2.5-flash"],
)
def test_a_gemini_model_gets_litellms_first_class_effort_instead(model: str) -> None:
    body, kwargs = render_thinking_switch(model, None, {**OFF, "seed": 42})
    assert THINKING_SWITCH_KEY not in body
    assert body == {"seed": 42}
    assert kwargs == {"reasoning_effort": GEMINI_THINKING_OFF_EFFORT}


def test_gemini_thinking_off_is_the_tightest_budget_every_gemini_accepts() -> None:
    # ``"disable"`` maps (in the installed litellm) to a thinking budget of 0, which
    # gemini-2.5-pro refuses outright; ``"minimal"`` maps per model to the smallest budget
    # the model takes (128 for 2.5-pro). The retry's job is to leave room for an answer,
    # not to hit a switch the model does not have.
    assert GEMINI_THINKING_OFF_EFFORT == "minimal"


def test_thinking_on_is_a_cloud_models_own_default_so_nothing_is_rendered() -> None:
    body, kwargs = render_thinking_switch("vertex_ai_beta/gemini-2.5-pro", None, dict(ON))
    assert body == {} and kwargs == {}


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o",  # hosted OpenAI: no api_base, rejects unknown arguments
        "anthropic/claude-sonnet-4",
        "vertex_ai/claude-sonnet-4",  # a Claude on Vertex takes Anthropic's mapping, not Gemini's
        "ollama_chat/qwen3:32b",
        "bedrock/anthropic.claude-3",
    ],
)
def test_every_other_destination_gets_no_switch_rather_than_a_guess(model: str) -> None:
    body, kwargs = render_thinking_switch(model, None, {**OFF, "seed": 42})
    assert body == {"seed": 42}
    assert kwargs == {}


def test_a_body_without_the_switch_is_returned_untouched() -> None:
    body, kwargs = render_thinking_switch("vertex_ai_beta/gemini-2.5-pro", None, {"seed": 1})
    assert body == {"seed": 1} and kwargs == {}
    assert render_thinking_switch("openai/gpt-4o", None, {}) == ({}, {})


def test_wire_names_are_reported_only_when_a_switch_was_rendered() -> None:
    assert rendered_thinking_wire_names({}) == ()
    names = rendered_thinking_wire_names({"reasoning_effort": "minimal"})
    assert "reasoning_effort" in names and "thinkingConfig" in names


# ── at the request chokepoint ─────────────────────────────────────────────────


def _gemini(**kw: object) -> LiteLLMProvider:
    return LiteLLMProvider(model="vertex_ai_beta/gemini-2.5-pro", context_window=1_000_000, **kw)


def test_the_overflow_retry_against_vertex_no_longer_carries_the_llama_cpp_key() -> None:
    """The incident's exact request: the ADR-0056 retry (a per-call thinking-off body)
    against a ``vertex_ai_beta`` Gemini model. ``chat_template_kwargs`` must be absent
    from the request that reaches litellm, and the switch expressed natively."""
    kwargs = _gemini()._build_kwargs(MSGS, None, extra_body=thinking_extra_body(False))
    assert "extra_body" not in kwargs
    assert kwargs["reasoning_effort"] == GEMINI_THINKING_OFF_EFFORT
    assert THINKING_SWITCH_KEY not in repr(kwargs)


def test_a_gemini_category_knob_renders_the_same_way_and_keeps_the_rest_of_the_body() -> None:
    provider = _gemini(extra_body={**thinking_extra_body(False), "seed": 42})
    kwargs = provider._build_kwargs(MSGS, None)
    assert kwargs["extra_body"] == {"seed": 42}
    assert kwargs["reasoning_effort"] == GEMINI_THINKING_OFF_EFFORT


def test_the_default_request_shape_is_unchanged_without_a_switch() -> None:
    kwargs = _gemini()._build_kwargs(MSGS, None)
    assert "extra_body" not in kwargs and "reasoning_effort" not in kwargs


def test_the_pod_request_is_byte_identical_to_before() -> None:
    provider = LiteLLMProvider(model="openai/zds-qwen3.8-27b", api_base=POD, context_window=131072)
    kwargs = provider._build_kwargs(MSGS, None, extra_body=thinking_extra_body(False))
    assert kwargs["extra_body"][THINKING_SWITCH_KEY] == {"enable_thinking": False}
    assert "reasoning_effort" not in kwargs


def test_a_field_the_provider_refused_stays_out_of_every_later_request() -> None:
    provider = _gemini(extra_body={"seed": 42, "top_k": 3})
    provider.rejected_request_fields.append("top_k")
    assert provider._build_kwargs(MSGS, None)["extra_body"] == {"seed": 42}
    provider.rejected_request_fields.append("reasoning_effort")
    kwargs = provider._build_kwargs(MSGS, None, extra_body=thinking_extra_body(False))
    assert "reasoning_effort" not in kwargs  # the rendered switch was refused too
    assert provider.extra_body == {"seed": 42, "top_k": 3}  # the operator's config is intact
