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

from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    _is_openai_gpt5_fixed_temperature_model,
)
from zakcode.providers.structured import complete_structured

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


# ── the integration path: complete_structured -> provider -> the wire ────────
#
# Everything above calls ``_build_kwargs`` DIRECTLY, so it pins the HANDLER and
# not the TRIGGER. Nothing above verifies that ``complete_structured`` actually
# routes its forced ``temperature=0`` THROUGH that chokepoint rather than
# reaching ``litellm.acompletion`` by some other path — a future refactor adding
# a second dispatch route would keep every test above green and regress silently
# for every gpt-5 caller. These two drive the real seam and assert at the WIRE,
# which is the only place a bypassing route would still be visible.


class _Obj:
    """Attribute-style stand-in for a litellm response object."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _wire_response() -> _Obj:
    message = _Obj(content='{"a": "x"}', tool_calls=None)
    return _Obj(
        choices=[_Obj(message=message, finish_reason="stop")],
        usage=_Obj(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        _hidden_params={"response_cost": 0.0},
    )


_SCHEMA = {
    "type": "object",
    "required": ["a"],
    "additionalProperties": False,
    "properties": {"a": {"type": "string"}},
}


async def _wire_kwargs_for(monkeypatch: pytest.MonkeyPatch, model: str) -> dict[str, Any]:
    """Drive the REAL complete_structured against `model`; return what hit the wire."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _Obj:
        captured.update(kwargs)
        return _wire_response()

    monkeypatch.setattr(lp.litellm, "acompletion", fake_acompletion)
    provider = LiteLLMProvider(model=model, context_window=400000)
    await complete_structured(provider, [Message.user("hi")], schema=_SCHEMA)
    assert captured, "litellm.acompletion was never reached — the test proves nothing"
    return captured


async def test_structured_path_drops_temperature_for_gpt5_at_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _wire_kwargs_for(monkeypatch, "openai/gpt-5.6-terra")
    assert "temperature" not in captured


async def test_structured_path_still_sends_temperature_for_a_normal_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The positive control, and it is what makes the assertion above
    # discriminating rather than vacuous: without it, deleting structured.py's
    # ``call_kwargs["temperature"] = 0.0`` would leave the gpt-5 test green while
    # silently retiring schema determinism for every other model.
    captured = await _wire_kwargs_for(monkeypatch, "openai/gpt-4o-mini")
    assert captured["temperature"] == 0.0
