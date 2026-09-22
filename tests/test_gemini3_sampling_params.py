"""Gemini 3+ gets no sampling parameters: Google deprecated temperature, top_p and top_k.

WHY THIS FILE EXISTS
Google's guidance from Gemini 3 on is to steer sampling through the system instructions.
The three parameters still function on Gemini 3, are already ignored on the newest 3.x
models, and are promised an error in a later generation. litellm forwards whatever it is
handed and logs, once per request-mapping, a DeprecationWarning naming all three — and its
trigger is the CALLER's parameters, not the request that finally goes out: the warning is
emitted inside its loop over the caller's ``non_default_params``, while the default it
supplies for these models (``temperature = 1.0`` when the request carries none) is applied
afterwards and says nothing.

So the line was ours. A Mind served on ``gemini-3.5-flash`` drew that warning on every
structured-output call, because the schema path REQUESTS ``temperature=0`` for determinism
(``providers/structured.py``, whose own docstring calls that a request and not a guarantee),
and on every judge, score and deep-think call, which pass an explicit temperature too.
Updating the harness could never have fixed it: no release changed who was sending them.

The fix drops all three at ``LiteLLMProvider._build_kwargs`` — the one chokepoint every
completion path funnels through — AFTER the per-call ``kw`` update, so it covers the
configured temperature, the structured path's forced zero, and any per-call value alike.
Same chokepoint and the same reasoning as the gpt-5 rule beside it: a backend's own
parameter map cannot be relied on to strip what it still supports. A LOCAL
OpenAI-compatible server behind a custom ``api_base`` has no such constraint and is left
alone.
"""

from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.litellm_provider import (
    LiteLLMProvider,
    _is_gemini_sampling_deprecated_model,
)
from zakcode.providers.structured import complete_structured

MSGS = [{"role": "user", "content": "hi"}]


# ── the predicate ─────────────────────────────────────────────────────────────


def test_predicate_matches_gemini_3_and_newer() -> None:
    for model in (
        "gemini-3.5-flash",
        "gemini-3-pro-preview",
        "gemini-3-flash",
        "gemini/gemini-3.5-flash",
        "vertex_ai/gemini-3-pro-preview",
        "gemini/gemini-3.6-flash",
        "gemini-4",  # the deprecation is forward-looking: a later generation errors
    ):
        assert _is_gemini_sampling_deprecated_model(model), model


def test_predicate_excludes_older_gemini_and_other_models() -> None:
    for model in (
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "vertex_ai/gemini-1.5-pro",
        "gemini/gemini-pro",  # un-numbered: an old model, not a future generation
        "openai/gpt-5.6-luna",
        "ollama_chat/llama3",
        "groq/openai/gpt-oss-120b",
    ):
        assert not _is_gemini_sampling_deprecated_model(model), model


# ── the drop at the build chokepoint ──────────────────────────────────────────


def test_configured_temperature_is_not_sent_to_gemini_3() -> None:
    provider = LiteLLMProvider(model="gemini/gemini-3.5-flash", temperature=0.2)
    assert "temperature" not in provider._build_kwargs(MSGS, None)


def test_structured_temperature_zero_is_not_sent_to_gemini_3() -> None:
    # structured.py forces temperature=0 through **kw; the chokepoint strips it too.
    provider = LiteLLMProvider(model="gemini/gemini-3.5-flash")
    assert "temperature" not in provider._build_kwargs(MSGS, None, temperature=0.0)


def test_top_p_and_top_k_are_not_sent_to_gemini_3() -> None:
    # Nothing in the product sets these today; they are dropped because the deprecation
    # covers all three, so an operator's extra kwarg cannot reintroduce the warning.
    provider = LiteLLMProvider(model="vertex_ai/gemini-3-pro-preview")
    kwargs = provider._build_kwargs(MSGS, None, top_p=0.9, top_k=40)
    assert "top_p" not in kwargs
    assert "top_k" not in kwargs


def test_an_older_gemini_keeps_its_sampling_parameters() -> None:
    # The positive control for the predicate: 2.5 is untouched, so the drop is scoped
    # to the generation Google deprecated and not to "anything called gemini".
    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash", temperature=0.2)
    kwargs = provider._build_kwargs(MSGS, None, top_p=0.9)
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.9


def test_a_local_gemini_3_named_model_keeps_its_sampling_parameters() -> None:
    # A self-hosted OpenAI-compatible server named after a Gemini generation has no such
    # constraint; same exclusion the gpt-5 rule makes.
    provider = LiteLLMProvider(
        model="gemini-3-local",
        api_base="http://10.0.0.205:9090/v1",
        temperature=0.2,
        context_window=131072,
    )
    assert provider._build_kwargs(MSGS, None)["temperature"] == 0.2


def test_the_drop_is_said_once_and_names_the_parameters(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An operator who configured a temperature this model ignores should learn why — once
    # per provider, not once per call.
    provider = LiteLLMProvider(model="gemini/gemini-3.5-flash", temperature=0.2)
    with caplog.at_level("INFO", logger="zakcode.providers"):
        provider._build_kwargs(MSGS, None)
        provider._build_kwargs(MSGS, None)
    said = [r for r in caplog.records if "not sent" in r.getMessage()]
    assert len(said) == 1, [r.getMessage() for r in said]
    assert "temperature" in said[0].getMessage()


def test_nothing_is_said_when_there_was_nothing_to_drop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The control for the line above: no configured temperature, no sampling parameter in
    # the request, so no explanation is owed.
    provider = LiteLLMProvider(model="gemini/gemini-3.5-flash")
    with caplog.at_level("INFO", logger="zakcode.providers"):
        kwargs = provider._build_kwargs(MSGS, None)
    assert "temperature" not in kwargs
    assert [r for r in caplog.records if "not sent" in r.getMessage()] == []


# ── the integration path: complete_structured -> provider -> the wire ─────────
#
# Everything above calls ``_build_kwargs`` directly, so it pins the HANDLER and not the
# TRIGGER. These drive the REAL structured seam and assert on what reached litellm, which
# is the input litellm's own deprecation check reads.


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
    provider = LiteLLMProvider(model=model, context_window=1000000)
    await complete_structured(provider, [Message.user("hi")], schema=_SCHEMA)
    assert captured, "litellm.acompletion was never reached — the test proves nothing"
    return captured


async def test_the_structured_path_sends_no_sampling_parameter_to_gemini_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = await _wire_kwargs_for(monkeypatch, "gemini/gemini-3.5-flash")
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured


async def test_the_structured_path_still_sends_temperature_to_an_older_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The positive control that makes the assertion above discriminating: without it,
    # deleting structured.py's forced zero would leave the Gemini 3 test green while
    # silently retiring schema determinism for every other model.
    captured = await _wire_kwargs_for(monkeypatch, "gemini/gemini-2.5-flash")
    assert captured["temperature"] == 0.0
