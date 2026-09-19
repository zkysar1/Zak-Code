"""A request field the provider refuses BY NAME is dropped and the call re-issued once
(ADR-0181, repair half).

The provider is told exactly what it did wrong — Vertex: ``Unknown name
"chat_template_kwargs": Cannot find field.``; OpenAI: ``Unrecognized request argument
supplied: x``; Anthropic: ``x: Extra inputs are not permitted`` — and until now threw the
sentence away as a generic ``RequestFailed`` that ended the turn. Now the field named in a
4xx, IF it is one we added ourselves (an ``extra_body`` key, or the rendered thinking
switch), joins the provider's ``rejected_request_fields`` for the rest of the session and
the same logical call is re-issued once without it. The "one we sent" gate is the whole
safety of the mechanism: a refusal naming ``tools`` or ``messages`` is a defect to surface.
Hermetic: ``litellm.acompletion`` is faked; the exception classes are named like litellm's.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from zakcode.messages import Message
from zakcode.providers.base import (
    RateLimited,
    RequestFailed,
    StreamDone,
    StreamTextDelta,
    rejected_request_field,
)
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.routing import thinking_extra_body

POD = "http://pod.local:8080/v1"

#: The measured refusal, byte-for-byte as ``str(exc)`` renders it (a bytes repr with the
#: JSON's quotes backslash-escaped) — 2026-09-17, a served Mind on vertex_ai_beta.
VERTEX_TEXT = (
    'litellm.BadRequestError: Vertex_ai_betaException BadRequestError - b\'{\\n  "error": {\\n'
    '    "code": 400,\\n    "message": "Invalid JSON payload received. Unknown name '
    '\\\\"chat_template_kwargs\\\\": Cannot find field.",\\n    "status": "INVALID_ARGUMENT"'
    "}}'"
)


class BadRequestError(Exception):
    """Named exactly: the provider matches litellm's class by MRO NAME (plus status)."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class InternalServerError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.status_code = 500


# ── the classifier ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        VERTEX_TEXT,
        'Invalid JSON payload received. Unknown name "chat_template_kwargs" at '
        "'generation_config': Cannot find field.",
        "Unrecognized request argument supplied: chat_template_kwargs",
        "Unrecognized request arguments supplied: foo, chat_template_kwargs",
        "chat_template_kwargs: Extra inputs are not permitted",
        "extra_body.chat_template_kwargs: Extra inputs are not permitted",
        "[{'type': 'extra_forbidden', 'loc': ('body', 'chat_template_kwargs'), "
        "'msg': 'Extra inputs are not permitted'}]",
        "got an unexpected keyword argument 'chat_template_kwargs'",
        "unknown parameter: chat_template_kwargs",
        "Unsupported parameter: 'chat_template_kwargs'",
    ],
)
def test_every_known_vendor_phrasing_names_the_field(text: str) -> None:
    assert rejected_request_field(text, ["chat_template_kwargs", "seed"]) == "chat_template_kwargs"


def test_a_field_we_did_not_send_is_never_ours_to_strip() -> None:
    assert rejected_request_field('Unknown name "tools": Cannot find field.', ["seed"]) is None
    assert rejected_request_field(VERTEX_TEXT, []) is None
    assert rejected_request_field("Invalid JSON payload received.", ["seed"]) is None


def test_the_classifier_reads_an_exception_too() -> None:
    assert rejected_request_field(BadRequestError(VERTEX_TEXT), ["chat_template_kwargs"]) == (
        "chat_template_kwargs"
    )


# ── the buffered call path ────────────────────────────────────────────────────


def _response(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        _hidden_params={"response_cost": 0.0},
    )


class _Litellm:
    """A scripted ``litellm.acompletion``: raises the queued exceptions first (each judged
    against the request that provoked it), then answers; every request is recorded."""

    def __init__(self, failures: list[Exception]) -> None:
        self.failures = list(failures)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        if kwargs.get("stream"):
            return self._stream()
        return _response()

    @staticmethod
    async def _stream() -> AsyncIterator[Any]:
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop"
                )
            ],
            usage=None,
        )


def _pod(**kw: Any) -> LiteLLMProvider:
    return LiteLLMProvider(
        model="openai/zds-qwen3.8-27b", api_base=POD, context_window=131072, **kw
    )


async def test_vertexs_refusal_drops_the_field_and_the_call_is_re_issued_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = _Litellm([BadRequestError(VERTEX_TEXT)])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod(extra_body={**thinking_extra_body(False), "seed": 42})
    with caplog.at_level(logging.WARNING, logger="zakcode.providers"):
        result = await provider.acomplete([Message.user("hi")])
    assert result.text == "ok"
    assert len(fake.calls) == 2
    assert fake.calls[0]["extra_body"] == {**thinking_extra_body(False), "seed": 42}
    assert fake.calls[1]["extra_body"] == {"seed": 42}  # the refused key only; the rest survives
    assert provider.rejected_request_fields == ["chat_template_kwargs"]
    assert provider.extra_body == {**thinking_extra_body(False), "seed": 42}  # config intact
    assert any(
        "chat_template_kwargs" in r.message and "dropping" in r.message for r in caplog.records
    )


async def test_the_refused_field_stays_out_for_the_rest_of_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm([BadRequestError(VERTEX_TEXT)])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod(extra_body=thinking_extra_body(False))
    await provider.acomplete([Message.user("hi")])
    await provider.acomplete([Message.user("again")])
    # 2 calls for the first turn (refuse + re-issue), ONE for the second: the same 400 is
    # never provoked twice.
    assert len(fake.calls) == 3
    assert "extra_body" not in fake.calls[2]


async def test_a_per_call_override_is_repaired_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loop's overflow retry passes the switch per call; the repair must catch that
    shape too, and remember it for the next per-call override."""
    fake = _Litellm([BadRequestError(VERTEX_TEXT)])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod()
    await provider.acomplete([Message.user("hi")], extra_body=thinking_extra_body(False))
    assert [("extra_body" in c) for c in fake.calls] == [True, False]
    await provider.acomplete([Message.user("hi")], extra_body=thinking_extra_body(False))
    assert len(fake.calls) == 3 and "extra_body" not in fake.calls[2]


async def test_a_refusal_naming_a_field_we_did_not_send_is_surfaced_not_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm([BadRequestError('Unknown name "tools": Cannot find field.')])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod(extra_body={"seed": 42})
    with pytest.raises(RequestFailed):
        await provider.acomplete([Message.user("hi")])
    assert len(fake.calls) == 1
    assert provider.rejected_request_fields == []


async def test_a_5xx_quoting_the_field_is_a_transient_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm([InternalServerError(VERTEX_TEXT)])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod(extra_body=thinking_extra_body(False))
    with pytest.raises(RateLimited):  # the transient-5xx arm, as before
        await provider.acomplete([Message.user("hi")])
    assert len(fake.calls) == 1
    assert provider.rejected_request_fields == []


async def test_the_repair_is_spent_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second refusal in the same call (another field, or the same one again) is reported
    as is — the mechanism never loops."""
    fake = _Litellm(
        [
            BadRequestError(VERTEX_TEXT),
            BadRequestError("Unrecognized request argument supplied: seed"),
        ]
    )
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod(extra_body={**thinking_extra_body(False), "seed": 42})
    with pytest.raises(RequestFailed) as ei:
        await provider.acomplete([Message.user("hi")])
    assert "seed" in str(ei.value)
    assert len(fake.calls) == 2
    # The first refusal was still recorded; the second was raised, not recorded.
    assert provider.rejected_request_fields == ["chat_template_kwargs"]


async def test_a_refused_rendered_thinking_switch_is_dropped_by_its_wire_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """litellm rewrites ``reasoning_effort`` into ``thinkingConfig`` before Vertex sees it, so
    a refusal names the WIRE field; the repair maps it back to the kwarg it rendered."""
    fake = _Litellm([BadRequestError('Unknown name "thinkingConfig" at generation_config')])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = LiteLLMProvider(model="vertex_ai_beta/gemini-2.5-pro", context_window=1_000_000)
    await provider.acomplete([Message.user("hi")], extra_body=thinking_extra_body(False))
    assert fake.calls[0]["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in fake.calls[1]
    assert provider.rejected_request_fields == ["reasoning_effort"]


# ── the streaming call path ───────────────────────────────────────────────────


async def test_streaming_repairs_a_refusal_that_arrives_before_any_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _Litellm([BadRequestError(VERTEX_TEXT)])
    monkeypatch.setattr(lp.litellm, "acompletion", fake)
    provider = _pod(extra_body=thinking_extra_body(False))
    events = [ev async for ev in provider.astream([Message.user("hi")])]
    assert [type(ev) for ev in events] == [StreamTextDelta, StreamDone]
    assert len(fake.calls) == 2
    assert fake.calls[0]["stream"] is True and "extra_body" in fake.calls[0]
    assert fake.calls[1]["stream"] is True and "extra_body" not in fake.calls[1]
    assert provider.rejected_request_fields == ["chat_template_kwargs"]


async def test_streaming_never_repairs_after_a_chunk_reached_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _dies_midstream(**kwargs: Any) -> Any:
        async def _aiter() -> AsyncIterator[Any]:
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="par", tool_calls=None), finish_reason=None
                    )
                ],
                usage=None,
            )
            raise BadRequestError(VERTEX_TEXT)

        calls.append(kwargs)
        return _aiter()

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(lp.litellm, "acompletion", _dies_midstream)
    provider = _pod(extra_body=thinking_extra_body(False))
    with pytest.raises(RequestFailed):
        _ = [ev async for ev in provider.astream([Message.user("hi")])]
    assert len(calls) == 1
    assert provider.rejected_request_fields == []
