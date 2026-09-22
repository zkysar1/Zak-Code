"""ADR-0214: every buffered call records the model the BACKEND said answered it.

The id a request NAMES and the id that answers it are two different facts. An endpoint may
publish several ids as aliases onto one set of weights, and a router answers with its own
canonical name -- so a run tagged only by what it ASKED for cannot be reconstructed
afterwards, and the server's access log cannot close the gap from its side either (it
records the name it resolved to, never the one the caller sent).

Two things are pinned here. The buffered path records the echo and announces each distinct
answerer once. The STREAMING path records nothing, on purpose: litellm's stream wrapper
stamps the REQUESTED model onto every chunk it constructs, so a capture there would file the
request as if it were the answer. Measured against the fleet's pod on 2026-09-22 -- the raw
SSE carried `/…/Qwen3.8-27B-UD-Q4_K_XL.gguf` on all 11 chunks and litellm handed back
`zds-qwen3.5-35b`, the requested id, on all 11.

Hermetic: ``litellm.acompletion`` is monkeypatched to return litellm-SHAPED plain objects, so
no network and no key is needed. They are ``SimpleNamespace``s, exercising the attribute path
of the provider's dict-or-attr ``_get`` helper.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

import zakcode.providers.litellm_provider as provider_mod
from zakcode.messages import Message
from zakcode.providers.litellm_provider import LiteLLMProvider

_MSGS = [Message.user("hello")]
_REQUESTED = "openai/zds-qwen3-8b"


def _make_provider() -> LiteLLMProvider:
    # An explicit window: the requested id is deliberately one the registry has never heard
    # of (that is the situation this ADR is about), and the provider refuses to guess a
    # window for such a model.
    return LiteLLMProvider(model=_REQUESTED, api_key="unused", context_window=131072)


def _response(served: Any) -> SimpleNamespace:
    """One buffered completion whose ``model`` field is whatever the backend chose."""
    return SimpleNamespace(
        model=served,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )


def _buffered(served: Any) -> Any:
    async def _acompletion(**_kwargs: Any) -> Any:
        return _response(served)

    return _acompletion


def _chunk(served: Any, *, content: str | None = None, finish_reason: str | None = None) -> Any:
    return SimpleNamespace(
        model=served,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _streamed(chunks: list[Any]) -> Any:
    async def _aiter() -> AsyncIterator[Any]:
        for c in chunks:
            yield c

    async def _acompletion(**_kwargs: Any) -> Any:
        return _aiter()

    return _acompletion


def _served_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "served by" in r.getMessage()]


# ── the buffered path records the echo ────────────────────────────────────────


async def test_the_result_carries_the_name_the_backend_answered_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``served_model`` is read off the RESPONSE, not echoed back from the request.

    The second assertion is the one that matters: a version that filled the field from
    ``self.model`` would satisfy "the field is populated" on every call and report a
    substitution as a match.
    """
    monkeypatch.setattr(provider_mod.litellm, "acompletion", _buffered("zds-qwen3.8-27b"))

    result = await _make_provider().acomplete(_MSGS)

    assert result.served_model == "zds-qwen3.8-27b"
    assert result.served_model != _REQUESTED


async def test_a_backend_that_echoes_no_model_leaves_the_field_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent, empty and non-string echoes all read ``None`` -- never a fabricated name.

    A provenance field that guesses is worse than one that admits it does not know, so every
    shape that is not a non-empty string resolves the same way.
    """
    for served in (None, "", 27, ["zds-qwen3.8-27b"]):
        monkeypatch.setattr(provider_mod.litellm, "acompletion", _buffered(served))
        result = await _make_provider().acomplete(_MSGS)
        assert result.served_model is None, f"{served!r} should not become a served name"


async def test_the_answerer_is_announced_once_per_distinct_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One INFO line per distinct answerer -- so a swap mid-run is audible, not buried.

    Three calls, two answerers: a repeat is silent, and the SECOND name gets its own line the
    moment it appears. That second line is the whole point of the announcement.
    """
    provider = _make_provider()
    caplog.set_level(logging.INFO, logger="zakcode.providers")

    for served in ("zds-qwen3.8-27b", "zds-qwen3.8-27b", "zds-qwen3-35b"):
        monkeypatch.setattr(provider_mod.litellm, "acompletion", _buffered(served))
        await provider.acomplete(_MSGS)

    assert _served_lines(caplog) == [
        f"{_REQUESTED}: served by zds-qwen3.8-27b",
        f"{_REQUESTED}: served by zds-qwen3-35b",
    ]


# ── the streaming path makes no claim ─────────────────────────────────────────


async def test_a_streamed_call_claims_no_answerer(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A stream announces nothing, because its chunks carry the REQUEST, not the answer.

    The chunks here are shaped as litellm really delivers them: every one stamped with the
    requested model, whatever the server put on the wire. Capturing that would record
    ``openai/zds-qwen3-8b`` as the thing that answered a call to ``openai/zds-qwen3-8b`` --
    a provenance record that can never disagree with the request and so can never detect a
    substitution. This test goes red the moment anything starts reading ``chunk.model``.
    """
    stamped = _REQUESTED.split("/", 1)[1]  # what litellm stamps: the id minus its prefix
    monkeypatch.setattr(
        provider_mod.litellm,
        "acompletion",
        _streamed(
            [
                _chunk(stamped, content="hi"),
                _chunk(stamped, content=" there"),
                _chunk(stamped, finish_reason="stop"),
            ]
        ),
    )
    provider = _make_provider()
    caplog.set_level(logging.INFO, logger="zakcode.providers")

    async for _ in provider.astream(_MSGS):
        pass

    assert _served_lines(caplog) == []
    assert provider.last_stream_sample is not None
    assert "served_model" not in provider.last_stream_sample
