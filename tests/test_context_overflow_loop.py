"""A gateway's context overflow, end to end through the agent loop (ADR-0293).

``test_context_overflow_classification`` pins the verdict on the error. These tests pin what
the loop does with it, so the chain that failed in the field is exercised whole: a real
``LiteLLMProvider`` over a fake gateway whose ``litellm.acompletion`` raises the exception
litellm builds for a Bedrock refusal, driving a real ``AgentLoop`` and the real ``Compactor``.
The summarizer is a call to the same gateway, as it is in production, so an overflow can hit
it too. Hermetic: no network, no credentials (``conftest`` points ``ZAKCODE_HOME`` at a
temporary folder, so no ``~/.zakcode/.env`` is read).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

import zakcode.providers.litellm_provider as lp
from tests.test_context_overflow_classification import INPUT_TOO_LONG, gateway_error
from tests.test_provider_edge import _make_response
from tests.test_provider_stream import _chunk
from zakcode.agent.compact import Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

#: The system prompt the loop's summarize call carries (``_summarize_for_compaction``), which
#: is how the fake tells a compaction's call from the turn's own.
SUMMARIZER_MARK = "compacting a long conversation"


class FakeGateway:
    """What ``litellm.acompletion`` does when it is pointed at a LiteLLM proxy over Bedrock.

    ``main_overflows`` of the turn's own calls are refused with the Bedrock wording; with
    ``summarizer_overflows`` the compaction's call is refused too. Anything else is answered.
    ``calls`` records ``"main"`` or ``"summary"`` for each request, in order.
    """

    def __init__(self, *, main_overflows: int = 0, summarizer_overflows: bool = False) -> None:
        self.main_overflows_left = main_overflows
        self.summarizer_overflows = summarizer_overflows
        self.calls: list[str] = []

    async def __call__(self, **kwargs: Any) -> Any:
        system: dict[str, Any] = next(
            (m for m in kwargs["messages"] if m.get("role") == "system"), {}
        )
        kind = "summary" if SUMMARIZER_MARK in str(system.get("content", "")) else "main"
        self.calls.append(kind)
        if kind == "summary" and self.summarizer_overflows:
            raise gateway_error(400, INPUT_TOO_LONG)
        if kind == "main" and self.main_overflows_left > 0:
            self.main_overflows_left -= 1
            raise gateway_error(400, INPUT_TOO_LONG)
        text = "<summary>Earlier questions and answers, condensed.</summary>"
        if not kwargs.get("stream"):
            return _make_response("recovered")

        async def _stream() -> AsyncIterator[Any]:
            yield _chunk(content=text if kind == "summary" else "recovered")
            yield _chunk(content="", finish_reason="stop")

        return _stream()


@pytest.fixture(autouse=True)
def _no_request_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider spaces request starts a second apart; that sleep is not under test, and
    this file makes up to five requests a turn."""
    monkeypatch.setattr(lp, "_MIN_REQUEST_INTERVAL_S", 0.0)


def _provider(window: int) -> LiteLLMProvider:
    # The window is declared, as an operator's config entry declares it: the provider refuses
    # to guess one for a gateway alias litellm has no metadata for (ADR-0066).
    return LiteLLMProvider(
        model="openai/Claude4.5",
        api_key="sk-test",
        api_base="https://gateway.invalid/v1",
        context_window=window,
        temperature=0.0,
    )


def _loop(provider: LiteLLMProvider, tmp_path: Path) -> AgentLoop:
    """A loop over a seven-exchange session: fourteen messages of about 1.5 KB each."""
    session = Session(cwd=str(tmp_path), model="openai/Claude4.5")
    for i in range(14):
        body = f"{'question' if i % 2 == 0 else 'answer'} {i} " + "x" * 1500
        session.add_message(Message.user(body) if i % 2 == 0 else Message.assistant_text(body))
    return AgentLoop(
        provider,
        ToolRegistry(),
        session,
        settings=load_settings(workspace_root=tmp_path),
        compactor=Compactor(),
    )


async def _stream_turn(loop: AgentLoop) -> tuple[list[Any], Any]:
    events = [event async for event in loop.astream_turn("continue")]
    return events, events[-1]


def _statuses(events: list[Any]) -> list[str]:
    return [ev.message for ev in events if type(ev).__name__ == "AgentStatus"]


# ── an overflow the gateway words its own way is recovered from ───────────────────────────────


async def test_a_gateway_overflow_compacts_once_and_the_turn_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = FakeGateway(main_overflows=1)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop = _loop(_provider(200_000), tmp_path)
    seeded = len(loop.session.messages)

    result = await loop.arun_turn("continue")

    assert result.stop_reason == "completed", result.error
    assert result.error == ""
    # Refused once, one compaction (one summarize call), then the SAME call answered.
    assert gateway.calls == ["main", "summary", "main"]
    assert loop.last_compaction.startswith("compacted"), loop.last_compaction
    assert len(loop.session.messages) < seeded
    assert result.iterations == 1  # the recovery is in place, not a new iteration


async def test_streaming_a_gateway_overflow_compacts_once_and_the_turn_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = FakeGateway(main_overflows=1)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop = _loop(_provider(200_000), tmp_path)
    seeded = len(loop.session.messages)

    events, done = await _stream_turn(loop)

    assert done.stop_reason == "completed", done.error
    assert gateway.calls == ["main", "summary", "main"]
    assert loop.last_compaction.startswith("compacted"), loop.last_compaction
    assert len(loop.session.messages) < seeded
    assert any(s.startswith("context window exceeded; compacted ") for s in _statuses(events))


# ── an overflow nothing can shrink ends the turn gracefully ───────────────────────────────────


async def test_when_the_summarizer_overflows_too_the_turn_ends_provider_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The transcript is bigger than the window, so the compaction's own call is refused with
    the same overflow. The loop falls back to eliding tool outputs, finds none in a
    conversation of plain text, and ends the turn as ``provider_error`` naming what was tried:
    no exception leaves the loop, and the conversation is left as it was."""
    gateway = FakeGateway(main_overflows=99, summarizer_overflows=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider(2048)
    loop = _loop(provider, tmp_path)
    before = [m.text for m in loop.session.messages]
    assert provider.count_tokens(loop.session.messages) > 2048  # the premise: it cannot fit

    result = await loop.arun_turn("continue")

    assert result.stop_reason == "provider_error"
    assert result.degraded
    assert "Input is too long for requested model" in result.error
    assert "(recovery: nothing left to elide; 0/2 attempts)" in result.error
    assert gateway.calls.count("main") == 1  # no retry once nothing could shrink it
    assert 1 <= gateway.calls.count("summary") <= 6  # the compaction was tried, and bounded
    # Nothing was lost: every seeded message is still there, then the turn's own request.
    assert [m.text for m in loop.session.messages][: len(before)] == before


async def test_streaming_when_the_summarizer_overflows_too_the_turn_ends_provider_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gateway = FakeGateway(main_overflows=99, summarizer_overflows=True)
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    provider = _provider(2048)
    loop = _loop(provider, tmp_path)
    before = [m.text for m in loop.session.messages]
    assert provider.count_tokens(loop.session.messages) > 2048

    _events, done = await _stream_turn(loop)

    assert done.stop_reason == "provider_error"
    assert "Input is too long for requested model" in done.error
    assert "(recovery: nothing left to elide; 0/2 attempts)" in done.error
    assert gateway.calls.count("main") == 1
    assert 1 <= gateway.calls.count("summary") <= 6
    assert [m.text for m in loop.session.messages][: len(before)] == before


# ── and an unrelated refusal is not mistaken for one ──────────────────────────────────────────


async def test_an_unrelated_bad_request_does_not_start_a_compaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The control: the same gateway, the same status, a different refusal. The turn ends
    ``provider_error`` on its first call and no summarize call is ever made."""

    class RefusesTheShape(FakeGateway):
        async def __call__(self, **kwargs: Any) -> Any:
            self.calls.append("main")
            raise gateway_error(
                400,
                "litellm.BadRequestError: BedrockException - "
                '{"message":"messages.0.content: Input should be a valid list"}',
            )

    gateway = RefusesTheShape()
    monkeypatch.setattr(lp.litellm, "acompletion", gateway)
    loop = _loop(_provider(200_000), tmp_path)

    result = await loop.arun_turn("continue")

    assert result.stop_reason == "provider_error"
    assert "Input should be a valid list" in result.error
    assert gateway.calls == ["main"]
    assert loop.last_compaction == ""
