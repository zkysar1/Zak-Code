"""A quota refusal ends the turns a client waits on, and only those (ADR-0393).

Why this file exists. ``POST /chat`` and ``POST /chat/stream`` hold a request open for the turn.
A spend-cap refusal ends such a turn as ``provider_error``, and a workspace Stop hook that vetoes
that ending re-entered the turn for up to 765 seconds (ADR-0181) with the request still open.
Measured in review of the spend gate, with the veto delays cut to 1 s and 2 s: the call was refused
at once, the hook vetoed, the client gave up at 1.36 s, a budget push landed at 1.70 s, and the
next re-entry was admitted at 3.56 s and billed 0.25 USD for a client that had left 2.2 s before.

These tests serve a REAL agent per turn, as the web app does, over a provider the spend gate
refuses on every call and a Stop hook that vetoes every ending. They count what reached the
provider and what reached the hook. The veto delay is patched to zero, so the unfixed behaviour
is seen as its call count, not waited out. Hermetic: tmp workspaces, no network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import zakcode.agent.loop as loop_module
from zakcode import Agent
from zakcode.agent.loop import _MAX_PROVIDER_ERROR_VETOES
from zakcode.config import Settings
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.providers.spend_gate import SpendCapReached
from zakcode.server.app import create_app
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore

MODEL = "scripted/test"
CAP = "spend cap reached"


class RefusingProvider(Provider):
    """Every call is refused as the spend gate refuses one: before anything is sent."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.calls += 1
        raise SpendCapReached(CAP)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


class VetoingStopHook:
    """A workspace Stop hook that vetoes every ending it is asked about."""

    def __init__(self) -> None:
        self.asked = 0

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult:
        self.asked += 1
        return TurnEndResult(vetoed=True, continuation_prompt="Keep going.")


@pytest.fixture(autouse=True)
def unpaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """The veto's backoff, patched to nothing: the re-entries it would pace are counted instead."""
    monkeypatch.setattr(loop_module, "_provider_error_veto_delay", lambda vetoes: 0.0)


def _serve(workspace: Path) -> tuple[Any, RefusingProvider, VetoingStopHook]:
    """A server on ``workspace`` whose every turn runs on a real agent of its own."""
    settings = Settings(default_model=MODEL, context_window=8192, workspace_root=workspace)
    store = SessionStore(base_dir=workspace / "sessions")
    provider = RefusingProvider()
    hook = VetoingStopHook()

    def factory(session: Session, model: str | None, prompter: Any) -> Agent:  # noqa: ARG001
        agent = Agent(session=session, settings=settings, session_store=store, provider=provider)
        agent.loop.hook_manager.register_turn_end(hook)
        return agent

    return create_app(settings=settings, store=store, agent_factory=factory), provider, hook


def _sse_events(text: str) -> list[dict[str, Any]]:
    """The JSON payloads of an SSE body's ``data:`` lines."""
    return [
        json.loads(line[len("data:") :]) for line in text.splitlines() if line.startswith("data:")
    ]


def test_chat_ends_on_the_first_quota_refusal(tmp_path: Path) -> None:
    app, provider, hook = _serve(tmp_path)
    response = TestClient(app).post("/chat", json={"message": "hi"})
    assert provider.calls == 1  # the refusal was the one call; nothing re-entered to bill
    assert hook.asked == 0
    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "provider_error"
    assert body["error"] == CAP


def test_chat_stream_ends_on_the_first_quota_refusal(tmp_path: Path) -> None:
    app, provider, hook = _serve(tmp_path)
    response = TestClient(app).post("/chat/stream", json={"message": "hi"})
    assert provider.calls == 1
    assert hook.asked == 0
    assert response.status_code == 200
    events = _sse_events(response.text)
    done = [e for e in events if e["event"] == "done"][-1]
    assert done["stop_reason"] == "provider_error"
    assert done["error"] == CAP
    statuses = [e["message"] for e in events if e["event"] == "status"]
    assert not any("the turn-end hook asked to continue" in s for s in statuses)


def test_a_say_turn_keeps_its_re_entry_on_the_same_refusal(tmp_path: Path) -> None:
    """The decision, pinned: a turn the workspace drives itself has no client waiting on it, and
    the budget that arrives while it waits is what its re-entry is for."""
    app, provider, hook = _serve(tmp_path)
    assert write_say(say_path(tmp_path), "hi")
    assert asyncio.run(app.state.consume_one_say()) is True
    assert provider.calls == _MAX_PROVIDER_ERROR_VETOES + 1
    assert hook.asked == _MAX_PROVIDER_ERROR_VETOES
