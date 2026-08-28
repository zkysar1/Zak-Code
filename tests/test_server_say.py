"""Tests for POST /say — the watch/talk unification inbox.

/say queues a single user message into ``<workspace>/.say`` (atomic, single-slot,
length-capped). Unlike a /nudge suggestion (folded into the preamble), a say is
delivered by the driver as the next turn's MESSAGE — talking to the driven mind is
just its next turn. Plain JSON (no streaming), so Starlette's TestClient drives it
directly; turn-side consumption is pinned in test_server_consumer.py.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import SAY_MAX_CHARS, create_app
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on this route)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _client(workspace: Path) -> TestClient:
    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    store = SessionStore(base_dir=workspace / "sessions")
    app: FastAPI = create_app(settings=settings, store=store, agent_factory=_factory)
    return TestClient(app)


def test_say_writes_single_slot_file(tmp_path: Path) -> None:
    resp = _client(tmp_path).post("/say", json={"text": "what did you find today?"})
    assert resp.status_code == 200
    assert resp.json() == {"queued": True}
    assert (tmp_path / ".say").read_text(encoding="utf-8").strip() == "what did you find today?"


def test_say_empty_text_is_400(tmp_path: Path) -> None:
    assert _client(tmp_path).post("/say", json={"text": "   "}).status_code == 400
    assert not (tmp_path / ".say").exists()


def test_say_second_pending_is_429(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.post("/say", json={"text": "first"}).status_code == 200
    assert client.post("/say", json={"text": "second"}).status_code == 429
    # the first message is preserved, not overwritten by the burst
    assert (tmp_path / ".say").read_text(encoding="utf-8").strip() == "first"


def test_say_length_capped(tmp_path: Path) -> None:
    _client(tmp_path).post("/say", json={"text": "x" * (SAY_MAX_CHARS + 3000)})
    assert len((tmp_path / ".say").read_text(encoding="utf-8").strip()) == SAY_MAX_CHARS


def test_say_and_nudge_are_independent_slots(tmp_path: Path) -> None:
    # A pending nudge must not block a say (and vice versa) — they are different seams
    # (preamble suggestion vs the turn's message) with independent single-slot queues.
    client = _client(tmp_path)
    assert client.post("/nudge", json={"text": "a suggestion"}).status_code == 200
    assert client.post("/say", json={"text": "a question"}).status_code == 200
    assert (tmp_path / ".nudge").read_text(encoding="utf-8").strip() == "a suggestion"
    assert (tmp_path / ".say").read_text(encoding="utf-8").strip() == "a question"
