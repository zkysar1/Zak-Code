"""Tests for the WebSocket channel: bidirectional input, the permission-approval
bridge, and cooperative interrupt.

Hermetic: fake agents (no provider/network) drive the socket through FastAPI's
``TestClient`` WebSocket support. Each test injects an agent factory tailored to
what it exercises.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from zakcode.agent.loop import TurnResult
from zakcode.config import PermissionTier, Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.permissions import PermissionPrompter, PermissionRequest
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


def _client(tmp_path: Path, factory) -> tuple[TestClient, SessionStore]:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(settings=settings, store=store, agent_factory=factory)
    return TestClient(app), store


def _make_session(store: SessionStore) -> str:
    session = Session(cwd=".", model="scripted/test")
    store.save(session)
    return session.id


# ── a basic streaming agent ────────────────────────────────────────────────────


class _EchoAgent:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str) -> TurnResult:  # pragma: no cover - unused here
        return TurnResult(stop_reason="completed", iterations=1)

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        yield AgentTextDelta(text=f"echo:{user_text}")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage(total_tokens=3))


def test_ws_unknown_session_errors(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, lambda s, m, p: _EchoAgent(s))
    with client.websocket_connect("/ws/does-not-exist") as ws:
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "does-not-exist" in frame["detail"]


def test_ws_input_streams_events(tmp_path: Path) -> None:
    client, store = _client(tmp_path, lambda s, m, p: _EchoAgent(s))
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "hi"})
        frames = [ws.receive_json(), ws.receive_json()]
    assert frames[0]["event"] == "text"
    assert frames[0]["text"] == "echo:hi"
    assert frames[1]["event"] == "done"
    assert frames[1]["stop_reason"] == "completed"


def test_ws_unrecognized_message_errors_without_dropping_connection(tmp_path: Path) -> None:
    client, store = _client(tmp_path, lambda s, m, p: _EchoAgent(s))
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "nonsense"})
        err = ws.receive_json()
        assert err["type"] == "error"
        # The socket is still usable: a real turn still works afterwards.
        ws.send_json({"type": "input", "message": "ok"})
        assert ws.receive_json()["event"] == "text"


# ── the permission-approval bridge ──────────────────────────────────────────────


class _ApprovalAgent:
    """An agent that escalates one permission decision through the WS prompter."""

    def __init__(self, session: Session, prompter: PermissionPrompter | None) -> None:
        self.session = session
        self.prompter = prompter

    async def arun_turn(self, user_text: str) -> TurnResult:  # pragma: no cover
        return TurnResult(stop_reason="completed", iterations=1)

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        assert self.prompter is not None
        outcome = await self.prompter.confirm(
            PermissionRequest(
                tool_name="bash",
                tier=PermissionTier.DANGER_FULL_ACCESS,
                arguments={"command": "ls"},
                reason="danger tier requires confirmation",
            )
        )
        yield AgentTextDelta(text=f"outcome:{outcome.value}")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def test_ws_permission_approval_round_trip(tmp_path: Path) -> None:
    client, store = _client(tmp_path, lambda s, m, p: _ApprovalAgent(s, p))
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "run ls"})
        # The server should ask for approval before the turn produces output.
        prompt = ws.receive_json()
        assert prompt["type"] == "action_required"
        assert prompt["tool_name"] == "bash"
        assert prompt["arguments"] == {"command": "ls"}
        # Approve for the session; the agent then resumes and reports the outcome.
        ws.send_json({"type": "approval", "outcome": "allow_session"})
        text = ws.receive_json()
        assert text["event"] == "text"
        assert text["text"] == "outcome:allow_session"
        assert ws.receive_json()["event"] == "done"


def test_ws_permission_denied_by_default_outcome(tmp_path: Path) -> None:
    client, store = _client(tmp_path, lambda s, m, p: _ApprovalAgent(s, p))
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "run ls"})
        assert ws.receive_json()["type"] == "action_required"
        # An unknown outcome string must not crash; it falls back to deny.
        ws.send_json({"type": "approval", "outcome": "garbage"})
        text = ws.receive_json()
        assert text["text"] == "outcome:deny_once"


def test_ws_permission_times_out_fail_closed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Regression: if the client never answers an approval prompt, the turn must NOT
    # hang forever — the bridge times out and fails CLOSED (deny_once). We shrink the
    # timeout so the test is fast; the production default is 120s.
    import zakcode.server.app as appmod

    monkeypatch.setattr(appmod, "APPROVAL_TIMEOUT_SECONDS", 0.5)
    client, store = _client(tmp_path, lambda s, m, p: _ApprovalAgent(s, p))
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "run ls"})
        assert ws.receive_json()["type"] == "action_required"
        # Deliberately send NO approval. The server times out and the agent resumes
        # with a deny — proving the turn completes rather than deadlocking.
        text = ws.receive_json()
        assert text["event"] == "text"
        assert text["text"] == "outcome:deny_once"
        assert ws.receive_json()["event"] == "done"


# ── cooperative interrupt ───────────────────────────────────────────────────────


class _SlowAgent:
    """Yields one delta, then blocks — so a turn can be interrupted mid-flight."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str) -> TurnResult:  # pragma: no cover
        return TurnResult(stop_reason="completed", iterations=1)

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        yield AgentTextDelta(text="working...")
        # Block long enough that the interrupt arrives first; if the cancel
        # machinery were broken the test would instead receive a normal 'done'.
        await asyncio.sleep(5)
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def test_ws_interrupt_cancels_in_flight_turn(tmp_path: Path) -> None:
    client, store = _client(tmp_path, lambda s, m, p: _SlowAgent(s))
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "go"})
        first = ws.receive_json()
        assert first["event"] == "text" and first["text"] == "working..."
        ws.send_json({"type": "interrupt"})
        cancelled = ws.receive_json()
        # The cancelled turn emits an 'interrupted' status, NOT a normal done.
        assert cancelled["event"] == "status"
        assert "interrupt" in cancelled["message"].lower()


def test_ws_turn_reserves_session_against_concurrent_rest(tmp_path: Path) -> None:
    # audit2 #4: a WS turn must hold the SAME per-session reservation REST uses, so a REST
    # turn on the same session is refused while the WS turn runs (no transcript-clobbering
    # interleave) — and the reservation is released when the WS turn ends.
    from fastapi.testclient import TestClient

    from zakcode.server.app import create_app

    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(settings=settings, store=store, agent_factory=lambda s, m, p: _SlowAgent(s))
    client = TestClient(app)
    sid = _make_session(store)
    with client.websocket_connect(f"/ws/{sid}") as ws:
        ws.send_json({"type": "input", "message": "go"})
        assert ws.receive_json()["text"] == "working..."  # the WS turn is now in flight
        # A REST turn on the SAME session is refused while the WS turn holds it.
        resp = client.post("/chat", json={"message": "b", "session_id": sid})
        assert resp.status_code == 409
        ws.send_json({"type": "interrupt"})
        assert ws.receive_json()["event"] == "status"  # interrupted -> releases the session
    # Once the WS turn has ended, the session is free for a REST turn again.
    assert client.post("/chat", json={"message": "c", "session_id": sid}).status_code == 200


# ── store.delete (added for the REST DELETE endpoint / general API) ──────────────


def test_session_store_delete(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path / "s")
    session = Session(cwd=".", model="m")
    store.save(session)
    assert session.id in store.list()
    assert store.delete(session.id) is True
    assert session.id not in store.list()
    # Deleting again is harmless and reports nothing was removed.
    assert store.delete(session.id) is False
