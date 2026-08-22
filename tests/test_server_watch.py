"""Tests for the read-only watch surface (GET /watch/{session_id}, P0-3).

The watch endpoint tails a session's event bus and streams each AgentEvent projected to a
secret-redacted SafeEvent. These tests exercise it end-to-end: run a real /chat/stream turn
(whose events are teed into the bus), then watch the buffered frames back. They pin the
endpoint contract — cursor-addressed replay, the ``?since=`` resume, the whitelist (tool
arguments/outputs never cross the wire), the unsafe-event drop, session 404, and bearer auth
— leaving the projection's field-level guarantees to test_safe_projection and the bus's
cursor/backpressure behaviour to test_event_bus.

Why a REAL server, not TestClient/ASGITransport: /watch is an infinite SSE stream that PARKS
after replaying the buffer. Both in-process transports buffer the whole response body before
returning (TestClient blocks its portal; ASGITransport collects all chunks), so a parking
generator hangs them forever. A uvicorn server on a loopback port streams incrementally over
real TCP, so we can read N frames and disconnect. asyncio_mode=auto runs the ``async def``
tests without a decorator.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from zakcode.config import Settings
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

CANNED = "Hello from the fake agent."
SECRET_ARG_PATH = "n.txt"
TOOL_OUTPUT = "wrote n.txt"


class _FakeAgent:
    """Minimal AgentLike: emits a fixed sequence of events (incl. an unsafe AgentUsage)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        events: list[AgentEvent] = [
            AgentTextDelta(text=CANNED),
            AgentToolCall(id="c1", name="write_file", arguments={"path": SECRET_ARG_PATH}),
            AgentUsage(usage=Usage(total_tokens=10)),  # unsafe → MUST be dropped by /watch
            AgentToolResult(tool_use_id="c1", output=TOOL_OUTPUT, is_error=False),
            AgentDone(stop_reason="completed", iterations=2, usage=Usage(total_tokens=10)),
        ]
        for event in events:
            yield event


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(tmp_path: Path, *, auth_token: str | None = None) -> FastAPI:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, auth_token=auth_token
    )
    store = SessionStore(base_dir=tmp_path / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _free_port() -> int:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


class _LiveServer:
    """Run ``app`` on a loopback uvicorn server in a daemon thread for the test's lifetime."""

    def __init__(self, app: FastAPI) -> None:
        self.port = _free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start within 10s")
            time.sleep(0.02)
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *exc: object) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10.0)


@pytest.fixture
def live_url(app: FastAPI) -> Iterator[str]:
    with _LiveServer(app) as base_url:
        yield base_url


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    return _make_app(tmp_path)


def _auth(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _run_turn(base_url: str, token: str | None = None) -> str:
    """Create a session and run one /chat/stream turn, teeing every event into the bus."""
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as ac:
        sid = (await ac.post("/sessions", headers=_auth(token))).json()["id"]
        async with ac.stream(
            "POST", "/chat/stream", json={"session_id": sid, "message": "hi"}, headers=_auth(token)
        ) as resp:
            assert resp.status_code == 200
            async for _ in resp.aiter_lines():  # drain fully so all events are teed
                pass
    return sid


async def _watch_frames(
    base_url: str, url: str, count: int, token: str | None = None
) -> list[dict[str, Any]]:
    """Open the SSE watch stream, return the first ``count`` data frames, then disconnect.

    The turn's events are already buffered, so replay yields them immediately; we break as
    soon as we have ``count`` (leaving the context disconnects — the generator parks after
    replay). The 5s read timeout turns a missing-frame bug into a fast failure, not a hang.
    """
    frames: list[dict[str, Any]] = []
    timeout = httpx.Timeout(10.0, read=5.0)
    async with (
        httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac,
        ac.stream("GET", url, headers=_auth(token)) as resp,
    ):
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                frames.append(json.loads(line[len("data:") :].strip()))
                if len(frames) >= count:
                    break
    return frames


async def test_watch_replays_buffered_turn_events_as_safe_events(live_url: str) -> None:
    sid = await _run_turn(live_url)
    # 5 raw events emitted; the AgentUsage one is dropped → 4 SafeEvent frames.
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=4)
    assert [f["event"] for f in frames] == ["text", "tool_summary", "tool_summary", "done"]
    assert frames[0] == {"event": "text", "text": CANNED}
    assert frames[1] == {
        "event": "tool_summary",
        "name": "write_file",
        "status": "running",
        "used_secrets": [],
    }
    assert frames[2] == {
        "event": "tool_summary",
        "name": "",
        "status": "completed",
        "used_secrets": [],
    }
    assert frames[3] == {"event": "done", "stop_reason": "completed"}


async def test_watch_whitelist_never_leaks_tool_args_or_output(live_url: str) -> None:
    sid = await _run_turn(live_url)
    blob = json.dumps(await _watch_frames(live_url, f"/watch/{sid}?since=0", count=4))
    # SafeToolSummary carries name+status only — the tool's arguments and output are stripped.
    assert SECRET_ARG_PATH not in blob
    assert TOOL_OUTPUT not in blob


async def test_watch_since_cursor_resumes_after_it(live_url: str) -> None:
    sid = await _run_turn(live_url)
    # Cursors 1..5 (usage=cursor 3). since=3 replays 4,5 → tool_result(completed) + done.
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=3", count=2)
    assert [f["event"] for f in frames] == ["tool_summary", "done"]
    assert frames[0]["status"] == "completed"


async def test_watch_drops_unsafe_usage_event(live_url: str) -> None:
    sid = await _run_turn(live_url)
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=4)
    assert all(f["event"] != "usage" for f in frames)


async def test_watch_unknown_session_is_404(live_url: str) -> None:
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        assert (await ac.get("/watch/does-not-exist")).status_code == 404


async def test_watch_current_alias_resolves_active_session(live_url: str, tmp_path: Path) -> None:
    """``/watch/current`` maps the ``.current-session`` marker to the live session id.

    The PEARL watch UI streams ``/watch/current`` without knowing the concrete id; the box
    resolves it to the session the sidecar-driver names in the marker. The marker is read
    fresh per request, so writing it after the server starts is enough.
    """
    sid = await _run_turn(live_url)
    (tmp_path / ".current-session").write_text(sid, encoding="utf-8")
    # Same buffered frames as watching the concrete id — the alias is transparent.
    frames = await _watch_frames(live_url, "/watch/current?since=0", count=4)
    assert [f["event"] for f in frames] == ["text", "tool_summary", "tool_summary", "done"]
    assert frames[0] == {"event": "text", "text": CANNED}


async def test_watch_current_with_no_active_session_is_404(live_url: str) -> None:
    """``/watch/current`` with no marker → 404 (nothing is active to watch yet)."""
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        resp = await ac.get("/watch/current")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "no active session to watch"


async def test_watch_requires_bearer_when_auth_configured(tmp_path: Path) -> None:
    token = "watch-secret-token"
    with _LiveServer(_make_app(tmp_path, auth_token=token)) as base_url:
        sid = await _run_turn(base_url, token=token)
        async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as ac:
            # No token → rejected by the bearer middleware before the route runs.
            assert (await ac.get(f"/watch/{sid}")).status_code == 401
        # With the token → streams the buffered frames.
        frames = await _watch_frames(base_url, f"/watch/{sid}?since=0", count=4, token=token)
        assert frames[0]["event"] == "text"


# ── watch markers (POST /watch/{session_id}/marker → session_rotated meta-event) ──────


async def test_watch_marker_publishes_session_rotated_to_observers(live_url: str) -> None:
    """POST /watch/{sid}/marker publishes a session_rotated meta-event onto the session's bus;
    a since=0 watcher then sees it as the allow-listed SafeSessionRotated frame. The turn buffers
    5 raw events (the AgentUsage one is dropped at projection → 4 safe frames), so the marker is
    the 5th safe frame."""
    sid = await _run_turn(live_url)  # buffers the turn's events AND creates the bus
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        resp = await ac.post(
            f"/watch/{sid}/marker",
            json={"event": "session_rotated", "reason": "daemon restarted; session re-minted"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["published"] is True and body["cursor"] == 6  # 5 turn events + the marker

    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=5)
    rotated = [f for f in frames if f.get("event") == "session_rotated"]
    assert len(rotated) == 1
    assert rotated[0]["reason"] == "daemon restarted; session re-minted"
    assert frames[-1] == rotated[0]  # published last, so highest cursor


async def test_watch_marker_for_session_with_no_bus_is_noop(live_url: str) -> None:
    """A marker for a session that has no live watch bus (nobody ran a turn or watched) is a
    no-op: it must never spawn a bus, so a bearer holder cannot grow buses for arbitrary ids."""
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        sid = (await ac.post("/sessions")).json()["id"]  # session exists, but no bus created yet
        resp = await ac.post(
            f"/watch/{sid}/marker", json={"event": "session_rotated", "reason": "x"}
        )
        assert resp.status_code == 201
        assert resp.json() == {"published": False, "cursor": None}


async def test_watch_marker_rejects_unknown_event_type(live_url: str) -> None:
    """`event` is a closed Literal allow-list; an unknown marker type is a 422 (pydantic
    validation), never published — the projection whitelist is not the only gate."""
    sid = await _run_turn(live_url)
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        resp = await ac.post(
            f"/watch/{sid}/marker", json={"event": "arbitrary_injected", "reason": "x"}
        )
        assert resp.status_code == 422


async def test_watch_marker_publishes_user_message_to_observers(live_url: str) -> None:
    """POST /watch/{sid}/marker with event=user_message (the watch/talk unification: the
    question the driver consumed) reaches a since=0 watcher as the allow-listed
    SafeUserMessage frame, so the shared transcript reads question-then-answer."""
    sid = await _run_turn(live_url)  # buffers the turn's events AND creates the bus
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        resp = await ac.post(
            f"/watch/{sid}/marker",
            json={"event": "user_message", "text": "what did you learn about volcanoes?"},
        )
        assert resp.status_code == 201
        assert resp.json()["published"] is True

    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=5)
    questions = [f for f in frames if f.get("event") == "user_message"]
    assert len(questions) == 1
    assert questions[0]["text"] == "what did you learn about volcanoes?"
