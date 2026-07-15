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
    assert frames[1] == {"event": "tool_summary", "name": "write_file", "status": "running"}
    assert frames[2] == {"event": "tool_summary", "name": "", "status": "completed"}
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
