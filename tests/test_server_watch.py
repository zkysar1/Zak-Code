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

import asyncio
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
from fastapi.testclient import TestClient

from zakcode import SkillInvocation
from zakcode.agent.loop import TurnResult
from zakcode.config import Settings
from zakcode.events import (
    AgentDone,
    AgentEvent,
    AgentTextDelta,
    AgentToolCall,
    AgentToolResult,
    AgentUsage,
)
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.server import app as server_app
from zakcode.server import event_bus
from zakcode.server.app import create_app
from zakcode.server.event_bus import EventBusRegistry
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
            AgentToolResult(
                tool_use_id="c1", output=TOOL_OUTPUT, is_error=False, name="write_file"
            ),
            AgentDone(stop_reason="completed", iterations=2, usage=Usage(total_tokens=10)),
        ]
        for event in events:
            yield event


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(tmp_path: Path, *, auth_token: str | None = None) -> FastAPI:
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        auth_token=auth_token,
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


def _unstamped(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pop every frame's publish stamp (ADR-0220), after pinning it: epoch seconds, in order."""
    stamps = [frame.pop("at") for frame in frames]
    assert all(isinstance(at, float) for at in stamps), stamps
    assert stamps == sorted(stamps), stamps
    return frames


async def test_watch_replays_buffered_turn_events_as_safe_events(live_url: str) -> None:
    sid = await _run_turn(live_url)
    # The user's line, then 5 raw events; the AgentUsage one is dropped → 5 SafeEvent frames.
    frames = _unstamped(await _watch_frames(live_url, f"/watch/{sid}?since=0", count=5))
    assert [f["event"] for f in frames] == [
        "user_message",
        "text",
        "tool_summary",
        "tool_summary",
        "done",
    ]
    # A /chat/stream turn puts the user's line on the bus first, as a say does (ADR-0325).
    assert frames[0] == {"event": "user_message", "text": "hi"}
    assert frames[1] == {"event": "text", "text": CANNED}
    # The terminal's call and receipt lines, from the name and a COUNT (ADR-0220): the
    # output ("wrote n.txt") and the argument ("n.txt") never cross, as the next test pins.
    assert frames[2] == {
        "event": "tool_summary",
        "name": "write_file",
        "status": "running",
        "used_secrets": [],
        "display_name": "Write",
        "receipt": "",
    }
    assert frames[3] == {
        "event": "tool_summary",
        "name": "write_file",
        "status": "completed",
        "used_secrets": [],
        "display_name": "Write",
        "receipt": "Written",
    }
    assert frames[4] == {
        "event": "done",
        "stop_reason": "completed",
        "iterations": 2,
        "label": "done",
    }


async def test_watch_whitelist_never_leaks_tool_args_or_output(live_url: str) -> None:
    sid = await _run_turn(live_url)
    blob = json.dumps(await _watch_frames(live_url, f"/watch/{sid}?since=0", count=5))
    # SafeToolSummary carries name+status only — the tool's arguments and output are stripped.
    assert SECRET_ARG_PATH not in blob
    assert TOOL_OUTPUT not in blob


async def test_watch_since_cursor_resumes_after_it(live_url: str) -> None:
    sid = await _run_turn(live_url)
    # Cursors 1..6 (the user's line=1, usage=4). since=3 replays 4..6 → usage (dropped),
    # tool_result(completed) + done.
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=3", count=2)
    assert [f["event"] for f in frames] == ["tool_summary", "done"]
    assert frames[0]["status"] == "completed"


async def _bus_epoch(base_url: str, sid: str) -> str:
    """The name a watch response gives its bus (``X-Bus-Epoch``), read from the headers alone."""
    timeout = httpx.Timeout(10.0, read=5.0)
    async with (
        httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac,
        ac.stream("GET", f"/watch/{sid}") as resp,
    ):
        assert resp.status_code == 200
        return resp.headers.get("x-bus-epoch", "")


async def test_watch_names_its_bus_in_a_response_header(live_url: str) -> None:
    """Cursors count from 1 on every bus, so a client resuming at one needs to know which bus it
    came from. Every connection to the same session's bus reports the same name (ADR-0289)."""
    sid = await _run_turn(live_url)
    epoch = await _bus_epoch(live_url, sid)
    assert epoch  # present, and not empty
    assert await _bus_epoch(live_url, sid) == epoch


async def test_watch_ignores_a_cursor_that_belongs_to_another_bus(live_url: str) -> None:
    """The same ``since=3`` that resumes after cursor 3 above means nothing to a bus the client
    never held a cursor from (a server restarted since): the whole buffer replays, not the
    two events past a number the new bus happens to reach."""
    sid = await _run_turn(live_url)
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=3&epoch=another-bus", count=5)
    assert [f["event"] for f in frames] == [
        "user_message",
        "text",
        "tool_summary",
        "tool_summary",
        "done",
    ]


async def test_watch_names_the_oldest_cursor_its_bus_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page resuming after a cursor this bus has since evicted cannot pick up where it was:
    what followed its cursor is gone. The response names the oldest cursor the bus still holds
    (``X-Bus-Oldest``), so such a page knows to start over; a bus holding nothing names none
    (ADR-0325)."""
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: EventBusRegistry(maxlen=3))
    with _LiveServer(_make_app(tmp_path)) as base_url:
        sid = await _run_turn(base_url)  # the user's line and 5 events: cursors 1..6, 4..6 kept
        timeout = httpx.Timeout(10.0, read=5.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac:
            fresh = (await ac.post("/sessions")).json()["id"]
            async with ac.stream("GET", f"/watch/{sid}") as resp:
                assert resp.headers.get("x-bus-oldest") == "4"
            async with ac.stream("GET", f"/watch/{fresh}") as resp:
                assert resp.status_code == 200
                assert resp.headers.get("x-bus-oldest") is None


async def test_watch_resumes_at_a_cursor_that_belongs_to_this_bus(live_url: str) -> None:
    """A client that names this bus resumes exactly as one that names none does
    (test_watch_since_cursor_resumes_after_it): only what follows its cursor."""
    sid = await _run_turn(live_url)
    epoch = await _bus_epoch(live_url, sid)
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=3&epoch={epoch}", count=2)
    assert [f["event"] for f in frames] == ["tool_summary", "done"]


async def test_the_transcript_names_the_bus_its_start_belongs_to(live_url: str) -> None:
    """``bus_from`` is a place on one bus's clock, and the page's cut of the conversation holds
    only against that bus: the transcript names it, with the name a watch response carries, so a
    page whose watch is answered by another bus knows to cut again (ADR-0289)."""
    sid = await _run_turn(live_url)
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        transcript = (await ac.get(f"/sessions/{sid}/transcript")).json()
    assert transcript["bus_from"] is not None
    assert transcript["bus_epoch"] == await _bus_epoch(live_url, sid)
    assert transcript["bus_epoch"]  # present, and not empty


async def test_the_transcript_of_a_session_with_no_bus_names_none(live_url: str) -> None:
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        sid = (await ac.post("/sessions")).json()["id"]
        transcript = (await ac.get(f"/sessions/{sid}/transcript")).json()
    assert transcript["bus_from"] is None
    assert transcript["bus_epoch"] is None


async def test_watch_drops_unsafe_usage_event(live_url: str) -> None:
    sid = await _run_turn(live_url)
    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=4)
    assert all(f["event"] != "usage" for f in frames)


async def test_watch_unknown_session_is_404(live_url: str) -> None:
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        assert (await ac.get("/watch/does-not-exist")).status_code == 404


async def test_watch_current_alias_resolves_active_session(live_url: str, tmp_path: Path) -> None:
    """``/watch/current`` maps the ``.current-session`` marker to the live session id.

    The watch UI streams ``/watch/current`` without knowing the concrete id; the box
    resolves it to the session the sidecar-driver names in the marker. The marker is read
    fresh per request, so writing it after the server starts is enough.
    """
    sid = await _run_turn(live_url)
    (tmp_path / ".current-session").write_text(sid, encoding="utf-8")
    # Same buffered frames as watching the concrete id — the alias is transparent.
    frames = _unstamped(await _watch_frames(live_url, "/watch/current?since=0", count=5))
    assert [f["event"] for f in frames] == [
        "user_message",
        "text",
        "tool_summary",
        "tool_summary",
        "done",
    ]
    assert frames[1] == {"event": "text", "text": CANNED}


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
        assert frames[0]["event"] == "user_message"


# ── watch markers (POST /watch/{session_id}/marker → session_rotated meta-event) ──────


async def test_watch_marker_publishes_session_rotated_to_observers(live_url: str) -> None:
    """POST /watch/{sid}/marker publishes a session_rotated meta-event onto the session's bus;
    a since=0 watcher then sees it as the allow-listed SafeSessionRotated frame. The turn buffers
    its user's line and 5 raw events (the AgentUsage one is dropped at projection → 5 safe
    frames), so the marker is the 6th safe frame."""
    sid = await _run_turn(live_url)  # buffers the turn's events AND creates the bus
    async with httpx.AsyncClient(base_url=live_url, timeout=10.0) as ac:
        resp = await ac.post(
            f"/watch/{sid}/marker",
            json={"event": "session_rotated", "reason": "daemon restarted; session re-minted"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["published"] is True and body["cursor"] == 7  # 6 turn events + the marker

    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=6)
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

    frames = await _watch_frames(live_url, f"/watch/{sid}?since=0", count=6)
    questions = [f for f in frames if f.get("event") == "user_message"]
    # The turn's own line comes first (ADR-0325); the driver's marker is the one published after.
    assert [q["text"] for q in questions] == ["hi", "what did you learn about volcanoes?"]


# ── a bus let go (ADR-0327) ──────────────────────────────────────────────────


async def test_deleting_a_session_ends_its_watch_and_its_reconnect_hears_it_is_gone(
    live_url: str,
) -> None:
    """The bus went on holding a deleted chat's events, and a page watching the chat was never
    let go: its stream stayed open on a chat that no longer existed."""
    sid = await _run_turn(live_url)
    timeout = httpx.Timeout(10.0, read=5.0)  # a stream left open fails here, not as a hang
    async with httpx.AsyncClient(base_url=live_url, timeout=timeout) as ac:
        async with ac.stream("GET", f"/watch/{sid}?full=1") as resp:
            assert resp.status_code == 200
            lines = resp.aiter_lines()
            async for line in lines:
                if line.startswith("data:"):
                    break  # replaying, so the watch is on the bus
            assert (await ac.delete(f"/sessions/{sid}")).status_code == 204
            async for _ in lines:  # the rest of the replay, then the end of the stream
                pass
        assert (await ac.get(f"/watch/{sid}")).status_code == 404


def test_the_current_chats_bus_is_never_let_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(event_bus, "_IDLE_BUSES", 0)
    registry = EventBusRegistry()
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: registry)
    _make_app(tmp_path)
    (tmp_path / ".current-session").write_text("current\n", encoding="utf-8")
    current = registry.get_or_create("current")
    left = registry.get_or_create("left")
    registry.get_or_create("new")  # a sweep that keeps no idle bus
    assert left.closed
    assert not current.closed


async def test_a_marker_for_a_bus_let_go_after_it_was_found_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker route finds the bus and then publishes on it, on a thread of its own; a sweep
    on the event loop can let the bus go in between. Nobody watched it, so there is nobody to
    tell: the route says it published nothing, where publishing on the closed bus raised."""

    class _LetGoOnceFound(EventBusRegistry):
        def get(self, session_id: str) -> event_bus.SessionEventBus | None:
            bus = super().get(session_id)
            if bus is not None:
                bus.close()  # the sweep, between the route's get and its publish
            return bus

    registry = _LetGoOnceFound()
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: registry)
    app = _make_app(tmp_path)
    registry.get_or_create("s1")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        resp = await ac.post(
            "/watch/s1/marker", json={"event": "session_rotated", "reason": "another chat"}
        )
    assert resp.status_code == 201
    assert resp.json() == {"published": False, "cursor": None}


async def test_a_running_turns_bus_is_let_go_neither_by_a_delete_nor_by_a_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn holds its bus to publish on, and publishing on a bus that was let go raises, so
    while the turn runs its chat's bus stays, deleted or idle."""
    monkeypatch.setattr(event_bus, "_IDLE_BUSES", 0)
    registry = EventBusRegistry()
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: registry)
    started, release = asyncio.Event(), asyncio.Event()

    class _BlockingAgent:
        def __init__(self, session: Session) -> None:
            self.session = session

        async def arun_turn(self, user_text: str) -> TurnResult:
            started.set()
            await release.wait()
            return TurnResult(stop_reason="completed", iterations=1)

    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings, store=store, agent_factory=lambda s, m, p: _BlockingAgent(s)
    )
    # Another chat is current, so only the running turn keeps this one's bus (a first chat
    # would otherwise become the current one, and stay for that reason alone).
    (tmp_path / ".current-session").write_text("elsewhere\n", encoding="utf-8")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        sid = (await ac.post("/sessions")).json()["id"]
        turn = asyncio.create_task(ac.post("/chat", json={"message": "a", "session_id": sid}))
        await asyncio.wait_for(started.wait(), timeout=2)
        bus = registry.get(sid)  # the turn's own: its prompter publishes on it
        assert bus is not None
        registry.get_or_create("another")  # a sweep that keeps no idle bus
        assert (await ac.delete(f"/sessions/{sid}")).status_code == 204
        assert not bus.closed
        release.set()
        assert (await turn).status_code == 200


# ── a POST /chat turn on the bus (ADR-0327) ──────────────────────────────────


class _BufferedAgent:
    """AgentLike that runs a turn the buffered way only, as ``POST /chat`` does: a reply that
    reads a file, the file, and a closing reply. With ``fail`` the turn raises instead, and the
    skill ``hidden`` is refused."""

    def __init__(self, session: Session, *, fail: bool = False) -> None:
        self.session = session
        self.fail = fail

    async def compose_skill_turn(self, name: str, args: str = "") -> SkillInvocation:  # noqa: ARG002
        if name == "hidden":
            return SkillInvocation(
                invoked=True, name="hidden", denied_reason="hidden is user-invocable: false"
            )
        return SkillInvocation(invoked=False)

    async def arun_turn(self, user_text: str) -> TurnResult:
        if self.fail:
            raise RuntimeError("the provider went away")
        self.session.add_message(Message.user(user_text))
        looks = Message(
            role="assistant",
            blocks=[
                TextBlock(text="let me look"),
                ToolUseBlock(id="c1", name="read", input={"path": "notes.md"}),
            ],
        )
        answer = Message.assistant_text("they say hello")
        self.session.add_message(looks)
        self.session.add_message(answer)
        return TurnResult(
            assistant_messages=[looks, answer],
            tool_results=[ToolResultBlock(tool_use_id="c1", output="the notes")],
            iterations=2,
            usage=Usage(total_tokens=7),
        )


def _chat_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail: bool = False
) -> tuple[TestClient, EventBusRegistry]:
    registry = EventBusRegistry()
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: registry)
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings, store=store, agent_factory=lambda s, m, p: _BufferedAgent(s, fail=fail)
    )
    return TestClient(app, raise_server_exceptions=False), registry


def _held(registry: EventBusRegistry, sid: str) -> list[Any]:
    """What the chat's bus holds, oldest first: its replay, read to the end and no further."""
    bus = registry.get(sid)
    assert bus is not None

    async def replay() -> list[Any]:
        agen = bus.subscribe(since=None)
        held = [(await agen.__anext__())[1] for _ in range(bus.latest_cursor)]
        await agen.aclose()
        return held

    return asyncio.run(replay())


def test_a_chat_turn_reaches_the_watch_bus_once_it_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``POST /chat`` put nothing on the bus, so a page watching the chat never saw its turns.
    Once a turn is done, the bus now gets the frames a streamed turn sends, in their order."""
    client, registry = _chat_app(tmp_path, monkeypatch)
    sid = client.post("/sessions").json()["id"]
    resp = client.post("/chat", json={"message": "what do the notes say?", "session_id": sid})
    assert resp.status_code == 200
    held = _held(registry, sid)
    assert [event.event for event in held] == [
        "user_message",
        "text",
        "tool_call",
        "tool_result",
        "text",
        "done",
    ]
    question, looks, call, read, answer, done = held
    assert question.text == "what do the notes say?"
    assert (looks.text, answer.text) == ("let me look", "they say hello")
    assert (call.id, call.name, call.arguments) == ("c1", "read", {"path": "notes.md"})
    assert (read.tool_use_id, read.output, read.name) == ("c1", "the notes", "read")
    assert (done.stop_reason, done.iterations, done.usage.total_tokens) == ("completed", 2, 7)


def test_a_refused_chat_puts_its_refusal_on_the_bus_as_a_streamed_one_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, registry = _chat_app(tmp_path, monkeypatch)
    sid = client.post("/sessions").json()["id"]
    assert client.post("/chat", json={"message": "/hidden", "session_id": sid}).status_code == 403
    held = _held(registry, sid)
    assert [event.event for event in held] == ["user_message", "status", "done"]
    assert held[2].stop_reason == "skill_refused"


def test_a_chat_turn_that_fails_still_ends_on_the_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page had the turn's line, and with no ``done`` after it showed the turn thinking
    until its next reconnect."""
    client, registry = _chat_app(tmp_path, monkeypatch, fail=True)
    sid = client.post("/sessions").json()["id"]
    assert client.post("/chat", json={"message": "hi", "session_id": sid}).status_code == 500
    held = _held(registry, sid)
    assert [event.event for event in held] == ["user_message", "done"]
    assert (held[1].stop_reason, held[1].degraded) == ("provider_error", True)
