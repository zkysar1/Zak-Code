"""The serve-side reactive say consumer — the web convergence's turn-runner.

The web page (and every other surface) is a pure viewer + say-writer: input
reaches the agent ONLY through the workspace say inbox, and the SERVER runs the
turn. These tests exercise the consumer beat through the ``app.state.consume_one_say``
seam (httpx's ASGITransport does not run lifespan events, so the background loop
never starts here — each test drives exactly the beats it means to).

Covers: a say becomes a turn on the workspace's current session (marker written,
session saved, inbox consumed exactly once); an idle inbox is a no-op beat; a beat
yields while a turn is in flight; ``POST /interrupt`` writes the contract's
interrupt file; the ``GET /sessions/current`` join flow the web page boots through;
and — against a REAL loopback uvicorn server, where the lifespan actually starts
the background loop — the full path the web page rides: POST /say → the consumer
runs the turn → a late-joining ``?full=1`` watcher replays ``user_message`` +
AgentEvents. (The live server is mandatory for any watch-stream read: the SSE
parks after replay and in-process transports buffer the whole body — see
test_server_watch.py.)
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.session.say_inbox import interrupt_path, say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


class _FakeAgent:
    """Instant scripted turn: records the user text, emits one delta + done."""

    def __init__(self, session: Session, gate: asyncio.Event | None = None) -> None:
        self.session = session
        self._gate = gate

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        if self._gate is not None:
            await self._gate.wait()
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _build(tmp_path: Path, gate: asyncio.Event | None = None) -> tuple[Any, SessionStore]:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: _FakeAgent(session, gate),
    )
    return app, store


def test_say_becomes_a_turn_on_the_current_session(tmp_path: Path) -> None:
    """One say → one server-run turn: marker written, session saved, slot consumed."""
    app, store = _build(tmp_path)
    assert write_say(say_path(tmp_path), "hello there")

    ran = asyncio.run(app.state.consume_one_say())
    assert ran is True
    assert not say_path(tmp_path).exists()  # consumed exactly once

    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    session = store.load(marker)
    assert [m.role for m in session.messages] == ["user", "assistant"]


def test_idle_inbox_is_a_noop_beat(tmp_path: Path) -> None:
    app, _ = _build(tmp_path)
    assert asyncio.run(app.state.consume_one_say()) is False
    assert not (tmp_path / ".current-session").exists()  # no phantom session


def test_beat_yields_while_a_turn_is_in_flight(tmp_path: Path) -> None:
    """The single-turn discipline: a say arriving mid-turn WAITS (stays queued in
    the slot) instead of racing a second concurrent turn."""
    gate = asyncio.Event()
    app, _ = _build(tmp_path, gate)

    async def go() -> tuple[bool, bool]:
        assert write_say(say_path(tmp_path), "first")
        first = asyncio.create_task(app.state.consume_one_say())
        await asyncio.sleep(0.05)  # first turn is now blocked on the gate
        assert write_say(say_path(tmp_path), "second")
        skipped = await app.state.consume_one_say()
        gate.set()
        return await first, skipped

    ran_first, skipped = asyncio.run(go())
    assert ran_first is True
    assert skipped is False
    assert say_path(tmp_path).read_text(encoding="utf-8").strip() == "second"  # still queued


def test_interrupt_route_writes_the_contract_file(tmp_path: Path) -> None:
    app, _ = _build(tmp_path)

    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/interrupt")

    resp = asyncio.run(go())
    assert resp.status_code == 200
    assert resp.json() == {"requested": True}
    assert interrupt_path(tmp_path).exists()


def test_sessions_current_join_flow(tmp_path: Path) -> None:
    """The web page's boot: 404 with no marker → create (writes the marker) →
    ``/sessions/current`` returns the same session thereafter."""
    app, _ = _build(tmp_path)

    async def go() -> tuple[int, str, str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.get("/sessions/current")
            created = await client.post("/sessions", json={})
            current = await client.get("/sessions/current")
            assert current.status_code == 200
            return missing.status_code, created.json()["id"], current.json()["id"]

    missing_status, created_id, current_id = asyncio.run(go())
    assert missing_status == 404
    assert created_id == current_id
    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    assert marker == created_id


# ── the real thing: live server, background loop ON, a watcher on ?full=1 ────────


class _LiveServer:
    """Loopback uvicorn in a daemon thread (copied per test_server_watch.py precedent)."""

    def __init__(self, app: Any) -> None:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = int(s.getsockname()[1])
        s.close()
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


async def test_web_page_path_say_to_full_watch(tmp_path: Path) -> None:
    """The exact path the served web page rides, end to end: POST /say → the
    lifespan-started consumer runs the turn → GET /sessions/current resolves →
    a late-joining ``?full=1`` watcher replays the ``user_message`` marker and
    the turn's AgentEvents in order."""
    app, _ = _build(tmp_path)  # serve_consume defaults ON — the loop runs for real
    with _LiveServer(app) as base_url:
        timeout = httpx.Timeout(10.0, read=5.0)
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac:
            resp = await ac.post("/say", json={"text": "hello there"})
            assert resp.status_code in (200, 202)

            sid = None
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                current = await ac.get("/sessions/current")
                if current.status_code == 200:
                    sid = current.json()["id"]
                    break
                await asyncio.sleep(0.05)
            assert sid is not None, "consumer never minted the current session"

            frames: list[dict[str, Any]] = []
            async with ac.stream("GET", f"/watch/{sid}?full=1") as watch:
                assert watch.status_code == 200
                async for line in watch.aiter_lines():
                    if line.startswith("data:"):
                        frames.append(json.loads(line[len("data:") :].strip()))
                        if any(f.get("event") == "done" for f in frames):
                            break

    assert frames[0]["event"] == "user_message"
    assert frames[0]["text"] == "hello there"
    assert {"event": "text", "text": "ok"} in frames
    assert frames[-1]["event"] == "done"
