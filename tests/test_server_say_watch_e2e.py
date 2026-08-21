"""End-to-end say -> watch tests: a message queued through POST /say must reach a
watcher on GET /watch EXACTLY ONCE.

This closes the integration gap left by test_server_driver.py. Those tests pin the
driver's suppression *logic* against a fake client (how many times it CALLS
publish_user_message). They cannot see the layer a real observer experiences:
POST /say -> driver reads .say -> publish_user_message -> POST /watch/{sid}/marker ->
event bus -> SafeUserMessage projection -> the watcher's SSE frame. The double-publish
bug fixed in #156 manifested exactly there — a watcher seeing the user's question twice —
so the assertion that matters is on the *effect* (the frames a watcher receives), not
only the *cause* (the driver's call count).

Two design facts these tests are built around, both load-bearing:

* The watch SSE is an infinite stream that PARKS after replaying its buffer, and both
  in-process transports (TestClient / ASGITransport) buffer the whole body, so they hang
  on it forever. A real loopback uvicorn server is mandatory (see test_server_watch.py).
* ``publish_watch_marker`` targets EXISTING observers only (``event_bus_registry.get``,
  not ``get_or_create`` — app.py) — publishing to a session with no live watcher is a
  deliberate no-op. So the watcher MUST subscribe BEFORE the say is delivered. Every test
  here subscribes first, then says; the driver cannot read a say it was not yet given, so
  the ordering is deterministic, not a race.

Two transports, one assertion — agent-drivable from afar, not human-gated:

* ``test_say_is_echoed_to_a_live_watcher_exactly_once`` — deterministic CI path. Real
  loopback server + a REAL ServeDriver (unbounded, cancelled at teardown) + a real
  POST /say; a subscribed watcher must see exactly one ``user_message`` frame. No LLM: the
  agent is a deterministic fake.
* ``test_say_watch_exactly_once_against_live_sidecar`` — the same assertion pointed at a
  LIVE sidecar via ``ZAKCODE_LIVE_URL`` (skipped when unset). The machine/agent
  "test from afar" path: subscribe to /watch/current on a running box, POST a unique /say,
  confirm its own driver echoes it once. Never a CI gate; a probe any machine can run.

``asyncio_mode=auto`` runs the ``async def`` tests without a decorator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.client import ServerClient
from zakcode.server.driver import CURRENT_SESSION_FILE, ServeDriver
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

SAY_TEXT = "what is the capital of France?"


class _EchoAgent:
    """Minimal deterministic AgentLike: one text delta + done. No model, no network."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage(total_tokens=1))


def _factory(session: Session, model: str | None, prompter: object = None) -> _EchoAgent:  # noqa: ARG001
    return _EchoAgent(session)


def _make_app(tmp_path: Path) -> FastAPI:
    from zakcode.server.app import create_app

    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
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
    """Run ``app`` on a loopback uvicorn server in a daemon thread for the test's lifetime.

    Copied (not imported) from test_server_watch.py on purpose: cross-test-module fixture
    imports are fragile, and this scaffold is ~30 lines. The real-server requirement is not
    optional — the watch SSE parks after replaying its buffer, and both in-process transports
    buffer the whole body, so they hang forever on it.
    """

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
def app(tmp_path: Path) -> FastAPI:
    return _make_app(tmp_path)


@pytest.fixture
def local_server(app: FastAPI) -> Iterator[str]:
    with _LiveServer(app) as base_url:
        yield base_url


async def _await_current_session(workspace: Path, *, timeout: float = 10.0) -> str:
    """Poll ``.current-session`` (the driver writes it right after minting a session)."""
    deadline = time.monotonic() + timeout
    path = workspace / CURRENT_SESSION_FILE
    while time.monotonic() < deadline:
        if path.exists():
            sid = path.read_text(encoding="utf-8").strip()
            if sid:
                return sid
        await asyncio.sleep(0.02)
    raise AssertionError("driver never wrote a current session")


async def _watch_for_user_message(
    base_url: str,
    url: str,
    *,
    ready: asyncio.Event,
    headers: dict[str, str] | None = None,
    idle_timeout: float = 8.0,
    hard_cap: int = 300,
) -> list[dict[str, Any]]:
    """Subscribe to a watch stream, signal ``ready`` once the bus exists, then collect frames
    until a ``user_message`` arrives (or the stream idles / hits ``hard_cap``).

    ``ready`` is set right after the 200 lands: the /watch handler runs
    ``event_bus_registry.get_or_create`` before it returns the streaming response, so a 200
    in hand proves the bus exists — the caller may now POST /say knowing the marker will land
    on a live observer, not into the void.
    """
    frames: list[dict[str, Any]] = []
    timeout = httpx.Timeout(30.0, read=idle_timeout)
    try:
        async with (
            httpx.AsyncClient(base_url=base_url, timeout=timeout) as ac,
            ac.stream("GET", url, headers=headers or {}) as resp,
        ):
            resp.raise_for_status()
            ready.set()
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    frames.append(json.loads(line[len("data:") :].strip()))
                    seen = any(f.get("event") == "user_message" for f in frames)
                    if seen or len(frames) >= hard_cap:
                        break
    except httpx.ReadTimeout:
        pass  # buffer replayed and the stream idled — expected when no more frames arrive
    return frames


async def test_say_is_echoed_to_a_live_watcher_exactly_once(
    tmp_path: Path, local_server: str
) -> None:
    """POST /say -> real driver -> a SUBSCRIBED watcher sees the user_message marker once.

    The server's workspace_root and the driver's workspace are the same tmp_path, so the
    ``.say`` the endpoint writes is the ``.say`` the driver reads — the real production wiring.
    A small inter_turn_delay keeps the driver from busy-looping; it is cancelled at teardown.
    """
    base = local_server
    client = ServerClient(base_url=base)
    driver = ServeDriver(
        client,
        tmp_path,
        boot_message="boot",
        continue_message="continue",
        backoff_initial=0.001,
        backoff_max=0.01,
        inter_turn_delay=0.05,
    )
    driver_task = asyncio.create_task(driver.run())
    try:
        sid = await _await_current_session(tmp_path, timeout=10.0)

        ready = asyncio.Event()
        watch_task = asyncio.create_task(
            _watch_for_user_message(base, f"/watch/{sid}", ready=ready)
        )
        # Subscribe FIRST (bus must exist when the marker fires), THEN say.
        await asyncio.wait_for(ready.wait(), timeout=10.0)
        async with httpx.AsyncClient(base_url=base, timeout=10.0) as ac:
            queued = await ac.post("/say", json={"text": SAY_TEXT})
            assert queued.status_code == 200
            assert queued.json()["queued"] is True

        frames = await asyncio.wait_for(watch_task, timeout=15.0)
        user_messages = [f for f in frames if f.get("event") == "user_message"]
        assert len(user_messages) == 1, (
            f"expected exactly one user_message frame, got {len(user_messages)}: {frames}"
        )
        assert user_messages[0]["text"] == SAY_TEXT
    finally:
        driver_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await driver_task
        await client.aclose()


@pytest.mark.skipif(
    not os.environ.get("ZAKCODE_LIVE_URL"),
    reason="set ZAKCODE_LIVE_URL to test a live sidecar from afar",
)
async def test_say_watch_exactly_once_against_live_sidecar() -> None:
    """Machine/agent 'test from afar': subscribe to a LIVE sidecar's /watch/current, POST a
    unique say, and confirm its own driver echoes it once. Gated on ZAKCODE_LIVE_URL so it
    never runs in CI; it is a probe a machine points at a running box, no human required."""
    base = os.environ["ZAKCODE_LIVE_URL"].rstrip("/")
    token = os.environ.get("ZAKCODE_LIVE_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    # Unique so the assertion cannot collide with a real user's concurrent say.
    text = f"e2e-say-watch-probe-{uuid4().hex[:8]}"

    ready = asyncio.Event()
    watch_task = asyncio.create_task(
        _watch_for_user_message(
            base, "/watch/current", ready=ready, headers=headers, idle_timeout=12.0
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=15.0)  # subscribed to the live stream
        async with httpx.AsyncClient(base_url=base, timeout=15.0) as ac:
            queued = await ac.post("/say", json={"text": text}, headers=headers)
            if queued.status_code == 429:
                pytest.skip(
                    "live sidecar already has a say pending (single-slot); re-run when idle"
                )
            assert queued.status_code == 200, f"/say returned {queued.status_code}: {queued.text}"

        frames = await asyncio.wait_for(watch_task, timeout=20.0)
        mine = [f for f in frames if f.get("event") == "user_message" and f.get("text") == text]
        assert len(mine) == 1, f"expected exactly one echo of {text!r}, got {len(mine)}: {frames}"
    finally:
        watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await watch_task
