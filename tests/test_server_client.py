"""Tests for ServerClient — including the M3 parity exit criterion: an
in-process turn and the same turn driven over the HTTP server produce an
identical AgentEvent transcript.

Hermetic: the ServerClient is pointed at the real FastAPI app through httpx's
in-memory ASGI transport (no socket), and the server runs a scripted agent.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from zakcode.agent.loop import TurnResult
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta, AgentToolCall, AgentToolResult
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.server.client import ServerClient
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

# A fixed transcript the scripted agent emits for ANY input, so the in-process
# and over-server runs are directly comparable.
SCRIPT: list[AgentEvent] = [
    AgentTextDelta(text="Reading then writing.\n"),
    AgentToolCall(id="c1", name="read_file", arguments={"path": "a.txt"}),
    AgentToolResult(tool_use_id="c1", output="contents of a", is_error=False),
    AgentToolCall(id="c2", name="write_file", arguments={"path": "b.txt", "content": "x"}),
    AgentToolResult(tool_use_id="c2", output="wrote b.txt", is_error=False),
    AgentTextDelta(text="Done."),
    AgentDone(stop_reason="completed", iterations=3, usage=Usage(total_tokens=42)),
]


async def test_auth_token_sets_bearer_header() -> None:
    # SRV-17: chat --server against a token-protected server must send Authorization: Bearer
    # (the remote-driver path 401'd before -- ServerClient had no auth surface).
    client = ServerClient("http://srv.example", auth_token="s3cret")
    assert client._http().headers.get("authorization") == "Bearer s3cret"
    await client.aclose()
    plain = ServerClient("http://srv.example")
    assert plain._http().headers.get("authorization") is None
    await plain.aclose()


class _ScriptedAgent:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str) -> TurnResult:  # pragma: no cover
        return TurnResult(stop_reason="completed", iterations=3)

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        # Mirror the real loop: record the user message on the session so the
        # server's post-turn persist has something to save (proving the streamed
        # turn lands on the addressed session).
        self.session.add_message(Message.user(user_text))
        for event in SCRIPT:
            yield event


def _app_client(tmp_path: Path) -> httpx.AsyncClient:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings, store=store, agent_factory=lambda s, m, p: _ScriptedAgent(s)
    )
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_server_client_health(tmp_path: Path) -> None:
    http = _app_client(tmp_path)
    try:
        client = ServerClient(http_client=http)
        health = await client.health()
        assert health["status"] == "ok"
    finally:
        await http.aclose()


async def test_server_client_create_session(tmp_path: Path) -> None:
    http = _app_client(tmp_path)
    try:
        client = ServerClient(http_client=http)
        sid = await client.create_session()
        assert isinstance(sid, str) and sid
    finally:
        await http.aclose()


async def test_server_client_streams_events(tmp_path: Path) -> None:
    http = _app_client(tmp_path)
    try:
        client = ServerClient(http_client=http)
        events = [e async for e in client.astream_turn("go")]
        assert [e.event for e in events] == [e.event for e in SCRIPT]
        assert isinstance(events[-1], AgentDone)
    finally:
        await http.aclose()


async def test_in_process_and_over_server_transcripts_match(tmp_path: Path) -> None:
    """The M3 parity exit criterion.

    Drive the SAME scripted agent (a) in-process and (b) through the HTTP server,
    and assert the resulting AgentEvent transcripts are byte-for-byte identical
    once serialized — proving the server adds transport, not behavior.
    """
    # (a) in-process
    in_proc_agent = _ScriptedAgent(Session(cwd=".", model="scripted/test"))
    in_proc = [e async for e in in_proc_agent.astream_turn("go")]

    # (b) over the server
    http = _app_client(tmp_path)
    try:
        client = ServerClient(http_client=http)
        over_server = [e async for e in client.astream_turn("go")]
    finally:
        await http.aclose()

    assert [e.model_dump(mode="json") for e in over_server] == [
        e.model_dump(mode="json") for e in in_proc
    ]


async def test_server_client_propagates_session_id(tmp_path: Path) -> None:
    # A turn against an explicit session id reaches that session on the server.
    http = _app_client(tmp_path)
    try:
        client = ServerClient(http_client=http)
        sid = await client.create_session()
        _ = [e async for e in client.astream_turn("go", session_id=sid)]
        # The server persisted the streamed turn onto that session.
        info = (await http.get(f"/sessions/{sid}")).json()
        assert info["message_count"] >= 1
    finally:
        await http.aclose()


async def test_server_client_raises_on_bad_status(tmp_path: Path) -> None:
    http = _app_client(tmp_path)
    try:
        client = ServerClient(http_client=http)
        with pytest.raises(httpx.HTTPStatusError):
            await client.astream_turn("go", session_id="nonexistent").__anext__()
    finally:
        await http.aclose()


async def test_publish_watch_marker_posts_expected_body() -> None:
    # The client's contract: POST /watch/{sid}/marker with the session_rotated body. A
    # MockTransport captures the wire form without needing a live bus (the driver's caller
    # supplies the sid; the server no-ops when no watcher is present).
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(201, json={"published": True, "cursor": 3})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        await ServerClient(http_client=http).publish_watch_marker(
            "sess-42", reason="daemon restarted"
        )
    finally:
        await http.aclose()
    assert seen["method"] == "POST"
    assert str(seen["url"]).endswith("/watch/sess-42/marker")
    assert seen["json"] == {"event": "session_rotated", "reason": "daemon restarted"}


async def test_publish_watch_marker_raises_on_error_status() -> None:
    # Like every other client call, a non-2xx raises — the driver decides whether to swallow it
    # (a rotation marker must never break the serve loop, so the driver wraps this call).
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://testserver")
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await ServerClient(http_client=http).publish_watch_marker("sess-42")
    finally:
        await http.aclose()
