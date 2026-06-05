"""Tests for the FastAPI app (REST + SSE), driven by a scripted agent factory.

Fully hermetic: a ``_FakeAgent`` stands in for the real Agent (no provider, no
network, no model), and a tmp-dir :class:`SessionStore` isolates persistence. The
app is exercised through FastAPI's ``TestClient``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from zakcode.agent.loop import TurnResult
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta, AgentToolCall, AgentToolResult
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.server.app import create_app
from zakcode.server.wire import event_from_dict
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

CANNED = "Hello from the fake agent."


class _FakeAgent:
    """Minimal AgentLike: mutates the session and emits canned output."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str) -> TurnResult:
        self.session.add_message(Message.user(user_text))
        assistant = Message(
            role="assistant",
            blocks=[
                TextBlock(text=CANNED),
                ToolUseBlock(id="c1", name="write_file", input={"path": "n.txt"}),
            ],
        )
        self.session.add_message(assistant)
        usage = Usage(prompt_tokens=4, completion_tokens=6, total_tokens=10, cost_usd=0.01)
        self.session.add_usage(usage)
        return TurnResult(
            assistant_messages=[assistant],
            tool_results=[ToolResultBlock(tool_use_id="c1", output="wrote n.txt")],
            iterations=2,
            usage=usage,
            stop_reason="completed",
        )

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        events: list[AgentEvent] = [
            AgentTextDelta(text=CANNED),
            AgentToolCall(id="c1", name="write_file", arguments={"path": "n.txt"}),
            AgentToolResult(tool_use_id="c1", output="wrote n.txt", is_error=False),
            AgentDone(stop_reason="completed", iterations=2, usage=Usage(total_tokens=10)),
        ]
        for event in events:
            yield event


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(settings=settings, store=store, agent_factory=_factory)
    return TestClient(app)


# ── meta ───────────────────────────────────────────────────────────────────────


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_config_has_no_secret_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    resp = client.get("/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "api_key" not in body
    assert "sk-must-not-leak" not in json.dumps(body)
    assert body["default_model"] == "scripted/test"


def test_config_omits_api_key_even_when_set(tmp_path: Path) -> None:
    """The Settings.api_key field is excluded from /config even when populated.

    Directly sets the (only) secret-bearing Settings field and asserts neither the
    key nor its value ever reaches the wire — i.e. the exclude=True convention
    holds, not just the defensive pop.
    """
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, api_key="super-secret-token"
    )
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(settings=settings, store=store, agent_factory=_factory)
    body = TestClient(app).get("/config").json()
    assert "api_key" not in body
    assert "super-secret-token" not in json.dumps(body)
    # The attribute itself is still readable in-process (providers rely on it).
    assert settings.api_key == "super-secret-token"


def test_default_factory_model_override_preserves_posture(tmp_path: Path) -> None:
    """A per-request model override swaps the model but keeps the operator posture.

    Regression guard for the bug where overriding ``model`` rebuilt Settings from
    the environment, silently dropping ``permission_mode`` / ``workspace_root``.
    """
    from zakcode import Agent
    from zakcode.server.app import _default_agent_factory

    settings = Settings(
        default_model="configured/base", permission_mode="allow", workspace_root=tmp_path
    )
    store = SessionStore(base_dir=tmp_path / "sessions")
    factory = _default_agent_factory(settings, store)
    session = Session(cwd=str(tmp_path), model="configured/base")

    overridden = factory(session, "override/model", None)
    assert isinstance(overridden, Agent)
    assert overridden.settings.default_model == "override/model"  # model swapped…
    assert overridden.settings.permission_mode == "allow"  # …posture preserved
    assert str(overridden.settings.workspace_root) == str(tmp_path)
    assert overridden.store is store  # store wired in for incremental persistence

    # With no override the exact configured Settings object is reused (no copy).
    plain = factory(session, None, None)
    assert isinstance(plain, Agent)
    assert plain.settings is settings


def test_tools_lists_builtins(client: TestClient) -> None:
    resp = client.get("/tools")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert {"read_file", "write_file", "edit_file", "bash", "glob", "grep"} <= names
    bash = next(t for t in resp.json() if t["name"] == "bash")
    assert bash["required_permission"] == "DANGER_FULL_ACCESS"


# ── session CRUD ────────────────────────────────────────────────────────────────


def test_session_crud(client: TestClient) -> None:
    created = client.post("/sessions")
    assert created.status_code == 201
    sid = created.json()["id"]

    listed = client.get("/sessions")
    assert sid in {s["id"] for s in listed.json()}

    got = client.get(f"/sessions/{sid}")
    assert got.status_code == 200
    assert got.json()["id"] == sid

    deleted = client.delete(f"/sessions/{sid}")
    assert deleted.status_code == 204

    assert client.get(f"/sessions/{sid}").status_code == 404
    assert client.delete(f"/sessions/{sid}").status_code == 404


# ── chat (buffered) ──────────────────────────────────────────────────────────────


def test_chat_creates_session_and_returns_response(client: TestClient) -> None:
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == CANNED
    assert body["stop_reason"] == "completed"
    assert body["iterations"] == 2
    assert body["usage"]["total_tokens"] == 10
    assert body["cost_usd"] == pytest.approx(0.01)
    assert body["tool_results"][0]["output"] == "wrote n.txt"
    assert body["session_id"]  # a session was created


def test_chat_reuses_existing_session(client: TestClient) -> None:
    sid = client.post("/sessions").json()["id"]
    r1 = client.post("/chat", json={"message": "one", "session_id": sid})
    r2 = client.post("/chat", json={"message": "two", "session_id": sid})
    assert r1.json()["session_id"] == sid
    assert r2.json()["session_id"] == sid
    # Both turns persisted onto the same session (2 user + 2 assistant messages).
    info = client.get(f"/sessions/{sid}").json()
    assert info["message_count"] == 4


def test_chat_unknown_session_404(client: TestClient) -> None:
    resp = client.post("/chat", json={"message": "hi", "session_id": "does-not-exist"})
    assert resp.status_code == 404


async def test_chat_rejects_concurrent_turn_on_same_session(tmp_path: Path) -> None:
    """A second turn on a session that already has one in flight is refused (409).

    Without this guard two overlapping turns would race on the same session's
    message list and ``store.save`` and corrupt the transcript — exactly the
    failure M4's session-sharing sub-agents would otherwise hit.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    class _BlockingAgent:
        def __init__(self, session: Session) -> None:
            self.session = session

        async def arun_turn(self, user_text: str) -> TurnResult:
            started.set()
            await release.wait()
            return TurnResult(stop_reason="completed", iterations=1)

        async def astream_turn(
            self, user_text: str
        ) -> AsyncIterator[AgentEvent]:  # pragma: no cover
            started.set()
            await release.wait()
            yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings, store=store, agent_factory=lambda s, m, p: _BlockingAgent(s)
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        sid = (await ac.post("/sessions")).json()["id"]
        # Start a turn that blocks inside the agent — the session is now in flight.
        first = asyncio.create_task(ac.post("/chat", json={"message": "a", "session_id": sid}))
        await asyncio.wait_for(started.wait(), timeout=2)
        # A concurrent turn on the SAME session is refused…
        second = await ac.post("/chat", json={"message": "b", "session_id": sid})
        assert second.status_code == 409
        # …and releasing the first frees the session for a later turn.
        release.set()
        assert (await first).status_code == 200
        third = await ac.post("/chat", json={"message": "c", "session_id": sid})
        assert third.status_code == 200


def test_chat_stream_factory_error_does_not_strand_inflight(tmp_path: Path) -> None:
    # audit2 #5: if the agent factory raises before the SSE stream starts, the per-session
    # reservation must still be released — else the session 409s forever (per-session DoS
    # reachable via a bad request.model). The factory raises once, then succeeds.
    calls = {"n": 0}

    def flaky_factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("bad model string")
        return _FakeAgent(session)

    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(settings=settings, store=store, agent_factory=flaky_factory)
    client = TestClient(app, raise_server_exceptions=False)
    sid = client.post("/sessions").json()["id"]
    first = client.post("/chat/stream", json={"message": "a", "session_id": sid})
    assert first.status_code == 500  # the factory raised
    # The reservation was released despite the error: a later turn is NOT 409'd.
    second = client.post("/chat", json={"message": "b", "session_id": sid})
    assert second.status_code == 200


# ── chat (SSE stream) ────────────────────────────────────────────────────────────


def _sse_events(text: str) -> list[dict]:
    """Extract the JSON payloads from SSE ``data:`` lines."""
    out = []
    for line in text.splitlines():
        if line.startswith("data:"):
            out.append(json.loads(line[len("data:") :].strip()))
    return out


def test_chat_stream_emits_agent_events(client: TestClient) -> None:
    resp = client.post("/chat/stream", json={"message": "go"})
    assert resp.status_code == 200
    payloads = _sse_events(resp.text)
    kinds = [p["event"] for p in payloads]
    assert kinds == ["text", "tool_call", "tool_result", "done"]
    # Each frame parses back into a real AgentEvent (wire contract holds).
    parsed = [event_from_dict(p) for p in payloads]
    assert isinstance(parsed[0], AgentTextDelta)
    assert parsed[0].text == CANNED
    assert isinstance(parsed[-1], AgentDone)
    assert parsed[-1].stop_reason == "completed"


def test_chat_stream_persists_session(client: TestClient) -> None:
    sid = client.post("/sessions").json()["id"]
    client.post("/chat/stream", json={"message": "go", "session_id": sid})
    info = client.get(f"/sessions/{sid}").json()
    # The streamed turn added the user message to the persisted session.
    assert info["message_count"] >= 1
