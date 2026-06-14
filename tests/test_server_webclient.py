"""Tests for the M10 web-client server surface: the AgentEvent wire-schema endpoint
and the static SPA mount. Hermetic (TestClient + scripted factory, no network)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.server.wire import event_type_names
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


class _FakeAgent:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def arun_turn(self, user_text: str):  # type: ignore[no-untyped-def]
        from zakcode.agent.loop import TurnResult

        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        return TurnResult(assistant_messages=[Message.assistant_text("ok")])

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _client(tmp_path: Path) -> TestClient:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: _FakeAgent(session),
    )
    return TestClient(app)


def test_events_schema_endpoint(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/schema/events")
    assert resp.status_code == 200
    schema = resp.json()
    assert isinstance(schema, dict)
    # It is a JSON Schema document (has the structural keys pydantic emits).
    assert "$defs" in schema or "oneOf" in schema or "anyOf" in schema


def test_event_type_names(tmp_path: Path) -> None:
    # The contract surface the web client must handle, asserted at its source.
    assert event_type_names() == [
        "done",
        "status",
        "task_update",
        "text",
        "tool_call",
        "tool_result",
        "usage",
    ]


def test_index_served_at_root(tmp_path: Path) -> None:
    resp = _client(tmp_path).get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Zak Code" in resp.text


def test_static_mount_serves_index(tmp_path: Path) -> None:
    # The /app mount (html=True) serves index.html for the SPA.
    resp = _client(tmp_path).get("/app/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_api_routes_not_shadowed_by_mount(tmp_path: Path) -> None:
    # The SPA mount must not break the API: /health still returns JSON.
    resp = _client(tmp_path).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
