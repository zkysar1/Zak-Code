"""Tests for the read-only watch surface (/watch, /workspace/summary, /sidecar/health).

Split by what the in-memory test transport can do. ``/chat/stream`` is a FINITE SSE stream, so
it is driven through ``TestClient`` and the projected frames it tees onto the watch bus are read
back via ``app.state.broadcaster`` — this asserts the security filtering over the real HTTP path
(tool args/output never buffered, secrets redacted, usage dropped). The ``/watch`` endpoint's
stream body is the module-level ``_watch_event_stream`` generator, unit-tested directly (history
replay + cursor + live) because httpx's ASGITransport buffers the whole body and would deadlock
on an endless SSE stream.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

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
from zakcode.server.app import (
    _resolve_watch_session,
    _watch_event_stream,
    create_app,
)
from zakcode.server.broadcast import SessionBroadcaster
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

SECRET = "gsk_watchsurfacekey0123456789abcdef"
TOOL_ARG_PATH = "n-secret-file.txt"
TOOL_OUTPUT = "wrote n-secret-file.txt with contents FOO"


class _WatchAgent:
    """Minimal AgentLike whose turn carries a secret + tool arg/output to be filtered out."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        events: list[AgentEvent] = [
            AgentTextDelta(text=f"Researching. My key is {SECRET} — do not show it."),
            AgentToolCall(id="c1", name="write_file", arguments={"path": TOOL_ARG_PATH}),
            AgentToolResult(tool_use_id="c1", output=TOOL_OUTPUT, is_error=False),
            AgentUsage(usage=Usage(total_tokens=10, cost_usd=0.5)),
            AgentDone(stop_reason="completed", iterations=2, usage=Usage(total_tokens=10)),
        ]
        for event in events:
            yield event


def _factory(session: Session, model: str | None, prompter: object = None) -> _WatchAgent:  # noqa: ARG001
    return _WatchAgent(session)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    return TestClient(create_app(settings=settings, store=store, agent_factory=_factory))


def test_chat_stream_tees_only_safe_frames_to_watch_bus(client: TestClient) -> None:
    session_id = client.post("/sessions").json()["id"]
    with client.stream(
        "POST", "/chat/stream", json={"message": "go", "session_id": session_id}
    ) as resp:
        assert resp.status_code == 200
        for _line in resp.iter_lines():
            pass  # consume so the turn runs to completion and tees every frame

    frames = [record["frame"] for record in client.app.state.broadcaster.history(session_id, 0)]
    events = [f["event"] for f in frames]
    assert "text" in events
    assert "tool_summary" in events
    assert "done" in events
    assert "usage" not in events  # cost/token data dropped

    blob = json.dumps(frames)
    assert TOOL_ARG_PATH not in blob  # tool argument never buffered
    assert "wrote n-secret-file" not in blob  # tool output never buffered
    assert SECRET not in blob  # planted secret redacted from text

    tool_frames = [f for f in frames if f["event"] == "tool_summary"]
    assert any(f["name"] == "write_file" and f["status"] == "running" for f in tool_frames)
    assert any(f["name"] == "write_file" and f["status"] == "completed" for f in tool_frames)
    assert all("arguments" not in f and "output" not in f for f in tool_frames)


async def test_watch_event_stream_replays_then_streams_live() -> None:
    bus = SessionBroadcaster()
    await bus.publish("s1", AgentTextDelta(text="a"))  # buffered before any subscriber
    gen = _watch_event_stream(bus, "s1", 0)
    try:
        first = await asyncio.wait_for(anext(gen), timeout=5)
        assert first["id"] == "1"
        assert json.loads(first["data"]) == {"event": "text", "text": "a"}
        # a frame published after the subscriber connected arrives live
        await bus.publish("s1", AgentTextDelta(text="b"))
        second = await asyncio.wait_for(anext(gen), timeout=5)
        assert json.loads(second["data"])["text"] == "b"
    finally:
        await gen.aclose()


async def test_watch_event_stream_since_cursor_skips_replayed() -> None:
    bus = SessionBroadcaster()
    for text in ("a", "b", "c"):
        await bus.publish("s1", AgentTextDelta(text=text))
    gen = _watch_event_stream(bus, "s1", 1)  # since seq 1 → skip "a"
    try:
        f1 = await asyncio.wait_for(anext(gen), timeout=5)
        f2 = await asyncio.wait_for(anext(gen), timeout=5)
        assert [json.loads(f1["data"])["text"], json.loads(f2["data"])["text"]] == ["b", "c"]
    finally:
        await gen.aclose()


def test_watch_rejects_unsafe_session_id(client: TestClient) -> None:
    assert client.get("/watch/bad:id").status_code == 404  # ':' can't appear in a minted id


def test_sidecar_health_reports_current_session(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / ".current-session").write_text("abc123\n", encoding="utf-8")
    body = client.get("/sidecar/health").json()
    assert body["status"] == "ok"
    assert body["active_session_id"] == "abc123"


def test_sidecar_health_no_session_file(client: TestClient) -> None:
    assert client.get("/sidecar/health").json()["active_session_id"] == ""


def test_workspace_summary_truncates_and_counts(client: TestClient, tmp_path: Path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    (research / "journal.md").write_text(
        "# Journal\n## Finding one\n## Finding two\n" + "x" * 6000, encoding="utf-8"
    )
    body = client.get("/workspace/summary").json()
    assert len(body["journal"]) == 5000  # first 5000 chars
    assert body["finding_count"] == 2  # two level-2 headings


def test_watch_current_returns_404_when_no_active_session(client: TestClient) -> None:
    # No .current-session in the fresh workspace → the "current" alias resolves to
    # nothing → 404 (finite, so TestClient does not hang on the SSE body).
    assert client.get("/watch/current").status_code == 404


def test_resolve_watch_session_maps_current_to_the_active_session(tmp_path: Path) -> None:
    (tmp_path / ".current-session").write_text("sess-live-1\n", encoding="utf-8")
    assert _resolve_watch_session("current", tmp_path) == "sess-live-1"


def test_resolve_watch_session_current_without_active_raises_404(tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as exc:
        _resolve_watch_session("current", tmp_path)
    assert exc.value.status_code == 404


def test_resolve_watch_session_passes_through_an_explicit_id(tmp_path: Path) -> None:
    assert _resolve_watch_session("sess-abc", tmp_path) == "sess-abc"


def test_resolve_watch_session_rejects_an_unsafe_id(tmp_path: Path) -> None:
    with pytest.raises(HTTPException):
        _resolve_watch_session("bad:id", tmp_path)


def test_nudge_queues_a_suggestion_file(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/nudge", json={"text": "explore volcanoes"})
    assert r.status_code == 200
    assert r.json()["queued"] is True
    assert (tmp_path / ".nudge").read_text(encoding="utf-8").strip() == "explore volcanoes"


def test_nudge_rejects_a_second_while_one_is_pending(client: TestClient) -> None:
    assert client.post("/nudge", json={"text": "first"}).status_code == 200
    assert client.post("/nudge", json={"text": "second"}).status_code == 429  # single-slot queue


def test_nudge_requires_non_empty_text(client: TestClient) -> None:
    assert client.post("/nudge", json={"text": "   "}).status_code == 400


def test_nudge_caps_length(client: TestClient, tmp_path: Path) -> None:
    client.post("/nudge", json={"text": "x" * 5000})
    assert len((tmp_path / ".nudge").read_text(encoding="utf-8").strip()) == 500


# ── knowledge base (PEARL §10.4) ─────────────────────────────────────────────

_BUNDLE = {
    "counts": {"tree": 2, "hypotheses": 1, "guardrails": 1, "lessons": 1},
    "tree": [
        {"key": "coral-reefs", "title": "Coral reefs", "summary": "Reefs host a quarter of marine life.",
         "parent": "marine-biology", "children": ["bleaching"]},
        {"key": "bleaching", "title": "Bleaching", "summary": "Warming expels the algae.",
         "parent": "coral-reefs", "children": []},
    ],
    "hypotheses": [
        {"statement": "Warmer water bleaches reefs faster.", "horizon": "short",
         "status": "resolved", "outcome": "Confirmed."},
    ],
    "guardrails": [{"rule": "Verify every claim against two sources."}],
    "lessons": [{"title": "Cross-check", "lesson": "One source was wrong once."}],
}


def _write_bundle(root: Path, bundle: dict = _BUNDLE) -> None:
    (root / ".knowledge-bundle.json").write_text(json.dumps(bundle), encoding="utf-8")


def test_knowledge_tree_returns_map_without_bodies(client: TestClient, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    r = client.get("/knowledge/tree")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    node = next(n for n in body["nodes"] if n["key"] == "coral-reefs")
    assert node["title"] == "Coral reefs"
    assert node["children"] == ["bleaching"]
    assert "summary" not in node  # the index is a map, not the bodies


def test_knowledge_node_returns_one_projected_node(client: TestClient, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    r = client.get("/knowledge/node/coral-reefs")
    assert r.status_code == 200
    assert r.json()["summary"].startswith("Reefs host")


def test_knowledge_node_404_when_absent(client: TestClient, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    assert client.get("/knowledge/node/no-such-node").status_code == 404


def test_knowledge_hypotheses_and_guardrails(client: TestClient, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    h = client.get("/knowledge/hypotheses").json()
    assert h["count"] == 1 and h["hypotheses"][0]["horizon"] == "short"
    g = client.get("/knowledge/guardrails").json()
    assert g["count"] == 1 and g["guardrails"][0]["rule"].startswith("Verify")


def test_knowledge_export_returns_whole_bundle(client: TestClient, tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    r = client.get("/knowledge/export")
    assert r.status_code == 200
    assert r.json()["counts"]["tree"] == 2


def test_knowledge_endpoints_fail_open_when_no_bundle(client: TestClient) -> None:
    # No .knowledge-bundle.json written — every route degrades to empty, never 500s.
    assert client.get("/knowledge/tree").json() == {"nodes": [], "count": 0}
    assert client.get("/knowledge/hypotheses").json() == {"hypotheses": [], "count": 0}
    assert client.get("/knowledge/export").json()["tree"] == []


def test_knowledge_tolerates_corrupt_bundle(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / ".knowledge-bundle.json").write_text("{not json", encoding="utf-8")
    assert client.get("/knowledge/tree").json()["count"] == 0  # malformed → empty, no 500
