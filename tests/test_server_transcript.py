"""``GET /sessions/{id}/transcript`` (ADR-0041) — the conversation as a reader sees it.

A session document persists every turn (ADR-0032), but the watch bus's retained buffer
starts empty on every daemon start, so a viewer joining a resumed session saw nothing.
The transcript is the persisted conversation projected for reading: user + assistant
TEXT only, tool/thinking/system frames omitted, secrets redacted like the watch stream,
``current`` resolved through the ``.current-session`` marker, ``limit`` keeping the tail.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from zakcode.config import Settings
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore


def _build(tmp_path: Path) -> tuple[Any, SessionStore]:
    settings = Settings(default_model="scripted/test", workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: None,  # no turn runs here
    )
    return app, store


def _stored(store: SessionStore, tmp_path: Path) -> Session:
    session = Session(cwd=str(tmp_path), model="scripted/test")
    session.messages = [
        Message.user("remember: the pearl holds"),
        Message(
            role="assistant",
            blocks=[
                TextBlock(text="Let me note that."),
                ToolUseBlock(id="t1", name="bash", input={"command": "echo noted"}),
            ],
        ),
        Message.tool_results([ToolResultBlock(tool_use_id="t1", output="noted")]),
        Message.assistant_text("The pearl holds — remembered."),
        Message.system("a system frame is never spoken"),
        Message(role="assistant", blocks=[ToolUseBlock(id="t2", name="bash", input={})]),
    ]
    store.save(session)
    return session


def _get(app: Any, path: str) -> httpx.Response:
    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(go())


def test_transcript_is_the_spoken_turns_only(tmp_path: Path) -> None:
    app, store = _build(tmp_path)
    session = _stored(store, tmp_path)
    resp = _get(app, f"/sessions/{session.id}/transcript")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session.id
    assert body["message_count"] == 6  # the full stored length, not the spoken count
    assert [(m["role"], m["text"]) for m in body["messages"]] == [
        ("user", "remember: the pearl holds"),
        ("assistant", "Let me note that."),
        ("assistant", "The pearl holds — remembered."),
    ]


def test_limit_keeps_the_tail(tmp_path: Path) -> None:
    app, store = _build(tmp_path)
    session = _stored(store, tmp_path)
    tail = _get(app, f"/sessions/{session.id}/transcript?limit=1").json()["messages"]
    assert [m["text"] for m in tail] == ["The pearl holds — remembered."]
    assert _get(app, f"/sessions/{session.id}/transcript?limit=0").json()["messages"] == []


def test_current_alias_resolves_the_marker_and_404s_without_one(tmp_path: Path) -> None:
    app, store = _build(tmp_path)
    assert _get(app, "/sessions/current/transcript").status_code == 404
    session = _stored(store, tmp_path)
    (tmp_path / ".current-session").write_text(session.id + "\n", encoding="utf-8")
    body = _get(app, "/sessions/current/transcript").json()
    assert body["session_id"] == session.id
    assert body["messages"][0]["text"] == "remember: the pearl holds"


def test_unknown_session_is_404(tmp_path: Path) -> None:
    app, _ = _build(tmp_path)
    assert _get(app, "/sessions/nope/transcript").status_code == 404


def test_text_is_redacted_like_the_watch_stream(tmp_path: Path, monkeypatch: Any) -> None:
    """A provider key that leaked into a spoken turn never leaves the box in clear."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-1234567890abcdefghijklmnop")
    app, store = _build(tmp_path)
    session = Session(cwd=str(tmp_path), model="scripted/test")
    session.messages = [Message.assistant_text("my key is sk-live-1234567890abcdefghijklmnop ok")]
    store.save(session)
    text = _get(app, f"/sessions/{session.id}/transcript").json()["messages"][0]["text"]
    assert "sk-live-1234567890abcdefghijklmnop" not in text
    assert text.endswith(" ok")
