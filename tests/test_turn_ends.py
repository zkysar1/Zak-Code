"""A turn's end is kept with the session, so a page reopened after a restart draws it (ADR-0361).

The web page draws each turn's end as a row of its own: "provider error — …", "stopped early —
max iterations", "interrupted". The row came only from the watch bus, which is in memory, so a
page reopened after the server restarted drew the turn's words and not how it ended. The session
now keeps every end but a plain "done", placed by how many messages came before it; the loop and
the say consumer keep them where they publish them, a compaction moves them with the messages, and
the transcript route hands them to the page. The page's drawing of them is tested with the rest of
its history in test_webclient_reload_history.py.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_loop_retry import FlakyProvider, FlakyStreamProvider, _make_loop
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import AuthError, ProviderStreamEvent, StreamDone
from zakcode.server import app as server_app
from zakcode.server.app import create_app
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import TURN_ENDS_KEPT, Session, SessionStore, TurnEnd


def _session(messages: int = 0) -> Session:
    session = Session(cwd="/work", model="test/model")
    for i in range(messages):
        session.add_message(Message.user(f"m{i}"))
    return session


# ── what the session keeps ────────────────────────────────────────────────────


def test_a_turn_that_simply_finished_keeps_no_end_and_every_other_end_is_kept() -> None:
    session = _session(2)
    session.keep_turn_end("completed")
    assert session.turn_ends == []
    session.keep_turn_end("completed", degraded=True)  # "done — recovered"
    session.keep_turn_end("completed", open_steps=2)  # "done — 2 plan step(s) left open"
    session.keep_turn_end("provider_error", error="HTTP 503\n" + "x" * 5000, degraded=True)
    kept = session.turn_ends
    assert [(e.stop_reason, e.after) for e in kept] == [
        ("completed", 2),
        ("completed", 2),
        ("provider_error", 2),
    ]
    assert kept[2].error.startswith("HTTP 503\n") and len(kept[2].error) == 2000


def test_a_session_keeps_only_its_last_turn_ends_and_saves_them(tmp_path: Path) -> None:
    session = _session(1)
    for n in range(TURN_ENDS_KEPT + 5):
        session.keep_turn_end("max_iterations", error=str(n))
    assert len(session.turn_ends) == TURN_ENDS_KEPT
    assert (session.turn_ends[0].error, session.turn_ends[-1].error) == (
        "5",
        str(TURN_ENDS_KEPT + 4),
    )
    store = SessionStore(base_dir=tmp_path)
    store.save(session)
    assert store.load(session.id).turn_ends == session.turn_ends


def test_a_document_an_older_build_saved_loads_with_no_turn_ends(tmp_path: Path) -> None:
    store = SessionStore(base_dir=tmp_path)
    session = _session(1)
    session.keep_turn_end("stuck")
    store.save(session)
    path = tmp_path / f"{session.id}.json"
    older = json.loads(path.read_text(encoding="utf-8"))
    del older["turn_ends"]
    path.write_text(json.dumps(older), encoding="utf-8")
    assert store.load(session.id).turn_ends == []


def _ends(*after: int) -> list[TurnEnd]:
    return [TurnEnd(after=a, stop_reason="stuck", error=str(a)) for a in after]


def test_a_summarizing_compaction_moves_each_end_with_the_messages_it_followed() -> None:
    """Ten messages, the first five summarized: the kept five follow the summary. The ends after
    the third and the fifth message belonged to turns now told only in the summary; those after
    the sixth and the tenth move four places up."""
    session = _session(10)
    session.turn_ends = _ends(3, 5, 6, 10)
    session.messages[:] = [Message.system("summary"), *session.messages[5:]]
    session.shift_turn_ends(10, summarized=True)
    assert [(e.after, e.error) for e in session.turn_ends] == [(2, "6"), (6, "10")]


def test_an_eliding_compaction_leaves_every_end_where_it_was() -> None:
    session = _session(4)
    session.turn_ends = _ends(0, 2, 4)  # 0: a turn that failed before its line was saved
    session.shift_turn_ends(4, summarized=False)
    assert [e.after for e in session.turn_ends] == [0, 2, 4]


# ── where the loop keeps them ─────────────────────────────────────────────────


@pytest.fixture()
def fast_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _instant(delay: float) -> None:
        return None

    monkeypatch.setattr("zakcode.agent.loop.asyncio.sleep", _instant)


def test_a_turn_the_provider_failed_keeps_its_end_after_the_users_line(fast_sleep: None) -> None:
    loop = _make_loop(FlakyProvider([AuthError("bad key")]))
    result = asyncio.run(loop.arun_turn("hi"))
    assert result.stop_reason == "provider_error"
    [end] = loop.session.turn_ends
    assert (end.after, end.stop_reason, end.degraded) == (1, "provider_error", True)
    assert end.error == result.error and "bad key" in end.error
    again = asyncio.run(loop.arun_turn("again"))  # a turn that simply finishes adds none
    assert again.stop_reason == "completed" and len(loop.session.turn_ends) == 1


def test_a_streamed_turn_the_provider_failed_keeps_the_end_its_done_frame_carried(
    fast_sleep: None,
) -> None:
    loop = _make_loop(FlakyStreamProvider([AuthError("bad key")]))

    async def run() -> list[AgentEvent]:
        return [event async for event in loop.astream_turn("hi")]

    done = asyncio.run(run())[-1]
    assert isinstance(done, AgentDone) and done.stop_reason == "provider_error"
    [end] = loop.session.turn_ends
    assert (end.after, end.stop_reason, end.error, end.degraded, end.open_steps) == (
        1,
        done.stop_reason,
        done.error,
        done.degraded,
        done.open_steps,
    )


class _StallingStream(FlakyStreamProvider):
    """Streams nothing until cancelled, as a call a person stops while it waits."""

    async def astream(self, *args: Any, **kw: Any) -> AsyncIterator[ProviderStreamEvent]:
        await asyncio.Event().wait()
        yield StreamDone(finish_reason="stop")  # pragma: no cover - never reached


def test_a_turn_stopped_mid_way_keeps_an_interrupted_end() -> None:
    loop = _make_loop(_StallingStream([]))

    async def run() -> None:
        async def drain() -> None:
            async for _ in loop.astream_turn("hi"):
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert [(e.after, e.stop_reason) for e in loop.session.turn_ends] == [(1, "interrupted")]


def test_a_compaction_in_the_loop_moves_the_ends_kept_before_it() -> None:
    # The end after m2 belongs to a turn the summary now tells, so it goes; the one after m5 moves.
    loop = _make_loop(FlakyProvider([]))
    loop.session.messages = [Message.user(f"m{i}") for i in range(6)]
    loop.session.turn_ends = _ends(3, 6)
    loop._adopt_compacted([Message.system("summary"), *loop.session.messages[3:]], summary="s")
    assert [e.after for e in loop.session.turn_ends] == [4]


# ── what the transcript route hands the page ──────────────────────────────────


def _get(app: Any, path: str) -> httpx.Response:
    async def go() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(path)

    return asyncio.run(go())


def _app(tmp_path: Path, agent: Any = None) -> tuple[Any, SessionStore]:
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings,
        store=store,
        agent_factory=lambda session, model, prompter: agent(session) if agent else None,
    )
    return app, store


def test_the_transcript_places_each_end_among_the_spoken_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool result between two spoken lines is not drawn, so the end after it goes after the
    spoken line before it; the error is redacted as the messages are."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-1234567890abcdefghijklmnop")
    app, store = _app(tmp_path)
    session = _session(0)
    session.add_message(Message.user("one"))
    session.add_message(Message.tool_results([ToolResultBlock(tool_use_id="t", output="x")]))
    session.keep_turn_end("provider_error", error="key sk-live-1234567890abcdefghijklmnop refused")
    session.add_message(Message.user("two"))
    session.add_message(Message.assistant_text("done two"))
    session.keep_turn_end("max_iterations")
    store.save(session)
    body = _get(app, f"/sessions/{session.id}/transcript").json()
    assert [m["text"] for m in body["messages"]] == ["one", "two", "done two"]
    assert [(e["after"], e["stop_reason"]) for e in body["ends"]] == [
        (1, "provider_error"),
        (3, "max_iterations"),
    ]
    error = body["ends"][0]["error"]
    assert "sk-live" not in error and error.endswith(" refused")
    assert isinstance(body["ends"][0]["at"], float)
    tail = _get(app, f"/sessions/{session.id}/transcript?limit=2").json()
    assert [m["text"] for m in tail["messages"]] == ["two", "done two"]
    assert [(e["after"], e["stop_reason"]) for e in tail["ends"]] == [(2, "max_iterations")]
    assert _get(app, f"/sessions/{session.id}/transcript?limit=0").json()["ends"] == []


def test_an_end_placed_past_the_messages_is_not_handed_over(tmp_path: Path) -> None:
    app, store = _app(tmp_path)
    session = _session(1)
    session.turn_ends = _ends(5)  # names messages the session does not have
    store.save(session)
    assert _get(app, f"/sessions/{session.id}/transcript").json()["ends"] == []


# ── where the say consumer keeps one ──────────────────────────────────────────


class _RaisingAgent:
    """A turn that raises past the agent loop once the user's line is saved."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        raise RuntimeError("the pod went away")
        yield  # pragma: no cover - makes this a generator


def test_a_say_turn_that_raised_keeps_the_end_the_consumer_published(tmp_path: Path) -> None:
    app, store = _app(tmp_path, _RaisingAgent)
    assert write_say(say_path(tmp_path), "hello?")
    assert asyncio.run(app.state.consume_one_say()) is True
    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    [end] = store.load(marker).turn_ends
    assert (end.after, end.stop_reason, end.degraded) == (1, "provider_error", True)
    # The words of the done frame the consumer published, which the page drew (nothing to redact).
    published = server_app._turn_failure_error(RuntimeError("the pod went away"), lambda s: s)
    assert end.error == published and "the pod went away" in end.error
