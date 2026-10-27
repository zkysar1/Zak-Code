"""``GET /sessions/{id}/transcript`` (ADR-0041) — the conversation as a reader sees it.

A session document persists every turn (ADR-0032), but the watch bus's retained buffer
starts empty on every daemon start, so a viewer joining a resumed session saw nothing.
The transcript is the persisted conversation projected for reading: user + assistant
TEXT only, tool/thinking/system frames omitted, secrets redacted like the watch stream,
``current`` resolved through the ``.current-session`` marker, ``limit`` keeping the tail.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

import httpx

from zakcode import messages as messages_mod
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.server import app as server_app
from zakcode.server import event_bus
from zakcode.server.app import create_app
from zakcode.server.wire import WatchMarkerRequest
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


def _build(tmp_path: Path) -> tuple[Any, SessionStore]:
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
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
        Message.user("remember: the data holds"),
        Message(
            role="assistant",
            blocks=[
                TextBlock(text="Let me note that."),
                ToolUseBlock(id="t1", name="bash", input={"command": "echo noted"}),
            ],
        ),
        Message.tool_results([ToolResultBlock(tool_use_id="t1", output="noted")]),
        Message.assistant_text("The data holds — remembered."),
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
        ("user", "remember: the data holds"),
        ("assistant", "Let me note that."),
        ("assistant", "The data holds — remembered."),
    ]


def test_limit_keeps_the_tail(tmp_path: Path) -> None:
    app, store = _build(tmp_path)
    session = _stored(store, tmp_path)
    tail = _get(app, f"/sessions/{session.id}/transcript?limit=1").json()["messages"]
    assert [m["text"] for m in tail] == ["The data holds — remembered."]
    assert _get(app, f"/sessions/{session.id}/transcript?limit=0").json()["messages"] == []


def test_current_alias_resolves_the_marker_and_404s_without_one(tmp_path: Path) -> None:
    app, store = _build(tmp_path)
    assert _get(app, "/sessions/current/transcript").status_code == 404
    session = _stored(store, tmp_path)
    (tmp_path / ".current-session").write_text(session.id + "\n", encoding="utf-8")
    body = _get(app, "/sessions/current/transcript").json()
    assert body["session_id"] == session.id
    assert body["messages"][0]["text"] == "remember: the data holds"


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


# An offset from UTC that no machine's own zone has, so a clock read that leaves the zone out
# comes back a different instant on every machine and never the right one by coincidence.
_LOCAL_OFFSET = timedelta(hours=5, minutes=17)


class _FrozenClock:
    """Stands in for ``datetime`` in a module that stamps with ``datetime.now(...)``.

    ``now(tz)`` is the one frozen instant read in ``tz``, as the real call reads it. With no ``tz``
    it is the naive wall-clock reading of a machine ``_LOCAL_OFFSET`` ahead of UTC, so a stamp cut
    without the zone is a different number, not the same one. Anything else (``fromtimestamp``,
    ``fromisoformat``) is the real class's, so a stamp cut from another source still runs and comes
    back wrong, rather than failing on a missing attribute.
    """

    def __init__(self, instant: datetime) -> None:
        self._instant = instant

    def now(self, tz: tzinfo | None = None) -> datetime:
        if tz is not None:
            return self._instant.astimezone(tz)
        return (self._instant + _LOCAL_OFFSET).replace(tzinfo=None)

    def __getattr__(self, name: str) -> Any:
        return getattr(datetime, name)


def test_a_message_and_a_bus_event_made_in_one_instant_come_back_as_one_number(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A message and a bus event made in one instant come back from the route as ONE number.

    The reopened page draws a saved message itself only when it is older than ``bus_from``, so a
    message made in the very instant of the bus's first event has to read as the bus's: ``at``
    must not fall below ``bus_from`` by even a microsecond (ADR-0288). Both stamps are taken under
    one frozen clock at 2026-10-14T00:44:03.985429Z, which is 1791938643.985429, the stamp of the
    message in the pair that drew a turn twice on Windows CI. The message is a real ``Message``,
    stamped by the messages module, saved, and read back by the real route, which turns its ISO
    string into epoch seconds. The event is a real publish on the bus the route reports
    ``bus_from`` from: the user line that begins the turn, where the cut falls (ADR-0325). The
    stand-in clock honours ``tz``, so a stamp cut from another clock source, or one that leaves the
    zone out, lands on a different number and the test fails on the value.
    """
    instant = datetime(2026, 10, 14, 0, 44, 3, 985429, tzinfo=UTC)
    assert instant.timestamp() == 1791938643.985429
    clock = _FrozenClock(instant)
    monkeypatch.setattr(messages_mod, "datetime", clock)
    # raising=False: a bus that does not read `datetime` at all must fail on the numbers below,
    # not on a name missing from this module.
    monkeypatch.setattr(event_bus, "datetime", clock, raising=False)
    registry = event_bus.EventBusRegistry()
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: registry)
    app, store = _build(tmp_path)

    session = Session(cwd=str(tmp_path), model="scripted/test")
    session.messages = [Message.user("the newest turn")]
    store.save(session)
    registry.get_or_create(session.id).publish(
        WatchMarkerRequest(event="user_message", text="the newest turn")
    )

    body = _get(app, f"/sessions/{session.id}/transcript").json()
    at = body["messages"][0]["at"]
    assert body["bus_from"] == instant.timestamp()  # the bus read the frozen clock
    assert at == body["bus_from"]  # the same reading is the same number
    assert at >= body["bus_from"]  # so the page counts the message as the bus's


class _SteppingClock:
    """Stands in for ``datetime`` in the bus: each reading is one second past the last, so events
    published one after another carry stamps that differ. The real clock need not: on Windows it
    moves in 15.6 ms steps, and four events published in a row read one stamp between them."""

    def __init__(self, start: datetime) -> None:
        self._next = start

    def now(self, tz: tzinfo | None = None) -> datetime:
        reading, self._next = self._next, self._next + timedelta(seconds=1)
        return reading.astimezone(tz) if tz is not None else reading


def test_the_cut_falls_where_the_oldest_turn_the_bus_holds_whole_begins(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The bus keeps only its last N events, so its oldest one is often in the middle of a turn.
    Cut there, a reply that began before it was hidden from the saved conversation and drawn from
    the bus without its beginning. The cut falls where the oldest turn the bus holds WHOLE begins,
    its user line: the page draws everything older from here and resumes the watch after
    ``bus_since``, so the replay starts at that line (ADR-0325)."""
    # Stamps one second apart on every platform, so the cut's stamp can be told from the oldest.
    monkeypatch.setattr(event_bus, "datetime", _SteppingClock(datetime(2026, 10, 27, tzinfo=UTC)))
    registry = event_bus.EventBusRegistry()
    monkeypatch.setattr(server_app, "EventBusRegistry", lambda: registry)
    app, store = _build(tmp_path)
    session = Session(cwd=str(tmp_path), model="scripted/test")
    session.messages = [Message.user("first"), Message.assistant_text("the first reply")]
    store.save(session)
    bus = registry.get_or_create(session.id)
    bus.publish(AgentTextDelta(text="reply"))  # the tail of the first turn: its start is gone
    bus.publish(AgentDone(stop_reason="completed", iterations=1, usage=Usage()))
    start = bus.publish(WatchMarkerRequest(event="user_message", text="second"))
    bus.publish(AgentTextDelta(text="the second reply"))
    stamps = {cursor: at for cursor, _, at in list(bus._buffer)}

    body = _get(app, f"/sessions/{session.id}/transcript").json()
    assert body["bus_since"] == start - 1
    assert body["bus_from"] == stamps[start]
    assert body["bus_from"] > stamps[1]  # not the oldest event the bus holds
    assert body["running"] is False


class _LongTurn:
    """A turn that streams more than a ring of four holds, then waits to be let go. It saves its
    reply after the stream, as the agent loop does."""

    def __init__(self, session: Session, gate: asyncio.Event) -> None:
        self.session = session
        self._gate = gate

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        parts = [f"part{i} " for i in range(6)]
        for part in parts:
            yield AgentTextDelta(text=part)
        await self._gate.wait()
        self.session.add_message(Message.assistant_text("".join(parts)))
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def test_a_turn_longer_than_the_bus_is_replayed_while_it_runs_and_drawn_from_disk_once_it_ends(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A turn whose user line the bus has already evicted. While it runs, the call in progress
    is in the bus alone, so the replay draws the bus's tail as before, and ``running`` says a
    turn is under way that the replay may not show starting. Once it has ended the whole turn is
    saved, so every saved message is drawn from here and the replay starts after the bus
    (ADR-0325)."""
    monkeypatch.setattr(
        server_app, "EventBusRegistry", lambda: event_bus.EventBusRegistry(maxlen=4)
    )

    async def go() -> tuple[dict[str, Any] | None, dict[str, Any]]:
        gate = asyncio.Event()
        settings = Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        )
        app = create_app(
            settings=settings,
            store=SessionStore(base_dir=tmp_path / "sessions"),
            agent_factory=lambda session, model, prompter: _LongTurn(session, gate),
        )
        assert write_say(say_path(tmp_path), "a long one")
        turn = asyncio.create_task(app.state.consume_one_say())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            running = None
            for _ in range(500):
                resp = await client.get("/sessions/current/transcript")
                if resp.status_code == 200 and resp.json().get("running"):
                    running = resp.json()
                    break
                await asyncio.sleep(0.01)
            gate.set()
            await turn
            ended = (await client.get("/sessions/current/transcript")).json()
        return running, ended

    running, ended = asyncio.run(go())

    assert running is not None, "the transcript never said a turn was running"
    assert running["bus_since"] is None  # the whole tail replays: it holds the call in progress
    assert running["bus_from"] is not None
    assert ended["running"] is False
    assert ended["bus_from"] is None  # every saved message is drawn from the transcript
    assert ended["bus_since"] == 8  # user line + six parts + done: the replay starts after them
    assert [m["text"] for m in ended["messages"]] == [
        "a long one",
        "part0 part1 part2 part3 part4 part5 ",
    ]
