"""The web app's chats: open a saved one or a new one, and the next turn runs on it (ADR-0319).

A served workspace keeps every conversation it has had (ADR-0032), and one of them is current: the
``.current-session`` marker names it, the page shows it, and the say consumer runs each turn on it.
Nothing could change which one that was. A page opened after a restart drew the last conversation
(ADR-0286), and there was no way to start another one, or to go back to an older one.

These hold the server's half. ``GET /sessions/recent`` lists the saved chats as a chat list shows
them. ``PUT /sessions/current`` makes one of them, or a new one, the current conversation, and tells
the pages watching the one it replaces. ``POST /say`` refuses a message a page wrote for a chat that
is no longer current. ``zakcode webapp --new-session`` opens the run on a new chat.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from zakcode import Agent
from zakcode.agent.loop import _elide_skill_body
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server import app as server_app
from zakcode.server.app import create_app
from zakcode.server.event_bus import EventBusRegistry
from zakcode.server.wire import WatchMarkerRequest
from zakcode.session.say_inbox import say_path, say_pending, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

_T0 = 1_790_000_000.0


class _Agent:
    """A scripted turn that records the conversation it was handed, then answers."""

    def __init__(self, session: Session, seen: list[tuple[str, list[str]]], gate: Any) -> None:
        self.session = session
        self._seen = seen
        self._gate = gate

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._seen.append((self.session.id, [m.text for m in self.session.messages]))
        self.session.add_message(Message.user(user_text))
        if self._gate is not None:
            await self._gate.wait()
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


class _Served:
    def __init__(self, tmp_path: Path, *, gate: Any = None, new_session: bool = False) -> None:
        self.root = tmp_path
        self.store = SessionStore(base_dir=tmp_path / "sessions")
        self.seen: list[tuple[str, list[str]]] = []
        settings = Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        )
        kwargs: dict[str, Any] = {"new_session": True} if new_session else {}
        self.app = create_app(
            settings=settings,
            store=self.store,
            agent_factory=lambda session, model, prompter: _Agent(session, self.seen, gate),
            **kwargs,
        )

    def saved(
        self, *turns: tuple[str, str], task: str = "", mtime: float | None = None, sid: str = ""
    ) -> Session:
        session = Session(cwd=str(self.root), model="scripted/test", task=task)
        if sid:
            session.id = sid
        for asked, answered in turns:
            session.add_message(Message.user(asked))
            session.add_message(Message.assistant_text(answered))
        self.store.save(session)
        if mtime is not None:
            os.utime(self.store.base_dir / f"{session.id}.json", (mtime, mtime))
        return session

    def mark(self, session: Session) -> None:
        (self.root / ".current-session").write_text(session.id + "\n", encoding="utf-8")

    def marker(self) -> str:
        return (self.root / ".current-session").read_text(encoding="utf-8").strip()

    def client(self) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=self.app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    def get(self, path: str) -> httpx.Response:
        async def go() -> httpx.Response:
            async with self.client() as client:
                return await client.get(path)

        return asyncio.run(go())

    def put_current(self, body: dict[str, Any]) -> httpx.Response:
        async def go() -> httpx.Response:
            async with self.client() as client:
                return await client.put("/sessions/current", json=body)

        return asyncio.run(go())

    def say(self, body: dict[str, Any]) -> httpx.Response:
        async def go() -> httpx.Response:
            async with self.client() as client:
                return await client.post("/say", json=body)

        return asyncio.run(go())


# ── GET /sessions/recent: the chat list ─────────────────────────────────────────────────────


def test_the_chat_list_is_the_saved_chats_newest_first_each_titled_by_its_first_ask(
    tmp_path: Path,
) -> None:
    served = _Served(tmp_path)
    # Ids in neither the order of the saves nor its reverse, so a list in id order fails.
    old = served.saved(("fix the login bug", "done"), mtime=_T0, sid="c" * 32)
    mid = served.saved(("write the release notes", "done"), mtime=_T0 + 60, sid="a" * 32)
    new = served.saved(("why is CI red", "a flaky test"), mtime=_T0 + 120, sid="b" * 32)
    served.saved(mtime=_T0 + 180)  # nothing asked in it, and not current: not a chat to go back to
    served.mark(mid)

    resp = served.get("/sessions/recent")

    assert resp.status_code == 200
    chats = resp.json()
    assert [c["id"] for c in chats] == [new.id, mid.id, old.id]
    assert [c["title"] for c in chats] == [
        "why is CI red",
        "write the release notes",
        "fix the login bug",
    ]
    assert [c["current"] for c in chats] == [False, True, False]
    assert [c["message_count"] for c in chats] == [2, 2, 2]
    stamps = [datetime.fromisoformat(c["updated_at"]).timestamp() for c in chats]
    assert stamps == [_T0 + 120, _T0 + 60, _T0]


def test_the_chat_list_shows_the_current_chat_before_anything_is_asked_in_it(
    tmp_path: Path,
) -> None:
    served = _Served(tmp_path)
    earlier = served.saved(("an earlier ask", "done"), mtime=_T0)
    fresh = served.saved(mtime=_T0 + 60)
    served.mark(fresh)

    chats = served.get("/sessions/recent").json()

    assert [(c["id"], c["title"], c["current"]) for c in chats] == [
        (fresh.id, "", True),
        (earlier.id, "an earlier ask", False),
    ]


def test_a_chat_title_is_its_pinned_ask_on_one_redacted_line_cut_at_a_word(
    tmp_path: Path,
) -> None:
    served = _Served(tmp_path)
    # A compacted chat: the first message is now the summary, and the pinned task (ADR-0233)
    # is still the operator's first ask.
    served.saved(("[summary of the conversation so far]", "ok"), task="deploy the\n  staging   box")
    word = "harbor"
    value = word + str(len(word)) * 6  # built here, never a real credential
    served.saved((f"use password={value} on the box", "ok"))
    served.saved((" ".join(["window"] * 40), "ok"))

    titles = {c["title"] for c in served.get("/sessions/recent").json()}

    assert "deploy the staging box" in titles
    secret = next(t for t in titles if t.startswith("use password"))
    assert (value in secret) is False
    long = next(t for t in titles if t.startswith("window"))
    assert long.endswith("…")
    assert len(long) <= 81
    assert set(long[:-1].split()) == {"window"}  # cut between words, never inside one


@pytest.mark.parametrize("ended", [False, True], ids=["while-it-runs", "once-it-ended"])
def test_a_skill_turn_reads_as_the_command_typed_in_the_chat_list_and_the_transcript(
    tmp_path: Path, ended: bool
) -> None:
    """A skill turn's user message is the frame it runs under, and the skill's body (or, once
    the turn is over, the marker that replaced it). The chat list titled such a chat
    "<command-message>start is running</command-message>…", every /start chat alike, and a chat
    opened again drew the frame and the marker as what the user said (ADR-0335). The frame here
    is the one the agent composes, so the two cannot drift apart."""
    skill = tmp_path / ".claude" / "skills" / "start" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: start\ndescription: Start an agent.\n---\nBind the agent and report.\n",
        encoding="utf-8",
    )
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        enable_skills=True,
    )
    composed = asyncio.run(agent.compose_skill_turn("start", "demo --mode assistant")).turn_text
    assert composed is not None
    asked = (_elide_skill_body(composed) or "") if ended else composed
    assert asked.startswith("<command-message>")
    served = _Served(tmp_path)
    chat = served.saved((asked, "started"), (" <command-name>/start</command-name>", "ok"))

    [listed] = served.get("/sessions/recent").json()
    spoken = served.get(f"/sessions/{chat.id}/transcript").json()["messages"]

    assert listed["title"] == "/start demo --mode assistant"
    assert [(m["role"], m["text"]) for m in spoken] == [
        ("user", "/start demo --mode assistant"),
        ("assistant", "started"),
        # Only a frame that opens the message is a command: one inside a message is its text.
        ("user", " <command-name>/start</command-name>"),
        ("assistant", "ok"),
    ]


def test_the_chat_list_stops_at_its_limit(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    made = [served.saved((f"ask {n}", "ok"), mtime=_T0 + n) for n in range(5)]

    chats = served.get("/sessions/recent?limit=2").json()

    assert [c["id"] for c in chats] == [made[4].id, made[3].id]


# ── PUT /sessions/current: open a chat, or a new one ────────────────────────────────────────


def test_opening_a_saved_chat_runs_the_next_turn_on_its_whole_history(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    first = served.saved(("first chat ask", "first reply"))
    second = served.saved(("second chat ask", "second reply"))
    served.mark(second)

    resp = served.put_current({"session_id": first.id})

    assert resp.status_code == 200
    assert resp.json()["id"] == first.id
    assert served.marker() == first.id
    assert write_say(say_path(tmp_path), "and then?")
    assert asyncio.run(served.app.state.consume_one_say()) is True
    assert served.seen == [(first.id, ["first chat ask", "first reply"])]
    assert len(served.store.load(first.id).messages) == 4
    assert len(served.store.load(second.id).messages) == 2


def test_a_new_chat_is_opened_and_one_left_empty_is_reused(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    used = served.saved(("an ask", "a reply"))
    served.mark(used)

    opened = served.put_current({"session_id": None})

    assert opened.status_code == 200
    new_id = opened.json()["id"]
    assert new_id != used.id
    assert served.marker() == new_id
    assert served.store.load(new_id).messages == []
    again = served.put_current({})
    assert again.json()["id"] == new_id
    assert sorted(served.store.list()) == sorted([used.id, new_id])


def test_opening_an_unknown_chat_is_404_and_leaves_the_current_one(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    served.mark(current)

    resp = served.put_current({"session_id": "0123456789abcdef"})

    assert resp.status_code == 404
    assert served.marker() == current.id


def test_no_chat_opens_while_a_turn_runs(tmp_path: Path) -> None:
    async def go() -> tuple[int, str]:
        gate = asyncio.Event()
        served = _Served(tmp_path, gate=gate)
        current = served.saved(("an ask", "a reply"))
        other = served.saved(("another ask", "another reply"))
        served.mark(current)
        assert write_say(say_path(tmp_path), "a long one")
        turn = asyncio.create_task(served.app.state.consume_one_say())
        for _ in range(500):
            if served.seen:
                break
            await asyncio.sleep(0.01)
        assert served.seen, "the turn never started"
        async with served.client() as client:
            resp = await client.put("/sessions/current", json={"session_id": other.id})
        gate.set()
        await turn
        assert served.marker() == current.id
        return resp.status_code, resp.json()["detail"]

    status, detail = asyncio.run(go())

    assert status == 409
    assert "turn" in detail


@pytest.mark.parametrize("queued", ["say", "nudge"])
def test_no_chat_opens_while_a_message_is_queued_for_the_current_one(
    tmp_path: Path, queued: str
) -> None:
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    other = served.saved(("another ask", "another reply"))
    served.mark(current)
    if queued == "say":
        assert write_say(say_path(tmp_path), "typed for the chat on screen")
    else:
        (tmp_path / ".nudge").write_text("a suggestion\n", encoding="utf-8")

    resp = served.put_current({"session_id": other.id})

    assert resp.status_code == 409
    # The queued message runs in the active chat, which is not the chat the page asks from.
    assert resp.json()["detail"] == (
        "a message is queued for the active chat; let it run, then switch"
    )
    assert served.marker() == current.id
    if queued == "say":
        assert say_pending(say_path(tmp_path)) is True


def test_the_pages_watching_the_replaced_chat_are_told_to_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[EventBusRegistry] = []

    class _Registry(EventBusRegistry):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(server_app, "EventBusRegistry", _Registry)
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    other = served.saved(("another ask", "another reply"))
    served.mark(current)
    watched = made[-1].get_or_create(current.id)  # a page is watching the current chat

    assert served.put_current({"session_id": other.id}).status_code == 200

    published = [event for _, event, _ in list(watched._buffer)]
    assert any(
        isinstance(event, WatchMarkerRequest) and event.event == "session_rotated"
        for event in published
    )
    assert made[-1].get(other.id) is None  # nothing is made for a chat nobody watches yet


def test_a_chat_made_current_again_forgets_the_rotation_it_was_told_when_it_was_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A, then another chat, then A again. The rotation A's watchers were told when it was left
    is over once A is current again; kept in A's bus, it replayed to every page that opened A
    afterwards, which read it as A restarting under it and ended the turn on screen
    (ADR-0325). Only the rotation goes: the rest of A's bus is kept."""
    made: list[EventBusRegistry] = []

    class _Registry(EventBusRegistry):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(server_app, "EventBusRegistry", _Registry)
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    other = served.saved(("another ask", "another reply"))
    served.mark(current)
    watched = made[-1].get_or_create(current.id)  # a page is watching the current chat
    watched.publish(WatchMarkerRequest(event="user_message", text="an ask"))

    def kinds() -> list[str]:
        return [event.event for _, event, _ in list(watched._buffer)]

    assert served.put_current({"session_id": other.id}).status_code == 200
    assert kinds() == ["user_message", "session_rotated"]  # its watchers were told to follow
    assert served.put_current({"session_id": current.id}).status_code == 200
    assert served.marker() == current.id
    assert kinds() == ["user_message"]


# ── POST /say: a message is for the chat it was written in ─────────────────────────────────


def test_a_message_written_for_a_chat_that_is_no_longer_current_is_refused(
    tmp_path: Path,
) -> None:
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    left = served.saved(("another ask", "another reply"))
    served.mark(current)

    refused = served.say({"text": "for the chat I see", "session_id": left.id})

    assert refused.status_code == 409
    assert say_pending(say_path(tmp_path)) is False
    accepted = served.say({"text": "for the chat I see", "session_id": current.id})
    assert accepted.status_code == 200
    assert say_pending(say_path(tmp_path)) is True


def test_a_message_that_names_no_chat_goes_to_the_current_one_as_before(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    served.mark(served.saved(("an ask", "a reply")))

    assert served.say({"text": "from zakcode say"}).status_code == 200
    assert say_pending(say_path(tmp_path)) is True


# ── zakcode webapp --new-session ────────────────────────────────────────────────────────────


def test_new_session_opens_the_run_on_a_new_chat(tmp_path: Path) -> None:
    served = _Served(tmp_path, new_session=True)
    earlier = served.saved(("an earlier ask", "a reply"))
    served.mark(earlier)

    with TestClient(served.app):
        opened = served.marker()

    assert opened != earlier.id
    assert served.store.load(opened).messages == []
    assert len(served.store.load(earlier.id).messages) == 2


def test_new_session_reuses_a_current_chat_nothing_was_asked_in(tmp_path: Path) -> None:
    served = _Served(tmp_path, new_session=True)
    fresh = served.saved()
    served.mark(fresh)

    with TestClient(served.app):
        assert served.marker() == fresh.id
    assert served.store.list() == [fresh.id]


def test_new_session_keeps_the_current_chat_while_a_message_is_queued_for_it(
    tmp_path: Path,
) -> None:
    served = _Served(tmp_path, new_session=True)
    earlier = served.saved(("an earlier ask", "a reply"))
    served.mark(earlier)
    assert write_say(say_path(tmp_path), "typed before the restart")

    with TestClient(served.app):
        assert served.marker() == earlier.id


def test_new_session_opens_nothing_on_a_restart_that_keeps_this_boots_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: "boot-A")
    served = _Served(tmp_path, new_session=True)
    earlier = served.saved(("an earlier ask", "a reply"))
    served.mark(earlier)
    (tmp_path / ".run-stop-reason").write_text(
        f"{earlier.id}\nduration_cap\nboot-A", encoding="utf-8"
    )

    with TestClient(served.app) as client:
        assert served.marker() == earlier.id
        assert client.get("/sidecar/health").json()["last_run_stop_reason"] == "duration_cap"


def test_without_new_session_the_run_continues_the_current_chat(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    earlier = served.saved(("an earlier ask", "a reply"))
    served.mark(earlier)

    with TestClient(served.app):
        assert served.marker() == earlier.id
    assert served.store.list() == [earlier.id]


def test_the_chat_list_reads_a_chat_again_only_once_it_is_saved_again(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    chat = served.saved(("the first ask", "ok"), mtime=_T0)
    served.mark(chat)
    loads: list[str] = []
    real_load = served.store.load

    def counting_load(session_id: str) -> Session:
        loads.append(session_id)
        return real_load(session_id)

    served.store.load = counting_load  # type: ignore[method-assign]
    served.get("/sessions/recent")
    served.get("/sessions/recent")
    assert loads.count(chat.id) == 1
    resaved = real_load(chat.id)
    resaved.add_message(Message.user("a second ask"))
    served.store.save(resaved)
    os.utime(served.store.base_dir / f"{chat.id}.json", (time.time(), time.time()))

    chats = served.get("/sessions/recent").json()

    assert loads.count(chat.id) == 2
    assert chats[0]["message_count"] == 3
