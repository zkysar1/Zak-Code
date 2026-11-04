"""A chat in the web app's list is renamed or deleted (ADR-0358).

The server held a ``DELETE /sessions/{id}`` that unlinked a chat's file and nothing else, and no way
to name a chat. A turn running on the chat saved it again a moment later, the current chat's
marker was left naming nothing, and the chat's transcript and uploads stayed on disk. These hold
the server's half: a deleted chat goes to the workspace's trash with its transcript and its
uploads, the current chat makes way for a new one, and neither a delete nor a name is accepted
while a turn runs on that chat, since the turn saves the chat as it holds it. A name keeps the
chat's place in the list and outlives a restart and a later turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_webapp_sessions import _T0, _Served
from zakcode.server import app as server_app
from zakcode.server.event_bus import EventBusRegistry
from zakcode.server.wire import WatchMarkerRequest
from zakcode.session.say_inbox import say_path, write_say


def _delete(served: _Served, sid: str) -> httpx.Response:
    async def go() -> httpx.Response:
        async with served.client() as client:
            return await client.delete(f"/sessions/{sid}")

    return asyncio.run(go())


def _rename(served: _Served, sid: str, title: str) -> httpx.Response:
    async def go() -> httpx.Response:
        async with served.client() as client:
            return await client.put(f"/sessions/{sid}/title", json={"title": title})

    return asyncio.run(go())


def _registries(monkeypatch: pytest.MonkeyPatch) -> list[EventBusRegistry]:
    """Every bus registry the app makes, to watch a chat's bus as a page would."""
    made: list[EventBusRegistry] = []

    class _Registry(EventBusRegistry):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(server_app, "EventBusRegistry", _Registry)
    return made


# ── DELETE /sessions/{id} ────────────────────────────────────────────────────────────────────


def test_a_deleted_chat_goes_to_the_trash_with_its_transcript_and_uploads(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    current = served.saved(("the chat on screen", "ok"))
    old = served.saved(("an old ask", "an old reply"))
    served.mark(current)
    document = served.store.base_dir / f"{old.id}.json"
    saved = document.read_bytes()
    transcript = served.store.transcript_path(old.id)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_text('{"role": "user"}\n', encoding="utf-8")
    uploads = tmp_path / "uploads" / old.id
    uploads.mkdir(parents=True)
    (uploads / "notes.txt").write_text("attached", encoding="utf-8")

    resp = _delete(served, old.id)

    assert resp.status_code == 204
    assert served.get(f"/sessions/{old.id}").status_code == 404
    assert [c["id"] for c in served.get("/sessions/recent").json()] == [current.id]
    folder = tmp_path / "trash" / old.id
    assert (folder / document.name).read_bytes() == saved
    assert (folder / transcript.name).read_text(encoding="utf-8") == '{"role": "user"}\n'
    assert (folder / "uploads" / "notes.txt").read_text(encoding="utf-8") == "attached"
    assert not document.exists()
    assert not transcript.exists()
    assert not uploads.exists()
    record = json.loads((folder / "deleted.json").read_text(encoding="utf-8"))
    assert record["id"] == old.id
    assert record["moved"] == {
        document.name: str(document),
        transcript.name: str(transcript),
        "uploads": str(uploads.resolve()),
    }
    assert record["left"] == {}
    assert served.marker() == current.id  # another chat deleted: the current one stays


def test_a_chat_deleted_twice_or_never_saved_is_not_found(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    served.mark(served.saved(("the chat on screen", "ok")))
    chat = served.saved(("an ask", "a reply"))

    assert _delete(served, chat.id).status_code == 204
    assert _delete(served, chat.id).status_code == 404
    assert _delete(served, "f" * 32).status_code == 404


def test_the_current_chat_deleted_makes_way_for_a_new_one_its_pages_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker named the deleted chat until a page asked for the current one and healed it,
    and the pages showing it were never told. A new chat is current now, as New chat makes one,
    and the pages watching the deleted chat are told to follow, as for any other change."""
    made = _registries(monkeypatch)
    served = _Served(tmp_path)
    current = served.saved(("the active chat", "ok"))
    served.mark(current)
    watched = made[-1].get_or_create(current.id)  # a page is watching it

    assert _delete(served, current.id).status_code == 204

    now = served.marker()
    assert now != current.id
    assert served.store.load(now).messages == []
    assert served.get("/sessions/current").json()["id"] == now
    published = [event for _, event, _ in list(watched._buffer)]
    rotations = [
        event.reason
        for event in published
        if isinstance(event, WatchMarkerRequest) and event.event == "session_rotated"
    ]
    assert rotations == ["this chat was deleted"]  # what a cockpit pane on it says (ADR-0352)
    assert made[-1].get(current.id) is None  # its bus went with it
    assert (tmp_path / "trash" / current.id / f"{current.id}.json").is_file()


def test_the_current_chat_is_not_deleted_while_a_message_is_queued_for_it(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    current = served.saved(("an ask", "a reply"))
    served.mark(current)
    assert write_say(say_path(tmp_path), "typed for the chat on screen")

    resp = _delete(served, current.id)

    assert resp.status_code == 409
    assert resp.json()["detail"] == (
        "a message is queued for the active chat; let it run, then delete it"
    )
    assert served.marker() == current.id
    assert served.store.exists(current.id)


def test_a_chat_with_a_turn_running_on_it_is_neither_deleted_nor_renamed(tmp_path: Path) -> None:
    """The turn holds the chat in memory and saves it whole: a delete came back a moment later,
    and a name would be dropped. Another chat is free meanwhile."""

    async def go() -> tuple[_Served, Any, list[httpx.Response]]:
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
            answers = [
                await client.delete(f"/sessions/{current.id}"),
                await client.put(f"/sessions/{current.id}/title", json={"title": "Trip"}),
                await client.put(f"/sessions/{other.id}/title", json={"title": "Other"}),
                await client.delete(f"/sessions/{other.id}"),
            ]
        gate.set()
        await turn
        return served, current, answers

    served, current, answers = asyncio.run(go())

    deleted, renamed, other_renamed, other_deleted = answers
    assert deleted.status_code == 409
    assert deleted.json()["detail"] == (
        "a turn is running in this chat; stop it or let it finish, then delete it"
    )
    assert renamed.status_code == 409
    assert renamed.json()["detail"] == (
        "a turn is running in this chat; stop it or let it finish, then rename it"
    )
    assert other_renamed.status_code == 200
    assert other_deleted.status_code == 204
    assert served.store.exists(current.id)
    assert served.store.load(current.id).title == ""


def test_a_turn_another_process_runs_on_a_chat_refuses_only_that_chat(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    served.mark(served.saved(("the chat on screen", "ok")))
    busy = served.saved(("an ask", "a reply"))
    idle = served.saved(("another ask", "another reply"))
    marker = {"pid": os.getpid() + 1, "session": busy.id, "since": time.time()}
    (tmp_path / ".busy").write_text(json.dumps(marker), encoding="utf-8")

    deleted = _delete(served, busy.id)
    renamed = _rename(served, busy.id, "Trip")

    for resp, then in ((deleted, "delete it"), (renamed, "rename it")):
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "a turn is running in this chat in another session on this workspace; "
            f"let it finish, then {then}"
        )
    assert served.store.exists(busy.id)
    assert _delete(served, idle.id).status_code == 204


# ── PUT /sessions/{id}/title ─────────────────────────────────────────────────────────────────


def test_a_renamed_chat_keeps_its_name_and_its_place_in_the_list(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    older = served.saved(("fix the login bug", "done"), mtime=_T0)
    newer = served.saved(("why is CI red", "a flaky test"), mtime=_T0 + 60)
    served.mark(newer)

    resp = _rename(served, older.id, "  Login\n  bug   triage ")

    assert resp.status_code == 200
    assert resp.json()["title"] == "Login bug triage"
    assert resp.json()["current"] is False
    chats = served.get("/sessions/recent").json()
    assert [(c["id"], c["title"]) for c in chats] == [
        (newer.id, "why is CI red"),
        (older.id, "Login bug triage"),
    ]
    # A name is not a message: the chat is dated as it was and keeps its place below.
    assert datetime.fromisoformat(chats[1]["updated_at"]).timestamp() == _T0
    # A server started again reads the name from the saved chat.
    again = _Served(tmp_path).get("/sessions/recent").json()
    assert [c["title"] for c in again] == ["why is CI red", "Login bug triage"]


def test_an_empty_name_gives_the_chat_back_its_first_ask(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    chat = served.saved(("fix the login bug", "done"))
    served.mark(chat)

    assert _rename(served, chat.id, "Login").json()["title"] == "Login"
    assert _rename(served, chat.id, "   ").json()["title"] == "fix the login bug"
    assert served.store.load(chat.id).title == ""


def test_a_named_chat_keeps_its_name_after_a_turn_runs_in_it(tmp_path: Path) -> None:
    async def go() -> tuple[_Served, Any]:
        served = _Served(tmp_path)
        chat = served.saved(("the first ask", "ok"))
        served.mark(chat)
        async with served.client() as client:
            named = await client.put(f"/sessions/{chat.id}/title", json={"title": "Trip"})
        assert named.status_code == 200
        assert write_say(say_path(tmp_path), "and then")
        await served.app.state.consume_one_say()
        return served, chat

    served, chat = asyncio.run(go())

    saved = served.store.load(chat.id)
    assert [m.text for m in saved.messages][-2:] == ["and then", "ok"]  # the turn was saved
    assert saved.title == "Trip"
    assert [c["title"] for c in served.get("/sessions/recent").json()] == ["Trip"]


def test_a_name_is_at_most_200_characters_and_the_list_cuts_it_at_a_word(tmp_path: Path) -> None:
    served = _Served(tmp_path)
    chat = served.saved(("an ask", "a reply"))
    served.mark(chat)

    assert _rename(served, chat.id, "x" * 201).status_code == 422
    title = _rename(served, chat.id, " ".join(["harbor"] * 28)).json()["title"]

    assert title.endswith("…")
    assert len(title) <= 81
    assert set(title[:-1].split()) == {"harbor"}


def test_an_unknown_chat_cannot_be_named(tmp_path: Path) -> None:
    served = _Served(tmp_path)

    assert _rename(served, "f" * 32, "Trip").status_code == 404
