"""The web page's chats: New chat, an old chat opened from the list, a page that follows (ADR-0319).

The page joined the workspace's current conversation and stayed on it: one opened after a restart
drew the last conversation (ADR-0286), with no way to start another or go back to an older one. It
now has a chat list beside the column, New chat above it, and each saved chat in it, newest first.
Opening one draws that chat alone, watches its own bus, and sends the next message to it, so the
turn runs on its whole history. A page also follows when the current chat changes under it, and a
message it could not send to the chat it shows stays in the composer.

These run the page's OWN script under node against a REAL loopback server, as
test_webclient_bus_epoch.py does, and drive it over stdin: CLICK presses a button by id, CHAT opens
the n-th chat in the list, TYPE and SEND write and send a message, OFFLINE and ONLINE take the
page's watch connection down and up, HOLD keeps the page's next GET of a path waiting until
RELEASE sends it, and SNAP reports what is drawn.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.test_server_consumer import _LiveServer
from tests.test_webclient_bus_epoch import _OpenPage, _spoken, _watches
from tests.test_webclient_reload_history import _app, _converse
from tests.test_webclient_turn_clock import _NODE
from zakcode.messages import Message
from zakcode.server import app as server_app
from zakcode.server.event_bus import EventBusRegistry
from zakcode.session.say_inbox import say_path
from zakcode.session.store import Session, SessionStore

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

_TAIL = """
const __BASE = process.argv[2];
const __realFetch = globalThis.fetch;
const __trace = [];
const __open = new Set();
let __offline = false;
let __hold = null;
let __held = null;
globalThis.fetch = (url, opts) => {
  const full = new URL(url, __BASE);
  __trace.push(full.pathname + full.search);
  if (full.pathname === __hold && !(opts && opts.method)) {
    __hold = null;  // HOLD catches one GET of its path, which RELEASE sends
    return new Promise((resolve) => { __held = () => resolve(__send(full, opts)); });
  }
  return __send(full, opts);
};
const __send = (full, opts) => {
  if (!full.pathname.startsWith("/watch/")) return __realFetch(full, opts);
  if (__offline) return Promise.reject(new TypeError("network is down"));
  const ctl = new AbortController();
  __open.add(ctl);
  const own = opts && opts.signal;
  if (own) own.addEventListener("abort", () => ctl.abort());
  return __realFetch(full, { ...opts, signal: ctl.signal });
};
const __log = document.getElementById("log");
const __say = (obj) => process.stdout.write(JSON.stringify(obj) + "\\n");
const __act = (fn) => {
  try { fn(); __say({ ok: true }); }
  catch (e) { __say({ ok: false, error: String(e) }); }
};
let __pending = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  __pending += chunk;
  let end;
  while ((end = __pending.indexOf("\\n")) >= 0) {
    const line = __pending.slice(0, end).replace(/\\r$/, "");  // a Windows pipe sends CR LF
    __pending = __pending.slice(end + 1);
    const space = line.indexOf(" ");
    const cmd = space < 0 ? line.trim() : line.slice(0, space);
    const arg = space < 0 ? "" : line.slice(space + 1);
    if (cmd === "OFFLINE") {
      __offline = true;
      for (const ctl of __open) ctl.abort();
      __say({ ok: true });
    } else if (cmd === "ONLINE") {
      __offline = false;
      __say({ ok: true });
    } else if (cmd === "CLICK") {
      __act(() => document.getElementById(arg).onclick());
    } else if (cmd === "CHAT") {
      __act(() => document.getElementById("chats").children[Number(arg)].onclick());
    } else if (cmd === "TYPE") {
      __act(() => { document.getElementById("input").value = arg; });
    } else if (cmd === "SEND") {
      __act(() => document.getElementById("send").onclick());
    } else if (cmd === "HOLD") {
      __hold = arg;
      __say({ ok: true });
    } else if (cmd === "RELEASE") {
      __act(() => { const send = __held; __held = null; send(); });
    } else if (cmd === "SNAP") {
      const rows = __log.children.map((r) => ({
        cls: [...r.classes].filter((c) => c !== "row").join(" "),
        text: r.textContent,
      }));
      const errs = __created
        .filter((n) => n.parent !== null && n.classes.has("errcard"))
        .map((n) => n.textContent);
      const chats = document.getElementById("chats").children.map((c) => ({
        title: c.children.length ? c.children[0].textContent : "",
        current: c.classes.has("current"),
      }));
      __say({
        rows,
        errs,
        trace: __trace,
        chats,
        chip: document.getElementById("chip-session").textContent,
        input: document.getElementById("input").value,
        status: document.getElementById("status-text").textContent,
        held: __held !== null,
      });
    }
  }
});
process.stdin.on("end", () => process.exit(0));
"""


def _current(base: str) -> str:
    resp = httpx.get(base + "/sessions/current", timeout=5)
    resp.raise_for_status()
    sid: str = resp.json()["id"]
    return sid


def _spoken_count(base: str, session_id: str) -> int:
    resp = httpx.get(base + f"/sessions/{session_id}/transcript", timeout=5)
    resp.raise_for_status()
    return len(resp.json()["messages"])


def _new_chat_from_elsewhere(base: str) -> str:
    """What another page's New chat does."""
    resp = httpx.put(base + "/sessions/current", json={"session_id": None}, timeout=5)
    assert resp.status_code == 200, resp.text
    sid: str = resp.json()["id"]
    return sid


def _act(page: _OpenPage, command: str) -> None:
    answer: dict[str, Any] = page.ask(command)
    assert answer["ok"], answer


@pytest.fixture
def buses(monkeypatch: pytest.MonkeyPatch) -> list[EventBusRegistry]:
    """Every watch-bus registry the app under test makes, so a test can count a chat's watchers."""
    made: list[EventBusRegistry] = []

    def make() -> EventBusRegistry:
        made.append(EventBusRegistry())
        return made[-1]

    monkeypatch.setattr(server_app, "EventBusRegistry", make)
    return made


def _watchers(buses: list[EventBusRegistry], session_id: str) -> int:
    """The watch connections the server holds open on a chat's bus."""
    bus = buses[0].get(session_id)
    assert bus is not None
    return bus.subscriber_count


def _watchers_drop_to_zero(buses: list[EventBusRegistry], session_id: str) -> bool:
    deadline = time.monotonic() + 15
    while _watchers(buses, session_id):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_new_chat_opens_an_empty_chat_and_lists_the_one_it_left(
    tmp_path: Path, buses: list[EventBusRegistry]
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        left = _current(base)
        with _OpenPage(base, tmp_path, _TAIL) as page:
            page.settle(4)
            assert _watchers(buses, left) == 1  # the page's own watch

            _act(page, "CLICK new-chat")
            snap = page._wait_for(
                lambda s: s["chip"] != left[:8] and len(s["chats"]) == 2 and not _spoken(s), 15
            )

            opened = _current(base)
            assert opened != left
            assert snap["chip"] == opened[:8]
            assert snap["chats"] == [
                {"title": "New chat", "current": True},
                {"title": "one", "current": False},
            ]
            # Nothing the page knew of the chat it left is read against the new one: the new
            # chat's watch opens at its start, with no cursor or bus of the old one.
            assert _watches(snap)[-1] == f"/watch/{opened}?full=1"
            # And the page ended its watch of the chat it left. A browser holds few connections
            # to one server, and a watch left open at every switch would use them up.
            assert _watchers_drop_to_zero(buses, left)
            assert snap["errs"] == []
            # Once the page is still, the new chat holds its start screen alone: no row of the
            # chat it left, and no "session restarted" note for a switch this page made itself.
            assert [row for row in page.settle(0)["rows"] if row["cls"]] == []
        assert _spoken_count(base, left) == 4
        assert _spoken_count(base, opened) == 0


def test_a_page_lists_the_saved_chats_as_soon_as_it_opens(tmp_path: Path) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
        _new_chat_from_elsewhere(base)  # the page opens on a chat with no turn to replay
        with _OpenPage(base, tmp_path, _TAIL) as page:
            snap = page._wait_for(lambda s: len(s["chats"]) == 2, 15)

            assert snap["chats"] == [
                {"title": "New chat", "current": True},
                {"title": "one", "current": False},
            ]


def test_an_old_chat_opened_from_the_list_is_drawn_and_the_next_message_continues_it(
    tmp_path: Path,
) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one", "two"])
        old = _current(base)
        newer = _new_chat_from_elsewhere(base)
        _converse(base, ["three"])
        with _OpenPage(base, tmp_path, _TAIL) as page:
            page.settle(2)
            page._wait_for(lambda s: len(s["chats"]) == 2, 15)

            _act(page, "CHAT 1")
            snap = page.settle(4)

            assert _spoken(snap) == [
                ("user", "one"),
                ("assistant", "echo one"),
                ("user", "two"),
                ("assistant", "echo two"),
            ]
            assert snap["chip"] == old[:8]
            _act(page, "TYPE and four")
            _act(page, "SEND")
            snap = page.settle(6)
            assert _spoken(snap)[4:] == [("user", "and four"), ("assistant", "echo and four")]
            assert snap["errs"] == []
            # The turn saved the chat, so the list asked for after it puts the chat on top.
            snap = page._wait_for(
                lambda s: s["chats"][:1] == [{"title": "one", "current": True}], 15
            )
            assert [chat["title"] for chat in snap["chats"]] == ["one", "three"]
        assert _current(base) == old
        assert _spoken_count(base, old) == 6
        assert _spoken_count(base, newer) == 2


def test_a_chat_opened_before_the_last_ones_history_lands_is_drawn_alone(tmp_path: Path) -> None:
    # A chat saved before this server started, as every chat is after a restart: no watch bus
    # holds its turns, so opening it draws its saved conversation, and nothing else would.
    old = Session(cwd=str(tmp_path), model="scripted/test")
    old.add_message(Message.user("an old ask"))
    old.add_message(Message.assistant_text("an old reply"))
    SessionStore(base_dir=tmp_path / "sessions").save(old)
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["two"])
        shown = _current(base)
        with _OpenPage(base, tmp_path, _TAIL) as page:
            page.settle(2)
            page._wait_for(lambda s: len(s["chats"]) == 2, 15)
            _act(page, f"HOLD /sessions/{old.id}/transcript")

            _act(page, "CHAT 1")  # the old chat, whose history is kept on its way ...
            page._wait_for(lambda s: s["held"], 15)
            _act(page, "CLICK new-chat")  # ... while another chat opens
            page._wait_for(lambda s: s["chip"] not in (old.id[:8], shown[:8]), 15)
            _act(page, "RELEASE")
            snap = page.settle(0)

            assert _spoken(snap) == []
            assert snap["chip"] == _current(base)[:8]
            assert snap["errs"] == []

            # Opened again, with nothing in its way, the old chat is drawn from what was saved.
            snap = page._wait_for(lambda s: len(s["chats"]) == 3, 15)
            assert [chat["title"] for chat in snap["chats"]] == ["New chat", "two", "an old ask"]
            _act(page, "CHAT 2")
            snap = page.settle(2)
            assert _spoken(snap) == [("user", "an old ask"), ("assistant", "an old reply")]
            assert snap["chip"] == old.id[:8]


def test_a_page_follows_when_another_page_opens_a_chat(tmp_path: Path) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
        with _OpenPage(base, tmp_path, _TAIL) as page:
            page.settle(2)
            page.wait_for_watches(1)

            opened = _new_chat_from_elsewhere(base)
            page._wait_for(lambda s: s["chip"] == opened[:8] and not _spoken(s), 15)
            _converse(base, ["two"])
            snap = page.settle(2)

            assert _spoken(snap) == [("user", "two"), ("assistant", "echo two")]
            assert snap["errs"] == []


def test_a_page_that_opens_a_chat_while_it_follows_a_move_marks_no_restart(tmp_path: Path) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
        with _OpenPage(base, tmp_path, _TAIL) as page:
            page.settle(2)
            page.wait_for_watches(1)
            _act(page, "HOLD /sessions/current")

            opened = _new_chat_from_elsewhere(base)  # the page hears of it and asks what is current
            page._wait_for(lambda s: s["held"], 15)
            _act(page, "CLICK new-chat")  # and opens a chat itself before the answer lands
            page._wait_for(lambda s: s["chip"] == opened[:8], 15)
            _act(page, "RELEASE")
            snap = page.settle(0)

            assert [row for row in snap["rows"] if row["cls"]] == []
            assert snap["chip"] == opened[:8]
            assert snap["errs"] == []


def test_a_page_left_on_another_chat_keeps_the_message_it_could_not_send(tmp_path: Path) -> None:
    with _LiveServer(_app(tmp_path)) as base:
        _converse(base, ["one"])
        shown = _current(base)
        with _OpenPage(base, tmp_path, _TAIL) as page:
            page.settle(2)
            _act(page, "OFFLINE")  # the page does not hear that the chat changed
            opened = _new_chat_from_elsewhere(base)

            _act(page, "TYPE meant for the chat on screen")
            _act(page, "SEND")
            snap = page._wait_for(
                lambda s: s["chip"] == opened[:8] and "not sent" in s["status"], 15
            )

            assert snap["input"] == "meant for the chat on screen"
            assert "not sent" in snap["status"]
            _act(page, "ONLINE")
        assert say_path(tmp_path).exists() is False
        assert _spoken_count(base, shown) == 2
        assert _spoken_count(base, opened) == 0
