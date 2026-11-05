"""Renaming and deleting a chat from the web page's chat list (ADR-0358).

Each chat in the list has ⋯, which opens, in the chat's own row, Rename and Delete. Rename puts a
name box in the row: Enter or Save names the chat, Escape or Cancel leaves it as it was. Delete
asks in the row whether to delete it, and only the second Delete sends the request. A refusal the
server gives (a turn running on that chat) is said in the status line, and the box keeps what was
typed. The list holds still while a name is typed. A deleted chat that was on screen gives way to
the active chat. A new chat is titled in the list as its first message is sent.

These run the page's OWN script, cut from the shipped page, under node, on the DOM the turn-clock
tests use, against a scripted server; test_webapp_chat_manage.py holds the server's half.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.test_webclient_turn_clock import _NODE, _PRELUDE, _page_script

pytestmark = pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")

_DOM = _PRELUDE.split("const __STREAMS = ")[0]
A, B, C = "aaaaaaaa-0001", "bbbbbbbb-0002", "cccccccc-0003"
NEW = "nnnnnnnn-0009"  # the chat the server makes when the current one is deleted

#: The scripted server and the timed steps. Runs after the DOM and before the page's script.
_SERVER = r"""
const __SCN = JSON.parse(process.argv[2]);
const __later = setTimeout;  // the real clock, for this script's own timing
globalThis.setTimeout = (fn, _ms, ...args) => __later(fn, 0, ...args);
const __asked = [];  // every request that changes something: [method, path, body]
const __watched = new Set();
function __answer(body, status) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: async () => body,
                           headers: { get: () => null } });
}
function __summary(chat) {
  return { id: chat.id, title: chat.title, message_count: 2,
           updated_at: "2026-11-04T10:00:00+00:00", current: chat.id === __SCN.current };
}
globalThis.fetch = async (url, opts) => {
  opts = opts || {};
  const method = opts.method || "GET";
  const u = new URL(url, "http://page.test");
  const chats = __SCN.chats;
  if (method !== "GET" && u.pathname !== "/say") {
    __asked.push([method, u.pathname, opts.body ? JSON.parse(opts.body) : null]);
  }
  if (u.pathname === "/sessions/current") return __answer({ id: __SCN.current, model: "m" }, 200);
  if (u.pathname === "/sessions/recent") return __answer(chats.map(__summary), 200);
  let m = u.pathname.match(/^\/sessions\/([^/]+)\/title$/);
  if (m && method === "PUT") {
    if (__SCN.refuse) return __answer({ detail: __SCN.refuse }, 409);
    const chat = chats.find((c) => c.id === m[1]);
    chat.title = JSON.parse(opts.body).title.split(/\s+/).filter(Boolean).join(" ") || chat.ask;
    return __answer(__summary(chat), 200);
  }
  m = u.pathname.match(/^\/sessions\/([^/]+)\/transcript$/);
  if (m) {
    return __answer({ session_id: m[1], messages: [], message_count: 0, bus_from: null,
                      bus_since: null, running: false }, 200);
  }
  m = u.pathname.match(/^\/sessions\/([^/]+)$/);
  if (m && method === "DELETE") {
    if (__SCN.refuse) return __answer({ detail: __SCN.refuse }, 409);
    const at = chats.findIndex((c) => c.id === m[1]);
    if (at < 0) return __answer({ detail: "no session" }, 404);
    chats.splice(at, 1);
    if (m[1] === __SCN.current) {  // a new chat is current, as the server makes one
      __SCN.current = __SCN.fresh;
      chats.unshift({ id: __SCN.fresh, title: "", ask: "" });
    }
    return __answer(null, 204);
  }
  if (m) return __answer({ id: m[1], model: "m" }, 200);
  m = u.pathname.match(/^\/watch\/([^/]+)$/);
  if (m) {
    // A chat's frames once, on its first connection; after that, and for a chat with none, a
    // live watch that says nothing.
    const frames = (__SCN.watch || {})[m[1]];
    if (!frames || __watched.has(m[1])) return new Promise(() => {});
    __watched.add(m[1]);
    const bytes = new TextEncoder().encode(frames.map(
      ([id, f]) => `id: ${id}\r\ndata: ${JSON.stringify(f)}\r\n\r\n`).join(""));
    let sent = false;
    const read = () => {
      if (sent) return new Promise(() => {});
      sent = true;
      // Sent once the page has stood a while (`watchAfter` ms), as a message typed into it is.
      const frame = { done: false, value: bytes };
      return new Promise((r) => __later(() => r(frame), __SCN.watchAfter || 0));
    };
    const body = { getReader: () => ({ read }) };
    return { ok: true, status: 200, headers: { get: () => null }, body };
  }
  return __answer({ queued: true }, 200);
};
const __click = { stopPropagation() {}, preventDefault() {} };
function __rows() { return document.getElementById("chats").children; }
function __row(title) {
  const row = __rows().find((c) => c.children[0] && c.children[0].textContent === title);
  if (!row) throw new Error("no chat in the list titled " + title);
  return row;
}
function __editing() {
  const row = __rows().find((c) => c.classes.has("editing"));
  if (!row) throw new Error("no chat row is open");
  return row;
}
function __box() { return __editing().children.find((k) => k.classes.has("chat-name")); }
const __seen = [];
function __look(label) {
  const statusOn = document.getElementById("stream-status").classes.has("show");
  return {
    label,
    rows: __rows().map((c) => ({
      first: c.children[0] ? c.children[0].textContent : "",
      editing: c.classes.has("editing"),
      viewing: c.classes.has("viewing"),
      buttons: c.children.filter((k) => k.tagName === "BUTTON").map((k) => k.textContent),
      box: c.children.filter((k) => k.classes.has("chat-name")).map((k) => k.value),
    })),
    status: statusOn ? document.getElementById("status-text").textContent : "",
    asked: [...__asked],
    chip: document.getElementById("chip-session").textContent,
  };
}
for (const step of __SCN.steps) {
  __later(() => {
    if (step.more !== undefined) {
      __row(step.more).children.find((k) => k.classes.has("chat-more")).onclick(__click);
    } else if (step.act !== undefined) {
      const button = __editing().children.find(
        (k) => k.tagName === "BUTTON" && k.textContent === step.act);
      if (!button) throw new Error("no button " + step.act + " in the open row");
      button.onclick(__click);
    } else if (step.type !== undefined) {
      __box().value = step.type;
    } else if (step.key !== undefined) {
      __box().onkeydown({ key: step.key, preventDefault() {} });
    } else if (step.open !== undefined) {
      __row(step.open).onclick();
    } else if (step.refresh) {
      refreshChats();  // what the end of a turn does to the list
    } else if (step.set) {
      Object.assign(__SCN, step.set);
    } else {
      __seen.push(__look(step.look));
    }
  }, step.at);
}
__later(() => {
  const errors = __created.filter((n) => n.parent !== null && n.classes.has("errcard"))
    .map((n) => n.textContent);
  process.stdout.write(JSON.stringify({ seen: __seen, errors }));
  process.exit(0);
}, Math.max(...__SCN.steps.map((s) => s.at)) + 100);
"""


def _page(
    tmp_path: Path,
    steps: list[dict[str, Any]],
    *,
    current: str = A,
    watch: dict[str, list[Any]] | None = None,
    watch_after: int = 0,
    chats: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Run the page on ``current`` against three saved chats (or ``chats``); what it showed at
    each ``look`` step. The page must draw no error card."""
    assert _NODE is not None
    script = tmp_path / f"page-{time.monotonic_ns()}.js"
    script.write_text(_DOM + _SERVER + _page_script(), encoding="utf-8")
    saved = chats or [
        {"id": A, "title": "deploy the staging box", "ask": "deploy the staging box"},
        {"id": B, "title": "why is CI red", "ask": "why is CI red"},
        {"id": C, "title": "fix the login bug", "ask": "fix the login bug"},
    ]
    scenario = {
        "current": current,
        "fresh": NEW,
        "chats": saved,
        "watch": watch or {},
        "watchAfter": watch_after,
        "steps": steps,
    }
    proc = subprocess.run(
        [_NODE, str(script), json.dumps(scenario)], capture_output=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    shown = json.loads(proc.stdout.decode("utf-8"))
    assert shown["errors"] == [], shown["errors"]
    seen: list[dict[str, Any]] = shown["seen"]
    return seen


def _titles(look: dict[str, Any]) -> list[str]:
    return [row["first"] for row in look["rows"]]


def test_a_chat_is_renamed_in_its_own_row(tmp_path: Path) -> None:
    seen = _page(
        tmp_path,
        [
            {"at": 300, "more": "why is CI red"},
            {"at": 350, "look": "menu"},
            {"at": 400, "act": "Rename"},
            {"at": 450, "look": "box"},
            {"at": 500, "type": "CI flake hunt"},
            {"at": 550, "key": "Enter"},
            {"at": 800, "look": "named"},
        ],
    )

    menu, box, named = seen
    assert [row["editing"] for row in menu["rows"]] == [False, True, False]
    assert menu["rows"][1]["buttons"] == ["Rename", "Delete", "Cancel"]
    assert box["rows"][1]["box"] == ["why is CI red"]  # the name it has, to change
    assert named["asked"] == [["PUT", f"/sessions/{B}/title", {"title": "CI flake hunt"}]]
    assert _titles(named) == ["deploy the staging box", "CI flake hunt", "fix the login bug"]
    assert not any(row["editing"] for row in named["rows"])


def test_escape_leaves_the_chat_as_it_was(tmp_path: Path) -> None:
    seen = _page(
        tmp_path,
        [
            {"at": 300, "more": "why is CI red"},
            {"at": 350, "act": "Rename"},
            {"at": 400, "type": "something else"},
            {"at": 450, "key": "Escape"},
            {"at": 700, "look": "after"},
        ],
    )

    (after,) = seen
    assert after["asked"] == []
    assert _titles(after) == ["deploy the staging box", "why is CI red", "fix the login bug"]
    assert not any(row["editing"] for row in after["rows"])


def test_a_refused_name_says_why_and_the_box_keeps_what_was_typed(tmp_path: Path) -> None:
    refusal = "a turn is running in this chat; stop it or let it finish, then rename it"
    seen = _page(
        tmp_path,
        [
            {"at": 200, "set": {"refuse": refusal}},
            {"at": 300, "more": "deploy the staging box"},
            {"at": 350, "act": "Rename"},
            {"at": 400, "type": "Staging deploy"},
            {"at": 450, "act": "Save"},
            {"at": 700, "look": "refused"},
        ],
    )

    (refused,) = seen
    assert refused["status"] == refusal
    assert refused["rows"][0]["editing"] is True
    assert refused["rows"][0]["box"] == ["Staging deploy"]


def test_the_list_holds_still_while_a_name_is_typed(tmp_path: Path) -> None:
    """A turn's end asks for the list again. Drawn while a name was being typed, the box went with
    what was typed in it."""
    seen = _page(
        tmp_path,
        [
            {"at": 300, "more": "fix the login bug"},
            {"at": 350, "act": "Rename"},
            {"at": 400, "type": "Login bug"},
            {"at": 450, "refresh": True},
            {"at": 700, "look": "typing"},
        ],
    )

    (typing,) = seen
    assert typing["rows"][2]["editing"] is True
    assert typing["rows"][2]["box"] == ["Login bug"]


def test_a_chat_is_deleted_only_once_the_question_in_its_row_is_answered(tmp_path: Path) -> None:
    seen = _page(
        tmp_path,
        [
            {"at": 300, "more": "fix the login bug"},
            {"at": 350, "act": "Delete"},
            {"at": 400, "look": "asked"},
            {"at": 450, "act": "Delete"},
            {"at": 700, "look": "deleted"},
        ],
    )

    asked, deleted = seen
    assert asked["asked"] == []  # nothing is deleted on the first Delete
    assert asked["rows"][2]["first"] == "Delete “fix the login bug”?"
    assert asked["rows"][2]["buttons"] == ["Delete", "Cancel"]
    assert deleted["asked"] == [["DELETE", f"/sessions/{C}", None]]
    assert _titles(deleted) == ["deploy the staging box", "why is CI red"]


def test_cancel_deletes_nothing(tmp_path: Path) -> None:
    seen = _page(
        tmp_path,
        [
            {"at": 300, "more": "fix the login bug"},
            {"at": 350, "act": "Delete"},
            {"at": 400, "act": "Cancel"},
            {"at": 700, "look": "after"},
        ],
    )

    (after,) = seen
    assert after["asked"] == []
    assert _titles(after) == ["deploy the staging box", "why is CI red", "fix the login bug"]


def test_a_refused_delete_says_why_and_keeps_the_chat(tmp_path: Path) -> None:
    refusal = "a turn is running in this chat; stop it or let it finish, then delete it"
    seen = _page(
        tmp_path,
        [
            {"at": 200, "set": {"refuse": refusal}},
            {"at": 300, "more": "deploy the staging box"},
            {"at": 350, "act": "Delete"},
            {"at": 400, "act": "Delete"},
            {"at": 700, "look": "refused"},
        ],
    )

    (refused,) = seen
    assert refused["status"] == refusal
    assert refused["asked"] == [["DELETE", f"/sessions/{A}", None]]
    assert refused["rows"][0]["first"] == "Delete “deploy the staging box”?"


def test_deleting_the_chat_on_screen_shows_the_active_chat(tmp_path: Path) -> None:
    seen = _page(
        tmp_path,
        [
            {"at": 300, "open": "why is CI red"},
            {"at": 500, "look": "viewing"},
            {"at": 550, "more": "why is CI red"},
            {"at": 600, "act": "Delete"},
            {"at": 650, "act": "Delete"},
            {"at": 900, "look": "deleted"},
        ],
    )

    viewing, deleted = seen
    assert viewing["chip"] == B[:8]
    assert deleted["chip"] == A[:8]  # the active chat, which the delete left as it was
    assert _titles(deleted) == ["deploy the staging box", "fix the login bug"]
    assert [row["viewing"] for row in deleted["rows"]] == [True, False]


def test_deleting_the_active_chat_shows_the_new_one_made_in_its_place(tmp_path: Path) -> None:
    seen = _page(
        tmp_path,
        [
            {"at": 300, "more": "deploy the staging box"},
            {"at": 350, "act": "Delete"},
            {"at": 400, "act": "Delete"},
            {"at": 700, "look": "deleted"},
        ],
    )

    (deleted,) = seen
    assert deleted["chip"] == NEW[:8]
    assert _titles(deleted) == ["New chat", "why is CI red", "fix the login bug"]


def test_a_new_chat_is_titled_as_its_first_message_is_sent(tmp_path: Path) -> None:
    """The server titles a chat by its first ask once the turn has saved it, and the page asked
    for the list again only when the answer ended: a new chat read "New chat" for as long as its
    first answer took."""
    fresh = {"id": NEW, "title": "", "ask": ""}
    older = {"id": B, "title": "why is CI red", "ask": "why is CI red"}
    sent = [["1", {"event": "user_message", "text": "plan the\n  trip to Lisbon", "at": 1.0}]]
    seen = _page(
        tmp_path,
        [{"at": 300, "look": "before"}, {"at": 800, "look": "sent"}],
        current=NEW,
        chats=[fresh, older],
        watch={NEW: sent},
        watch_after=500,
    )

    before, shown = seen
    assert _titles(before) == ["New chat", "why is CI red"]
    assert _titles(shown) == ["plan the trip to Lisbon", "why is CI red"]
    assert shown["asked"] == []  # named on the page alone: the server titles it when it saves
