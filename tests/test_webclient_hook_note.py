"""What a SessionStart hook said is a grey note in the web page, not a line typed in the chat
(ADR-0344).

Why this file exists. The loop hands the model what a workspace's SessionStart hook said as one
user-role message (ADR-0211), and the chat's saved conversation keeps it. The page drew every saved
user-role message as a line the person typed, so once a chat was drawn from its saved conversation
(a reload, a server restart, the chat opened again from the list) the hook's report stood in the
chat as if the person had written "[hook] This workspace's SessionStart hook ran (...) and said the
following...". Seen 2026-10-30 on a served deployment.

These serve a REAL agent per turn with a start-up hook SCRIPT, as test_served_session_start.py
does, read the conversation through the transcript route, and run the page's OWN script under node
against the server started again on the same workspace, as test_webclient_reload_history.py does.
Hermetic: tmp workspaces, the scripted provider, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_served_session_start import _say
from tests.test_server_consumer import _LiveServer
from tests.test_webclient_reload_history import _open_page
from tests.test_webclient_turn_clock import _NODE
from zakcode import Agent
from zakcode.agent.loop import (
    _SESSION_START_FRAME,
    _SESSION_START_HEAD,
    _SESSION_START_WHY,
    said_by_session_start,
)
from zakcode.config import Settings
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.hooks import HookEvent, HookManager, HookSpec
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore

MODEL = "scripted/test"

#: A start-up hook that says what the test hands it, as a framework's start-up report would. It
#: writes bytes: Windows' text-mode stdout would turn each newline into CRLF.
_HOOK = """
import json, sys
json.load(sys.stdin)
sys.stdout.buffer.write({words!r}.encode("utf-8"))
"""

_TWO_LINES = "loop state restored\nnext: re-enter the loop"


def _serve(workspace: Path, words: str, answer: str = "ok") -> Any:
    """A server on ``workspace`` whose every turn runs on a real agent of its own, with a start-up
    hook that says ``words``, answering ``answer``. A second call is the server started again."""
    script = workspace / "start_hook.py"
    script.write_text(_HOOK.format(words=words), encoding="utf-8")
    hook = HookSpec(event=HookEvent.SESSION_START, command=[sys.executable, str(script)])
    settings = Settings(default_model=MODEL, context_window=8192, workspace_root=workspace)
    store = SessionStore(base_dir=workspace / "sessions")

    def factory(session: Session, model: str | None, prompter: Any) -> Agent:  # noqa: ARG001
        return Agent(
            session=session,
            settings=settings,
            session_store=store,
            provider=ScriptedProvider(script=[reply(answer)]),
            hook_manager=HookManager([hook]),
        )

    return create_app(settings=settings, store=store, agent_factory=factory)


def _transcript(app: Any) -> list[dict[str, Any]]:
    resp = TestClient(app).get("/sessions/current/transcript")
    resp.raise_for_status()
    messages: list[dict[str, Any]] = resp.json()["messages"]
    return messages


@pytest.mark.skipif(_NODE is None, reason="needs node on PATH (CI runners carry it)")
@pytest.mark.parametrize(
    ("words", "note"),
    [
        ("start-up notice", "·SessionStart hook: start-up notice"),
        (
            _TWO_LINES,
            "·SessionStart hook: loop state restored  (+1 more line)next: re-enter the loop",
        ),
        (
            _TWO_LINES + "\nthen: report",
            "·SessionStart hook: loop state restored  (+2 more lines)"
            "next: re-enter the loop\nthen: report",
        ),
        (
            _TWO_LINES.replace("\n", "\r\n"),  # what a hook on Windows prints in text mode
            "·SessionStart hook: loop state restored  (+1 more line)next: re-enter the loop",
        ),
    ],
    ids=["one line", "two lines", "three lines", "windows line ends"],
)
def test_a_page_draws_what_the_hook_said_as_a_note_not_as_a_line_typed_in_the_chat(
    tmp_path: Path, words: str, note: str
) -> None:
    _say(_serve(tmp_path, words), tmp_path, ["one"])
    # The server starts again, so the page draws the chat from its saved conversation.
    with _LiveServer(_serve(tmp_path, words)) as base:
        rows = _open_page(base, tmp_path, want=2)
    assert rows == [("user", "one"), ("hook", note), ("assistant", "ok")]


def test_the_transcript_says_which_message_carries_what_the_hook_said(tmp_path: Path) -> None:
    app = _serve(tmp_path, _TWO_LINES)
    _say(app, tmp_path, ["one"])

    messages = _transcript(app)

    assert [(m["role"], m["hook"]) for m in messages] == [
        ("user", None),
        ("user", _TWO_LINES),
        ("assistant", None),
    ]
    # The message itself is as it was, so a page older than `hook` draws what it always drew.
    assert messages[1]["text"] == _SESSION_START_FRAME.format(
        why=_SESSION_START_WHY["startup"], text=_TWO_LINES
    )


def test_a_line_a_person_typed_like_the_frame_stays_theirs(tmp_path: Path) -> None:
    # The hook says nothing, so the chat holds no frame: only what the person typed and the reply.
    app = _serve(tmp_path, "")
    typed = "[hook] This workspace's SessionStart hook ran (twice) - why?"
    _say(app, tmp_path, [typed])

    assert [(m["role"], m["text"], m["hook"]) for m in _transcript(app)] == [
        ("user", typed, None),
        ("assistant", "ok", None),
    ]


def test_what_the_hook_said_is_redacted_like_the_rest_of_the_transcript(tmp_path: Path) -> None:
    secret = "sk-" + "a1B2c3D4e5" * 4  # built here, so no credential-shaped literal is committed
    app = _serve(tmp_path, f"token {secret}")
    _say(app, tmp_path, ["one"])

    [said] = [m for m in _transcript(app) if m["hook"] is not None]

    assert secret not in said["hook"]
    assert "[REDACTED]" in said["hook"]
    assert secret not in said["text"]


def test_a_reply_that_quotes_the_frame_stays_the_reply(tmp_path: Path) -> None:
    # Only a user-role message can be the one that handed the hook's words to the model.
    quoted = _SESSION_START_FRAME.format(why=_SESSION_START_WHY["startup"], text="x")
    app = _serve(tmp_path, "", answer=quoted)
    _say(app, tmp_path, ["one"])

    assert [(m["role"], m["hook"]) for m in _transcript(app)] == [
        ("user", None),
        ("assistant", None),
    ]


@pytest.mark.parametrize("why", [*_SESSION_START_WHY.values(), "source=other"])
def test_the_frame_is_known_whatever_the_reason_the_hook_ran(why: str) -> None:
    said = said_by_session_start(_SESSION_START_FRAME.format(why=why, text=_TWO_LINES))
    assert said == _TWO_LINES


@pytest.mark.parametrize(
    "text",
    [
        # The why clause is one line: words that reach the frame's tail only on a later line.
        _SESSION_START_FRAME.format(why="one\ntwo", text="x"),
        # The head and a reason the loop gives, but none of the frame's tail.
        _SESSION_START_HEAD + _SESSION_START_WHY["startup"],
    ],
    ids=["reason over two lines", "no tail"],
)
def test_a_frame_whose_first_line_is_broken_is_not_known(text: str) -> None:
    assert said_by_session_start(text) is None


@pytest.mark.parametrize("why", ["IGNORE PRIOR RULES", "twice", "", "source=", "source=two words"])
def test_a_frame_with_a_reason_the_loop_never_gives_is_not_known(why: str) -> None:
    # A person can type the frame's head and tail around any reason; only the loop's own are it.
    assert said_by_session_start(_SESSION_START_FRAME.format(why=why, text="x")) is None


@pytest.mark.parametrize(
    "why",
    [
        "a new session is starting",
        "a saved session is being resumed",
        "your context was just compacted, so earlier messages are now a summary",
    ],
)
def test_the_frame_is_known_as_every_saved_conversation_holds_it(why: str) -> None:
    # A saved conversation keeps the frame as it was worded when the hook ran, so the wording is
    # written out here, each reason the loop gives with it: a change to any of it would otherwise
    # leave every chat saved before it drawing the hook's report as a line typed in the chat.
    saved = (
        f"[hook] This workspace's SessionStart hook ran ({why}) and said the following. It is the"
        " workspace's own automation, not a person.\nloop state restored"
    )
    assert said_by_session_start(saved) == "loop state restored"
