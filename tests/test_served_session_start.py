"""A served chat runs the workspace's SessionStart hooks once, not on every message (ADR-0342).

Why this file exists. The web app runs every turn on an agent of its own, and an agent's loop
fires SessionStart before its first turn. So on a served workspace every message counted as "a
saved session is being resumed": the start-up hooks ran again, and what they printed was added
to the chat again, one "[hook]" line under every message. Seen 2026-10-30 on a served deployment
whose framework hook printed a one-line notice: three messages, three copies of it in the chat.

These tests serve a REAL agent per turn, as the app does, and a hook SCRIPT writes down every
run it makes. Hermetic: tmp workspaces, the scripted provider, no network.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from zakcode import Agent, SkillInvocation
from zakcode.agent.loop import AgentLoop
from zakcode.config import Settings
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.hooks import HookEvent, HookManager, HookSpec
from zakcode.server.app import create_app
from zakcode.session.say_inbox import interrupt_path, request_interrupt, say_path, write_say
from zakcode.session.store import Session, SessionStore

MODEL = "scripted/test"

#: A start-up hook in the shape a framework's takes: it writes down that it ran, and says a line.
#: While the workspace holds a ``slow-hook`` file, the run that finds it deletes it and sleeps, so
#: a test can press Stop while the hook is still running.
_HOOK = """
import json, os, sys, time
doc = json.load(sys.stdin)
with open({log!r}, "a", encoding="utf-8") as out:
    out.write(json.dumps({{"source": doc.get("source"), "session": doc.get("session_id")}}) + "\\n")
if os.path.exists({slow!r}):
    os.remove({slow!r})
    time.sleep(60)
print("start-up notice (%s)" % doc.get("source"))
"""


def _serve(workspace: Path) -> tuple[Any, SessionStore]:
    """A server on ``workspace`` whose every turn runs on a real agent of its own, as the web
    app's does, with the start-up hook registered. A second call is the server started again."""
    script = workspace / "start_hook.py"
    script.write_text(
        _HOOK.format(log=str(workspace / "hook-runs.jsonl"), slow=str(workspace / "slow-hook")),
        encoding="utf-8",
    )
    hook = HookSpec(event=HookEvent.SESSION_START, command=[sys.executable, str(script)])
    settings = Settings(default_model=MODEL, context_window=8192, workspace_root=workspace)
    store = SessionStore(base_dir=workspace / "sessions")

    def factory(session: Session, model: str | None, prompter: Any) -> Agent:  # noqa: ARG001
        return Agent(
            session=session,
            settings=settings,
            session_store=store,
            provider=ScriptedProvider(script=[reply("ok")]),
            hook_manager=HookManager([hook]),
        )

    return create_app(settings=settings, store=store, agent_factory=factory), store


def _say(app: Any, workspace: Path, texts: list[str]) -> None:
    async def turns() -> None:
        for text in texts:
            assert write_say(say_path(workspace), text)
            assert await app.state.consume_one_say() is True

    asyncio.run(turns())


def _say_stopped_then(app: Any, workspace: Path, until: Callable[[], bool], then: str) -> None:
    """Send "first", press Stop once ``until()`` holds, then send ``then`` as the next message."""

    async def turns() -> None:
        assert write_say(say_path(workspace), "first")
        first = asyncio.create_task(app.state.consume_one_say())
        deadline = time.monotonic() + 30
        while not until():
            assert time.monotonic() < deadline, "the first turn never reached the Stop point"
            await asyncio.sleep(0.05)
        request_interrupt(interrupt_path(workspace))
        await first
        assert write_say(say_path(workspace), then)
        assert await app.state.consume_one_say() is True

    asyncio.run(turns())


def _runs(workspace: Path) -> list[str]:
    log = workspace / "hook-runs.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [json.loads(line)["source"] for line in lines if line]


def _words(store: SessionStore, workspace: Path) -> list[str]:
    """Every message of the current chat that carries the hook's words, in order."""
    current = (workspace / ".current-session").read_text(encoding="utf-8").strip()
    texts = [m.text or "" for m in store.load(current).messages]
    return [text for text in texts if "SessionStart hook ran" in text]


def test_three_messages_in_one_chat_run_its_start_up_hook_once(tmp_path: Path) -> None:
    app, store = _serve(tmp_path)
    _say(app, tmp_path, ["first", "second", "third"])

    assert _runs(tmp_path) == ["startup"]
    words = _words(store, tmp_path)
    assert len(words) == 1, words
    assert "a new session is starting" in words[0]


def test_the_server_started_again_runs_it_once_more_as_a_resume(tmp_path: Path) -> None:
    app, store = _serve(tmp_path)
    _say(app, tmp_path, ["first", "second"])
    again, _ = _serve(tmp_path)  # the same workspace, served by a process started again
    _say(again, tmp_path, ["third", "fourth"])

    assert _runs(tmp_path) == ["startup", "resume"]
    words = _words(store, tmp_path)
    assert len(words) == 2, words
    assert "a saved session is being resumed" in words[1]


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_a_chat_a_driver_runs_through_the_api_runs_it_once(tmp_path: Path, path: str) -> None:
    app, _ = _serve(tmp_path)
    client = TestClient(app)
    sid = client.post("/sessions").json()["id"]
    for text in ("first", "second", "third"):
        assert client.post(path, json={"message": text, "session_id": sid}).status_code == 200

    assert _runs(tmp_path) == ["startup"]


def test_a_refused_slash_command_leaves_the_hook_to_the_first_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused slash command runs no turn, so the chat has not started: the message after
    it is the one that runs the start-up hook."""

    async def refuse(self: Agent, name: str, args: str = "", **_: Any) -> SkillInvocation:
        return SkillInvocation(invoked=True, name=name, denied_reason=f"{name} is refused")

    monkeypatch.setattr(Agent, "compose_skill_turn", refuse)
    app, store = _serve(tmp_path)
    _say(app, tmp_path, ["/hidden", "first", "second"])

    assert _runs(tmp_path) == ["startup"]
    assert len(_words(store, tmp_path)) == 1


def test_a_first_message_stopped_while_the_hook_runs_leaves_it_to_the_next(
    tmp_path: Path,
) -> None:
    """Stop pressed while the start-up hook runs kills it before it says anything: the chat has
    not heard it, so the next message runs it again."""
    app, store = _serve(tmp_path)
    slow = tmp_path / "slow-hook"
    slow.touch()
    _say_stopped_then(app, tmp_path, until=lambda: not slow.exists(), then="second")

    assert _runs(tmp_path) == ["startup", "startup"]
    assert len(_words(store, tmp_path)) == 1


def test_a_first_message_stopped_before_the_hook_s_words_are_added_leaves_them_to_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook ran to its end, but Stop came while the turn compacted, before what the hook
    said was added to the chat: the words died with that turn's agent, so the next message
    runs the hook again and they reach the chat once."""
    compacting: list[bool] = []
    compact = AgentLoop._maybe_compact

    async def stall_first(self: AgentLoop) -> Any:
        if not compacting:
            compacting.append(True)
            await asyncio.sleep(60)  # the Stop cancels the turn here
        return await compact(self)

    monkeypatch.setattr(AgentLoop, "_maybe_compact", stall_first)
    app, store = _serve(tmp_path)
    _say_stopped_then(app, tmp_path, until=lambda: bool(compacting), then="second")

    assert _runs(tmp_path) == ["startup", "startup"]
    assert len(_words(store, tmp_path)) == 1
