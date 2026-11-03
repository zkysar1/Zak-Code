"""Every turn of a served chat hands the workspace's tool hooks that chat's one session id.

Why this file exists. A framework written for Claude Code can keep who an agent is in a record
filed under the session id: its start command files the agent under the id a PreToolUse[Bash]
hook hands every command, and every later command finds the agent through that same id. The web
app runs each message on an agent of its own (ADR-0342), so if a turn's agent handed its hooks an
id of its own, the agent a person started with one message would be gone by the next. Nothing
would fail; the record would just never be found again.

So these tests serve a REAL agent per turn, as the app does, with a hook in that framework's
shape: it writes down the session id it is handed and puts it into the command, which prints
it. Hermetic: a tmp workspace, the scripted provider, no network.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from zakcode import Agent
from zakcode.config import Settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.hooks import HookEvent, HookManager, HookSpec
from zakcode.permissions import PermissionPolicy
from zakcode.server.app import create_app
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore

MODEL = "scripted/test"
MESSAGES = ("start the agent", "which agent are you?")

#: A PreToolUse[Bash] hook in the shape a framework's takes: it writes down the session id it is
#: handed and puts it into the command as an exported variable.
_HOOK = """
import json, sys
doc = json.load(sys.stdin)
sid = doc.get("session_id") or ""
with open({log!r}, "a", encoding="utf-8") as out:
    out.write(json.dumps({{"event": doc.get("hook_event_name"), "session": sid}}) + "\\n")
command = (doc.get("tool_input") or {{}}).get("command", "")
print(json.dumps({{"hookSpecificOutput": {{
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "updatedInput": {{"command": "export BOUND_SID=%s; %s" % (sid, command)}},
}}}}))
"""


def _serve(workspace: Path) -> tuple[Any, SessionStore]:
    """A server on ``workspace`` whose every turn runs on a real agent of its own, as the web
    app's does. Each turn runs one shell command, which prints the id its hook handed it."""
    script = workspace / "bind_hook.py"
    script.write_text(_HOOK.format(log=str(workspace / "hook-runs.jsonl")), encoding="utf-8")
    hook = HookSpec(
        event=HookEvent.PRE_TOOL_USE, command=[sys.executable, str(script)], matcher="Bash"
    )
    settings = Settings(default_model=MODEL, context_window=8192, workspace_root=workspace)
    store = SessionStore(base_dir=workspace / "sessions")

    def factory(session: Session, model: str | None, prompter: Any) -> Agent:  # noqa: ARG001
        return Agent(
            session=session,
            settings=settings,
            session_store=store,
            provider=ScriptedProvider(
                script=[call_tool("bash", {"command": 'echo "bound=$BOUND_SID"'}), reply("ok")]
            ),
            permission_policy=PermissionPolicy("allow"),
            hook_manager=HookManager([hook]),
        )

    return create_app(settings=settings, store=store, agent_factory=factory), store


def _hook_runs(workspace: Path) -> list[dict[str, str]]:
    log = workspace / "hook-runs.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return [json.loads(line) for line in lines if line]


def _printed(store: SessionStore, chat: str) -> list[str]:
    """What each turn's command printed, in order: the id its hook handed it."""
    outputs = [
        block.output
        for message in store.load(chat).messages
        for block in message.blocks
        if block.type == "tool_result"
    ]
    return [
        line.removeprefix("bound=")
        for output in outputs
        for line in output.splitlines()
        if line.startswith("bound=")
    ]


def test_two_messages_in_one_chat_hand_its_tool_hooks_one_session_id(tmp_path: Path) -> None:
    """The web page's path: each message goes through the say slot to a turn of its own."""
    app, store = _serve(tmp_path)

    async def turns() -> None:
        for text in MESSAGES:
            assert write_say(say_path(tmp_path), text)
            assert await app.state.consume_one_say() is True

    asyncio.run(turns())

    chat = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    assert chat
    runs = _hook_runs(tmp_path)
    assert [run["event"] for run in runs] == ["PreToolUse", "PreToolUse"]
    assert [run["session"] for run in runs] == [chat, chat]
    assert _printed(store, chat) == [chat, chat]


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_two_messages_a_driver_sends_to_one_chat_hand_its_hooks_one_session_id(
    tmp_path: Path, path: str
) -> None:
    """A driver's path: both messages name the chat they belong to."""
    app, store = _serve(tmp_path)
    client = TestClient(app)
    chat = client.post("/sessions").json()["id"]
    for text in MESSAGES:
        assert client.post(path, json={"message": text, "session_id": chat}).status_code == 200

    assert [run["session"] for run in _hook_runs(tmp_path)] == [chat, chat]
    assert _printed(store, chat) == [chat, chat]
