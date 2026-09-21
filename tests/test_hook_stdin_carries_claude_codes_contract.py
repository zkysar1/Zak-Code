"""Every shell hook's stdin carries what Claude Code hands a hook, and PostToolUse carries
``tool_response`` (ADR-0210).

Why this file exists. A framework written for Claude Code hangs a reminder on its
PostToolUse[Bash] hook: when the script that closes a lap of its loop has really finished, the
hook reads the script's closing marker in ``tool_response.stdout`` and tells the model what
to call next. Missing field, no reminder, BY DESIGN (a false reminder would pull the model
off live work). Zak Code sent ``output`` and ``is_error`` and no ``tool_response`` at all, so
under Zak Code that reminder never fired and nothing anywhere said so. The compat map had
claimed the field since its first version; no test had ever asked for it.

So the tests here do not inspect a dict the harness built. A REAL scripted agent runs the
REAL shell tool, and a hook SCRIPT reads its own stdin the way a Claude Code hook does:
the consumer's predicate, fed the product's own bytes. Hermetic: tmp workspaces, scripted
provider, no network. The hook bodies are synthesized; no framework's text is quoted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.hooks import (
    HookEvent,
    HookManager,
    HookPayload,
    HookSpec,
    claude_code_event_name,
    claude_code_tool_response,
    wire_payload,
)
from zakcode.permissions import PermissionPolicy
from zakcode.tools.base import STDOUT_CHARS, ToolContext
from zakcode.tools.builtins.bash import BashTool

#: A hook that writes down what it was handed, plus what the transcript it was pointed at
#: held AT THAT MOMENT (later flushes add rows, so only the hook itself can say).
_RECORDER = """
import json, sys
from pathlib import Path
doc = json.load(sys.stdin)
path = doc.get("transcript_path") or ""
rows = Path(path).read_text(encoding="utf-8").splitlines() if path and Path(path).exists() else []
doc["_transcript_exists"] = bool(path) and Path(path).exists()
doc["_transcript_names_the_call"] = any("echo hello" in row for row in rows)
with open({seen!r}, "a", encoding="utf-8") as out:
    out.write(json.dumps(doc) + "\\n")
"""

#: The consumer the change is for, in its own shape: speak ONLY when the result is Claude
#: Code's object and its stdout carries the marker. Anything else, say nothing.
_REMINDER = """
import json, re, sys
doc = json.load(sys.stdin)
response = doc.get("tool_response")
if isinstance(response, dict) and isinstance(response.get("stdout"), str):
    if re.search(r"LAP CLOSED", response["stdout"]):
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
              "additionalContext": "REMINDER: call the loop again"}}))
"""

#: A hook that PARSES the command's stdout, as a hook written for Claude Code may.
_PARSER = """
import json, sys
doc = json.load(sys.stdin)
parsed = json.loads(doc["tool_response"]["stdout"])
print(json.dumps({"hookSpecificOutput": {"additionalContext": "PARSED ok=%s" % parsed["ok"]}}))
"""


def _hook(tmp_path: Path, name: str, body: str, event: HookEvent) -> HookSpec:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return HookSpec(event=event, command=[sys.executable, str(path)])


def _agent(tmp_path: Path, script: list, hooks: list[HookSpec]) -> Agent:
    return Agent(
        provider=ScriptedProvider(script=script),
        permission_policy=PermissionPolicy("allow"),
        hook_manager=HookManager(hooks),
        default_model="scripted/test",
        workspace_root=str(tmp_path),
        max_iterations=6,
    )


def _seen(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ── PostToolUse: the shell tool's result, in Claude Code's shape ──────────────────────────


async def test_a_posttooluse_hook_is_handed_claude_codes_tool_response(tmp_path: Path) -> None:
    seen = tmp_path / "seen.jsonl"
    recorder = _hook(tmp_path, "rec.py", _RECORDER.format(seen=str(seen)), HookEvent.POST_TOOL_USE)
    agent = _agent(
        tmp_path, [call_tool("bash", {"command": "echo hello"}), reply("done")], [recorder]
    )
    await agent.arun_turn("run it")
    await agent.aclose()
    (doc,) = _seen(seen)
    response = doc["tool_response"]
    assert set(response) == {"stdout", "stderr", "interrupted", "isImage"}, response
    assert response["stdout"].strip() == "hello"
    assert "[exit code" not in response["stdout"]  # the harness's footer is not the command's
    assert (response["stderr"], response["interrupted"], response["isImage"]) == ("", False, False)
    assert doc["hook_event_name"] == "PostToolUse" and doc["tool_name"] == "Bash"
    # Zak Code's own keys stay beside it, unchanged: the text the model reads, footer and all.
    assert doc["is_error"] is False and doc["output"].rstrip().endswith("[exit code: 0]")
    assert doc["output"].startswith(response["stdout"])


async def test_the_reminder_fires_on_a_finished_command_and_not_on_a_failed_one(
    tmp_path: Path,
) -> None:
    """The consumer's own predicate, both ways. The second half is the control that must NOT
    flip: a command that printed the marker and then FAILED is no finished lap, Claude Code
    hands a failed call's result over as text, and so do we."""
    reminder = _hook(tmp_path, "rem.py", _REMINDER, HookEvent.POST_TOOL_USE)
    agent = _agent(
        tmp_path,
        [
            call_tool("bash", {"command": 'echo "LAP CLOSED"'}, id="c1"),
            call_tool("bash", {"command": 'echo "LAP CLOSED"; exit 3'}, id="c2"),
            reply("done"),
        ],
        [reminder],
    )
    result = await agent.arun_turn("run it")
    await agent.aclose()
    by_id = {block.tool_use_id: block.output or "" for block in result.tool_results}
    assert "LAP CLOSED" in by_id["c1"] and "LAP CLOSED" in by_id["c2"], by_id  # both really ran
    assert "REMINDER: call the loop again" in by_id["c1"], by_id["c1"]
    assert "REMINDER" not in by_id["c2"], by_id["c2"]


async def test_a_hook_that_parses_stdout_never_meets_the_exit_code_line(tmp_path: Path) -> None:
    (tmp_path / "emit.py").write_text(
        'import json; print(json.dumps({"ok": True}))', encoding="utf-8"
    )
    parser = _hook(tmp_path, "parse.py", _PARSER, HookEvent.POST_TOOL_USE)
    command = f'"{Path(sys.executable).as_posix()}" emit.py'
    agent = _agent(tmp_path, [call_tool("bash", {"command": command}), reply("done")], [parser])
    result = await agent.arun_turn("run it")
    await agent.aclose()
    outputs = [block.output or "" for block in result.tool_results]
    assert any("PARSED ok=True" in out for out in outputs), outputs


async def test_a_tool_other_than_the_shell_hands_over_its_result_text(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("a line worth reading\n", encoding="utf-8")
    seen = tmp_path / "seen.jsonl"
    recorder = _hook(tmp_path, "rec.py", _RECORDER.format(seen=str(seen)), HookEvent.POST_TOOL_USE)
    agent = _agent(tmp_path, [call_tool("Read", {"path": "note.txt"}), reply("done")], [recorder])
    await agent.arun_turn("read it")
    await agent.aclose()
    (doc,) = _seen(seen)
    assert doc["tool_name"] == "Read"
    assert isinstance(doc["tool_response"], str) and doc["tool_response"] == doc["output"]
    assert "a line worth reading" in doc["tool_response"]


# ── PreToolUse: the event's name, the transcript, and no response yet ──────────────────────


async def test_a_pretooluse_hook_reads_the_call_in_the_transcript_it_is_handed(
    tmp_path: Path,
) -> None:
    seen = tmp_path / "seen.jsonl"
    recorder = _hook(tmp_path, "rec.py", _RECORDER.format(seen=str(seen)), HookEvent.PRE_TOOL_USE)
    agent = _agent(
        tmp_path, [call_tool("bash", {"command": "echo hello"}), reply("done")], [recorder]
    )
    await agent.arun_turn("run it")
    await agent.aclose()
    (doc,) = _seen(seen)
    assert doc["hook_event_name"] == "PreToolUse"
    assert "tool_response" not in doc  # Claude Code's PreToolUse stdin has no such key
    assert doc["transcript_path"] and doc["_transcript_exists"]
    # As under Claude Code, the file already holds the assistant message that made this call.
    assert doc["_transcript_names_the_call"]


# ── every shell hook: the fields Claude Code hands all of them ─────────────────────────────


async def test_every_shell_hooks_stdin_carries_the_common_fields(tmp_path: Path) -> None:
    seen = tmp_path / "seen.jsonl"
    body = _RECORDER.format(seen=str(seen))
    events = [
        HookEvent.SESSION_START,
        HookEvent.USER_PROMPT_SUBMIT,
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.TURN_END,
        HookEvent.SESSION_END,
    ]
    hooks = [_hook(tmp_path, f"rec{n}.py", body, event) for n, event in enumerate(events)]
    agent = _agent(tmp_path, [call_tool("bash", {"command": "echo hello"}), reply("done")], hooks)
    await agent.arun_turn("run it")
    await agent.aclose()
    docs = _seen(seen)
    names = [doc.get("hook_event_name") for doc in docs]
    # Claude Code's names, in the order a one-call turn fires them. The turn's end is "Stop".
    assert names == [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "SessionEnd",
    ], names
    for doc in docs:
        name = doc["hook_event_name"]
        assert doc.get("session_id") == agent.session.id, name
        assert doc.get("cwd") == str(tmp_path), name
        assert doc.get("transcript_path") and doc["_transcript_exists"], name
    assert len({doc["transcript_path"] for doc in docs}) == 1  # one session, one file


# ── the builders, at their edges ───────────────────────────────────────────────────────────


def test_the_event_names_are_claude_codes() -> None:
    assert claude_code_event_name(HookEvent.TURN_END) == "Stop"
    same = (
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.SESSION_START,
        HookEvent.SESSION_END,
        HookEvent.PRE_COMPACT,
        HookEvent.USER_PROMPT_SUBMIT,
    )
    assert [claude_code_event_name(e) for e in same] == [e.value for e in same]


@pytest.mark.parametrize(
    ("count", "stdout"),
    [
        pytest.param(4, "out\n", id="the-tools-count"),  # the command's own output, no footer
        pytest.param(0, "", id="printed-nothing"),
        pytest.param(None, "out\n[exit code: 0]", id="no-count"),  # the whole text, nothing guessed
        pytest.param(99, "out\n[exit code: 0]", id="past-the-end"),
        pytest.param(-1, "out\n[exit code: 0]", id="negative"),
        pytest.param(True, "out\n[exit code: 0]", id="a-bool"),  # an int subclass, and no count
    ],
)
def test_stdout_is_cut_where_the_tool_says_and_nowhere_else(count: object, stdout: str) -> None:
    text = "out\n[exit code: 0]"
    response = claude_code_tool_response("Bash", text, False, stdout_chars=count)  # type: ignore[arg-type]
    assert response == {"stdout": stdout, "stderr": "", "interrupted": False, "isImage": False}


def test_a_failed_call_and_every_other_tool_carry_the_text() -> None:
    assert claude_code_tool_response("Bash", "boom\n[exit code: 3]", True, stdout_chars=5) == (
        "boom\n[exit code: 3]"
    )
    assert claude_code_tool_response("Read", "1\tline", False) == "1\tline"
    # a payload built before ADR-0190 still names the shell tool the old way
    assert claude_code_tool_response("bash", "x", False, stdout_chars=1)["stdout"] == "x"


def test_the_wire_places_tool_response_on_posttooluse_only(tmp_path: Path) -> None:
    base = {"tool_name": "Bash", "arguments": {"command": "true"}, "cwd": str(tmp_path)}
    pre = json.loads(wire_payload(HookPayload(event=HookEvent.PRE_TOOL_USE, **base)))
    assert "tool_response" not in pre and pre["hook_event_name"] == "PreToolUse"
    # built outside the loop, with no response object: the result text stands in
    post = json.loads(
        wire_payload(
            HookPayload(event=HookEvent.POST_TOOL_USE, output="text", is_error=False, **base)
        )
    )
    assert post["tool_response"] == "text" and post["hook_event_name"] == "PostToolUse"
    shaped = HookPayload(
        event=HookEvent.POST_TOOL_USE,
        output="o",
        is_error=False,
        tool_response={"stdout": "o"},
        **base,
    )
    assert json.loads(wire_payload(shaped))["tool_response"] == {"stdout": "o"}


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("echo hello", id="echo"),
        pytest.param("printf no-newline", id="printf"),
        pytest.param("true", id="true"),
    ],
)
async def test_the_shell_tool_counts_its_own_output_exactly(tmp_path: Path, command: str) -> None:
    result = await BashTool().execute({"command": command}, ToolContext(workspace_root=tmp_path))
    assert result.data is not None and not result.is_error
    own = result.output[: result.data[STDOUT_CHARS]]
    assert "[exit code" not in own
    assert result.output[result.data[STDOUT_CHARS] :].strip() == "[exit code: 0]"
    assert (
        own.strip()
        == {"echo hello": "hello", "printf no-newline": "no-newline", "true": ""}[command]
    )
