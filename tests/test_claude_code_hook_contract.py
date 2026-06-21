"""Zak Code honors the **Claude Code hook contract**, so Claude-Code-targeted frameworks
(e.g. claude-mind) run on it unmodified.

The contract, in five parts (each a fix proven here):
  1. shell hooks run at the **workspace cwd** (so a hook's relative ``bash core/scripts/...``
     resolves), guarded against a bogus cwd;
  2. a hook's stdout ``hookSpecificOutput.updatedInput`` rewrites the tool args, and
     ``permissionDecision: deny`` blocks on exit 0 (Claude Code blocks via JSON, not exit 2);
  3. shell executables resolve to a real path on Windows (dodging the WSL app-exec stub);
  4. the hook stdin carries a top-level ``session_id``;
  5. the hook stdin names tool args ``tool_input`` (not Zak-native ``arguments``).

All hermetic: tmp-dir workspaces, scripted Python hooks, no network.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from zakcode._subprocess import find_bash, resolve_executable
from zakcode.hooks import (
    HookEvent,
    HookManager,
    HookPayload,
    HookSpec,
    _valid_cwd,
)
from zakcode.tools.builtins._proc import run_capturing


def _script(tmp_path: Path, name: str, body: str) -> list[str]:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return [sys.executable, str(path)]


def _payload(tmp_path: Path, **kw: object) -> HookPayload:
    base: dict = {
        "event": HookEvent.PRE_TOOL_USE,
        "tool_name": "bash",
        "arguments": {"command": "load-conventions.sh"},
        "cwd": str(tmp_path),
        "session_id": "sid-123",
    }
    base.update(kw)
    return HookPayload(**base)


# ── (1) cwd guard ────────────────────────────────────────────────────────────────


def test_valid_cwd(tmp_path: Path) -> None:
    assert _valid_cwd(str(tmp_path)) == str(tmp_path)  # real dir → used
    assert _valid_cwd(str(tmp_path / "nope")) is None  # missing → inherit
    assert _valid_cwd("") is None  # empty → inherit


# ── (2) Claude Code stdout shape: _parse_stdout returns (message, mutated, deny) ──


def test_parse_stdout_native_arguments() -> None:
    msg, mut, deny = HookManager._parse_stdout(
        json.dumps({"message": "m", "arguments": {"a": 1}}).encode()
    )
    assert msg == "m" and mut == {"a": 1} and deny is False


def test_parse_stdout_claude_updated_input() -> None:
    doc = {"hookSpecificOutput": {"updatedInput": {"command": "safe"}}}
    _msg, mut, deny = HookManager._parse_stdout(json.dumps(doc).encode())
    assert mut == {"command": "safe"} and deny is False


def test_parse_stdout_claude_deny_blocks_on_exit_zero() -> None:
    doc = {"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "nope"}}
    msg, _mut, deny = HookManager._parse_stdout(json.dumps(doc).encode())
    assert deny is True and msg == "nope"


def test_parse_stdout_plain_and_empty() -> None:
    assert HookManager._parse_stdout(b"hi") == ("hi", None, False)
    assert HookManager._parse_stdout(b"") == ("", None, False)


# ── (3) executable resolution (Windows WSL-stub avoidance; pure on POSIX) ─────────


def test_find_bash_returns_path_or_none() -> None:
    b = find_bash()
    assert b is None or (isinstance(b, str) and b)


def test_resolve_executable_passes_through_paths_and_unknowns(tmp_path: Path) -> None:
    p = str(tmp_path / "x")
    assert resolve_executable(p) == p  # already a path → unchanged
    assert resolve_executable("not-a-real-cmd-zzz") == "not-a-real-cmd-zzz"  # unresolvable → as-is


def test_resolve_executable_resolves_bash_when_present() -> None:
    if shutil.which("bash"):
        r = resolve_executable("bash")
        assert os.path.isabs(r), f"bash should resolve to an absolute path, got {r!r}"


# ── (5) HookPayload serializes the Claude Code wire shape ────────────────────────


def test_hookpayload_serializes_tool_input_and_session_id(tmp_path: Path) -> None:
    p = _payload(tmp_path)
    wire = json.loads(p.model_dump_json(by_alias=True))
    assert wire["tool_input"] == {"command": "load-conventions.sh"}  # not "arguments"
    assert wire["session_id"] == "sid-123"
    assert "arguments" not in wire
    assert p.arguments == {"command": "load-conventions.sh"}  # in-process attribute unchanged


# ── end-to-end shell hooks (1 + 2 + 4 + 5 together) ──────────────────────────────


async def test_shell_hook_receives_claude_code_stdin(tmp_path: Path) -> None:
    # The hook denies IFF it received tool_input.command AND session_id — the Claude Code shape.
    body = (
        "import sys, json\n"
        "d = json.load(sys.stdin)\n"
        "cmd = (d.get('tool_input') or {}).get('command')\n"
        "sid = d.get('session_id')\n"
        "if cmd and sid:\n"
        "    print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'deny',\n"
        "        'permissionDecisionReason': sid + ':' + cmd}}))\n"
    )
    mgr = HookManager(
        [HookSpec(event=HookEvent.PRE_TOOL_USE, command=_script(tmp_path, "h.py", body))]
    )
    res = await mgr.run(_payload(tmp_path))
    assert res.blocked
    assert "sid-123:load-conventions.sh" in res.message


async def test_shell_hook_updated_input_rewrites_command(tmp_path: Path) -> None:
    body = (
        "import sys, json; sys.stdin.read()\n"
        "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'allow',\n"
        "    'updatedInput': {'command': 'export X=1; load-conventions.sh'}}}))\n"
    )
    mgr = HookManager(
        [HookSpec(event=HookEvent.PRE_TOOL_USE, command=_script(tmp_path, "inj.py", body))]
    )
    res = await mgr.run(_payload(tmp_path))
    assert not res.blocked
    assert res.mutated_arguments == {"command": "export X=1; load-conventions.sh"}


async def test_shell_hook_runs_at_workspace_cwd(tmp_path: Path) -> None:
    # The hook reports its own cwd; assert it ran at the workspace, not Zak Code's directory.
    body = (
        "import sys, json, os; sys.stdin.read()\n"
        "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'deny',\n"
        "    'permissionDecisionReason': os.getcwd()}}))\n"
    )
    mgr = HookManager(
        [HookSpec(event=HookEvent.PRE_TOOL_USE, command=_script(tmp_path, "pwd.py", body))]
    )
    res = await mgr.run(_payload(tmp_path))
    assert res.blocked
    assert Path(res.message).resolve() == tmp_path.resolve()


# ── the shell tool runner executes a real command (Git Bash on Windows) ──────────


async def test_run_capturing_executes_shell_command(tmp_path: Path) -> None:
    out, code = await run_capturing(
        shell_command="echo claude_code_contract", cwd=str(tmp_path), timeout=30
    )
    assert code == 0
    assert "claude_code_contract" in out
