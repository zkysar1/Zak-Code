"""Zak Code honors the **Claude Code hook contract**, so Claude-Code-targeted frameworks
(e.g. claude-mind) run on it unmodified.

The contract, in five parts (each a fix proven here):
  1. shell hooks run at the **workspace cwd** (so a hook's relative ``bash core/scripts/...``
     resolves), guarded against a bogus cwd;
  2. a hook's stdout ``hookSpecificOutput.updatedInput`` rewrites the tool args, and
     ``permissionDecision: deny`` blocks on exit 0 (Claude Code blocks via JSON, not exit 2);
  3. shell executables resolve to a real path on Windows (dodging the WSL app-exec stub);
  4. the hook stdin carries a top-level ``session_id``;
  5. the hook stdin names tool args ``tool_input`` (not Zak-native ``arguments``);
  6. the hook stdin names the tool as Claude Code does (``write_file`` → ``Write``) and the
     file tools' argument ``file_path`` (workspace-resolved); an ``updatedInput`` rewrite
     maps back onto ``path`` (ADR-0071).

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
    wire_payload,
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


# ── (2) Claude Code stdout shape: _parse_stdout returns (message, mutated, deny, additional) ──


def test_parse_stdout_native_arguments() -> None:
    msg, mut, deny, _extra = HookManager._parse_stdout(
        json.dumps({"message": "m", "arguments": {"a": 1}}).encode()
    )
    assert msg == "m" and mut == {"a": 1} and deny is False


def test_parse_stdout_claude_updated_input() -> None:
    doc = {"hookSpecificOutput": {"updatedInput": {"command": "safe"}}}
    _msg, mut, deny, _extra = HookManager._parse_stdout(json.dumps(doc).encode())
    assert mut == {"command": "safe"} and deny is False


def test_parse_stdout_claude_deny_blocks_on_exit_zero() -> None:
    doc = {"hookSpecificOutput": {"permissionDecision": "deny", "permissionDecisionReason": "nope"}}
    msg, _mut, deny, _extra = HookManager._parse_stdout(json.dumps(doc).encode())
    assert deny is True and msg == "nope"


def test_parse_stdout_plain_and_empty() -> None:
    assert HookManager._parse_stdout(b"hi") == ("hi", None, False, "")
    assert HookManager._parse_stdout(b"") == ("", None, False, "")


def test_parse_stdout_posttooluse_additional_context() -> None:
    # PostToolUse hooks inject context via hookSpecificOutput.additionalContext (4th tuple element).
    doc = {"hookSpecificOutput": {"additionalContext": "more info"}}
    _msg, _mut, _deny, additional = HookManager._parse_stdout(json.dumps(doc).encode())
    assert additional == "more info"


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


# ── (6) a Claude-Code matcher fires on the equivalent Zak Code tool ──────────────


def _spec(matcher: str) -> HookSpec:
    return HookSpec(event=HookEvent.PRE_TOOL_USE, command=["x"], matcher=matcher)


def test_matcher_fires_on_claude_code_tool_names() -> None:
    # A matcher written for Claude Code fires on the corresponding Zak Code tool call, so a
    # framework's PreToolUse gates (e.g. claude-mind's skill-dedup gate) apply unchanged.
    assert _spec("Skill").matches("use_skill")
    assert _spec("Read").matches("read_file")
    assert _spec("Bash").matches("bash")  # case-correct cross-platform (fnmatch is POSIX-sensitive)
    assert _spec("Edit").matches("edit_file")
    assert _spec("MultiEdit").matches("edit_file")
    # Specificity preserved: a Claude-Code matcher does NOT fire on an unrelated tool.
    assert not _spec("Skill").matches("read_file")
    assert not _spec("Bash").matches("read_file")
    # The tool's own (Zak Code) name still matches, and "*" still matches everything.
    assert _spec("use_skill").matches("use_skill")
    assert _spec("*").matches("anything")


# ── (7) Claude Code's $CLAUDE_PROJECT_DIR (the project root) ──────────────────────


def test_hook_command_expands_claude_project_dir(tmp_path: Path) -> None:
    # A Claude-Code hook command referencing $CLAUDE_PROJECT_DIR is substituted with the workspace
    # root (forward-slash) at load — we run hook argv without a shell, so nothing else expands it.
    from zakcode.hooks.settings_loader import load_settings_hooks

    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {"type": "command", "command": "bash $CLAUDE_PROJECT_DIR/core/x.sh"}
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    specs, _errors = load_settings_hooks(tmp_path)
    assert specs
    expanded = specs[0].command[-1]
    assert "$CLAUDE_PROJECT_DIR" not in expanded
    assert expanded == f"{tmp_path.as_posix()}/core/x.sh"


async def test_shell_hook_env_has_claude_project_dir(tmp_path: Path) -> None:
    # The hook reports $CLAUDE_PROJECT_DIR from its environment; assert it is the workspace root.
    body = (
        "import sys, json, os; sys.stdin.read()\n"
        "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'deny',\n"
        "    'permissionDecisionReason': os.environ.get('CLAUDE_PROJECT_DIR', '<unset>')}}))\n"
    )
    mgr = HookManager(
        [HookSpec(event=HookEvent.PRE_TOOL_USE, command=_script(tmp_path, "env.py", body))]
    )
    res = await mgr.run(_payload(tmp_path))
    assert res.blocked
    assert res.message == tmp_path.as_posix()


# ── (8) the wire names the tool and its file argument the way Claude Code does ──────


def test_wire_payload_uses_claude_code_tool_name_and_file_path(tmp_path: Path) -> None:
    # A Claude-Code hook keys off tool_name == "Write" and tool_input.file_path; Zak Code's
    # write_file/path shape was approved unread. The relative path is resolved against the
    # workspace (the way the tool resolves it) so a path gate judges the real target.
    wire = json.loads(
        wire_payload(
            _payload(
                tmp_path,
                tool_name="write_file",
                arguments={"path": "world/knowledge/x.md", "content": "c"},
            )
        )
    )
    assert wire["tool_name"] == "Write"
    assert wire["tool_input"] == {
        "file_path": os.path.normpath(os.path.join(str(tmp_path), "world/knowledge/x.md")),
        "content": "c",
    }
    assert "path" not in wire["tool_input"] and "arguments" not in wire
    assert wire["session_id"] == "sid-123"
    # An absolute path passes through untouched; edit_file is named "Edit" (its first alias).
    wire = json.loads(
        wire_payload(
            _payload(
                tmp_path,
                tool_name="edit_file",
                arguments={"path": str(tmp_path / "a.md"), "old_string": "x", "new_string": "y"},
            )
        )
    )
    assert wire["tool_name"] == "Edit"
    assert wire["tool_input"] == {
        "file_path": str(tmp_path / "a.md"),
        "old_string": "x",
        "new_string": "y",
    }
    # bash keeps `command` under its CC name; a tool with no counterpart keeps its own shape.
    wire = json.loads(wire_payload(_payload(tmp_path)))
    assert wire["tool_name"] == "Bash"
    assert wire["tool_input"] == {"command": "load-conventions.sh"}
    wire = json.loads(wire_payload(_payload(tmp_path, tool_name="deep_think", arguments={"q": 1})))
    assert wire["tool_name"] == "deep_think"
    assert wire["tool_input"] == {"q": 1}


async def test_shell_hook_gates_write_file_by_file_path_and_maps_the_rewrite_back(
    tmp_path: Path,
) -> None:
    # A Claude-Code path gate: denies a write under <workspace>/world/ (the cruft claude-mind's
    # L1 hook refuses), otherwise rewrites file_path. Zak Code must deliver the deny for its
    # own write_file call, and map the rewrite back onto the tool's `path` argument.
    body = (
        "import sys, json\n"
        "d = json.load(sys.stdin)\n"
        "fp = (d.get('tool_input') or {}).get('file_path', '')\n"
        "if d.get('tool_name') != 'Write' or not fp:\n"
        "    sys.exit(0)\n"
        "if '/world/' in fp or '\\\\world\\\\' in fp:\n"
        "    print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'deny',\n"
        "        'permissionDecisionReason': 'cruft: ' + fp}}))\n"
        "else:\n"
        "    print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'allow',\n"
        "        'updatedInput': {'file_path': fp + '.rewritten', 'content': 'c'}}}))\n"
    )
    mgr = HookManager(
        [HookSpec(event=HookEvent.PRE_TOOL_USE, command=_script(tmp_path, "gate.py", body))]
    )
    denied = await mgr.run(
        _payload(
            tmp_path,
            tool_name="write_file",
            arguments={"path": "world/knowledge/x.md", "content": "c"},
        )
    )
    assert denied.blocked
    assert "cruft: " in denied.message and "world" in denied.message

    allowed = await mgr.run(
        _payload(tmp_path, tool_name="write_file", arguments={"path": "notes/x.md", "content": "c"})
    )
    assert not allowed.blocked
    expected = os.path.normpath(os.path.join(str(tmp_path), "notes/x.md")) + ".rewritten"
    assert allowed.mutated_arguments == {"path": expected, "content": "c"}
