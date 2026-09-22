"""ADR-0212: the workspace's settings ``env`` block reaches every child a session starts.

Claude Code sets the ``env`` object of a workspace's settings files "for every session and
its subprocesses". A framework written for that harness keeps its own switches there,
because that block is the one configuration that travels with the repository. Zak Code had
no reader for it, so on a hosted workspace every such switch was silently OFF in every
shell command, background task, hook, status line and framework-script call — and nothing
said so.

These tests spawn REAL children and read back what the child actually saw. They also pin
the ORDER, which is the safety half of the change: the block is overlaid on the inherited
environment, and everything the product sets for a child — the no-colour pair, the egress
proxy, ``CLAUDE_PROJECT_DIR``, the agent's name — is applied after it, with the provider-key
scrub last. A workspace's settings file can switch a framework on; it cannot point a child
around the egress sandbox, redirect it to another project, or hand it a model credential.

The variable names here are synthesized (``ZC_PROBE`` and friends). Which names a real
framework keeps in its own settings file is that framework's business, not a fixture's.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

from zakcode.background import BackgroundTasks
from zakcode.hooks import HookEvent, HookManager, HookPayload, HookSpec
from zakcode.session.framework_signal import MODE_GET_SCRIPT, framework_agent_mode
from zakcode.session.store import Session
from zakcode.status_line import StatusLineSpec, build_status_input, render_status_line
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.bash import BashTool
from zakcode.workspace_env import settings_env

PROBE = "ZC_PROBE"
SHARED = ".claude/settings.json"
LOCAL = ".claude/settings.local.json"
ZAKCODE = ".zakcode/settings.json"


def _settings(workspace: Path, env: dict[str, object], *, where: str = SHARED) -> Path:
    """Write one of the three settings files the hook loader reads, carrying an ``env`` block."""
    path = workspace / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"env": env}), encoding="utf-8")
    return path


def _report(*names: str) -> str:
    """A shell command printing ``NAME[value]`` per name — ``<unset>`` when the child lacks it.

    One child reports every name the test cares about, so a precedence case is one spawn.
    """
    fmt = "".join(f"{name}[%s]" for name in names)
    args = " ".join(f'"${{{name}:-<unset>}}"' for name in names)
    return f'printf "{fmt}" {args}'


async def _shell(workspace: Path, command: str, ctx: ToolContext | None = None) -> str:
    """Run *command* through the real Bash tool and return its output."""
    res = await BashTool().execute(
        {"command": command}, ctx or ToolContext(workspace_root=workspace)
    )
    assert not res.is_error, res.output
    return res.output


async def _hook_sees(workspace: Path, name: str, *, drop_env: list[str] | None = None) -> str:
    """What a real shell hook's child sees in *name*: it reports the value as its deny reason."""
    script = workspace / f"hook_reporting_{name}.py"
    script.write_text(
        "import json, os, sys\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'hookSpecificOutput': {'permissionDecision': 'deny',\n"
        f"    'permissionDecisionReason': os.environ.get({name!r}, '<unset>')}}}}))\n",
        encoding="utf-8",
    )
    mgr = HookManager(
        [
            HookSpec(
                event=HookEvent.PRE_TOOL_USE,
                command=[sys.executable, str(script)],
                drop_env=drop_env or [],
            )
        ]
    )
    res = await mgr.run(
        HookPayload(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="bash",
            arguments={"command": "true"},
            cwd=str(workspace),
            session_id="sid-0212",
        )
    )
    assert res.blocked
    return res.message or ""


async def _status_line_sees(workspace: Path, name: str) -> str:
    """What the status-line command's child sees in *name*."""
    script = workspace / "status_reporting.py"
    script.write_text(
        f"import os\nprint('{name}[' + os.environ.get({name!r}, '<unset>') + ']')\n",
        encoding="utf-8",
    )
    spec = StatusLineSpec(command=[sys.executable, str(script)], timeout=15.0)
    line = await render_status_line(
        spec,
        build_status_input(
            session_id="sid-0212",
            cwd=str(workspace),
            model_id="openai/gpt-4o",
            cost_usd=0.0,
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            version="0.0.0",
        ),
    )
    return line or ""


def _plant_mode_get(workspace: Path, body: str) -> None:
    """Plant a stand-in for the framework's ``session-mode-get.sh`` at the real path.

    ``framework_agent_mode`` returns the script's stdout, so a script that echoes one
    variable turns that call into a reader for the framework-script child's environment.
    """
    script = workspace / MODE_GET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}\n", encoding="utf-8")


# ── the block reaches every kind of child ─────────────────────────────────────


async def test_a_shell_command_sees_the_block(tmp_path: Path) -> None:
    _settings(tmp_path, {PROBE: "from-the-settings"})
    assert f"{PROBE}[from-the-settings]" in await _shell(tmp_path, _report(PROBE))


async def test_a_background_shell_task_sees_the_block(tmp_path: Path) -> None:
    # ADR-0191's background lane spawns through the same builder as the foreground one;
    # this is the assertion that keeps the claim "every child" true of both.
    _settings(tmp_path, {PROBE: "from-the-settings"})
    session = Session(cwd=str(tmp_path), model="test")
    tasks = BackgroundTasks(session, tasks_dir=tmp_path / "tasks")
    task = await tasks.start(_report(PROBE), cwd=str(tmp_path))
    status, _ = await tasks.wait(task, 20.0)
    assert status != "running", "the background task never exited"
    assert f"{PROBE}[from-the-settings]" in tasks.output(task)


async def test_a_hook_script_sees_the_block(tmp_path: Path) -> None:
    _settings(tmp_path, {PROBE: "from-the-settings"})
    assert await _hook_sees(tmp_path, PROBE) == "from-the-settings"


async def test_a_status_line_command_sees_the_block(tmp_path: Path) -> None:
    _settings(tmp_path, {PROBE: "from-the-settings"})
    assert await _status_line_sees(tmp_path, PROBE) == f"{PROBE}[from-the-settings]"


def test_a_framework_script_sees_the_block(tmp_path: Path) -> None:
    _settings(tmp_path, {PROBE: "from-the-settings"})
    _plant_mode_get(tmp_path, f'printf "%s" "${{{PROBE}:-<unset>}}"')
    assert framework_agent_mode(tmp_path, "alpha") == "from-the-settings"


# ── precedence: the block beats what was inherited, and loses to what we set ───


async def test_the_block_beats_the_inherited_environment(tmp_path: Path, monkeypatch) -> None:
    # Claude Code's own precedence: the settings value wins over a variable the host exported.
    monkeypatch.setenv(PROBE, "from-the-shell")
    _settings(tmp_path, {PROBE: "from-the-settings"})
    assert f"{PROBE}[from-the-settings]" in await _shell(tmp_path, _report(PROBE))


async def test_the_files_merge_with_the_later_file_winning(tmp_path: Path) -> None:
    # Read order: .claude/settings.json, then .claude/settings.local.json (per-machine),
    # then .zakcode/settings.json. Later files win per NAME; names only an earlier file
    # carries survive, so a local override does not wipe the shared block.
    _settings(tmp_path, {"ZC_A": "shared", "ZC_B": "shared", "ZC_C": "shared", "ZC_D": "shared"})
    _settings(tmp_path, {"ZC_A": "local", "ZC_C": "local"}, where=LOCAL)
    _settings(tmp_path, {"ZC_B": "zakcode", "ZC_C": "zakcode"}, where=ZAKCODE)
    out = await _shell(tmp_path, _report("ZC_A", "ZC_B", "ZC_C", "ZC_D"))
    assert "ZC_A[local]" in out, "the per-machine file must beat the shared one"
    assert "ZC_B[zakcode]" in out
    assert "ZC_C[zakcode]" in out, "the last file read must win"
    assert "ZC_D[shared]" in out, "a name only the shared file carries must survive the merge"


async def test_the_products_own_child_variables_win(tmp_path: Path) -> None:
    # The no-colour pair exists because the child's output is fed to the model; a settings
    # file must not be able to turn ANSI escapes back on in what the model reads.
    _settings(tmp_path, {"NO_COLOR": "", "TERM": "xterm-256color"})
    out = await _shell(tmp_path, _report("NO_COLOR", "TERM"))
    assert "NO_COLOR[1]" in out
    assert "TERM[dumb]" in out


async def test_the_egress_variables_win_over_the_block(tmp_path: Path) -> None:
    _settings(tmp_path, {"HTTPS_PROXY": "http://the-block-says-this:1"})
    ctx = ToolContext(
        workspace_root=tmp_path, egress_env={"HTTPS_PROXY": "http://the-sandbox:8080"}
    )
    out = await _shell(tmp_path, _report("HTTPS_PROXY"), ctx)
    assert "HTTPS_PROXY[http://the-sandbox:8080]" in out, "a block must not reroute a child"


async def test_the_block_cannot_hand_a_shell_child_a_provider_key(tmp_path: Path) -> None:
    # The scrub is applied LAST, after the block, so a settings file cannot resurrect a
    # credential the session removed.
    _settings(tmp_path, {"OPENAI_API_KEY": "offline-test-value"})
    ctx = ToolContext(workspace_root=tmp_path, scrub_env=["OPENAI_API_KEY"])
    out = await _shell(tmp_path, _report("OPENAI_API_KEY"), ctx)
    assert "OPENAI_API_KEY[<unset>]" in out


async def test_the_block_cannot_hand_a_hook_a_provider_key(tmp_path: Path) -> None:
    _settings(tmp_path, {"OPENAI_API_KEY": "offline-test-value"})
    seen = await _hook_sees(tmp_path, "OPENAI_API_KEY", drop_env=["OPENAI_API_KEY"])
    assert seen == "<unset>"


async def test_the_block_cannot_tell_a_hook_the_project_is_elsewhere(tmp_path: Path) -> None:
    _settings(tmp_path, {"CLAUDE_PROJECT_DIR": "/somewhere/else"})
    assert await _hook_sees(tmp_path, "CLAUDE_PROJECT_DIR") == tmp_path.as_posix()


def test_the_agent_name_is_set_after_the_block(tmp_path: Path) -> None:
    # The framework's scripts resolve the agent from these two names; a settings file does
    # not get to say which agent is being read or stopped.
    _settings(tmp_path, {"AYOAI_AGENT": "not-this-one", "MIND_AGENT": "nor-this-one"})
    _plant_mode_get(tmp_path, 'printf "%s/%s" "${AYOAI_AGENT}" "${MIND_AGENT}"')
    assert framework_agent_mode(tmp_path, "alpha") == "alpha/alpha"


# ── freshness, and what a bad edit does ───────────────────────────────────────


async def test_a_saved_edit_reaches_the_next_child_with_no_restart(tmp_path: Path) -> None:
    # Claude Code applies a saved settings change to the running session; so does this.
    # The two values differ in LENGTH as well as content, so the cache's (path, mtime, size)
    # signature changes even where the clock's resolution would not settle it.
    _settings(tmp_path, {PROBE: "before-the-edit"})
    assert f"{PROBE}[before-the-edit]" in await _shell(tmp_path, _report(PROBE))
    _settings(tmp_path, {PROBE: "after-the-edit-and-then-some"})
    assert f"{PROBE}[after-the-edit-and-then-some]" in await _shell(tmp_path, _report(PROBE))


async def test_a_broken_edit_keeps_the_last_good_block(tmp_path: Path) -> None:
    # ADR-0079's rule for hooks, applied to the block: a half-saved file must not silently
    # switch a framework's configuration off in the middle of a run.
    path = _settings(tmp_path, {PROBE: "the-last-good-value"})
    assert f"{PROBE}[the-last-good-value]" in await _shell(tmp_path, _report(PROBE))
    path.write_text('{"env": {"' + PROBE + '": "half-writt', encoding="utf-8")
    assert f"{PROBE}[the-last-good-value]" in await _shell(tmp_path, _report(PROBE))


async def test_a_first_read_parse_error_still_honours_the_files_that_parse(tmp_path: Path) -> None:
    # Nothing to fall back to, so the files that DO parse are the answer — not an empty block.
    broken = tmp_path / SHARED
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{not json", encoding="utf-8")
    _settings(tmp_path, {PROBE: "from-the-file-that-parses"}, where=ZAKCODE)
    assert f"{PROBE}[from-the-file-that-parses]" in await _shell(tmp_path, _report(PROBE))


def test_an_impossible_name_or_a_non_string_value_is_skipped_and_named(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A name with a space cannot go into a child's environment at all, and the schema is
    # string to string: guessing whether 3 meant "3" is not this reader's job. Both are
    # skipped and reported by name, and the rest of the block still stands.
    _settings(tmp_path, {PROBE: "kept", "not a name": "x", "ZC_NUMBER": 3})
    with caplog.at_level(logging.WARNING, logger="zakcode.workspace_env"):
        block = settings_env(tmp_path)
    assert block == {PROBE: "kept"}
    assert "not a name" in caplog.text
    assert "ZC_NUMBER" in caplog.text


# ── what the block must never touch ───────────────────────────────────────────


def test_the_products_own_environment_is_never_written(tmp_path: Path) -> None:
    # A workspace's settings file configures that workspace's CHILDREN, never the host
    # process — so a served repository cannot reconfigure the harness running it.
    _settings(tmp_path, {PROBE: "for-children-only"})
    assert settings_env(tmp_path) == {PROBE: "for-children-only"}
    assert PROBE not in os.environ


async def test_one_workspaces_block_never_reaches_another_workspaces_child(
    tmp_path: Path,
) -> None:
    # One process can serve two workspaces; each child gets its own workspace's block.
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _settings(first, {PROBE: "the-first-workspace"})
    _settings(second, {PROBE: "the-second-workspace"})
    assert f"{PROBE}[the-first-workspace]" in await _shell(first, _report(PROBE))
    assert f"{PROBE}[the-second-workspace]" in await _shell(second, _report(PROBE))


async def test_a_workspace_with_no_settings_files_leaves_the_child_unchanged(
    tmp_path: Path, monkeypatch
) -> None:
    # The positive control for every test above: with no settings file the reader adds
    # nothing, and what the host exported still reaches the child as it always did.
    monkeypatch.setenv("ZC_FROM_THE_SHELL", "inherited")
    out = await _shell(tmp_path, _report(PROBE, "ZC_FROM_THE_SHELL"))
    assert f"{PROBE}[<unset>]" in out
    assert "ZC_FROM_THE_SHELL[inherited]" in out


def test_the_log_names_the_variables_and_never_their_values(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A block may hold a token, so the line that says a workspace has one names it only.
    _settings(tmp_path, {PROBE: "a-value-that-must-not-be-logged"})
    with caplog.at_level(logging.INFO, logger="zakcode.workspace_env"):
        settings_env(tmp_path)
    assert PROBE in caplog.text
    assert "a-value-that-must-not-be-logged" not in caplog.text
