"""Tests for the shared shell-tool subprocess runner (audit3 #3).

These prove that a timeout and a turn cancellation both tear the child down promptly
(killing the process tree) instead of waiting out a long-running command or orphaning it.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
import time
from typing import Any

import pytest

from zakcode._subprocess import new_group_kwargs, terminate_process_tree
from zakcode.tools.builtins._proc import CommandTimeout, run_capturing


def _sleep_cmd(seconds: int) -> str:
    """A portable shell command that blocks for ~``seconds`` (cmd.exe vs POSIX sh).

    Deliberately NOT redirected to NUL. ``run_capturing`` captures stdout already, so the
    redirect bought nothing — and when the shell resolves to bash rather than cmd.exe (Git
    Bash / MSYS, the normal dev shell on this project's Windows boxes), ``>NUL`` is not a
    device at all: it creates a real 93-byte file called ``NUL`` in the repo root. Git then
    cannot index it, because NUL is a reserved Windows device name, so every subsequent
    ``git add -A`` in the repo dies with "short read while indexing NUL" until someone
    deletes it by hand. A test artifact that silently blocks commits is worth one comment.
    """
    if sys.platform == "win32":
        return f"ping -n {seconds + 1} 127.0.0.1"
    return f"sleep {seconds}"


async def test_run_capturing_normal_command() -> None:
    output, code = await run_capturing(shell_command="echo hello", cwd=".", timeout=10)
    assert "hello" in output
    assert code == 0


@pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell's `&` child")
async def test_a_command_that_leaves_a_redirected_child_running_returns_when_it_exits() -> None:
    # A mind world's playbooks do this all the time: start a daemon, `nohup ... &`. With all
    # three of the child's streams redirected nothing holds the command's pipe, so the call is
    # over when the command is. True on the stdlib loop, which is why the served process is
    # pinned to it (ADR-0197): under uvloop this exact command waits out its whole timeout.
    start = time.monotonic()
    output, code = await run_capturing(
        shell_command="sleep 8 </dev/null >/dev/null 2>&1 & echo started", cwd=".", timeout=6
    )
    assert "started" in output and code == 0
    assert time.monotonic() - start < 4  # did NOT wait for the sleeping child


async def test_run_capturing_times_out_promptly() -> None:
    # A 30s command with a 0.5s timeout must raise CommandTimeout almost immediately —
    # proving wait_for fired and the child tree was killed rather than waited out.
    start = time.monotonic()
    with pytest.raises(CommandTimeout):
        await run_capturing(shell_command=_sleep_cmd(30), cwd=".", timeout=0.5)
    assert time.monotonic() - start < 15  # did NOT block for the full 30s


async def test_run_capturing_cancel_propagates_promptly() -> None:
    # Cancelling the turn (as a WS interrupt / disconnect does) must re-raise CancelledError
    # quickly after tearing the child down, not run the command to completion in a stuck thread.
    task = asyncio.ensure_future(run_capturing(shell_command=_sleep_cmd(30), cwd=".", timeout=30))
    await asyncio.sleep(0.5)  # let the child actually spawn
    task.cancel()
    start = time.monotonic()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.monotonic() - start < 15


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid (util-linux)")
async def test_a_timeout_is_not_held_open_by_a_descendant_that_left_the_group() -> None:
    # `setsid` puts the sleep in a new session, out of reach of the group kill, and it keeps
    # the command's output pipe open. The timeout must still end the call: measured
    # 2026-09-23, a 2-second timeout returned after 12 seconds (OpenCode issue #49169).
    start = time.monotonic()
    with pytest.raises(CommandTimeout):
        await run_capturing(shell_command="setsid sleep 10 & sleep 60", cwd=".", timeout=1)
    assert time.monotonic() - start < 7  # timeout + the bounded reap, never the sleep's 10s


async def test_terminate_tree_reaps_a_running_child() -> None:
    # Spawn the child the way EVERY production spawner does -- in its own process
    # group/session via new_group_kwargs(). terminate_process_tree's POSIX path is
    # os.killpg(getpgid(child), SIGKILL); without an own group the child shares
    # pytest's process group, so killpg SIGKILLs the test runner itself (ubuntu-only
    # exit 137 -- Windows uses `taskkill /PID /T` and is unaffected). This honors the
    # documented terminate_process_tree contract; do NOT drop new_group_kwargs().
    proc = await asyncio.create_subprocess_shell(
        _sleep_cmd(30),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        **new_group_kwargs(),
    )
    assert proc.returncode is None  # alive
    await terminate_process_tree(proc)
    assert proc.returncode is not None  # killed + reaped, not orphaned


class _FakeTaskkill:
    """What ``taskkill`` looks like to the teardown: an exit code and its stderr."""

    def __init__(self, returncode: int, stderr: bytes) -> None:
        self.returncode = returncode
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", self._stderr


async def _terminate_through_a_fake_taskkill(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stderr: bytes
) -> asyncio.subprocess.Process:
    """Run the Windows branch of ``terminate_process_tree`` on a real child, with taskkill
    replaced by a stand-in that answers ``returncode`` and ``stderr``. The child is spawned
    BEFORE the platform is faked, so it gets this platform's real process group."""
    proc = await asyncio.create_subprocess_shell(
        _sleep_cmd(30),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        **new_group_kwargs(),
    )
    assert proc.returncode is None  # alive

    async def fake_exec(*argv: str, **_kwargs: Any) -> _FakeTaskkill:
        assert argv[:2] == ("taskkill", "/PID") and argv[2] == str(proc.pid)
        return _FakeTaskkill(returncode, stderr)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await terminate_process_tree(proc)
    return proc


async def test_a_failed_taskkill_is_reported_and_the_child_is_still_killed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Measured 2026-09-23 on CI (windows-latest, twice in twelve runs): the shell of a
    # cancelled call outlived the tree kill by more than 5 s and nothing said why, because
    # taskkill's exit and stderr both went to DEVNULL. Now the exit and the reason are logged,
    # and the direct child is killed through the handle asyncio holds, whatever taskkill did.
    caplog.set_level(logging.WARNING, logger="zakcode._subprocess")
    stderr = (
        b"ERROR: The process with PID 7 could not be terminated.\r\nReason: Access is denied.\r\n"
    )
    proc = await _terminate_through_a_fake_taskkill(monkeypatch, 1, stderr)
    assert proc.returncode is not None  # killed by the fallback and reaped, not left running
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [f"taskkill /T on pid {proc.pid} exited 1: Reason: Access is denied."]


async def test_a_clean_taskkill_says_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Positive control for the warning: a taskkill that exited 0 logs nothing, and the child
    # is dead either way.
    caplog.set_level(logging.WARNING, logger="zakcode._subprocess")
    proc = await _terminate_through_a_fake_taskkill(monkeypatch, 0, b"")
    assert proc.returncode is not None
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs setsid (util-linux)")
async def test_terminate_tree_lets_go_of_pipes_a_stray_descendant_holds() -> None:
    # The escaped sleep outlives the group kill and keeps our pipes open. Returning on time
    # is half the fix; the other half is closing our ends, or a long-lived server leaks two
    # descriptors per stray until it runs out. Closed ends read as EOF at once.
    proc = await asyncio.create_subprocess_shell(
        "setsid sh -c 'echo escaped; exec sleep 10' & sleep 60",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **new_group_kwargs(),
    )
    assert proc.stdout is not None
    # Kill only once the sleep has left the group. Killed sooner, it dies with the group and
    # this test passes without ever meeting a stray.
    assert await asyncio.wait_for(proc.stdout.readline(), timeout=5) == b"escaped\n"
    start = time.monotonic()
    await terminate_process_tree(proc)
    assert time.monotonic() - start < 6  # the bounded reap, never the sleep's 10s
    assert proc.returncode is not None
    assert await asyncio.wait_for(proc.stdout.read(), timeout=1) == b""


async def test_shell_commands_run_under_real_bash(tmp_path) -> None:
    # The tool is NAMED bash; /bin/sh is dash on Debian/Ubuntu, where bashisms fail
    # (mind-world playbooks assume bash — 2026-08-25 field report from a live box).
    # [[ ]] is a bashism dash rejects, so this passes only under real bash.
    out, code = await run_capturing(
        shell_command='[[ -n "x" ]] && echo real-bash', cwd=str(tmp_path), timeout=10.0
    )
    assert code == 0
    assert "real-bash" in out


async def test_workspace_env_hook_extends_path(tmp_path) -> None:
    # <workspace>/.zakcode/env is sourced (via BASH_ENV) by every shell command, so a
    # workspace can put its own script dirs on PATH — a mind world's bare script
    # names then resolve without the model re-deriving the bash-prefix form.
    bin_dir = tmp_path / "myscripts"
    bin_dir.mkdir()
    script = bin_dir / "hello-from-workspace.sh"
    script.write_text("#!/usr/bin/env bash\necho workspace-script-ran\n", encoding="utf-8")
    script.chmod(0o755)
    hook_dir = tmp_path / ".zakcode"
    hook_dir.mkdir()
    (hook_dir / "env").write_text('PATH="$PWD/myscripts:$PATH"\n', encoding="utf-8")

    out, code = await run_capturing(
        shell_command="hello-from-workspace.sh", cwd=str(tmp_path), timeout=10.0
    )
    assert code == 0
    assert "workspace-script-ran" in out
