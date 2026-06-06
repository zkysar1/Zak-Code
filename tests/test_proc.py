"""Tests for the shared shell-tool subprocess runner (audit3 #3).

These prove that a timeout and a turn cancellation both tear the child down promptly
(killing the process tree) instead of waiting out a long-running command or orphaning it.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from zakcode.tools.builtins._proc import CommandTimeout, _terminate_tree, run_capturing


def _sleep_cmd(seconds: int) -> str:
    """A portable shell command that blocks for ~``seconds`` (cmd.exe vs POSIX sh)."""
    if sys.platform == "win32":
        return f"ping -n {seconds + 1} 127.0.0.1 >NUL"
    return f"sleep {seconds}"


async def test_run_capturing_normal_command() -> None:
    output, code = await run_capturing(shell_command="echo hello", cwd=".", timeout=10)
    assert "hello" in output
    assert code == 0


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


async def test_terminate_tree_reaps_a_running_child() -> None:
    proc = await asyncio.create_subprocess_shell(
        _sleep_cmd(30), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    assert proc.returncode is None  # alive
    await _terminate_tree(proc)
    assert proc.returncode is not None  # killed + reaped, not orphaned
