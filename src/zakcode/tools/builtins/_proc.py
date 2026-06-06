"""Shared subprocess runner for the shell tools (bash / powershell).

Spawns a child via :mod:`asyncio` so that BOTH a cancelled turn (a WebSocket interrupt or
client disconnect that cancels ``astream_turn``) and a timeout tear down the **whole
process tree** instead of orphaning grandchildren that hold ports / file locks. The
previous ``subprocess.run``-via-``asyncio.to_thread`` approach kept no process handle: a
``to_thread`` worker can't be interrupted (so a cancel just abandons the thread while the
child runs on), and ``subprocess.run``'s own timeout terminates only the *parent*, leaving
the tree alive. The hooks and MCP layers already reap children correctly via
``create_subprocess_exec`` + kill; this brings the two shell tools — the only other
subprocess spawns in the repo — in line. (audit3 #3)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from typing import Any


class CommandTimeout(Exception):
    """Raised by :func:`run_capturing` when the child exceeds its timeout.

    The child's entire process tree has already been killed by the time this propagates.
    """


def _new_group_kwargs() -> dict[str, Any]:
    """Spawn the child in its own process group/session so the whole tree is killable."""
    if sys.platform == "win32":
        # A new process group lets taskkill /T find the tree by the child's PID.
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def _terminate_tree(proc: asyncio.subprocess.Process) -> None:
    """Forcibly kill ``proc`` AND its descendants (best-effort), then reap it."""
    if proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            # taskkill walks and kills the whole tree (/T) forcibly (/F); killing only the
            # parent (proc.kill) would orphan grandchildren on Windows.
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(proc.pid),
                "/T",
                "/F",
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # already gone / race — nothing to reap
    with contextlib.suppress(Exception):
        await proc.wait()


async def run_capturing(
    *,
    argv: list[str] | None = None,
    shell_command: str | None = None,
    cwd: str,
    timeout: float,
    stdin_text: str | None = None,
) -> tuple[str, int]:
    """Run a child to completion, capturing combined stdout+stderr; enforce ``timeout``.

    Exactly one of ``argv`` (direct exec) or ``shell_command`` (platform shell) must be
    given. Returns ``(combined_output, exit_code)``. On timeout raises
    :class:`CommandTimeout`; on cancellation re-raises ``CancelledError`` — in BOTH cases
    the child's entire process tree is killed first, never orphaned.
    """
    if (argv is None) == (shell_command is None):
        raise ValueError("exactly one of argv or shell_command is required")

    stdin = subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL
    spawn_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": stdin,
        **_new_group_kwargs(),
    }
    if shell_command is not None:
        proc = await asyncio.create_subprocess_shell(shell_command, **spawn_kwargs)
    else:
        assert argv is not None
        proc = await asyncio.create_subprocess_exec(*argv, **spawn_kwargs)

    input_bytes = stdin_text.encode("utf-8") if stdin_text is not None else None
    try:
        out_bytes, _ = await asyncio.wait_for(proc.communicate(input=input_bytes), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError) as exc:
        # wait_for cancelled communicate() but the child is still running — kill the tree.
        await _terminate_tree(proc)
        if isinstance(exc, asyncio.CancelledError):
            raise  # a turn cancel: propagate after teardown
        raise CommandTimeout from exc
    output = out_bytes.decode("utf-8", errors="replace") if out_bytes else ""
    return output, proc.returncode if proc.returncode is not None else -1
