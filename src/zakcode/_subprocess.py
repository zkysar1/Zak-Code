"""Shared subprocess group-spawn + tree-teardown helpers.

Every place that spawns a child process (the shell tools, the shell hook runners, the MCP
stdio transport) uses these so teardown is UNIFORM: spawn the child in its own process group
/ session (:func:`new_group_kwargs`) and, on timeout or cancellation, kill the WHOLE tree
(:func:`terminate_process_tree`) rather than orphaning grandchildren that hold ports / file
locks. Centralizing the two primitives means a fix or platform quirk is handled once for all
spawners. (audit3 #3 / audit4 #2 / #3)
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
    """Raised when a child exceeds its timeout — its process tree is killed first."""


def new_group_kwargs() -> dict[str, Any]:
    """``create_subprocess_*`` kwargs that isolate the child in its own group/session.

    This is what makes the whole tree killable: Windows ``CREATE_NEW_PROCESS_GROUP`` (so
    ``taskkill /T`` reaches it by PID), POSIX ``start_new_session`` (so ``killpg`` reaches the
    group). Spread into every spawn that may launch descendants.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Forcibly kill ``proc`` AND its descendants (best-effort), then reap it.

    Killing only the parent (``proc.kill()``) orphans grandchildren — wrappers like
    ``sh -c '... &'``, ``npx``/``uvx`` launchers, or a dev server — so this kills the tree:
    ``taskkill /PID <pid> /T /F`` on Windows, ``os.killpg(getpgid, SIGKILL)`` on POSIX (the
    child must have been spawned with :func:`new_group_kwargs`). No-op if already exited.
    """
    if proc.returncode is not None:
        return
    try:
        if sys.platform == "win32":
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
        pass  # already gone / race
    with contextlib.suppress(Exception):
        await proc.wait()
