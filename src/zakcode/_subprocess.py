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
import logging
import os
import shutil
import signal
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


class CommandTimeout(Exception):
    """Raised when a child exceeds its timeout — its process tree is killed first."""


def find_bash() -> str | None:
    """Absolute path to a real Bash interpreter, or ``None`` if none is found.

    On Windows this deliberately AVOIDS the WindowsApps app-execution-alias stub (the WSL
    ``bash.exe`` launcher): a bare ``create_subprocess_exec("bash")`` is hijacked by that stub
    even when Git Bash is first on PATH (``CreateProcess`` consults app-exec aliases, unlike
    ``shutil.which``), so a caller must spawn the ABSOLUTE path this returns. Prefers Git for
    Windows. On POSIX it is just ``shutil.which("bash")``.
    """
    if sys.platform != "win32":
        return shutil.which("bash")
    bases = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMW6432", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
    ]
    for base in bases:
        cand = os.path.join(base, "Git", "usr", "bin", "bash.exe") if base else ""
        if cand and os.path.isfile(cand):
            return cand
    found = shutil.which("bash")
    if found and "windowsapps" not in found.lower():  # skip the WSL app-exec stub
        return found
    return None


def resolve_executable(name: str) -> str:
    """Resolve a bare command name to a real absolute path, dodging the Windows app-exec stubs.

    Returns ``name`` unchanged when it is already a path, or can't be confidently resolved (let
    the OS try). The motivating case: a shell-hook ``argv[0]`` of ``bash`` on Windows resolving
    to the WSL stub instead of the Git Bash actually on PATH.
    """
    if os.path.isabs(name) or os.sep in name or (os.altsep and os.altsep in name):
        return name  # already a path
    if sys.platform == "win32" and os.path.splitext(os.path.basename(name))[0].lower() == "bash":
        return find_bash() or name
    found = shutil.which(name)
    if found and "windowsapps" not in found.lower():
        return found
    return name


def new_group_kwargs() -> dict[str, Any]:
    """``create_subprocess_*`` kwargs that isolate the child in its own group/session.

    This is what makes the whole tree killable: Windows ``CREATE_NEW_PROCESS_GROUP`` (so
    ``taskkill /T`` reaches it by PID), POSIX ``start_new_session`` (so ``killpg`` reaches the
    group). Spread into every spawn that may launch descendants.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


#: How long the teardown waits to reap a killed child before it stops waiting on the child's
#: pipes. See :func:`terminate_process_tree`.
_REAP_GRACE_S = 2.0


async def terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Forcibly kill ``proc`` AND its descendants (best-effort), then reap it.

    Killing only the parent (``proc.kill()``) orphans grandchildren — wrappers like
    ``sh -c '... &'``, ``npx``/``uvx`` launchers, or a dev server — so this kills the tree:
    ``taskkill /PID <pid> /T /F`` on Windows, ``os.killpg(getpgid, SIGKILL)`` on POSIX (the
    child must have been spawned with :func:`new_group_kwargs`). No-op if already exited.

    The reap is BOUNDED, and it lets go of the child's pipes. A descendant that left the
    child's group (``setsid``, a server that daemonizes itself) survives the group kill with
    those pipes still open. Before Python 3.13, asyncio's ``Process.wait()`` settles only once
    every pipe has closed, so an unbounded reap lasted as long as that descendant: measured
    2026-09-23 on 3.11, a command that started ``setsid sleep 12`` under a 2-second timeout
    returned after 12 seconds, and a daemon would have held the turn indefinitely (OpenCode
    issue #49169 is the same defect). 3.13 returns at the exit (gh-119710), but our ends of the
    pipes stay open until the stray exits. So the wait is bounded by :data:`_REAP_GRACE_S`, and
    then the transport is closed on every version. The stray is not killed: it left the group.
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
                stderr=subprocess.PIPE,
            )
            _, err = await killer.communicate()
            if killer.returncode != 0:
                # taskkill walks the tree by parent pid and exits non-zero when it could not end
                # a member. Measured 2026-09-23 on CI (windows-latest, main at 3c0b430): the
                # shell of a cancelled call outlived this kill by more than 5 s, and nothing said
                # why, because the exit and the reason both went to DEVNULL. Say both: pytest
                # prints this line under a failing test, and a session's log keeps it.
                reason = err.decode("utf-8", errors="replace").strip().splitlines()
                logger.warning(
                    "taskkill /T on pid %s exited %s: %s",
                    proc.pid,
                    killer.returncode,
                    reason[-1][:200] if reason else "",
                )
            # Whatever taskkill reached, the direct child is ours to end: TerminateProcess on
            # the handle asyncio holds needs no tree walk. A child already gone raises here.
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # already gone / race
    with contextlib.suppress(Exception):  # TimeoutError: a stray holds the pipes (see above)
        await asyncio.wait_for(proc.wait(), timeout=_REAP_GRACE_S)
    # asyncio exposes no public way to let go of a child's pipes, so close its transport. After
    # a normal exit this is a no-op; with a stray it closes our ends.
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        transport.close()
