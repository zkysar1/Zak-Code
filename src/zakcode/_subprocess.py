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
import shutil
import signal
import subprocess
import sys
from typing import Any


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
