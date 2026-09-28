"""Shared subprocess group-spawn + tree-teardown helpers.

Every place that spawns a child process (the shell tools, the shell hook runners, the MCP
stdio transport) uses these so teardown is UNIFORM: spawn the child in its own process group
/ session via :func:`create_group_subprocess_exec` / :func:`create_group_subprocess_shell`,
and, on timeout or cancellation, kill the WHOLE tree (:func:`terminate_process_tree`) rather
than orphaning grandchildren that hold ports / file locks. Centralizing the two primitives
means a fix or platform quirk is handled once for all spawners. (audit3 #3 / audit4 #2 / #3)

Windows job objects
~~~~~~~~~~~~~~~~~~~

Under Git for Windows (Git Bash), each program runs in a new Windows process whose Windows
*parent* is a short-lived forked process, not the shell that started it. ``taskkill /T``
walks Windows parent links, so from our shell it reaches nothing below the first ``exec``.
Measured: after ``terminate_process_tree``, a command's inner bash and its sleep were still
running 5 s later. Only the outer shell died. On Linux, ``killpg`` left nothing.

The fix: put every child into its own Windows job object at spawn time. Every descendant
inherits the job regardless of its Windows parent link, so ``TerminateJobObject`` kills
the entire tree. To make the assignment race-free (a fast child could fork before the
assignment), the child is created ``CREATE_SUSPENDED``, assigned to the job, then resumed
via ``NtResumeProcess`` (asyncio's ``Popen`` closes the thread handle, so ``ResumeThread``
is not available). ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` is deliberately NOT set: a
background task must keep running after the zakcode process exits.
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
import weakref
from typing import Any

logger = logging.getLogger(__name__)


class CommandTimeout(Exception):
    """Raised when a child exceeds its timeout — its process tree is killed first."""


# ---------------------------------------------------------------------------
# Windows job-object plumbing (loaded only on Windows, never imported on Linux)
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)

    # -- CreateProcess flags -------------------------------------------------
    _CREATE_SUSPENDED = 0x00000004

    # -- Process access rights -----------------------------------------------
    _PROCESS_SUSPEND_RESUME = 0x0800
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_TERMINATE = 0x0001

    # -- ctypes signatures (explicit argtypes/restype for every function) ----

    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    _kernel32.CreateJobObjectW.restype = ctypes.c_void_p

    _kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _kernel32.AssignProcessToJobObject.restype = ctypes.wintypes.BOOL

    _kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    _kernel32.TerminateJobObject.restype = ctypes.wintypes.BOOL

    _kernel32.OpenProcess.argtypes = [
        ctypes.wintypes.DWORD,
        ctypes.wintypes.BOOL,
        ctypes.wintypes.DWORD,
    ]
    _kernel32.OpenProcess.restype = ctypes.c_void_p

    _kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    _kernel32.CloseHandle.restype = ctypes.wintypes.BOOL

    # NtResumeProcess — undocumented but stable (ntoskrnl). Resumes all threads of a process.
    # We need this because asyncio's Popen closes the thread handle from CreateProcess, so
    # ResumeThread is not available. A process handle with PROCESS_SUSPEND_RESUME is enough.
    _ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
    _ntdll.NtResumeProcess.restype = ctypes.c_long  # NTSTATUS

    class _JobRef:
        """Mutable wrapper so the job handle is closed exactly once.

        Either :meth:`terminate_and_close` (the kill path) or :meth:`close` (the GC
        finalizer) may fire first; the other becomes a no-op.  Closing the handle does
        NOT kill anything because ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` is never set:
        a background task must outlive the zakcode process.
        """

        __slots__ = ("_handle",)

        def __init__(self, handle: int) -> None:
            self._handle = handle

        @property
        def handle(self) -> int:
            return self._handle

        def terminate_and_close(self) -> bool:
            """Kill every process in the job, then close the handle."""
            h = self._handle
            if h:
                self._handle = 0
                result = _kernel32.TerminateJobObject(h, 1)
                _kernel32.CloseHandle(h)
                return bool(result)
            return False

        def close(self) -> None:
            """Close the handle without killing anything (the GC / finalizer path)."""
            h = self._handle
            if h:
                self._handle = 0
                _kernel32.CloseHandle(h)

    def _resume_or_kill(proc_handle: int, pid: int, proc: asyncio.subprocess.Process) -> None:
        """Resume a suspended child via NtResumeProcess.

        If the resume fails (NTSTATUS < 0) the child is killed immediately so it is
        never left suspended. Raises :class:`OSError` on failure so the spawn helper
        can propagate the error to the caller.
        """
        status = _ntdll.NtResumeProcess(proc_handle)
        if status < 0:  # NTSTATUS failure (negative = error)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise OSError(
                f"NtResumeProcess(pid={pid}) returned 0x{status & 0xFFFFFFFF:08x}; "
                "child killed to avoid leaving it suspended"
            )

    def _adopt_into_job(proc: asyncio.subprocess.Process) -> None:
        """Assign a ``CREATE_SUSPENDED`` child to a fresh job object, then resume it.

        If a Win32 step before resume fails the child is still resumed (without a job),
        so :func:`terminate_process_tree` degrades to the ``taskkill /T /F`` fallback
        (today's behavior). A warning is logged for diagnostics.

        If the resume itself fails the child is killed and :class:`OSError` is raised —
        a suspended child must never be left frozen.
        """
        pid = proc.pid
        if pid is None:
            # No pid means the process failed to start; nothing to resume or kill.
            raise OSError("CREATE_SUSPENDED child has no pid; cannot resume")

        # Open a handle to the suspended child with the rights AssignProcessToJobObject
        # and NtResumeProcess need. OpenProcess should never fail for a process we own,
        # but if it does we fall back to the Popen's internal handle (which has
        # PROCESS_ALL_ACCESS from CreateProcess).
        proc_handle = _kernel32.OpenProcess(
            _PROCESS_SUSPEND_RESUME | _PROCESS_SET_QUOTA | _PROCESS_TERMINATE,
            False,
            pid,
        )
        owns_handle = True
        if not proc_handle:
            try:
                # asyncio Process -> SubprocessTransport -> subprocess.Popen -> handle
                proc_handle = proc._transport._proc._handle  # type: ignore[attr-defined]
                owns_handle = False  # the Popen owns this handle; don't close it
            except AttributeError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                raise OSError(
                    f"OpenProcess(pid={pid}) failed (error {ctypes.get_last_error()}) "
                    "and no internal handle; child killed to avoid leaving it suspended"
                ) from None

        try:
            job = _kernel32.CreateJobObjectW(None, None)
            if not job:
                logger.warning(
                    "job-object: CreateJobObjectW failed (error %d), pid %d uses taskkill fallback",
                    ctypes.get_last_error(),
                    pid,
                )
                _resume_or_kill(proc_handle, pid, proc)
                return

            if not _kernel32.AssignProcessToJobObject(job, proc_handle):
                logger.warning(
                    "job-object: AssignProcessToJobObject(pid=%d) failed"
                    " (error %d), taskkill fallback",
                    pid,
                    ctypes.get_last_error(),
                )
                _kernel32.CloseHandle(job)
                _resume_or_kill(proc_handle, pid, proc)
                return

            # Everything succeeded — resume the child and store the job ref. A failed resume
            # has already killed the child; close the job too so the raise leaks no handle.
            try:
                _resume_or_kill(proc_handle, pid, proc)
            except OSError:
                _kernel32.CloseHandle(job)
                raise

            ref = _JobRef(job)
            proc._job_ref = ref  # type: ignore[attr-defined]
            # When the Process is garbage-collected, close the job handle. This does NOT kill
            # anything (no kill-on-close), so a background task that outlives us is unaffected.
            # If terminate_and_close already ran, close() is a no-op.
            weakref.finalize(proc, ref.close)
        finally:
            if owns_handle:
                _kernel32.CloseHandle(proc_handle)


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
    group).

    A raw spawn with these kwargs gets no job object on Windows, so descendants that
    re-parent (the Git Bash pattern) escape ``taskkill /T``. Spawners that need a
    killable tree should use :func:`create_group_subprocess_exec` /
    :func:`create_group_subprocess_shell` instead — those add the job-object dance
    on top of these same flags.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# ---------------------------------------------------------------------------
# One-contract spawn helpers (the preferred API)
# ---------------------------------------------------------------------------


async def create_group_subprocess_exec(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
    """Spawn a child in its own killable group, with a job object on Windows.

    On Windows the child is created ``CREATE_SUSPENDED`` inside a new process group
    (via :func:`new_group_kwargs` plus ``CREATE_SUSPENDED``), assigned to a fresh job
    object, then resumed. This guarantees every descendant is in the job regardless of
    its Windows parent link (the Git Bash parent-link problem described in the module
    docstring). On POSIX this is exactly
    ``create_subprocess_exec(..., start_new_session=True)``.

    Callers MUST NOT pass ``creationflags`` or ``start_new_session`` — this helper owns
    both. Every spawn site that needs a killable tree uses this instead of calling
    ``asyncio.create_subprocess_exec`` + ``new_group_kwargs()`` directly.
    """
    group_kw = new_group_kwargs()
    if sys.platform == "win32":
        # new_group_kwargs() gives CREATE_NEW_PROCESS_GROUP; add CREATE_SUSPENDED so
        # the child is frozen until it's in the job object.
        group_kw["creationflags"] = group_kw["creationflags"] | _CREATE_SUSPENDED
        kwargs.update(group_kw)
        proc = await asyncio.create_subprocess_exec(*args, **kwargs)
        _adopt_into_job(proc)
        return proc
    kwargs.update(group_kw)
    return await asyncio.create_subprocess_exec(*args, **kwargs)


async def create_group_subprocess_shell(cmd: str, **kwargs: Any) -> asyncio.subprocess.Process:
    """Like :func:`create_group_subprocess_exec` but for a platform-shell command.

    On Windows the same suspended-spawn + job-object dance applies. On POSIX this is
    ``create_subprocess_shell(..., start_new_session=True)``.
    """
    group_kw = new_group_kwargs()
    if sys.platform == "win32":
        group_kw["creationflags"] = group_kw["creationflags"] | _CREATE_SUSPENDED
        kwargs.update(group_kw)
        proc = await asyncio.create_subprocess_shell(cmd, **kwargs)
        _adopt_into_job(proc)
        return proc
    kwargs.update(group_kw)
    return await asyncio.create_subprocess_shell(cmd, **kwargs)


# ---------------------------------------------------------------------------
# Tree teardown
# ---------------------------------------------------------------------------

#: How long the teardown waits to reap a killed child before it stops waiting on the child's
#: pipes. See :func:`terminate_process_tree`.
_REAP_GRACE_S = 2.0


async def terminate_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Forcibly kill ``proc`` AND its descendants (best-effort), then reap it.

    Killing only the parent (``proc.kill()``) orphans grandchildren — wrappers like
    ``sh -c '... &'``, ``npx``/``uvx`` launchers, or a dev server — so this kills the tree.

    On Windows, if the child was spawned with :func:`create_group_subprocess_exec` (or
    ``_shell``), it has a job object: ``TerminateJobObject`` kills every process in the
    job. When there is no job (the child was spawned without the helper, or the job
    assignment failed at spawn time), it falls back to ``taskkill /PID <pid> /T /F``.
    Either way, ``proc.kill()`` is called for the direct child.

    On POSIX, ``os.killpg(getpgid, SIGKILL)`` kills the process group (the child must
    have been spawned with ``start_new_session=True``). No-op if already exited.

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
            # Primary path: kill via the job object when the child has one. This reaches
            # every descendant regardless of its Windows parent link (the Git Bash fix).
            job_ref: _JobRef | None = getattr(proc, "_job_ref", None)
            if job_ref is not None:
                job_ref.close()  # MUTATION-PROOF-ADR0261: close without killing the job
            else:
                # Fallback: walk the Windows parent-pid tree. taskkill /T walks parent
                # links, which misses descendants under Git Bash (they re-parent to a
                # short-lived fork), but it's better than nothing when the job object
                # was not available.
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
                    # taskkill exits non-zero when it could not end a member of
                    # the tree. Log the exit and the reason: pytest prints this
                    # under a failing test, and a session's log keeps it.
                    reason = err.decode("utf-8", errors="replace").strip().splitlines()
                    logger.warning(
                        "taskkill /T on pid %s exited %s: %s",
                        proc.pid,
                        killer.returncode,
                        reason[-1][:200] if reason else "",
                    )
            # Whatever the job/taskkill reached, the direct child is ours to end:
            # TerminateProcess on the handle asyncio holds needs no tree walk.
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
