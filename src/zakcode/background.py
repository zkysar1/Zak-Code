"""Background commands: Claude Code's ``Bash(run_in_background=true)`` for a Zak Code session
(ADR-0191).

Claude Code runs a command detached when asked to, returns at once with a task id and an
output file, and re-invokes the model with a ``<task-notification>`` when the command exits;
``TaskOutput`` reads the output, ``TaskStop`` kills it. A Mind's playbooks are written
against exactly that: "background the suite, END the turn; the harness notifies" — and the
framework's rules forbid polling a background job with ``ScheduleWakeup`` because the
harness reports on it. Zak Code had no such thing: a suite run held the turn for its whole
duration, a framework had to branch on WHICH harness ran it to know whether a background
job would ever be reported, and that branch (``background_job_notify``) was the last
harness-capability table left in the framework. This module removes the reason for it.

The contract, matching Claude Code's:

* ``start`` spawns the command in its own process group, stdout+stderr to ONE output file,
  and records the task on the session (id, command, files, pid). The tool result is the
  id and the file; nothing waits.
* The exit is observed two ways, so it survives the ADR-0034 restart into a new build: an
  in-process watcher reaps the child and writes ``<output>.exit``; where a real bash runs
  the command, a wrapper shell writes that file itself, so a process that did not spawn
  the task still learns how it ended. Status is DERIVED from the files and the pid — with
  the OS's own start time for that pid, so a pid reused after the task exited is not the
  task — never stored, so no stale record can call a dead task alive.
* The session's idle doors (the REPL's idle wait, the served consumer's beat) ask
  :meth:`BackgroundTasks.take_notifications`: every exited task not yet reported is
  reported ONCE, as one harness line carrying a ``<task-notification>`` block per task —
  the shape Claude Code delivers, so a rule written for it ("read the log before
  accepting the exit code") reads the same text here. Never mid-turn.
* ``TaskOutput`` returns status, exit code and the latest output (a bounded tail);
  ``TaskStop`` kills the whole process group. A session's end kills what it started.

Pure where it can be: the record is a pydantic model on the session, the status a function
of files and a pid; only ``start``/``stop``/``kill_all`` touch processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from zakcode._subprocess import find_bash, new_group_kwargs, terminate_process_tree
from zakcode.config import zakcode_home

#: The output tail ``TaskOutput`` returns. It was the foreground bash tool's budget until ADR-0234.
MAX_OUTPUT_CHARS = 64_000
#: ``TaskOutput(block=true)`` waits this long by default, and at most this long (ms).
DEFAULT_BLOCK_MS = 30_000
MAX_BLOCK_MS = 600_000
#: How often a blocking wait re-reads the task's status.
_POLL_SECONDS = 0.2

#: The harness line every notification opens with: provenance first (this is not a person
#: speaking), then the one thing a small model must do before it trusts the exit code.
NOTIFICATION_HEAD = (
    "[harness] a background command you started has exited. This is a notification from the "
    "harness, not a message from a person: read the output file before acting on the exit "
    "code."
)

#: Live handles for the tasks THIS process spawned: the watcher task (so it is never
#: garbage-collected before the child is reaped) and the process (for an in-process kill).
#: Module-level on purpose — a served mind builds an agent per turn, and the child outlives it.
_WATCHERS: dict[str, asyncio.Task[None]] = {}
_PROCS: dict[str, asyncio.subprocess.Process] = {}


class BackgroundTask(BaseModel):
    """One background command as the session records it. Status is never stored here —
    see :func:`task_status` — so the record cannot go stale."""

    id: str
    command: str
    description: str = ""
    cwd: str
    output_file: str
    exit_file: str
    pid: int
    started_at: str
    #: The OS's start time for ``pid``, read at spawn — an identity beyond the pid, so a pid
    #: the OS reused after the task exited (a restart in between, no exit file) is never
    #: mistaken for it. ``None`` where the platform cannot say, or on an older record.
    start_token: str | None = None
    #: Its exit was reported to the session (a notification is delivered once).
    notified: bool = False
    #: ``TaskStop`` asked for it: a missing exit code then means "killed", not "lost".
    stopped: bool = False

    @property
    def label(self) -> str:
        """What the notification calls it: the model's description, else the command."""
        text = (self.description or self.command).strip().splitlines()[0]
        return text if len(text) <= 120 else text[:117] + "..."


def tasks_dir_for(store_base_dir: Path | None, session_id: str) -> Path:
    """Where a session's task output lives: beside its session store (``~/.zakcode/tasks/<sid>``
    by default; ``<workspace>/.zakcode/tasks/<sid>`` for a served mind, ADR-0032)."""
    home = store_base_dir.parent if store_base_dir is not None else zakcode_home()
    return home / "tasks" / session_id


def pid_alive(pid: int) -> bool:
    """Whether a process with ``pid`` exists (best-effort; a zombie counts as alive until
    reaped, which the in-process watcher does)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return False
        process_query_limited_information = 0x1000
        still_active = 259
        handle = windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return int(code.value) == still_active
        finally:
            windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]  # noqa: RUF012


def _windows_start_token(pid: int) -> str | None:
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    process_query_limited_information = 0x1000
    handle = windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        creation, exited, kernel, user = (_FileTime() for _ in range(4))
        ok = windll.kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            return None
        return str((int(creation.high) << 32) | int(creation.low))
    finally:
        windll.kernel32.CloseHandle(handle)


def process_start_token(pid: int) -> str | None:
    """The OS's start time for the live process ``pid``, as an opaque string — the identity
    :func:`task_is_live` checks beside the pid. Linux reads ``/proc/<pid>/stat`` (start time
    in clock ticks since boot), Windows asks ``GetProcessTimes``, other POSIX ``ps -o
    lstart=``. ``None`` when the process is gone or the platform cannot say — never a guess.
    """
    if pid <= 0:
        return None
    try:
        if sys.platform == "win32":
            return _windows_start_token(pid)
        if sys.platform.startswith("linux"):
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            # After the ")" that closes the (space-bearing) comm: state is field 3 of the
            # documented layout, starttime field 22.
            fields = stat.rsplit(")", 1)[1].split()
            return fields[19] if len(fields) > 19 else None
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        token = out.stdout.strip()
        return token or None
    except Exception:  # noqa: BLE001 — an unreadable identity is "cannot say", never a crash
        return None


def task_is_live(task: BackgroundTask) -> bool:
    """Whether the process at ``task.pid`` is still the task: alive, and — for a task this
    process did not spawn, or has already reaped — carrying the start token the record took
    at spawn. Our own unreaped child cannot have had its pid reused. A record without a
    token, or a platform that cannot read one, keeps the pid's word: the token is only ever
    held against ANOTHER process, never against the task."""
    if not pid_alive(task.pid):
        return False
    if task.id in _PROCS or task.start_token is None:
        return True
    observed = process_start_token(task.pid)
    return observed is None or observed == task.start_token


def _read_exit_code(exit_file: str) -> int | None:
    try:
        text = Path(exit_file).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def task_status(task: BackgroundTask) -> tuple[str, int | None]:
    """``(status, exit_code)`` derived from the exit file and the pid: ``running``,
    ``completed`` (with its code), ``killed`` (``TaskStop`` asked; a code if the wrapper
    still wrote one), or ``lost`` (gone without a recorded code — the process that spawned
    it, on a platform with no bash wrapper, restarted before it exited; or the pid now
    names another process, see :func:`task_is_live`)."""
    code = _read_exit_code(task.exit_file)
    if code is not None:
        return ("killed" if task.stopped else "completed", code)
    if task_is_live(task):
        return ("running", None)
    return ("killed" if task.stopped else "lost", None)


def output_tail(output_file: str, *, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """The task's output so far — the LAST ``max_chars`` of it when longer (the end is where
    a verdict lands), with a note saying how much was skipped."""
    try:
        data = Path(output_file).read_bytes()
    except OSError:
        return ""
    text = data.decode("utf-8", errors="replace")
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return (
        f"[... {hidden} earlier chars omitted; showing the last {max_chars} ...]\n"
        + text[-max_chars:]
    )


def notification_block(task: BackgroundTask, status: str, code: int | None) -> str:
    """One ``<task-notification>`` — Claude Code's shape, so a rule that greps for it (or for
    ``completed (exit code N)``) reads the same text on either harness."""
    if status == "completed":
        summary = f'Background command "{task.label}" completed (exit code {code})'
    elif status == "killed":
        summary = f'Background command "{task.label}" was stopped' + (
            f" (exit code {code})" if code is not None else ""
        )
    else:
        summary = (
            f'Background command "{task.label}" exited without a recorded exit code (the '
            "process that started it restarted before the exit was observed); the output "
            "file may still be complete — read it before deciding how it ended"
        )
    return (
        "<task-notification>\n"
        f"<task-id>{task.id}</task-id>\n"
        f"<output-file>{task.output_file}</output-file>\n"
        f"<status>{status}</status>\n"
        f"<summary>{summary}</summary>\n"
        "</task-notification>"
    )


def _kill_pid_tree(pid: int) -> None:
    """Kill a task's whole process group by pid — the path for a task THIS process did not
    spawn (a restart in between). The spawn used :func:`new_group_kwargs`, so the group
    is the task's own."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


async def _watch(task_id: str, proc: asyncio.subprocess.Process, exit_file: str) -> None:
    """Reap the child and record its exit — unless the bash wrapper already did."""
    try:
        code = await proc.wait()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — a watcher must never take the event loop down
        return
    finally:
        _PROCS.pop(task_id, None)
    path = Path(exit_file)
    if not path.exists():
        with contextlib.suppress(OSError):
            path.write_text(f"{code}\n", encoding="utf-8")


class BackgroundTasks:
    """The session's background commands: start, read, stop, and — for the idle doors —
    report the ones that exited. ``on_change`` (the loop's persist) runs after every
    mutation of the record so the table is on disk before the turn that started a task
    ends; ``tasks_dir`` is where output files go (:func:`tasks_dir_for`)."""

    def __init__(
        self,
        session: Any,
        *,
        on_change: Callable[[], None] | None = None,
        tasks_dir: Path | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._session = session
        self._on_change = on_change
        self._tasks_dir = tasks_dir
        self._clock = clock

    # ── the record ────────────────────────────────────────────────────────────────

    def records(self) -> list[BackgroundTask]:
        tasks = getattr(self._session, "background_tasks", None)
        return list(tasks) if isinstance(tasks, list) else []

    def get(self, task_id: str) -> BackgroundTask | None:
        for task in self.records():
            if task.id == task_id:
                return task
        return None

    def status(self, task: BackgroundTask) -> tuple[str, int | None]:
        return task_status(task)

    def output(self, task: BackgroundTask, *, max_chars: int = MAX_OUTPUT_CHARS) -> str:
        return output_tail(task.output_file, max_chars=max_chars)

    # ── the output directory ──────────────────────────────────────────────────────

    @property
    def directory(self) -> Path:
        """Where this session's command output is written (:func:`tasks_dir_for`). Asking
        creates nothing: Read asks whenever a path falls outside the workspace (ADR-0234)."""
        if self._tasks_dir is not None:
            return self._tasks_dir
        return tasks_dir_for(None, str(getattr(self._session, "id", "session")))

    def _prepared_directory(self) -> Path:
        """:attr:`directory`, created, with an ignore-everything file beside it."""
        tasks_dir = self.directory
        tasks_dir.mkdir(parents=True, exist_ok=True)
        ignore = tasks_dir.parent / ".gitignore"  # a served workspace may be a checkout
        if not ignore.exists():
            with contextlib.suppress(OSError):
                ignore.write_text("*\n", encoding="utf-8")
        return tasks_dir

    def save_output(self, text: str) -> Path:
        """Write a foreground command's whole output to a file of its own and return the
        path (ADR-0234). The shell tool calls this when an output is too long to show
        whole; the result then carries a preview and this path. Raises ``OSError``."""
        path = self._prepared_directory() / f"output-{uuid.uuid4().hex[:9]}.txt"
        # Bytes, not write_text: on Windows text mode would turn every "\n" into "\r\n".
        path.write_bytes(text.encode("utf-8", errors="replace"))
        return path

    # ── start / wait / stop ───────────────────────────────────────────────────────

    async def start(
        self,
        command: str,
        *,
        cwd: str,
        description: str = "",
        extra_env: dict[str, str] | None = None,
        drop_env: list[str] | None = None,
    ) -> BackgroundTask:
        """Spawn ``command`` detached and record it. Returns at once."""
        from zakcode.tools.builtins._proc import child_environment

        tasks_dir = self._prepared_directory()
        task_id = uuid.uuid4().hex[:9]
        output_file = tasks_dir / f"{task_id}.out"
        exit_file = tasks_dir / f"{task_id}.exit"
        child_env = child_environment(cwd, extra_env=extra_env, drop_env=drop_env)
        spawn_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "stdin": subprocess.DEVNULL,
            "env": child_env,
            **new_group_kwargs(),
        }
        bash = find_bash()
        if bash is not None:
            # A wrapper shell runs the command, owns the redirect, and writes the exit code
            # itself — so the exit is recorded even if THIS process is gone by then.
            script = '"$4" -c "$1" >"$2" 2>&1; printf "%s\\n" "$?" >"$3"'
            proc = await asyncio.create_subprocess_exec(
                bash,
                "-c",
                script,
                "zakcode-task",
                command,
                output_file.as_posix(),
                exit_file.as_posix(),
                bash,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **spawn_kwargs,
            )
        else:
            # No bash anywhere: the platform shell, output straight into the file; only the
            # in-process watcher records the exit (a restart in between loses the code).
            with output_file.open("wb") as handle:
                proc = await asyncio.create_subprocess_shell(
                    command, stdout=handle, stderr=subprocess.STDOUT, **spawn_kwargs
                )
        task = BackgroundTask(
            id=task_id,
            command=command,
            description=description,
            cwd=cwd,
            output_file=str(output_file),
            exit_file=str(exit_file),
            pid=proc.pid,
            started_at=datetime.fromtimestamp(self._clock(), tz=UTC).isoformat(),
            start_token=process_start_token(proc.pid),
        )
        _PROCS[task_id] = proc
        watcher = asyncio.create_task(_watch(task_id, proc, str(exit_file)))
        _WATCHERS[task_id] = watcher
        watcher.add_done_callback(lambda _t: _WATCHERS.pop(task_id, None))
        tasks = getattr(self._session, "background_tasks", None)
        if isinstance(tasks, list):
            tasks.append(task)
        else:
            self._session.background_tasks = [task]
        self._changed()
        return task

    async def wait(self, task: BackgroundTask, timeout_seconds: float) -> tuple[str, int | None]:
        """Block until the task is no longer running, or ``timeout_seconds`` pass; either way
        the current ``(status, exit_code)``."""
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            status, code = self.status(task)
            if status != "running" or time.monotonic() >= deadline:
                return status, code
            await asyncio.sleep(_POLL_SECONDS)

    async def stop(self, task_id: str) -> tuple[bool, str]:
        """Kill a running task's whole process group. ``(stopped, reason)`` — ``False`` with
        the reason when there is no such task or it had already exited."""
        task = self.get(task_id)
        if task is None:
            return False, f"no background task with id {task_id!r}"
        status, _ = self.status(task)
        if status != "running":
            return False, f"background task {task_id} is not running (status: {status})"
        task.stopped = True
        self._changed()
        proc = _PROCS.get(task.id)
        if proc is not None:
            await terminate_process_tree(proc)
        else:
            _kill_pid_tree(task.pid)
        return True, "stopped"

    async def kill_all(self) -> int:
        """Kill every task still running — the session is ending (Claude Code does the
        same when it exits). Returns how many were killed."""
        killed = 0
        for task in self.records():
            stopped, _ = await self.stop(task.id)
            killed += int(stopped)
        return killed

    # ── the idle doors ────────────────────────────────────────────────────────────

    def take_notifications(self) -> str | None:
        """One harness line reporting every exited task not yet reported — each marked
        reported — else ``None``. Called from an idle door; never mid-turn."""
        blocks: list[str] = []
        for task in self.records():
            if task.notified:
                continue
            status, code = self.status(task)
            if status == "running":
                continue
            task.notified = True
            blocks.append(notification_block(task, status, code))
        if not blocks:
            return None
        self._changed()
        return NOTIFICATION_HEAD + "\n" + "\n".join(blocks)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()
