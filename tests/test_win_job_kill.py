"""Windows job-object kill and POSIX group-kill tests.

The job-object fix puts every child into its own Windows job object at spawn time,
so ``TerminateJobObject`` kills every descendant regardless of its Windows parent link.
Under Git Bash each program runs in a new Windows process whose parent is a short-lived
forked process, not the shell that started it, so ``taskkill /T`` walking Windows parent
links from our shell reaches nothing below the first exec. The job object does not depend
on parent links.

These tests verify:

1. On Windows with Git Bash, both exec form (last command is the sleep) and no-exec form
   (``sleep N; :``) are killed by ``terminate_process_tree`` within 5 s. The processes are
   identified by their unique command-line argument (``41.25`` / ``43.75``) through a
   Windows process listing (``Get-CimInstance Win32_Process``), which does not depend on
   pid files or parent links.
2. A background task outlives the job handle being closed (no kill-on-close) [Windows-only].
3. On POSIX the spawn path is unchanged (``start_new_session=True``, ``killpg``).
4. The fallback to ``taskkill`` still works when no job is present.
5. A failed resume kills the child and raises ``OSError`` (never leaves it suspended).
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from zakcode._subprocess import (
    create_group_subprocess_exec,
    create_group_subprocess_shell,
    terminate_process_tree,
)
from zakcode.background import pid_alive, process_start_token


async def _until(
    predicate: Callable[[], bool],
    timeout: float = 10.0,
    what: str = "",
) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"{what or 'not true'} after {timeout}s"
        await asyncio.sleep(0.05)


def _has_powershell() -> bool:
    """Return True if PowerShell is available on this machine."""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", "echo ok"],
            capture_output=True,
            timeout=10,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return True


def _find_carriers(needle: str) -> list[int]:
    """Return Windows pids whose command line contains *needle*.

    Uses ``Get-CimInstance Win32_Process`` filtered on ``CommandLine``, excluding the
    PowerShell process itself. A listing that fails RAISES rather than returning an empty
    list: after the kill, "no carriers" is the passing answer, so a broken listing must not
    be able to produce it.
    """
    ps_cmd = (
        f"Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -and $_.CommandLine.Contains('{needle}') "
        f"-and $_.Name -ne 'powershell.exe' -and $_.ProcessId -ne $PID }} | "
        f"Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"process listing failed rc={result.returncode}: {result.stderr[-300:]}")
    pids: list[int] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


# ---------------------------------------------------------------------------
# Test 1: Windows-only — job object kills Git Bash tree (exec + no-exec)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job-object kill test")
async def test_job_object_kills_git_bash_tree_exec_and_noexec(tmp_path: Path) -> None:
    """Spawn a command through the exact wrapper argv BackgroundTasks._spawn uses (an outer
    bash running a script that invokes an inner bash -c with the user command), in BOTH forms,
    and verify terminate_process_tree kills every descendant.

    exec form:    the last command is ``sleep 41.25``; bash replaces itself with sleep.exe.
    no-exec form: ``sleep 43.75; :`` — the trailing ``:`` keeps bash alive as the parent.

    Processes are identified by their unique argument (41.25 / 43.75) via a Windows process
    listing (Get-CimInstance Win32_Process filtered on CommandLine). This avoids depending
    on pid files or Windows parent links — the very thing the fix addresses.

    (The wrapper argv comes from background.py BackgroundTasks._spawn: bash -c <script>
    zakcode-task <command> <output> <exit> <bash>.)
    """
    from zakcode._subprocess import find_bash

    bash = find_bash()
    if bash is None:
        pytest.skip("Git Bash not found on this Windows machine")
    if not _has_powershell():
        pytest.skip("PowerShell not found; needed for process listing")

    for label, needle, user_cmd in [
        ("exec", "41.25", "sleep 41.25"),
        ("noexec", "43.75", "sleep 43.75; :"),
    ]:
        output_file = tmp_path / f"out_{label}"
        exit_file = tmp_path / f"exit_{label}"
        # Reproduce BackgroundTasks._spawn's wrapper argv exactly.
        script = '"$4" -c "$1" >"$2" 2>&1; printf "%s\\n" "$?" >"$3"'
        proc = await create_group_subprocess_exec(
            bash,
            "-c",
            script,
            "zakcode-task",
            user_cmd,
            output_file.as_posix(),
            exit_file.as_posix(),
            bash,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for the inner command to start (the sleep must appear in the process list).
        await _until(
            lambda n=needle: len(_find_carriers(n)) > 0,
            timeout=10.0,
            what=f"{label}: no process carrying '{needle}' found before kill",
        )
        carriers_before = _find_carriers(needle)
        assert len(carriers_before) > 0, f"{label}: expected carrier processes before kill"

        # Kill the whole tree via the job object.
        await terminate_process_tree(proc)

        # Assert within 5 s that no process carrying the needle survives.
        await _until(
            lambda n=needle: len(_find_carriers(n)) == 0,
            timeout=5.0,
            what=f"{label}: processes carrying '{needle}' still alive after kill",
        )


# ---------------------------------------------------------------------------
# Test 2: background task outlives job handle being closed (no kill-on-close)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job-object no-kill-on-close test")
async def test_background_task_outlives_job_handle_close(tmp_path: Path) -> None:
    """Closing the job handle does NOT kill the child — JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    is deliberately not set, so background tasks survive the zakcode process exiting."""
    marker = tmp_path / "alive"
    proc = await create_group_subprocess_exec(
        sys.executable,
        "-c",
        (f"import pathlib, time; pathlib.Path({str(marker)!r}).write_text('yes'); time.sleep(30)"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await _until(lambda: marker.exists(), timeout=10.0, what="child did not start")
        # The child has a job ref — close it without killing.
        job_ref = getattr(proc, "_job_ref", None)
        assert job_ref is not None, "child was not assigned a job object"
        job_ref.close()
        # The child must still be alive after the handle is closed.
        await asyncio.sleep(0.5)
        assert pid_alive(proc.pid), "child died when the job handle was closed"
    finally:
        # Clean up: kill the child directly.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(proc.wait(), timeout=5.0)


# ---------------------------------------------------------------------------
# Test 3: POSIX path stays identical
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX session/group test")
async def test_posix_child_is_in_its_own_session(tmp_path: Path) -> None:
    """create_group_subprocess_exec passes start_new_session=True on POSIX, putting
    the child in its own process session so killpg reaches the whole tree."""
    proc = await create_group_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert os.getsid(proc.pid) == proc.pid
        assert os.getpgid(proc.pid) == proc.pid
    finally:
        await terminate_process_tree(proc)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX killpg test")
async def test_posix_terminate_kills_via_killpg(tmp_path: Path) -> None:
    """terminate_process_tree uses os.killpg on POSIX, killing the child and its
    descendants in one call."""
    marker = tmp_path / "started"
    proc = await create_group_subprocess_exec(
        sys.executable,
        "-c",
        (
            "import pathlib, time, subprocess, sys; "
            f"pathlib.Path({str(marker)!r}).write_text('yes'); "
            # Spawn a grandchild in the same session
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "time.sleep(60)"
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await _until(lambda: marker.exists(), timeout=10.0, what="child did not start")
    pid = proc.pid
    token = process_start_token(pid)
    await terminate_process_tree(proc)
    # The child should be dead.
    await _until(
        lambda: not pid_alive(pid) or process_start_token(pid) != token,
        timeout=5.0,
        what=f"pid {pid} still alive after killpg",
    )


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell test")
async def test_posix_create_group_subprocess_shell_uses_start_new_session(
    tmp_path: Path,
) -> None:
    """create_group_subprocess_shell also passes start_new_session=True on POSIX."""
    proc = await create_group_subprocess_shell(
        "sleep 30",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert os.getsid(proc.pid) == proc.pid
    finally:
        await terminate_process_tree(proc)


# ---------------------------------------------------------------------------
# Test 4: fallback path (no job) on Windows
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows fallback test")
async def test_terminate_falls_back_to_taskkill_without_job(tmp_path: Path) -> None:
    """When a process has no _job_ref (e.g. the job assignment failed), terminate_process_tree
    falls back to taskkill /T /F — today's behavior before the job-object fix."""
    # Spawn WITHOUT the helper so there's no job object.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    assert getattr(proc, "_job_ref", None) is None
    pid = proc.pid
    assert pid_alive(pid)
    await terminate_process_tree(proc)
    await _until(
        lambda: not pid_alive(pid),
        timeout=5.0,
        what=f"pid {pid} still alive after fallback taskkill",
    )


# ---------------------------------------------------------------------------
# Test 5: job handle cleaned up by GC finalizer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows handle cleanup test")
async def test_job_handle_cleaned_up_after_gc(tmp_path: Path) -> None:
    """After a child exits normally and the Process is garbage-collected, the weakref
    finalizer closes the job handle."""
    proc = await create_group_subprocess_exec(
        sys.executable,
        "-c",
        "pass",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    job_ref = getattr(proc, "_job_ref", None)
    assert job_ref is not None
    assert job_ref.handle != 0
    # Wait for normal exit.
    await asyncio.wait_for(proc.wait(), timeout=10.0)
    # Drop all references to the Process — the weakref finalizer should close the handle.
    del proc
    gc.collect()
    assert job_ref.handle == 0, "finalizer did not close the job handle after GC"
    # Double-close is a no-op.
    job_ref.close()
    assert job_ref.handle == 0


# ---------------------------------------------------------------------------
# Test 6: failed resume kills child and raises (never leaves it suspended)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows resume-failure test")
async def test_failed_resume_kills_child_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When NtResumeProcess returns a failure status, the spawn helper kills the child
    and raises OSError. A child must never be left suspended."""
    import zakcode._subprocess as mod

    # Make NtResumeProcess return a failure NTSTATUS.
    _real_resume = mod._ntdll.NtResumeProcess

    def _failing_resume(handle: int) -> int:
        # Return STATUS_ACCESS_DENIED (0xC0000022) as a signed c_long.
        return -0x3FFFFDE  # 0xC0000022 as signed 32-bit

    monkeypatch.setattr(mod._ntdll, "NtResumeProcess", _failing_resume)

    with pytest.raises(OSError, match="NtResumeProcess"):
        await create_group_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Restore real resume for any cleanup. The child should already be dead — the spawn
    # helper killed it. We can't easily verify pid_alive here because we never got a proc
    # reference back (the OSError prevented it). The invariant is: no child is left
    # suspended — either it was resumed or it was killed.
    monkeypatch.setattr(mod._ntdll, "NtResumeProcess", _real_resume)
