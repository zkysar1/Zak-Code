"""Replace a uv tool environment from OUTSIDE it, and put the old one back if that fails.

``zakcode update`` reinstalls with ``uv tool install --force --reinstall``, and uv deletes the
tool's environment before it builds the new one. Windows will not delete the files of a program
that is running, so run from inside the environment it is replacing the delete fails half way: uv
had already removed every dependency and then stopped at ``Scripts\\python.exe``, leaving an
environment that held nothing, and ``zakcode`` could not start again (ADR-0291).

This file is the second process that does the replacement once ``zakcode update`` has exited. It
is COPIED out of the package before it runs and imports nothing from it: the package it sits
beside is the thing being replaced. It needs only the standard library of the interpreter the
environment was built from, which lives outside the environment.

The order is the whole design:

1. wait for the updater to exit;
2. move the environment aside. A rename fails, and nothing has changed, while any other process
   still holds a file in it, so a program that is running is found out here and not half way
   through a delete;
3. run the reinstall into the now-empty place;
4. start the new environment and import the package from it;
5. only then delete the old one.

A failure after step 2 deletes whatever the reinstall made and moves the old environment back, so a
failed update leaves the install that was there. Everything is written to the log, whose last line
says how it ended: ``update complete``, or ``update failed`` and what was done about it.

Exit codes: 0 updated, 1 failed and the old install is back, 2 refused and nothing was touched,
3 failed and the old install could NOT be put back (the log says how to).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

#: How long to wait for the updater to exit, and for the environment to come free, by default.
_WAIT_SECONDS = 120
_MOVE_SECONDS = 60


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _wait_for_exit(pid: int, seconds: float) -> None:
    """Block until process ``pid`` exits or ``seconds`` pass; ``pid <= 0`` means do not wait."""
    if pid <= 0:
        return
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE  # not the default int: a handle is 64 bits
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        synchronize = 0x00100000
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return  # already gone, or not ours to wait on: the move below is the real test
        try:
            kernel32.WaitForSingleObject(handle, int(seconds * 1000))
        finally:
            kernel32.CloseHandle(handle)
        return
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass  # exists, and is not ours: keep waiting
        time.sleep(0.2)


def _move(src: Path, dst: Path, seconds: float) -> OSError | None:
    """Rename ``src`` to ``dst``, retrying for ``seconds``: ``None`` on success, else the error.

    The retry covers the tail of the updater's own process tree: the launcher that started it
    exits a moment after it does. A process that stays holds its files and the rename keeps failing.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            os.replace(src, dst)
            return None
        except OSError as exc:
            if time.monotonic() >= deadline:
                return exc
            time.sleep(0.5)


def _rmtree(path: Path) -> None:
    """Delete a tree, clearing the read-only bit Windows refuses to delete through."""

    def clear_and_retry(func: Any, name: str, _exc: Any) -> None:
        os.chmod(name, stat.S_IWRITE)
        func(name)

    # `onexc` replaced `onerror` in Python 3.12 and this runs under whichever interpreter built the
    # environment, so try the new name and fall back to the old one.
    rmtree: Any = shutil.rmtree
    try:
        rmtree(path, onexc=clear_and_retry)
    except TypeError:
        rmtree(path, onerror=clear_and_retry)


def _starts(verify: list[str]) -> tuple[bool, str]:
    """Run the verify command; ``(ok, what it printed or the tail of why not)``."""
    try:
        proc = subprocess.run(verify, capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[-1500:]
    return True, proc.stdout.strip()


def run(job: dict[str, Any], log: TextIO) -> int:
    """Carry out one update job (see the module docstring). Returns the exit code."""

    def say(message: str) -> None:
        log.write(f"{_stamp()} {message}\n")
        log.flush()

    env_dir, backup = Path(job["env_dir"]), Path(job["backup_dir"])
    verify: list[str] = job["verify"]
    say(f"update started for {env_dir}")
    _wait_for_exit(int(job.get("wait_pid", 0)), float(job.get("wait_seconds", _WAIT_SECONDS)))

    if backup.exists():
        # A run that was cut off between moving the install aside and finishing leaves this behind.
        # Delete it only when the install in place works; otherwise it may be the only good copy.
        if env_dir.exists() and _starts(verify)[0]:
            _rmtree(backup)
            say(f"removed {backup}, left by an earlier update, since the install in place works")
        else:
            say(
                f"update failed: {backup} is left from an earlier update and {env_dir} does not "
                f"start. Nothing was touched. To restore by hand: delete {env_dir} if it exists, "
                f"then rename {backup} to {env_dir}."
            )
            return 2

    had_old = env_dir.exists()
    if had_old:
        error = _move(env_dir, backup, float(job.get("move_seconds", _MOVE_SECONDS)))
        if error is not None:
            say(
                f"update failed: {env_dir} cannot be moved ({error}). Another program is still "
                "running from it, so the install was left as it was. Close every zakcode session, "
                "the web app included, and run zakcode update again."
            )
            return 2
        say(f"moved the install to {backup}")
    else:
        say(f"no install at {env_dir}; installing fresh")

    def roll_back(reason: str) -> int:
        say(f"update failed: {reason}")
        try:
            if env_dir.exists():
                _rmtree(env_dir)
            if had_old:
                os.replace(backup, env_dir)
        except OSError as exc:
            say(
                f"the old install could NOT be put back ({exc}). It is intact in {backup}: delete "
                f"{env_dir} if it exists, then rename {backup} to {env_dir}."
            )
            return 3
        say(
            "restored the previous install, as it was"
            if had_old
            else "no previous install to restore"
        )
        return 1

    try:
        for command in job["commands"]:
            say("running: " + " ".join(command))
            proc = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                cwd=job.get("cwd") or None,
                check=False,
            )
            if proc.returncode != 0:
                return roll_back(f"{Path(command[0]).name} exited with status {proc.returncode}")
        ok, printed = _starts(verify)
        if not ok:
            return roll_back(f"the new install does not start: {printed}")
    except BaseException as exc:  # noqa: BLE001 - whatever stopped us, the old install comes back
        say(traceback.format_exc().rstrip())
        return roll_back(f"{type(exc).__name__}: {exc}")

    new = printed
    if job.get("new_sha"):
        new = new.replace("(local path)", f"(git {job['new_sha']})")
    say(f"updated  {job.get('old', '?')}  ->  {new}")
    if had_old:
        try:
            _rmtree(backup)
        except OSError as exc:
            say(f"could not delete {backup} ({exc}); it is the old install, safe to delete by hand")
    say("update complete")
    return 0


def main(argv: list[str]) -> int:
    job_file = Path(argv[1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    with open(job["log"], "a", encoding="utf-8") as log:
        return run(job, log)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
