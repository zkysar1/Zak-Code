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
2. write to the log how to put the old environment back by hand, then move it aside. A rename
   fails, and nothing has changed, while another process has a file in it open or has it as its
   working directory. A program that is merely running from it does NOT stop the rename (measured
   on Windows, ADR-0291): the updater's listing is what refuses that case, before anything is
   pulled;
3. run the reinstall into the now-empty place;
4. start the new environment and import the package from it;
5. only then delete the old one. A program still running from it keeps its files from being
   deleted: the log says so and leaves the folder, which the next update removes.

A failure after step 2 deletes whatever the reinstall made and moves the old environment back, so a
failed update leaves the environment that was there. The checkout the updater pulled stays pulled:
the job carries a note saying so, and how to undo it, for the log. Everything is written to the log,
whose last lines say how it ended: ``update complete``, or ``update failed`` and what was done about
it. If this process is itself killed, the line written at step 2 is what is left to follow.

Exit codes: 0 updated, 1 failed and the old environment is back, 2 refused and nothing was moved,
3 failed and the old environment may NOT be back (the log says how to check and what to do).
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

#: How long to wait for the updater to exit, and by default for the environment to come free.
_WAIT_SECONDS = 120
_MOVE_SECONDS = 60
#: Win32 ``CREATE_NO_WINDOW``. This process is detached, so it has no console, and a console
#: program it starts would be given a new visible one; closing that window would kill the reinstall
#: half way and the person would never learn why. Zero elsewhere, where the flag does not exist.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


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
    exits a moment after it does. A process that stays, with a file in the folder open or the folder
    as its working directory, keeps the rename failing.
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


def _is_link(path: Path) -> bool:
    """True for a symlink or, on Windows, a directory junction: something that points elsewhere."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    junction = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)
    return stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) == junction


def _rmtree(path: Path) -> None:
    """Delete a tree, clearing the read-only bit Windows refuses to delete through.

    A link is removed as the link it is and never followed: ``shutil.rmtree`` refuses one, and the
    handler below would then ``chmod`` the directory it points at and report success with the link
    still in place.
    """
    if _is_link(path):
        try:
            os.unlink(path)
        except OSError:
            os.rmdir(path)  # a directory link on Windows is removed like an empty directory
        return

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
        # `errors="replace"`: a byte the locale cannot decode must not turn a healthy install into
        # a failed one (the reader thread would drop the output and `.strip()` would raise).
        proc = subprocess.run(
            verify,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=120,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[-1500:]
    return True, proc.stdout.strip()


def _by_hand(env_dir: Path, backup: Path) -> str:
    """The two steps that put the old environment back without this program.

    The paths are quoted: a Windows profile folder holds a space, and unquoted the reader cannot
    tell where a path ends and the next word begins. The new name is the folder's own name,
    because the rename commands of both shells take a name there and not a path.
    """
    return f'delete "{env_dir}" if it exists, then rename the folder "{backup}" to "{env_dir.name}"'


def run(job: dict[str, Any], log: TextIO) -> int:
    """Carry out one update job (see the module docstring). Returns the exit code.

    Optional job keys: ``checkout_note``, a sentence about the checkout the updater pulled, which
    stays pulled when the update stops and is written as the last word of every failure; and
    ``move_seconds``, how long the environment is given to come free.
    """

    def say(message: str) -> None:
        log.write(f"{_stamp()} {message}\n")
        log.flush()

    note = str(job.get("checkout_note") or "")

    def fail(code: int) -> int:
        if note:
            say(note)
        return code

    env_dir, backup = Path(job["env_dir"]), Path(job["backup_dir"])
    verify: list[str] = job["verify"]
    say(f"update started for {env_dir}")
    _wait_for_exit(int(job.get("wait_pid", 0)), _WAIT_SECONDS)

    if backup.exists():
        # A run that was cut off between moving the install aside and finishing leaves this behind,
        # and so does one that could not delete the old install because a program ran from it.
        # Delete it only when the install in place works; otherwise it may be the only good copy.
        if env_dir.exists() and _starts(verify)[0]:
            try:
                _rmtree(backup)
            except OSError as exc:
                say(
                    f"update failed: {backup}, left by an earlier update, cannot be deleted "
                    f"({exc}). A program is probably still running from it. Close that program "
                    "and run zakcode update again, or delete the folder by hand. The install in "
                    "place was not touched."
                )
                return fail(2)
            say(f"removed {backup}, left by an earlier update, since the install in place works")
        else:
            say(
                f"update failed: {backup} is left from an earlier update and {env_dir} is "
                f"missing or does not start. Nothing was touched. To restore by hand: "
                f"{_by_hand(env_dir, backup)}."
            )
            return fail(2)

    had_old = env_dir.exists()
    if had_old:
        # Written before the move, so that a process killed after it (a reboot, a logoff, Task
        # Manager) still leaves the way back in the log: nothing after this line can run then.
        say(
            "if this log stops here, the update was cut off: to put the old install back, "
            f"{_by_hand(env_dir, backup)}"
        )
        error = _move(env_dir, backup, float(job.get("move_seconds", _MOVE_SECONDS)))
        if error is not None:
            say(
                f"update failed: {env_dir} cannot be moved ({error}). Another program has a file "
                "in it open or is working in that folder, so the environment was left as it was. "
                "Close every zakcode session, the web app included, and any window or editor open "
                "in that folder, and run zakcode update again."
            )
            return fail(2)
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
                f"the old install could NOT be put back ({exc}). It is intact in {backup}: "
                f"{_by_hand(env_dir, backup)}."
            )
            return fail(3)
        say(
            "restored the previous environment, as it was"
            if had_old
            else "no previous install to restore"
        )
        return fail(1)

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
                creationflags=_NO_WINDOW,
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
            say(
                f"could not delete {backup} ({exc}). It is the old install, and a program is "
                "probably still running from it: restart that program to use the new install. "
                "The folder is safe to delete by hand once it has exited, and the next update "
                "deletes it."
            )
    say("update complete")
    return 0


def main(argv: list[str]) -> int:
    job_file = Path(argv[1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    with open(job["log"], "a", encoding="utf-8") as log:
        try:
            return run(job, log)
        except BaseException:  # noqa: BLE001 - this process is detached: the log is all there is
            env_dir, backup = Path(job["env_dir"]), Path(job["backup_dir"])
            log.write(
                f"{_stamp()} update failed unexpectedly:\n{traceback.format_exc().rstrip()}\n"
                f"{_stamp()} {env_dir} may or may not have been moved aside. If it is missing or "
                f"does not start and {backup} exists, {_by_hand(env_dir, backup)}.\n"
            )
            log.flush()
            return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
