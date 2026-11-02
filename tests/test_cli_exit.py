"""A clean /exit leaves the process with exit code 0, whoever holds its stdin (ADR-0353)."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest


def test_a_clean_exit_is_exit_code_0_while_the_stdin_pipe_stays_open(tmp_path: Path) -> None:
    """The input pump is a daemon thread, still blocked in a stdin read when the REPL returns.
    When stdin and stdout are not both terminals, ``input()`` reads through stdin's buffered
    reader and holds its lock. On CPython 3.11 ``input()`` keeps no reference of its own to
    stdin, so interpreter shutdown freed stdin, and closing it needs that lock: the process
    aborted after the goodbye line with "Fatal Python error: _enter_buffered_busy", and a
    wrapper that keeps the pipe open until the process exits read a failure from a clean /exit.
    """
    out, err = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    env = dict(os.environ, ZAKCODE_WORKSPACE_ROOT=str(tmp_path), PYTHONIOENCODING="utf-8")
    with out.open("wb") as stdout, err.open("wb") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-m", "zakcode", "cli"],
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        assert proc.stdin is not None
        try:
            proc.stdin.write(b"/exit\n")
            proc.stdin.flush()
            code = proc.wait(timeout=60)  # the pipe stays open until the process has gone
        finally:
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
    said = err.read_text(encoding="utf-8", errors="replace")
    assert "goodbye" in out.read_text(encoding="utf-8", errors="replace"), said
    assert (code, "Fatal Python error" in said) == (0, False), said


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX pseudo-terminal")
def test_a_clean_exit_on_a_terminal_is_exit_code_0(tmp_path: Path) -> None:
    """On a terminal ``input()`` reads through readline, which takes no lock on stdin's
    buffered reader, so the abort above has no way in; this pins the same clean exit for the
    terminal a person types into. ``--no-rules`` keeps the REPL inline: a plain chat on a
    terminal with tmux available opens the cockpit instead.
    """
    import fcntl
    import pty
    import struct
    import termios

    primary, secondary = pty.openpty()
    # A fresh pseudo-terminal reports 0 rows by 0 columns; give it a real size.
    fcntl.ioctl(secondary, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
    err = tmp_path / "stderr.txt"
    env = dict(os.environ, ZAKCODE_WORKSPACE_ROOT=str(tmp_path), PYTHONIOENCODING="utf-8")
    with err.open("wb") as stderr:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "zakcode", "cli", "--no-rules"],
                cwd=tmp_path,
                stdin=secondary,
                stdout=secondary,
                stderr=stderr,
                env=env,
            )
        finally:
            os.close(secondary)  # the child holds its own copies
    heard = b""
    try:
        os.write(primary, b"/exit\n")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if select.select([primary], [], [], 0.2)[0]:
                try:
                    chunk = os.read(primary, 65536)
                except OSError:  # every holder of the terminal has closed it
                    break
                if not chunk:
                    break
                heard += chunk
            elif proc.poll() is not None:
                break
        code = proc.wait(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        os.close(primary)
        if proc.poll() is None:
            proc.kill()
    said = err.read_text(encoding="utf-8", errors="replace")
    assert b"goodbye" in heard, said
    assert (code, "Fatal Python error" in said) == (0, False), said
