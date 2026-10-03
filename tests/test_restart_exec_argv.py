"""The self-restart's exec must hand the fresh process the arguments it was given.

zakcode restarts into a new build by replacing itself with ``os.execv`` (ADR-0034). On
Windows there is no exec: the C runtime starts a new process from ONE command line, which it
builds by joining the arguments with spaces and quoting none of them, and the old process
exits. The new process splits that line back into arguments, so an argument with a space in
it, a double quote, or nothing at all can arrive split, merged or dropped.

Each case here execs for real, in the restart's own call shape (the interpreter's path, then
the interpreter as argv[0], then the arguments) and through its quoting (``_exec_argv``),
into a helper that writes down the argv it received. A case sends one kind of argument
between two plain ones, so a failure names the kind that broke. The control sends an
argument that needs no quoting anywhere: it must pass on every platform, or the probe itself
is broken.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zakcode.cli import _exec_argv

#: One of each kind of argument the quoting must carry, the interpreter's path first.
_KINDS = [
    "C:\\py dir\\python.exe",
    "-m",
    "plain",
    "two words",
    "C:\\some dir\\",
    'say"hi',
    "",
]


def test_posix_hands_execv_the_argv_as_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert _exec_argv(list(_KINDS)) == _KINDS


def test_windows_quotes_each_argument_for_the_command_line_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert _exec_argv(list(_KINDS)) == [
        '"C:\\py dir\\python.exe"',  # argv[0] too: the new process splits it like the rest
        "-m",
        "plain",  # needs no quoting, so gets none
        '"two words"',
        '"C:\\some dir\\\\"',  # a backslash before the closing quote is doubled
        'say\\"hi',  # an inner quote is escaped, so it opens nothing
        '""',
    ]


#: Replaces itself with ``os.execv`` in the restart's call shape. The interpreter and the
#: argv come in a JSON file, so no command line on the way here can re-split them.
_EXEC_INTO = """\
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    interpreter, argv = json.load(fh)
os.execv(interpreter, argv)
"""

#: The fresh process. Writes down every argument after its own two (this script's path and
#: the report's), to a temporary name first so the test never reads a half-written report.
_REPORT_ARGV = """\
import json
import os
import sys

out = sys.argv[1]
with open(out + ".tmp", "w", encoding="utf-8") as fh:
    json.dump(sys.argv[2:], fh)
os.replace(out + ".tmp", out)
"""


def _received(tmp_path: Path, args: list[str], interpreter: str = sys.executable) -> list[str]:
    """Exec ``interpreter`` with ``args`` the way the restart does; return what arrived.

    On Windows the exec'd process outlives the one ``subprocess.run`` waits for, so the
    report is waited for rather than assumed present when the run returns.
    """
    exec_into = tmp_path / "exec_into.py"
    report = tmp_path / "report_argv.py"
    out = tmp_path / "argv.json"
    spec = tmp_path / "exec.json"
    exec_into.write_text(_EXEC_INTO, encoding="utf-8")
    report.write_text(_REPORT_ARGV, encoding="utf-8")
    argv = _exec_argv([interpreter, str(report), str(out), *args])
    spec.write_text(json.dumps([interpreter, argv]), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(exec_into), str(spec)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    deadline = time.monotonic() + 30
    while not out.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert out.exists(), (
        f"the exec'd process never reported (exit {run.returncode}): {run.stderr[-600:]!r}"
    )
    received: list[str] = json.loads(out.read_text(encoding="utf-8"))
    return received


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param("plain", id="control"),
        pytest.param("two words", id="a-space"),
        pytest.param("C:\\Users\\some user\\project", id="a-spaced-path"),
        pytest.param("C:\\some dir\\", id="a-spaced-path-ending-in-a-backslash"),
        pytest.param('say"hi', id="a-quote-and-no-space"),
        pytest.param("", id="an-empty-argument"),
    ],
)
def test_an_argument_reaches_the_fresh_process_intact(tmp_path: Path, arg: str) -> None:
    sent = ["first", arg, "last"]
    got = _received(tmp_path, sent)
    assert got == sent, f"sent {sent!r}, the fresh process received {got!r}"


def _interpreter_under_a_space(tmp_path: Path) -> str:
    """A working copy of this virtual environment's interpreter, at a path with a space in it.

    A venv's interpreter finds the real one through the ``pyvenv.cfg`` above it, so copying
    the interpreter's own files (on Windows a launcher, or the interpreter and its DLLs;
    elsewhere a symlink) and that file is enough to run it from another directory.
    """
    if sys.prefix == sys.base_prefix:
        pytest.skip("needs a virtual environment's interpreter to copy")
    here = Path(sys.executable)
    home = tmp_path / "python with space"
    bindir = home / here.parent.name
    bindir.mkdir(parents=True)
    exe = bindir / here.name
    if os.name == "nt":
        for f in here.parent.iterdir():
            if f.suffix.lower() == ".dll" or f.name.lower().startswith("python"):
                shutil.copy2(f, bindir / f.name)
    else:
        exe.symlink_to(os.path.realpath(here))
    shutil.copy2(Path(sys.prefix) / "pyvenv.cfg", home / "pyvenv.cfg")
    # The copy must start when it is started correctly (subprocess quotes for Windows), so a
    # failure below is the exec's, not the copy's.
    started = subprocess.run(
        [str(exe), "-c", "print('ok')"], capture_output=True, text=True, timeout=60
    )
    assert started.stdout.strip() == "ok", f"the copy does not start: {started.stderr[-600:]!r}"
    return str(exe)


def test_an_interpreter_path_with_a_space_survives_the_exec(tmp_path: Path) -> None:
    interpreter = _interpreter_under_a_space(tmp_path)
    sent = ["first", "last"]
    got = _received(tmp_path, sent, interpreter=interpreter)
    assert got == sent, f"exec'd {interpreter!r} with {sent!r}; the fresh process got {got!r}"
