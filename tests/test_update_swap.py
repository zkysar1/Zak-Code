"""``swap_install``: the second process that replaces a uv tool environment (ADR-0291).

These run the real file against a real directory tree. The reinstall and "does it start" commands
are small Python programs standing in for ``uv tool install`` and the new environment's
interpreter, so a failure can be put at any step and the tree compared byte for byte afterwards.
What only Windows can show (a rename that fails while a file in the directory is open) has its own
Windows-only test here. What a real program does to the rename and to the delete, whether it runs
from the directory or only works in it, is the job of ``test_update_e2e.py``: measured there, a
program that merely runs from the directory does not stop the rename.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from zakcode.cli import swap_install

PY = sys.executable
#: What `uv tool install` leaves: a new environment where the old one was.
_INSTALL = (
    "import pathlib, sys; e = pathlib.Path(sys.argv[1]); e.mkdir(); "
    "(e / 'version.txt').write_text('new')"
)
_INSTALL_THEN_FAIL = _INSTALL + "; print('boom: resolver failed'); sys.exit(3)"
#: The environment's interpreter importing the package: prints what is in the environment.
_SHOW = "import pathlib, sys; print(pathlib.Path(sys.argv[1], 'version.txt').read_text())"
_SHOW_BUILD = (
    "import pathlib, sys; "
    "print('0.0.1 (local path)' if pathlib.Path(sys.argv[1], 'version.txt').read_text() == 'new' "
    "else 'STALE')"
)
#: What the updater tells the helper to say about the checkout it pulled, if the update stops.
_NOTE = (
    "The checkout /work/zak was pulled from abc1234 to def5678 and stays there. An editable "
    'install runs the checkout\'s code, so to run the old code again: git -C "/work/zak" '
    "reset --hard abc1234"
)


def _old_env(env: Path) -> None:
    """An environment with nested directories and binary files, to compare after a restore."""
    (env / "Lib" / "site-packages").mkdir(parents=True)
    (env / "Scripts").mkdir()
    (env / "version.txt").write_text("old")
    (env / "pyvenv.cfg").write_text("home = elsewhere\n")
    (env / "Lib" / "site-packages" / "dep.py").write_bytes(bytes(range(256)) * 4)
    (env / "Scripts" / "zakcode.exe").write_bytes(b"MZ\x00\x01" * 100)


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Every path under ``root`` with its bytes (``None`` for a directory)."""
    return {
        p.relative_to(root).as_posix(): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def _job(tmp_path: Path, env: Path, *commands: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "env_dir": str(env),
        "backup_dir": str(env.with_name(env.name + ".update-backup")),
        "commands": list(commands),
        "verify": [PY, "-c", _SHOW, str(env)],
        "cwd": str(tmp_path),
        "log": str(tmp_path / "update.log"),
        **extra,
    }


def _run(job: dict[str, Any]) -> tuple[int, str]:
    log_path = Path(job["log"])
    with open(log_path, "a", encoding="utf-8") as log:
        code = swap_install.run(job, log)
    return code, log_path.read_text(encoding="utf-8")


def _install(env: Path) -> list[str]:
    return [PY, "-c", _INSTALL, str(env)]


def test_a_successful_update_replaces_the_install_and_removes_the_backup(tmp_path: Path) -> None:
    env = tmp_path / "toolenv"
    _old_env(env)
    job = _job(
        tmp_path,
        env,
        _install(env),
        verify=[PY, "-c", _SHOW_BUILD, str(env)],
        old="0.0.1 (git 1111111)",
        new_sha="abc1234",
        checkout_note=_NOTE,
    )
    code, log = _run(job)
    assert code == 0, log
    assert (env / "version.txt").read_text() == "new"
    assert not Path(job["backup_dir"]).exists()
    assert "updated  0.0.1 (git 1111111)  ->  0.0.1 (git abc1234)" in log
    # The old install is moved aside BEFORE anything is built, and deleted only at the end.
    assert log.index("moved the install") < log.index("running:") < log.index("updated  ")
    assert log.rstrip().endswith("update complete")
    assert _NOTE not in log, "the note about the checkout belongs to a failure"


def test_the_way_back_is_on_disk_before_the_install_is_moved(tmp_path: Path, monkeypatch) -> None:
    """A process killed after the move (a reboot, a logoff, Task Manager) runs nothing more, so the
    line that says how to undo it has to be written before the move starts. The paths hold a space,
    as a Windows profile folder does, and are quoted so the reader can tell where each one ends."""
    env, backup = tmp_path / "tool env", tmp_path / "tool env.update-backup"
    _old_env(env)
    job = _job(tmp_path, env, _install(env))
    at_the_move: list[str] = []
    real_move = swap_install._move

    def spy(src: Path, dst: Path, seconds: float) -> OSError | None:
        at_the_move.append(Path(job["log"]).read_text(encoding="utf-8"))
        return real_move(src, dst, seconds)

    monkeypatch.setattr(swap_install, "_move", spy)
    code, log = _run(job)
    assert code == 0, log
    (written_by_then,) = at_the_move
    by_hand = f'delete "{env}" if it exists, then rename the folder "{backup}" to "{env.name}"'
    assert "if this log stops here" in written_by_then
    assert by_hand in written_by_then


def test_the_updater_is_waited_for_before_the_install_is_touched(
    tmp_path: Path, monkeypatch
) -> None:
    """The retry in the move would hide a helper that never waited: it keeps trying until the
    updater's own process tree has gone. Only the order shows it."""
    env = tmp_path / "toolenv"
    _old_env(env)
    order: list[tuple[str, int]] = []
    real_move = swap_install._move

    def spy(src: Path, dst: Path, seconds: float) -> OSError | None:
        order.append(("move", 0))
        return real_move(src, dst, seconds)

    monkeypatch.setattr(
        swap_install, "_wait_for_exit", lambda pid, seconds: order.append(("wait", pid))
    )
    monkeypatch.setattr(swap_install, "_move", spy)
    code, log = _run(_job(tmp_path, env, _install(env), wait_pid=4242))
    assert code == 0, log
    assert order == [("wait", 4242), ("move", 0)]


def test_the_programs_it_starts_get_no_console_window_on_windows(
    tmp_path: Path, monkeypatch
) -> None:
    """The helper is detached, so it has no console to share: a console program it starts is given
    a new visible one, and closing that window would kill the reinstall."""
    env = tmp_path / "toolenv"
    _old_env(env)
    seen: list[int] = []
    real_run = subprocess.run

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("creationflags", -1))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(swap_install.subprocess, "run", spy)
    code, log = _run(_job(tmp_path, env, _install(env)))
    assert code == 0, log
    no_window = 0x08000000 if sys.platform == "win32" else 0
    assert seen == [no_window, no_window]  # the reinstall, then the check that it starts


def test_a_failed_command_puts_the_old_install_back_exactly(tmp_path: Path) -> None:
    env = tmp_path / "toolenv"
    _old_env(env)
    before = _snapshot(env)
    code, log = _run(_job(tmp_path, env, [PY, "-c", _INSTALL_THEN_FAIL, str(env)]))
    assert code == 1, log
    assert _snapshot(env) == before  # the half-built new environment is gone, the old one whole
    assert not (tmp_path / "toolenv.update-backup").exists()
    assert "boom: resolver failed" in log  # the command's own output is in the log
    assert f"update failed: {Path(PY).name} exited with status 3" in log
    assert log.rstrip().endswith("restored the previous environment, as it was")


@pytest.mark.parametrize(
    "verify",
    [
        [PY, "-c", "import sys; print('No module named zakcode', file=sys.stderr); sys.exit(1)"],
        ["no-such-program-for-zakcode-tests"],
    ],
    ids=["exits nonzero", "cannot be run"],
)
def test_an_install_that_does_not_start_is_rolled_back(tmp_path: Path, verify: list[str]) -> None:
    env = tmp_path / "toolenv"
    _old_env(env)
    before = _snapshot(env)
    code, log = _run(_job(tmp_path, env, _install(env), verify=verify))
    assert code == 1, log
    assert _snapshot(env) == before
    assert "the new install does not start" in log
    assert log.rstrip().endswith("restored the previous environment, as it was")


def test_an_exception_while_installing_rolls_back_too(tmp_path: Path) -> None:
    env = tmp_path / "toolenv"
    _old_env(env)
    before = _snapshot(env)
    code, log = _run(_job(tmp_path, env, ["no-such-program-for-zakcode-tests"]))
    assert code == 1, log
    assert _snapshot(env) == before
    assert "FileNotFoundError" in log and "Traceback" in log  # what stopped it is on record


def test_an_install_another_program_holds_is_left_alone(tmp_path: Path, monkeypatch) -> None:
    env = tmp_path / "toolenv"
    _old_env(env)
    before = _snapshot(env)
    marker = tmp_path / "ran.txt"
    monkeypatch.setattr(swap_install, "_move", lambda src, dst, seconds: PermissionError("in use"))
    command = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    code, log = _run(_job(tmp_path, env, command))
    assert code == 2, log
    assert _snapshot(env) == before
    assert not marker.exists(), "a command ran although the install could not be moved aside"
    assert "Another program has a file in it open or is working in that folder" in log
    assert "Close every zakcode session" in log


def test_a_move_is_retried_until_the_install_comes_free(tmp_path: Path, monkeypatch) -> None:
    attempts: list[int] = []
    real_replace = swap_install.os.replace

    def flaky(src: Any, dst: Any) -> None:
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(
                "in use"
            )  # the launcher that started the updater is still exiting
        real_replace(src, dst)

    monkeypatch.setattr(swap_install.os, "replace", flaky)
    monkeypatch.setattr(swap_install.time, "sleep", lambda seconds: None)
    (tmp_path / "a").mkdir()
    assert swap_install._move(tmp_path / "a", tmp_path / "b", 60) is None
    assert len(attempts) == 3 and (tmp_path / "b").is_dir()


def test_a_move_that_never_works_returns_the_error(tmp_path: Path, monkeypatch) -> None:
    def refuse(src: Any, dst: Any) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(swap_install.os, "replace", refuse)
    error = swap_install._move(tmp_path / "a", tmp_path / "b", 0)
    assert isinstance(error, PermissionError)


def test_a_backup_from_a_cut_off_update_is_cleared_when_the_install_works(tmp_path: Path) -> None:
    env, backup = tmp_path / "toolenv", tmp_path / "toolenv.update-backup"
    _old_env(env)
    backup.mkdir()
    (backup / "stale.txt").write_text("an earlier run's leftovers")
    code, log = _run(_job(tmp_path, env, _install(env)))
    assert code == 0, log
    assert "removed" in log and "left by an earlier update" in log
    assert (env / "version.txt").read_text() == "new" and not backup.exists()


def _locked(path: Path) -> None:
    raise PermissionError("[WinError 5] Access is denied")


def test_an_old_install_a_program_still_holds_is_left_and_the_update_still_completes(
    tmp_path: Path, monkeypatch
) -> None:
    """Measured on Windows (ADR-0291): a program running from the old install does not stop the
    move, and does stop the delete at the end. The new install is already in place and works, so
    the update is complete; the log says what is left and why, and the next update clears it."""
    env, backup = tmp_path / "toolenv", tmp_path / "toolenv.update-backup"
    _old_env(env)
    old = _snapshot(env)
    with monkeypatch.context() as held:
        held.setattr(swap_install, "_rmtree", _locked)
        code, log = _run(_job(tmp_path, env, _install(env)))
    assert code == 0, log
    assert (env / "version.txt").read_text() == "new"
    assert _snapshot(backup) == old  # only the delete that failed could have removed anything
    assert "could not delete" in log and "safe to delete by hand" in log
    assert "restart that program" in log
    assert log.rstrip().endswith("update complete")
    # The next update finds the leftover, sees the install in place works, and deletes it.
    (tmp_path / "update.log").unlink()
    code, log = _run(_job(tmp_path, env, _install(env)))
    assert code == 0, log
    assert "removed" in log and "left by an earlier update" in log
    assert not backup.exists()


def test_a_leftover_backup_that_cannot_be_deleted_stops_the_update_before_anything_moves(
    tmp_path: Path, monkeypatch
) -> None:
    """The leftover may still have a program running from it. That is not a crash and not a reason
    to touch the install in place: the update says so and stops, with the note on the checkout."""
    env, backup = tmp_path / "toolenv", tmp_path / "toolenv.update-backup"
    _old_env(env)
    backup.mkdir()
    (backup / "stale.txt").write_text("an earlier run's leftovers")
    before = _snapshot(env)
    marker = tmp_path / "ran.txt"
    command = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    monkeypatch.setattr(swap_install, "_rmtree", _locked)
    code, log = _run(_job(tmp_path, env, command, checkout_note=_NOTE))
    assert code == 2, log
    assert _snapshot(env) == before
    assert not marker.exists(), "a command ran although the leftover could not be cleared"
    assert "cannot be deleted" in log and "run zakcode update again" in log
    assert "The install in place was not touched" in log
    assert "unexpectedly" not in log and "Traceback" not in log
    assert log.rstrip().endswith(_NOTE)


@pytest.mark.parametrize("env_state", ["missing", "does not start"])
def test_a_backup_is_kept_when_the_install_in_place_cannot_vouch_for_it(
    tmp_path: Path, env_state: str
) -> None:
    """The backup may be the only good copy: nothing is touched and the log says how to restore."""
    env, backup = tmp_path / "toolenv", tmp_path / "toolenv.update-backup"
    _old_env(backup)
    if env_state == "does not start":
        env.mkdir()  # no version.txt, so the verify command fails
    before = _snapshot(backup)
    marker = tmp_path / "ran.txt"
    command = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    code, log = _run(_job(tmp_path, env, command))
    assert code == 2, log
    assert _snapshot(backup) == before
    assert not marker.exists()
    assert "To restore by hand" in log and str(backup) in log and str(env) in log
    assert "is missing or does not start" in log


def test_when_the_old_install_cannot_be_put_back_the_log_says_how(
    tmp_path: Path, monkeypatch
) -> None:
    env, backup = tmp_path / "toolenv", tmp_path / "toolenv.update-backup"
    _old_env(env)
    before = _snapshot(env)

    def locked(path: Path) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(swap_install, "_rmtree", locked)
    code, log = _run(_job(tmp_path, env, [PY, "-c", _INSTALL_THEN_FAIL, str(env)]))
    assert code == 3, log
    assert _snapshot(backup) == before  # the old install is intact where it was moved
    assert "could NOT be put back" in log and "rename" in log and str(backup) in log


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [
        ("rolled back", 1),
        ("cannot be moved", 2),
        ("cannot be put back", 3),
        ("stale backup", 2),
        ("undeletable backup", 2),
    ],
)
def test_every_failure_ends_with_the_note_about_the_checkout(
    tmp_path: Path, monkeypatch, failure: str, exit_code: int
) -> None:
    """The updater pulled the checkout and nothing here un-pulls it, so a failure that says the
    install was left as it was must say what is true of the checkout (ADR-0291)."""
    env, backup = tmp_path / "toolenv", tmp_path / "toolenv.update-backup"
    _old_env(env)
    command = [PY, "-c", _INSTALL_THEN_FAIL, str(env)]
    if failure == "cannot be moved":
        monkeypatch.setattr(swap_install, "_move", lambda src, dst, seconds: PermissionError("x"))
    if failure == "cannot be put back":

        def locked(path: Path) -> None:
            raise PermissionError("locked")

        monkeypatch.setattr(swap_install, "_rmtree", locked)
    if failure == "stale backup":
        backup.mkdir()
        (env / "version.txt").unlink()  # the install in place cannot vouch for the backup
    if failure == "undeletable backup":
        backup.mkdir()
        monkeypatch.setattr(swap_install, "_rmtree", _locked)
    code, log = _run(_job(tmp_path, env, command, checkout_note=_NOTE))
    assert code == exit_code, log
    assert log.rstrip().endswith(_NOTE)


def test_an_exception_outside_the_install_steps_still_ends_in_the_log(
    tmp_path: Path, monkeypatch
) -> None:
    """Whatever raises in ``run`` before the step that rolls back escapes it (here the wait for the
    updater). The helper is detached and its stderr goes nowhere, so without this the log would stop
    at "update started" and say nothing."""
    env, backup = tmp_path / "tool env", tmp_path / "tool env.update-backup"
    _old_env(env)

    def broken(pid: int, seconds: float) -> None:
        raise RuntimeError("the wait broke")

    monkeypatch.setattr(swap_install, "_wait_for_exit", broken)
    job = _job(tmp_path, env, _install(env), wait_pid=4242)
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    code = swap_install.main(["swap_install.py", str(tmp_path / "job.json")])
    log = Path(job["log"]).read_text(encoding="utf-8")
    assert code == 3, log
    assert "update failed unexpectedly" in log and "Traceback" in log and "RuntimeError" in log
    assert f'delete "{env}" if it exists, then rename the folder "{backup}" to "{env.name}"' in log


def test_an_update_with_no_install_to_replace_installs_fresh(tmp_path: Path) -> None:
    env = tmp_path / "toolenv"
    code, log = _run(_job(tmp_path, env, _install(env)))
    assert code == 0, log
    assert (env / "version.txt").read_text() == "new"
    assert "no install at" in log and not (tmp_path / "toolenv.update-backup").exists()


def test_a_failed_fresh_install_leaves_nothing_behind(tmp_path: Path) -> None:
    env = tmp_path / "toolenv"
    code, log = _run(_job(tmp_path, env, [PY, "-c", _INSTALL_THEN_FAIL, str(env)]))
    assert code == 1, log
    assert not env.exists()
    assert log.rstrip().endswith("no previous install to restore")


def test_main_appends_to_the_log_and_returns_the_exit_code(tmp_path: Path) -> None:
    env = tmp_path / "toolenv"
    _old_env(env)
    job = _job(tmp_path, env, _install(env))
    Path(job["log"]).write_text("handed over\n", encoding="utf-8")
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    assert swap_install.main(["swap_install.py", str(tmp_path / "job.json")]) == 0
    log = Path(job["log"]).read_text(encoding="utf-8")
    assert log.startswith("handed over\n") and log.rstrip().endswith("update complete")


def test_the_copy_the_updater_runs_works_with_nothing_beside_it(tmp_path: Path) -> None:
    """The file is copied out of the package and run alone, by the env's base interpreter."""
    work = tmp_path / "work"
    work.mkdir()
    copy = work / "swap_install.py"
    shutil.copyfile(swap_install.__file__, copy)
    env = tmp_path / "toolenv"
    _old_env(env)
    job = _job(tmp_path, env, _install(env), log=str(work / "update.log"))
    (work / "job.json").write_text(json.dumps(job), encoding="utf-8")
    proc = subprocess.run(
        [PY, "-I", str(copy), str(work / "job.json")],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert (env / "version.txt").read_text() == "new"
    assert (work / "update.log").read_text(encoding="utf-8").rstrip().endswith("update complete")


def test_it_imports_nothing_but_the_standard_library() -> None:
    """The package beside it is the thing being replaced, so it may import none of it."""
    tree = ast.parse(Path(swap_install.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and node.module, "a relative import needs the package"
            roots.add(node.module.split(".")[0])
    assert roots <= set(sys.stdlib_module_names), roots - set(sys.stdlib_module_names)


def test_rmtree_deletes_read_only_files(tmp_path: Path) -> None:
    """Windows refuses to delete through the read-only bit; a posix delete never notices it."""
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    locked = tree / "sub" / "locked.txt"
    locked.write_text("x")
    locked.chmod(0o444)
    swap_install._rmtree(tree)
    assert not tree.exists()


def _tree_to_keep(root: Path) -> Path:
    keep = root / "target" / "sub" / "keep.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("x")
    return keep


def test_rmtree_removes_a_symlink_and_leaves_what_it_points_at(tmp_path: Path) -> None:
    """``shutil.rmtree`` refuses a link, and the old handler then ``chmod``-ed the directory the
    link pointed at and reported success with the link still in place."""
    keep = _tree_to_keep(tmp_path)
    link = tmp_path / "link"
    try:
        link.symlink_to(keep.parent.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this machine does not let the test create a symlink")
    mode = (tmp_path / "target").stat().st_mode
    swap_install._rmtree(link)
    assert not link.is_symlink() and not link.exists()
    assert keep.read_text() == "x"
    assert (tmp_path / "target").stat().st_mode == mode


@pytest.mark.skipif(sys.platform != "win32", reason="a directory junction exists on Windows only")
def test_rmtree_removes_a_junction_and_leaves_what_it_points_at(tmp_path: Path) -> None:
    keep = _tree_to_keep(tmp_path)
    junction = tmp_path / "junction"
    made = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(keep.parent.parent)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert made.returncode == 0, made.stdout + made.stderr
    swap_install._rmtree(junction)
    assert not junction.exists()
    assert keep.read_text() == "x"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows refuses to move a directory with a file open in it",
)
def test_a_file_held_open_in_the_install_blocks_the_move_and_nothing_changes(
    tmp_path: Path,
) -> None:
    """Measured here on the real filesystem (ADR-0291): the rename of the environment fails while
    any process has a file in it open, and then nothing has changed. A program that only RUNS from
    the directory holds no file open this way, and its rename went through when
    ``test_update_e2e.py`` measured it."""
    env = tmp_path / "toolenv"
    _old_env(env)
    before = _snapshot(env)
    marker = tmp_path / "ran.txt"
    command = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    with open(env / "Lib" / "site-packages" / "dep.py", "rb"):
        code, log = _run(_job(tmp_path, env, command, move_seconds=1))
    assert code == 2, log
    assert "cannot be moved" in log
    assert not marker.exists(), "a command ran although the install could not be moved aside"
    assert _snapshot(env) == before


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="only Windows refuses to move a directory that is a process's working directory",
)
def test_a_process_working_in_the_install_blocks_the_move_and_nothing_changes(
    tmp_path: Path,
) -> None:
    """A terminal left open in the environment's folder holds it the same way, and a listing of the
    programs that run from the install cannot see it, so the move is the only thing that does."""
    env = tmp_path / "toolenv"
    _old_env(env)
    before = _snapshot(env)
    marker = tmp_path / "ran.txt"
    command = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    # The base interpreter, not the venv's launcher: killing the launcher need not end the process
    # behind it, and that process is the one holding the folder.
    base_python = getattr(sys, "_base_executable", None) or PY
    decoy = subprocess.Popen(
        [base_python, "-c", "print('up', flush=True); import time; time.sleep(120)"],
        cwd=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert decoy.stdout is not None and decoy.stdout.readline().strip() == "up"
        code, log = _run(_job(tmp_path, env, command, move_seconds=1))
    finally:
        decoy.kill()
        decoy.wait()
        if decoy.stdout is not None:
            decoy.stdout.close()
    assert code == 2, log
    assert "cannot be moved" in log
    assert not marker.exists(), "a command ran although the install could not be moved aside"
    assert _snapshot(env) == before


def test_a_wait_without_a_pid_does_not_wait() -> None:
    started = time.monotonic()
    swap_install._wait_for_exit(0, 60)
    assert time.monotonic() - started < 5


@pytest.mark.skipif(
    sys.platform != "win32", reason="the Windows branch waits on the process handle"
)
def test_wait_for_exit_returns_when_the_process_ends() -> None:
    proc = subprocess.Popen([PY, "-c", "import time; time.sleep(2)"])
    try:
        swap_install._wait_for_exit(proc.pid, 120)
        assert proc.poll() is not None, "returned while the process was still running"
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(
    sys.platform != "win32", reason="the Windows branch waits on the process handle"
)
def test_wait_for_exit_gives_up_at_its_limit() -> None:
    proc = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
    try:
        started = time.monotonic()
        swap_install._wait_for_exit(proc.pid, 1)
        assert 0.5 < time.monotonic() - started < 30
        assert proc.poll() is None  # still running: it waited, it did not stop it
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="the posix branch polls for the process")
def test_wait_for_exit_polls_until_the_process_is_gone(monkeypatch) -> None:
    polls: list[int] = []

    def kill(pid: int, sig: int) -> None:
        polls.append(pid)
        if len(polls) == 3:
            raise ProcessLookupError

    monkeypatch.setattr(swap_install.os, "kill", kill)
    monkeypatch.setattr(swap_install.time, "sleep", lambda seconds: None)
    swap_install._wait_for_exit(4242, 60)
    assert polls == [4242, 4242, 4242]
