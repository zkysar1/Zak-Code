"""``zakcode update`` end to end, with a real uv, a real git remote and a real tool environment.

The unit tests for the update command fake uv and git. They cannot show what Windows does when the
command replaces the environment it is running from, nor what uv installs after a pull, and both
went wrong for a user (ADR-0291): the reinstall deleted its own program and left an environment
that held nothing, and an editable install came back as a copy at the newest version of every
dependency. This drives the real thing.

It installs the project into a throwaway uv tool directory the way a user does (editable, from a
local checkout, unpinned), pushes commits to a local remote, and runs ``zakcode update`` from the
installed program. It needs uv and the network and takes minutes, so it is skipped unless
``ZAKCODE_UPDATE_E2E=1``; the ``update-e2e`` CI job sets that, the offline suite does not.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ZAKCODE_UPDATE_E2E") != "1",
        reason="set ZAKCODE_UPDATE_E2E=1 to run (needs uv and the network, takes minutes)",
    ),
    pytest.mark.timeout(1800),
]

REPO = Path(__file__).resolve().parents[1]
WINDOWS = sys.platform == "win32"
#: What a user does not get from a checkout: tests, docs and benchmarks are not needed to install.
_SKIP_TOP = ("tests", "docs", "bench")


def _run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 900
) -> subprocess.CompletedProcess[str]:
    # UTF-8 on purpose: the programs print it (PYTHONUTF8=1 below), and the Windows default is
    # cp1252, which cannot decode the glyphs in `zakcode update`'s messages.
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _ok(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    proc = _run(cmd, cwd=cwd, env=env)
    assert proc.returncode == 0, f"{cmd} -> rc {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _git(cwd: Path, *args: str) -> str:
    return _ok(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd)


def _seed_remote(base: Path, uv: str) -> tuple[Path, Path, Path]:
    """A remote, a working clone to push new commits from, and the checkout that gets installed.

    The project's own tracked files are copied into a fresh repository rather than cloning the CI
    checkout: that one is shallow and detached, and a pull needs a branch with an upstream.
    """
    author, remote, checkout = base / "author", base / "remote.git", base / "checkout"
    tracked = _ok(["git", "ls-files", "-z"], cwd=REPO).split("\0")
    for name in filter(None, tracked):
        if name.split("/", 1)[0] in _SKIP_TOP:
            continue
        target = author / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    _hold_one_package_back(uv, author)
    _git(author, "init", "-q", "-b", "main")
    _git(author, "add", "-A")
    _git(author, "commit", "-q", "-m", "seed")
    _git(base, "init", "-q", "--bare", "-b", "main", str(remote))
    _git(author, "remote", "add", "origin", str(remote))
    _git(author, "push", "-q", "origin", "main")
    _git(base, "clone", "-q", str(remote), str(checkout))
    return author, remote, checkout


def _push_commit(author: Path, note: str) -> str:
    """A harmless commit on the remote, so the next update has something to pull."""
    readme = author / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + f"\n<!-- {note} -->\n", encoding="utf-8")
    _git(author, "add", "-A")
    _git(author, "commit", "-q", "-m", note)
    _git(author, "push", "-q", "origin", "main")
    return _git(author, "rev-parse", "HEAD").strip()


def _head(checkout: Path) -> str:
    return _git(checkout, "rev-parse", "HEAD").strip()


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed(tool_python: Path) -> dict[str, str]:
    """``{normalised name: version}`` of every distribution in the tool environment."""
    code = (
        "import importlib.metadata as m, json;"
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
    )
    got: dict[str, str] = json.loads(_ok([str(tool_python), "-c", code]))
    return {_norm(name): version for name, version in got.items()}


def _locked(uv: str, checkout: Path) -> dict[str, str]:
    """The project's own lock as ``{normalised name: pinned version}``."""
    out = _ok(
        [
            uv,
            "export",
            "--locked",
            "--all-extras",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--no-annotate",
        ],
        cwd=checkout,
    )
    pins: dict[str, str] = {}
    for line in out.splitlines():
        found = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
        if found:
            pins[_norm(found.group(1))] = found.group(2)
    return pins


#: Plain-Python packages near the leaves of the project's dependency tree. One held a release
#: behind cannot stop the project importing; the first one the lock has is used.
_LEAVES = ("idna", "certifi", "sniffio", "h11", "click", "pygments")


def _hold_one_package_back(uv: str, project: Path) -> str:
    """Re-lock ``project`` with one package a release behind its newest, and name it.

    A reinstall that ignores the lock resolves to the newest releases. On the day the lock already
    names them, "pinned to the lock" cannot be told from "not pinned" and the pin check would
    prove nothing (or the job would go red for a reason that is not a defect). Holding one package
    back makes the two differ whatever today's releases are.
    """
    pins = _locked(uv, project)
    tried: list[str] = []
    for name in _LEAVES:
        pin = pins.get(name)
        if pin is None:
            continue
        proc = _run([uv, "lock", "--upgrade-package", f"{name}<{pin}"], cwd=project)
        if proc.returncode == 0 and _locked(uv, project).get(name, pin) != pin:
            return name
        tried.append(
            f"{name} (rc {proc.returncode}): {(proc.stderr or proc.stdout).strip()[-300:]}"
        )
    raise AssertionError(f"could not hold any package behind its newest release: {tried}")


def _read_log(log: Path) -> str:
    return log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""


def _wait_for_log(log: Path, markers: tuple[str, ...], seconds: int = 600) -> str:
    """Poll the helper's log until one of ``markers`` shows, and return the log."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        text = _read_log(log)
        if any(marker in text for marker in markers):
            return text
        time.sleep(2)
    raise AssertionError(f"none of {markers} in {log} after {seconds}s:\n{_read_log(log)}")


def test_update_replaces_the_install_it_runs_from(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv, "uv is not on PATH"
    # A space in the path on purpose: a Windows profile folder has one, and so did the report.
    base = tmp_path / "zak code up"
    base.mkdir()
    author, _remote, checkout = _seed_remote(base, uv)
    tool_dir, bin_dir, home = base / "tools", base / "bin", base / "home"
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tool_dir),
        "UV_TOOL_BIN_DIR": str(bin_dir),
        "ZAKCODE_HOME": str(home),
        "PYTHONUTF8": "1",
    }
    scripts = "Scripts" if WINDOWS else "bin"
    exe = ".exe" if WINDOWS else ""
    zakcode = bin_dir / f"zakcode{exe}"
    tool_env = tool_dir / "zakcode"
    tool_python = tool_env / scripts / f"python{exe}"
    log = home / "update" / "update.log"

    # ── the install a user has: editable, from a local checkout, dependencies at the newest ──
    _ok(
        [uv, "tool", "install", "--python", "3.11", "--editable", f"zakcode @ {checkout.as_uri()}"],
        env=env,
    )
    assert zakcode.is_file(), f"no {zakcode} after the install"
    before = _installed(tool_python)
    pins = _locked(uv, checkout)
    off_lock = {n for n, v in pins.items() if n in before and before[n] != v}
    assert off_lock, (
        "the unpinned install already matches the lock; the pin check would prove nothing"
    )
    assert "editable" in (tool_env / "uv-receipt.toml").read_text(encoding="utf-8")

    if WINDOWS:
        # ── 1. another process runs from this install: refuse before touching anything ──
        old_head = _head(checkout)
        _push_commit(author, "first")
        decoy = subprocess.Popen([str(tool_python), "-c", "import time; time.sleep(900)"], env=env)
        try:
            time.sleep(2)
            blocked = _run([str(zakcode), "update"], env=env)
        finally:
            decoy.kill()
            decoy.wait()
        shown = blocked.stdout + blocked.stderr
        assert blocked.returncode != 0, (
            f"update ran while another process held the install:\n{shown}"
        )
        held = sorted(p.name for p in tool_env.iterdir()) if tool_env.is_dir() else "nothing"
        assert str(decoy.pid) in shown, (
            f"the busy process is not named:\n{shown}\nthe install holds: {held}"
        )
        assert _head(checkout) == old_head, "update pulled before it knew it could finish"
        assert not (tool_dir / "zakcode.update-backup").exists()
        _ok([str(tool_python), "-c", "import zakcode"])  # the install is untouched

        # ── 1b. the programs cannot be listed: the second process still refuses a busy install ──
        # PowerShell is what lists them. With it off PATH the update goes ahead and pulls, and the
        # move of the environment is what finds the program running from it: nothing is replaced,
        # and the log says the checkout was pulled and how to go back.
        receipt_before = (tool_env / "uv-receipt.toml").read_text(encoding="utf-8")
        no_powershell = os.pathsep.join(
            p for p in env["PATH"].split(os.pathsep) if "powershell" not in p.lower()
        )
        decoy = subprocess.Popen([str(tool_python), "-c", "import time; time.sleep(900)"], env=env)
        try:
            time.sleep(2)
            proc = _run([str(zakcode), "update"], env={**env, "PATH": no_powershell})
            assert proc.returncode == 0, f"the listing was still available:\n{proc.stdout}"
            text = _wait_for_log(log, ("cannot be moved", "update complete"))
        finally:
            decoy.kill()
            decoy.wait()
        assert "cannot be moved" in text, text
        assert "was pulled from" in text and "reset --hard" in text, text
        assert _head(checkout) != old_head, "the update should have pulled before it found out"
        assert (tool_env / "uv-receipt.toml").read_text(encoding="utf-8") == receipt_before
        assert not (tool_dir / "zakcode.update-backup").exists(), "the backup was left behind"
        _ok([str(tool_python), "-c", "import zakcode"])  # the install is untouched

        # ── 2. the reinstall fails: the old install comes back, whole ──
        offline = {**env, "UV_OFFLINE": "1", "UV_NO_CACHE": "1"}
        proc = _run([str(zakcode), "update"], env=offline)
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        # The last lines, not the first: "update failed" is written before the old install is put
        # back, so a test that stopped there would read the tree while it was still being restored.
        text = _wait_for_log(
            log, ("restored the previous environment", "could NOT be put back", "update complete")
        )
        assert "update failed" in text and "restored the previous environment" in text, text
        assert tool_python.is_file() and _ok([str(tool_python), "-c", "import zakcode"]) is not None
        assert (tool_env / "uv-receipt.toml").read_text(encoding="utf-8") == receipt_before
        assert not (tool_dir / "zakcode.update-backup").exists(), "the backup was left behind"

    # ── the update itself: pulled, pinned to the lock, still editable, working ──
    new_head = _push_commit(author, "second")
    proc = _run([str(zakcode), "update"], env=env)
    shown = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"update failed:\n{shown}"
    if WINDOWS:
        # The program cannot replace the files it runs from, so the reinstall finishes in a
        # second process once the command has exited, and says so in its log.
        assert str(log) in shown or log.name in shown, f"the log is not named:\n{shown}"
        text = _wait_for_log(log, ("update failed", "update complete"))
        assert "update complete" in text, text
    assert _head(checkout) == new_head, "the checkout was not pulled"
    assert tool_python.is_file(), f"no environment at {tool_env}:\n{shown}"
    _ok([str(tool_python), "-c", "import zakcode, zakcode.cli"])
    receipt = tomllib.loads((tool_env / "uv-receipt.toml").read_text(encoding="utf-8"))
    requirement = next(r for r in receipt["tool"]["requirements"] if r["name"] == "zakcode")
    assert "editable" in requirement, f"the install is no longer editable: {requirement}"
    after = _installed(tool_python)
    moved = {n: (after.get(n), v) for n, v in pins.items() if n in after and after[n] != v}
    assert not moved, f"dependencies not at the version the lock pins: {moved}"
    assert {n for n in off_lock if n in after}, "none of the off-lock packages is installed"
    assert not (tool_dir / "zakcode.update-backup").exists(), "the backup was left behind"
