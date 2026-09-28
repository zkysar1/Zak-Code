"""PROOF-ONLY PROBE, DO NOT MERGE: which pid names a command's shell on Windows, and whether the
kill reaches that shell. It fails on purpose so that its readings reach the CI log."""

from __future__ import annotations

import asyncio
import base64
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zakcode._subprocess import find_bash, new_group_kwargs, terminate_process_tree
from zakcode.background import pid_alive, process_start_token

_WRAPPER = '"$4" -c "$1" >"$2" 2>&1; printf "%s\\n" "$?" >"$3"'


def _snapshot(marker: str) -> list[str]:
    """pid/parent/name/created for each live process whose command line holds ``marker``."""
    if sys.platform == "win32":
        script = (
            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*"
            + marker
            + "*' } | ForEach-Object { '{0}/{1}/{2}/{3}' -f $_.ProcessId, $_.ParentProcessId, "
            "$_.Name, $_.CreationDate.ToString('HH:mm:ss.fff') }"
        )
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout.split() or [f"none(rc={out.returncode})"]
    found = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            parent = (proc_dir / "stat").read_text().rsplit(")", 1)[1].split()[1]
            name = (proc_dir / "comm").read_text().strip()
        except (OSError, IndexError):
            continue
        if marker in cmdline:
            found.append(f"{proc_dir.name}/{parent}/{name}")
    return found or ["none"]


async def _reading(tmp_path: Path, name: str, tail: str, marker: str) -> str:
    bash = find_bash()
    assert bash is not None, "no bash"
    pid_file = tmp_path / f"{name}.pid"
    command = (
        f"{{ cat /proc/$$/winpid 2>/dev/null || echo $$; }} > {pid_file.name}; sleep {marker}{tail}"
    )
    proc = await asyncio.create_subprocess_exec(
        bash,
        "-c",
        _WRAPPER,
        "probe",
        command,
        (tmp_path / f"{name}.out").as_posix(),
        (tmp_path / f"{name}.exit").as_posix(),
        bash,
        cwd=str(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **new_group_kwargs(),
    )
    deadline = time.monotonic() + 10
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, f"{name}: no pid file"
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.0)  # time for bash to exec its last command, if it does
    pid = int(pid_file.read_text().strip())
    token = process_start_token(pid)
    before = f"alive={pid_alive(pid)} token={token is not None} procs={_snapshot(marker)}"
    await terminate_process_tree(proc)
    await asyncio.sleep(5.0)
    survived = pid_alive(pid) and token is not None and process_start_token(pid) == token
    after = f"file_pid_survived={survived} procs={_snapshot(marker)}"
    return f"[{name}] shell_pid={proc.pid} file_pid={pid} BEFORE {before} AFTER {after}"


async def test_zz_proof_winpid_probe(tmp_path: Path) -> None:
    readings = [
        await _reading(tmp_path, "exec", "", "41.25"),
        await _reading(tmp_path, "noexec", "; :", "43.75"),
    ]
    pytest.fail("PROBE " + " || ".join(readings))
