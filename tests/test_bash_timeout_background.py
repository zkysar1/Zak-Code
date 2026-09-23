"""ADR-0236: a foreground ``Bash`` command still running at its timeout is moved to the
background, not killed.

Claude Code does this (probed 2026-09-23: a command past its timeout, default or explicit,
keeps running as a background task). Zak Code killed it, and a Mind's closing step can run 12
to 15 minutes: measured 2026-09-23 on two worker Bodies, the models' calls to it were killed at
their timeouts 8 times, and then they went around the step instead of waiting for it. These
tests pin the contract: a quick
command leaves no trace; a slow one keeps running, recorded like ``run_in_background``, with
its output still landing in its file and its exit reported once; a failure is still an error;
a cancelled turn still kills the whole tree; and the wait is for the command's shell, not its
output, so ``cmd &`` returns at once as in Claude Code. Hermetic: short real shell commands in
a tmp workspace.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from zakcode.background import BackgroundTasks, pid_alive
from zakcode.session.store import Session
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.task_output import TaskOutputTool


def _tasks(tmp_path: Path, changes: list[int]) -> tuple[Session, BackgroundTasks]:
    session = Session(cwd=str(tmp_path), model="test")
    tasks = BackgroundTasks(
        session, on_change=lambda: changes.append(1), tasks_dir=tmp_path / "tasks"
    )
    return session, tasks


def _ctx(tmp_path: Path, tasks: BackgroundTasks) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, background_tasks=tasks)


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"not true after {timeout}s"
        await asyncio.sleep(0.05)


async def test_a_command_that_finishes_in_time_leaves_no_trace(tmp_path: Path) -> None:
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)

    res = await BashTool().execute(
        {"command": "echo out; echo err 1>&2", "timeout": 30}, _ctx(tmp_path, tasks)
    )

    assert not res.is_error, res.output
    assert "out" in res.output and "err" in res.output  # stderr still combined
    assert res.data is not None and res.data["exit_code"] == 0
    assert session.background_tasks == [] and changes == []
    assert list((tmp_path / "tasks").iterdir()) == []  # its output file went with it
    assert tasks.take_notifications() is None


async def test_a_failed_command_is_still_an_error_with_its_exit_code(tmp_path: Path) -> None:
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)

    res = await BashTool().execute({"command": "echo boom; exit 3"}, _ctx(tmp_path, tasks))

    assert res.is_error
    assert "boom" in res.output and "[exit code: 3]" in res.output
    assert res.data is not None and res.data["exit_code"] == 3
    assert session.background_tasks == []


async def test_a_command_still_running_at_its_timeout_is_moved_not_killed(
    tmp_path: Path,
) -> None:
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)
    ctx = _ctx(tmp_path, tasks)

    started = time.monotonic()
    res = await BashTool().execute(
        {"command": "echo before; sleep 2; echo after", "timeout": 1, "description": "slow"},
        ctx,
    )

    assert time.monotonic() - started < 1.9, "the tool waited for the command"
    assert not res.is_error, res.output
    assert res.data is not None and res.data["moved_to_background"] is True
    task_id = res.data["task_id"]
    assert "moved to the background" in res.output
    assert f'TaskOutput(task_id="{task_id}", timeout=600000)' in res.output
    # Recorded like a run_in_background task, and persisted before the tool returned.
    (record,) = session.background_tasks
    assert record.id == task_id and record.description == "slow" and changes
    assert tasks.status(record) == ("running", None)
    assert Path(record.output_file).read_text(encoding="utf-8").strip() == "before"

    # It was not killed: it finishes, its later output lands in the same file, and the exit
    # is recorded.
    waited = await TaskOutputTool().execute({"task_id": task_id, "timeout": 15_000}, ctx)
    assert "status: completed" in waited.output, waited.output
    assert tasks.status(record) == ("completed", 0)
    assert Path(record.output_file).read_text(encoding="utf-8").split() == ["before", "after"]

    # Its exit is reported once, like any background task's.
    note = tasks.take_notifications()
    assert note is not None and "completed (exit code 0)" in note and "slow" in note
    assert tasks.take_notifications() is None


async def test_a_background_command_names_the_same_wait(tmp_path: Path) -> None:
    # Its exit notice comes at an idle prompt too, so its result names TaskOutput as well.
    changes: list[int] = []
    _session, tasks = _tasks(tmp_path, changes)

    res = await BashTool().execute(
        {"command": "sleep 1", "run_in_background": True}, _ctx(tmp_path, tasks)
    )

    assert res.data is not None and "notified when it completes" in res.output
    assert f'TaskOutput(task_id="{res.data["task_id"]}", timeout=600000)' in res.output
    await tasks.wait(tasks.records()[0], 15.0)


async def test_a_moved_command_keeps_its_own_exit_code(tmp_path: Path) -> None:
    changes: list[int] = []
    _session, tasks = _tasks(tmp_path, changes)

    moved = await tasks.run_foreground("sleep 1; exit 7", cwd=str(tmp_path), timeout_seconds=0.2)

    assert not isinstance(moved, tuple), "a command still running was not moved"
    status, _ = await tasks.wait(moved, 15.0)
    assert (status, tasks.status(moved)[1]) == ("completed", 7)


async def test_a_cancelled_call_still_kills_the_command_and_records_nothing(
    tmp_path: Path,
) -> None:
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)
    pid_file = tmp_path / "pid"

    call = asyncio.create_task(
        BashTool().execute(
            {"command": f"echo $$ > {pid_file.name}; sleep 30", "timeout": 60},
            _ctx(tmp_path, tasks),
        )
    )
    await _until(lambda: pid_file.exists() and pid_file.read_text().strip() != "")
    pid = int(pid_file.read_text().strip())
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    await _until(lambda: not pid_alive(pid), timeout=5.0)
    assert session.background_tasks == [] and changes == []
    assert list((tmp_path / "tasks").iterdir()) == []


async def test_a_command_ending_in_an_ampersand_returns_when_its_shell_exits(
    tmp_path: Path,
) -> None:
    # Claude Code returns here at once and leaves the job running (probed 2026-09-23). Through
    # a pipe, the call waited until the job closed its output: 8 seconds, or the timeout.
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)

    started = time.monotonic()
    res = await BashTool().execute(
        {"command": "sleep 8 & echo started", "timeout": 20}, _ctx(tmp_path, tasks)
    )

    assert time.monotonic() - started < 6, "the call waited for the backgrounded job"
    assert not res.is_error, res.output
    assert "started" in res.output
    assert session.background_tasks == []
