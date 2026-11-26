"""A whole wait on a background task still running is the wait going on, not a stall (ADR-0410).

Bash's result tells a model to wait for a command it moved to the background by calling
``TaskOutput(task_id=..., timeout=600000)``, and to call again while the command runs (ADR-0236).
Each answer reads "status: running" in the same words, so the exact-repeat guard refused the
third wait and the stuck ladder read the third answer as a re-measured result: a model could wait
twenty to thirty minutes and no longer (ADR-0407 named this the known cost). The tool now marks
an answer after a whole wait of a minute or more on a task younger than six hours, and both guards
skip what it marks. These tests pin the mark and its bounds at the tool, the age it is measured
by, the ladder's skip, and the guard's skip on both turn paths; ``tests/test_stuck_poll.py`` keeps
pinning the unmarked waits, which still end the turn.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.test_stuck_decompose import _c, _drain, _rails, _ScriptByCallProvider
from zakcode.agent.loop import AgentLoop
from zakcode.agent.stuck import SIG_REPEATED_OUTCOME, StuckAction, StuckTracker
from zakcode.background import (
    MAX_BLOCK_MS,
    WAIT_MARK_MAX_AGE_S,
    WAIT_MARK_MIN_MS,
    BackgroundTask,
    BackgroundTasks,
)
from zakcode.config import PermissionTier, Settings
from zakcode.events import AgentDone
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import LLMResult, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import (
    WAITED_ON_RUNNING,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.task_output import TaskOutputTool

#: What the tool that waits on a background task answers while the task is still running.
STILL_RUNNING = (
    "[task bg-1] status: running\ncommand: sleep 7200\noutput file: /tmp/bg-1.out\n\n"
    "(no output yet)"
)


def _shell_loop(
    provider: _ScriptByCallProvider,
    tmp_path: Path,
    *,
    max_iterations: int = 15,
    also: Tool | None = None,
) -> AgentLoop:
    registry = ToolRegistry()
    if also is not None:
        registry.register(also)
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=max_iterations,
        settings=Settings(permission_mode="allow"),
    )


# ── the mark, at the tool ───────────────────────────────────────────────────────


def _table(tmp_path: Path, now: list[float]) -> BackgroundTasks:
    session = Session(cwd=str(tmp_path), model="test")
    return BackgroundTasks(session, tasks_dir=tmp_path / "tasks", clock=lambda: now[0])


async def _answer(
    tmp_path: Path,
    args: dict[str, Any],
    *,
    age_s: float,
    monkeypatch: pytest.MonkeyPatch,
) -> ToolResult:
    """The tool's answer for a task still running that is ``age_s`` old, with its wait cut
    short so a whole ten-minute wait costs nothing here."""
    now = [1_000_000.0]
    tasks = _table(tmp_path, now)
    task = await tasks.start("sleep 30", cwd=str(tmp_path))
    try:
        now[0] += age_s

        async def whole_wait(_task: BackgroundTask, _timeout_s: float) -> tuple[str, int | None]:
            return "running", None

        monkeypatch.setattr(tasks, "wait", whole_wait)
        ctx = ToolContext(workspace_root=tmp_path, background_tasks=tasks)
        return await TaskOutputTool().execute({"task_id": task.id, **args}, ctx)
    finally:
        await tasks.stop(task.id)


async def test_a_whole_wait_on_a_running_task_is_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = await _answer(tmp_path, {"timeout": MAX_BLOCK_MS}, age_s=600, monkeypatch=monkeypatch)
    assert not res.is_error
    assert res.data is not None and res.data[WAITED_ON_RUNNING] is True
    assert res.data["status"] == "running"
    # The mark is for the guards: the model reads the same words as before.
    assert "status: running" in res.output and "waited" not in res.output


@pytest.mark.parametrize(
    ("args", "age_s"),
    [
        ({}, 600),  # no timeout named: the tool's default wait, 30 seconds, is a check
        ({"timeout": WAIT_MARK_MIN_MS - 1}, 600),  # just under a minute
        ({"timeout": MAX_BLOCK_MS, "block": False}, 600),  # a look, not a wait
        ({"timeout": MAX_BLOCK_MS}, WAIT_MARK_MAX_AGE_S),  # a task six hours old
    ],
    ids=["default-wait", "under-a-minute", "block-false", "six-hours-old"],
)
async def test_a_short_wait_a_look_and_an_old_task_are_not_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: dict[str, Any], age_s: float
) -> None:
    res = await _answer(tmp_path, args, age_s=age_s, monkeypatch=monkeypatch)
    assert res.data is not None and res.data["status"] == "running"
    assert WAITED_ON_RUNNING not in res.data


async def test_the_shortest_marked_wait_is_one_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    res = await _answer(tmp_path, {"timeout": WAIT_MARK_MIN_MS}, age_s=600, monkeypatch=monkeypatch)
    assert res.data is not None and res.data[WAITED_ON_RUNNING] is True


async def test_an_answer_about_a_task_that_exited_is_not_marked(tmp_path: Path) -> None:
    now = [1_000_000.0]
    tasks = _table(tmp_path, now)
    task = await tasks.start("echo done", cwd=str(tmp_path))
    ctx = ToolContext(workspace_root=tmp_path, background_tasks=tasks)
    res = await TaskOutputTool().execute({"task_id": task.id, "timeout": MAX_BLOCK_MS}, ctx)
    assert res.data is not None and res.data["status"] == "completed"
    assert WAITED_ON_RUNNING not in res.data


def test_a_tasks_age_is_counted_from_its_spawn_by_the_tables_clock(tmp_path: Path) -> None:
    now = [2_000_000.0]
    tasks = _table(tmp_path, now)
    started = datetime.fromtimestamp(now[0] - 3600, tz=UTC).isoformat()
    task = BackgroundTask(
        id="t1",
        command="sleep 9",
        cwd=str(tmp_path),
        output_file="o",
        exit_file="e",
        pid=1,
        started_at=started,
    )
    assert tasks.age_seconds(task) == pytest.approx(3600.0)
    assert tasks.age_seconds(task.model_copy(update={"started_at": "not a time"})) is None


# ── the ladder skips what is marked ─────────────────────────────────────────────


def _wait_call(n: int) -> list[ToolCall]:
    return [
        ToolCall(id=f"w{n}", name="TaskOutput", arguments={"task_id": "bg-1", "timeout": 600000})
    ]


def _wait_result(n: int, *, marked: bool) -> list[ToolResultBlock]:
    data: dict[str, Any] = {"task_id": "bg-1", "status": "running", "exit_code": None}
    if marked:
        data[WAITED_ON_RUNNING] = True
    return [ToolResultBlock(tool_use_id=f"w{n}", output=STILL_RUNNING, data=data)]


def test_the_ladder_never_climbs_on_marked_waits() -> None:
    tracker = StuckTracker()
    for n in range(1, 13):  # two hours of ten-minute waits
        tracker.observe(_wait_call(n), _wait_result(n, marked=True))
        assert SIG_REPEATED_OUTCOME not in tracker.last_signals
        assert tracker.next_action() is StuckAction.CONTINUE
    assert not tracker.took_action


def test_the_ladder_still_climbs_on_the_same_waits_unmarked() -> None:
    # The control: the same answers without the mark are a re-measured result at the third.
    tracker = StuckTracker()
    actions = []
    for n in range(1, 4):
        tracker.observe(_wait_call(n), _wait_result(n, marked=False))
        actions.append(tracker.next_action())
    assert SIG_REPEATED_OUTCOME in tracker.last_signals
    assert actions == [StuckAction.CONTINUE, StuckAction.CONTINUE, StuckAction.NUDGE]


# ── the guard skips what is marked, on both turn paths ──────────────────────────

DONE = (
    "[task bg-1] status: completed, exit code: 0\ncommand: sleep 7200\n"
    "output file: /tmp/bg-1.out\n\nall 12 checks passed"
)


class _LongTask(Tool):
    """Stands in for TaskOutput on a long task: the first ``waits`` calls each wait the whole
    timeout and answer that the task is still running, in the same words, marked as the real
    tool marks them; the next answers that it exited."""

    spec = ToolSpec(
        name="TaskOutput",
        description="waits for a background task",
        required_permission=PermissionTier.READ_ONLY,
    )

    def __init__(self, waits: int) -> None:
        self.waits = waits
        self.calls = 0

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls += 1
        if self.calls <= self.waits:
            return ToolResult.ok(
                STILL_RUNNING,
                data={"task_id": "bg-1", "status": "running", WAITED_ON_RUNNING: True},
            )
        return ToolResult.ok(DONE, data={"task_id": "bg-1", "status": "completed", "exit_code": 0})


@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
def test_a_model_that_waits_out_a_long_task_is_never_refused(
    tmp_path: Path, streamed: bool
) -> None:
    # The way Bash's result says to wait, followed for two hours: the same call each time, until
    # the task exits and the model answers. No wait is refused, no rail is sent, the turn ends
    # on the answer.
    waits = 12
    tool = _LongTask(waits)

    def script(n: int) -> LLMResult:
        if n <= waits + 1:
            return LLMResult(
                tool_calls=[_c(f"c{n}", "TaskOutput", task_id="bg-1", timeout=MAX_BLOCK_MS)]
            )
        return LLMResult(text="The suite finished: all 12 checks passed.")

    loop = _shell_loop(_ScriptByCallProvider(script), tmp_path, max_iterations=30, also=tool)
    if streamed:
        events = _drain(loop, "wait for the build job to finish")
        stop_reason = next(e for e in events if isinstance(e, AgentDone)).stop_reason
    else:
        stop_reason = asyncio.run(loop.arun_turn("wait for the build job to finish")).stop_reason
    results = [
        block.output or ""
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert stop_reason == "completed"
    assert tool.calls == waits + 1
    assert results == [STILL_RUNNING] * waits + [DONE]
    assert [rail for rail in _rails(loop) if rail.startswith("[harness]")] == []
