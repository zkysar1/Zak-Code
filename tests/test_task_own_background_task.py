"""ADR-0397: the ``task`` tool refuses a delegation that names a still-running background task of
THIS session. A sub-agent has its own task table, so it cannot wait on, read the status of or stop
that task; the parent waits with ``TaskOutput`` itself, in one call.

Measured on the worker fleet, 24 h to 2026-11-19 19:30Z: 4 of 18 sub-agents were sent to wait on
their parent's own background command (a selection walk the Bash tool had moved to the background at
its 120 s timeout). They spent 131 tool calls and about 100 minutes of the parents' turns on it; one
spent 40 minutes and 70 calls on a 17-line output file while its parent held no work.

Driven by a fake spawner and a real :class:`BackgroundTasks` over records whose status is derived as
in production (a live pid and no exit file is ``running``; an exit file is ``completed``).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from zakcode.agent.subagent import SubAgentResult
from zakcode.background import BackgroundTask, BackgroundTasks
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.task import TaskTool
from zakcode.usage import Usage

TASK_ID = "d8be08327"


class _Spawner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
        self.calls.append((type_name, prompt))
        return SubAgentResult(name=type_name, summary="done", usage=Usage(total_tokens=1))

    def available_types(self) -> list[str]:
        return ["general-purpose"]

    def default_type(self) -> str:
        return "general-purpose"


def _record(tmp_path: Path, task_id: str = TASK_ID, *, exited: bool = False) -> BackgroundTask:
    out = tmp_path / f"{task_id}.out"
    out.write_text("SELECTWALK_RC=0\n", encoding="utf-8")
    exit_file = tmp_path / f"{task_id}.exit"
    if exited:
        exit_file.write_text("0\n", encoding="utf-8")
    return BackgroundTask(
        id=task_id,
        command="py -3 core/scripts/worker_execute.py select-walk --top 10",
        cwd=str(tmp_path),
        output_file=str(out),
        exit_file=str(exit_file),
        pid=os.getpid(),  # alive for the whole test, so no exit file reads as running
        started_at="2026-11-19T18:44:47Z",
    )


def _ctx(tmp_path: Path, spawner: _Spawner, records: list[BackgroundTask] | None) -> ToolContext:
    if records is None:
        return ToolContext(workspace_root=tmp_path, spawner=spawner)
    tasks = BackgroundTasks(SimpleNamespace(background_tasks=records), tasks_dir=tmp_path)
    return ToolContext(workspace_root=tmp_path, spawner=spawner, background_tasks=tasks)


async def test_a_running_task_named_by_id_is_refused_before_any_child(tmp_path: Path) -> None:
    rec = _record(tmp_path)
    tasks = BackgroundTasks(SimpleNamespace(background_tasks=[rec]), tasks_dir=tmp_path)
    assert tasks.status(rec)[0] == "running"  # control: the fixture is the live case
    spawner = _Spawner()
    prompt = f"Wait for background task {TASK_ID} to finish and report which goal it selected."
    result = await TaskTool().execute({"tasks": [{"prompt": prompt}]}, _ctx(tmp_path, spawner, [rec]))
    assert result.is_error
    assert spawner.calls == [], "no sub-agent may start on a refused delegation"
    assert f'TaskOutput(task_id="{TASK_ID}", timeout=600000)' in result.output
    assert rec.output_file in result.output
    assert result.data == {"refused": "names_own_running_background_task", "task_id": TASK_ID}


async def test_a_running_task_named_by_its_output_file_is_refused(tmp_path: Path) -> None:
    rec = _record(tmp_path)
    spawner = _Spawner()
    prompt = f"Read {rec.output_file} once it is complete and summarise the candidates."
    result = await TaskTool().execute({"tasks": [{"prompt": prompt}]}, _ctx(tmp_path, spawner, [rec]))
    assert result.is_error and spawner.calls == []


async def test_the_single_delegation_shape_is_refused_too(tmp_path: Path) -> None:
    rec = _record(tmp_path)
    spawner = _Spawner()
    args = {"description": "wait", "prompt": f"TaskOutput {TASK_ID} and report it."}
    result = await TaskTool().execute(args, _ctx(tmp_path, spawner, [rec]))
    assert result.is_error and spawner.calls == []


async def test_a_batch_is_refused_whole_and_names_the_subtask(tmp_path: Path) -> None:
    rec = _record(tmp_path)
    spawner = _Spawner()
    tasks = [{"prompt": "survey the tests"}, {"prompt": f"poll task {TASK_ID} until it exits"}]
    result = await TaskTool().execute({"tasks": tasks}, _ctx(tmp_path, spawner, [rec]))
    assert result.is_error
    assert result.output.startswith("task[1] names background task")
    assert spawner.calls == [], "the innocent sibling does not start either"


async def test_a_finished_task_may_be_delegated(tmp_path: Path) -> None:
    rec = _record(tmp_path, exited=True)
    spawner = _Spawner()
    prompt = f"Summarise the output of task {TASK_ID} in {rec.output_file}."
    result = await TaskTool().execute({"tasks": [{"prompt": prompt}]}, _ctx(tmp_path, spawner, [rec]))
    assert not result.is_error
    assert spawner.calls == [("general-purpose", prompt)]


async def test_an_id_inside_a_longer_token_is_not_a_match(tmp_path: Path) -> None:
    rec = _record(tmp_path)
    spawner = _Spawner()
    for prompt in (f"commit {TASK_ID}1 broke it", f"see x{TASK_ID} in the log", f"{TASK_ID}-old"):
        result = await TaskTool().execute(
            {"tasks": [{"prompt": prompt}]}, _ctx(tmp_path, spawner, [rec])
        )
        assert not result.is_error, prompt
    assert len(spawner.calls) == 3


async def test_a_prompt_naming_no_task_spawns_as_before(tmp_path: Path) -> None:
    spawner = _Spawner()
    rec = _record(tmp_path)
    result = await TaskTool().execute(
        {"tasks": [{"prompt": "find every caller of goal-selector"}]}, _ctx(tmp_path, spawner, [rec])
    )
    assert not result.is_error and len(spawner.calls) == 1


async def test_no_task_table_on_the_context_spawns_as_before(tmp_path: Path) -> None:
    spawner = _Spawner()
    result = await TaskTool().execute(
        {"tasks": [{"prompt": f"wait for {TASK_ID}"}]}, _ctx(tmp_path, spawner, None)
    )
    assert not result.is_error and len(spawner.calls) == 1
