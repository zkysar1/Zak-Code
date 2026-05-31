"""Tests for the `task` delegation tool (M4-4b), driven by a fake spawner.

No real agents: a fake :class:`SubAgentSpawner` stands in so these stay fast and
deterministic while still proving the tool's contract — concurrent dispatch,
summary formatting, graceful errors, and type validation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from zakcode.agent.subagent import SubAgentResult
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.task import TaskTool
from zakcode.usage import Usage


class _FakeSpawner:
    """Records concurrency and returns a canned summary per subtask."""

    def __init__(self, *, types: list[str] | None = None) -> None:
        self.types = types if types is not None else ["general-purpose"]
        self.active = 0
        self.max_active = 0
        self.calls: list[tuple[str, str]] = []

    async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
        self.calls.append((type_name, prompt))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)  # yield so concurrent siblings can interleave
        self.active -= 1
        return SubAgentResult(name=type_name, summary=f"{prompt}!", usage=Usage(total_tokens=1))

    def available_types(self) -> list[str]:
        return self.types

    def default_type(self) -> str:
        return self.types[0]


def _ctx(tmp_path: Path, spawner: object | None) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, spawner=spawner)


async def test_single_task_returns_summary(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    result = await TaskTool().execute({"tasks": [{"prompt": "do X"}]}, _ctx(tmp_path, spawner))
    assert not result.is_error
    assert "do X!" in result.output
    assert spawner.calls == [("general-purpose", "do X")]
    assert result.data["count"] == 1
    assert result.data["errors"] == 0
    assert result.data["child_total_tokens"] == 1  # the fake reports 1 token/child


async def test_batch_runs_concurrently(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    tasks = [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
    result = await TaskTool().execute({"tasks": tasks}, _ctx(tmp_path, spawner))
    assert not result.is_error
    # All three were in flight at once → true concurrency, not sequential.
    assert spawner.max_active == 3
    for letter in ("a!", "b!", "c!"):
        assert letter in result.output


async def test_explicit_subagent_type_is_passed(tmp_path: Path) -> None:
    spawner = _FakeSpawner(types=["general-purpose", "plan"])
    await TaskTool().execute(
        {"tasks": [{"subagent_type": "plan", "prompt": "outline"}]}, _ctx(tmp_path, spawner)
    )
    assert spawner.calls == [("plan", "outline")]


async def test_unknown_subagent_type_errors(tmp_path: Path) -> None:
    spawner = _FakeSpawner(types=["general-purpose"])
    result = await TaskTool().execute(
        {"tasks": [{"subagent_type": "nope", "prompt": "x"}]}, _ctx(tmp_path, spawner)
    )
    assert result.is_error
    assert "unknown subagent_type" in result.output
    assert spawner.calls == []  # validation happens before any spawn


async def test_no_spawner_is_graceful_error(tmp_path: Path) -> None:
    result = await TaskTool().execute({"tasks": [{"prompt": "x"}]}, _ctx(tmp_path, None))
    assert result.is_error
    assert "delegation is not available" in result.output


async def test_missing_tasks_errors(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    result = await TaskTool().execute({}, _ctx(tmp_path, spawner))
    assert result.is_error
    assert "non-empty array" in result.output


async def test_task_without_prompt_errors(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    result = await TaskTool().execute(
        {"tasks": [{"subagent_type": "general-purpose"}]}, _ctx(tmp_path, spawner)
    )
    assert result.is_error
    assert "must be an object containing a 'prompt'" in result.output


async def test_child_failure_is_reported_not_raised(tmp_path: Path) -> None:
    class _BoomSpawner(_FakeSpawner):
        async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
            raise RuntimeError("child cap reached")

    result = await TaskTool().execute({"tasks": [{"prompt": "x"}]}, _ctx(tmp_path, _BoomSpawner()))
    assert result.is_error  # the only task failed
    assert "ERROR: RuntimeError: child cap reached" in result.output


async def test_partial_failure_keeps_successes(tmp_path: Path) -> None:
    class _FlakySpawner(_FakeSpawner):
        async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
            if prompt == "bad":
                raise RuntimeError("boom")
            return SubAgentResult(name=type_name, summary=f"{prompt}-ok")

    result = await TaskTool().execute(
        {"tasks": [{"prompt": "good"}, {"prompt": "bad"}]}, _ctx(tmp_path, _FlakySpawner())
    )
    assert not result.is_error  # not ALL failed
    assert "good-ok" in result.output
    assert "ERROR: RuntimeError: boom" in result.output
    assert result.data["count"] == 2
    assert result.data["errors"] == 1
