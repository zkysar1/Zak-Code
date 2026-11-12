"""Tests for the `task` delegation tool (M4-4b), driven by a fake spawner.

No real agents: a fake :class:`SubAgentSpawner` stands in so these stay fast and
deterministic while still proving the tool's contract — concurrent dispatch,
summary formatting, graceful errors, and type validation.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.agent.subagent import SubAgentResult
from zakcode.messages import ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.task import ALREADY_RUN, TaskTool
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
    assert "omit subagent_type for the default" in result.output  # steers the retry
    assert spawner.calls == []  # validation happens before any spawn


async def test_default_alias_coerces_to_default_type(tmp_path: Path) -> None:
    # Models routinely guess "default" (measured: a 27B did, 2026-08-22); on slow local
    # inference a bounced call costs a full model round-trip, so that one alias is absorbed.
    spawner = _FakeSpawner(types=["general-purpose", "plan"])
    result = await TaskTool().execute(
        {"tasks": [{"subagent_type": "default", "prompt": "go"}]}, _ctx(tmp_path, spawner)
    )
    assert not result.is_error
    assert spawner.calls == [("general-purpose", "go")]


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


# --- ADR-0382: a batch identical to one that ran to its end this turn gets a pointer ---------


def _ctx_seen(tmp_path: Path, spawner: object, *, seen: bool = True) -> ToolContext:
    """A turn's context whose conversation does (``seen``) or does not still hold each result."""
    return ToolContext(workspace_root=tmp_path, spawner=spawner, in_context=lambda _text: seen)


async def test_an_identical_batch_that_ran_to_its_end_gets_a_pointer(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    ctx = _ctx_seen(tmp_path, spawner)
    batch = {"tasks": [{"prompt": "a"}, {"prompt": "b"}]}
    first = await TaskTool().execute(batch, ctx)
    again = await TaskTool().execute(batch, ctx)
    assert spawner.calls == [("general-purpose", "a"), ("general-purpose", "b")]  # ran once
    (at, _head, chars) = next(iter(ctx.task_runs.values()))
    assert re.fullmatch(r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z", at)
    assert chars == len(first.output)
    assert not again.is_error
    assert again.output == ALREADY_RUN.format(at=at, chars=len(first.output))


async def test_the_pointer_asks_whether_the_head_of_the_earlier_result_is_in_context(
    tmp_path: Path,
) -> None:
    """A clamp keeps a result's head and a compaction takes the whole result: the head answers."""
    asked: list[str] = []

    def in_context(text: str) -> bool:
        asked.append(text)
        return True

    spawner = _FakeSpawner()
    ctx = ToolContext(workspace_root=tmp_path, spawner=spawner, in_context=in_context)
    batch = {"tasks": [{"prompt": "x" * 1000}]}
    first = await TaskTool().execute(batch, ctx)
    again = await TaskTool().execute(batch, ctx)
    assert asked == [first.output[:400]]
    assert len(spawner.calls) == 1
    # The pointer names the whole result's length, not the probed head's.
    assert len(first.output) > 400
    assert f"({len(first.output):,} characters)" in again.output


async def test_a_batch_whose_result_left_the_conversation_runs_again(tmp_path: Path) -> None:
    spawner = _FakeSpawner()
    ctx = _ctx_seen(tmp_path, spawner, seen=False)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    again = await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    assert spawner.calls == [("general-purpose", "a")] * 2
    assert again.output == "[task 1] general-purpose:\na!"


async def test_a_bare_context_runs_every_batch(tmp_path: Path) -> None:
    """No conversation to check (``in_context`` is None): nothing is answered with a pointer."""
    spawner = _FakeSpawner()
    ctx = _ctx(tmp_path, spawner)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    assert len(spawner.calls) == 2


async def test_rerun_true_runs_the_batch_again_and_the_next_repeat_points_at_it(
    tmp_path: Path,
) -> None:
    spawner = _FakeSpawner()
    ctx = _ctx_seen(tmp_path, spawner)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    again = await TaskTool().execute({"tasks": [{"prompt": "a"}], "rerun": True}, ctx)
    assert len(spawner.calls) == 2
    assert again.output == "[task 1] general-purpose:\na!"
    third = await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    assert third.output.startswith("[already run this turn] ")
    assert len(spawner.calls) == 2


async def test_one_batch_whatever_its_spelling_and_another_type_is_another(
    tmp_path: Path,
) -> None:
    """The key is what the batch runs: the resolved type and the prompt, in order."""
    spawner = _FakeSpawner(types=["general-purpose", "plan"])
    ctx = _ctx_seen(tmp_path, spawner)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    alias = await TaskTool().execute({"tasks": [{"prompt": "a", "subagent_type": "default"}]}, ctx)
    single = await TaskTool().execute({"prompt": "a", "description": "a label"}, ctx)
    assert alias.output.startswith("[already run this turn] ")
    assert single.output.startswith("[already run this turn] ")
    other = await TaskTool().execute({"tasks": [{"prompt": "a", "subagent_type": "plan"}]}, ctx)
    more = await TaskTool().execute({"tasks": [{"prompt": "a"}, {"prompt": "b"}]}, ctx)
    assert not other.output.startswith("[already run")
    assert not more.output.startswith("[already run")
    assert spawner.calls == [
        ("general-purpose", "a"),
        ("plan", "a"),
        ("general-purpose", "a"),
        ("general-purpose", "b"),
    ]


async def test_a_batch_with_a_failed_subtask_runs_again(tmp_path: Path) -> None:
    class _FlakySpawner(_FakeSpawner):
        async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
            self.calls.append((type_name, prompt))
            if prompt == "bad":
                raise RuntimeError("boom")
            return SubAgentResult(name=type_name, summary=f"{prompt}-ok")

    spawner = _FlakySpawner()
    ctx = _ctx_seen(tmp_path, spawner)
    batch = {"tasks": [{"prompt": "good"}, {"prompt": "bad"}]}
    await TaskTool().execute(batch, ctx)
    again = await TaskTool().execute(batch, ctx)
    assert len(spawner.calls) == 4
    assert "ERROR: RuntimeError: boom" in again.output


async def test_a_batch_with_a_stopped_subtask_runs_again(tmp_path: Path) -> None:
    class _StoppedSpawner(_FakeSpawner):
        async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
            self.calls.append((type_name, prompt))
            return SubAgentResult(name=type_name, summary="half", stop_reason="max_iterations")

    spawner = _StoppedSpawner()
    ctx = _ctx_seen(tmp_path, spawner)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    again = await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    assert len(spawner.calls) == 2
    assert "(stopped: max_iterations)" in again.output


async def test_a_rerun_that_fails_leaves_the_next_repeat_to_run(tmp_path: Path) -> None:
    class _ThenBoomSpawner(_FakeSpawner):
        async def spawn(self, *, type_name: str, prompt: str) -> SubAgentResult:
            self.calls.append((type_name, prompt))
            if len(self.calls) == 2:
                raise RuntimeError("boom")
            return SubAgentResult(name=type_name, summary=f"{prompt}!")

    spawner = _ThenBoomSpawner()
    ctx = _ctx_seen(tmp_path, spawner)
    await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    failed = await TaskTool().execute({"tasks": [{"prompt": "a"}], "rerun": True}, ctx)
    assert failed.is_error
    third = await TaskTool().execute({"tasks": [{"prompt": "a"}]}, ctx)
    assert len(spawner.calls) == 3
    assert third.output == "[task 1] general-purpose:\na!"


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _task_call(call_id: str, prompt: str = "a") -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id=call_id, name="task", arguments={"tasks": [{"prompt": prompt}]})],
        usage=Usage(total_tokens=1),
    )


_DONE = LLMResult(text="done", usage=Usage(total_tokens=1))


def _outputs(loop: AgentLoop) -> list[str]:
    return [
        block.output
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]


@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_answers_a_repeat_in_the_turn_and_runs_it_in_the_next(
    tmp_path: Path, streaming: bool
) -> None:
    """Both turn paths: a repeat in the same turn gets the pointer, through the loop's own
    conversation check; the next turn's context starts with no record, so the batch runs."""
    spawner = _FakeSpawner()
    registry = ToolRegistry()
    registry.register(TaskTool())
    provider = _Scripted([_task_call("t1"), _task_call("t2"), _DONE])
    loop = AgentLoop(provider, registry, Session(cwd=str(tmp_path), model="t"), spawner=spawner)

    async def turn(text: str) -> None:
        if streaming:
            async for _event in loop.astream_turn(text):
                pass
        else:
            await loop.arun_turn(text)

    await turn("go")
    outputs = _outputs(loop)
    assert len(spawner.calls) == 1
    assert outputs[0].startswith("[task 1] general-purpose:\na!")
    assert outputs[1].startswith("[already run this turn] This exact task batch ran to its end at ")
    provider._results = [_task_call("t3"), _DONE]
    provider.calls = 0
    await turn("again")
    assert len(spawner.calls) == 2
    assert _outputs(loop)[-1].startswith("[task 1] general-purpose:\na!")
