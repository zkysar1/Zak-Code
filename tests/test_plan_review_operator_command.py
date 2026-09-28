"""ADR-0265: no fresh-eyes review after an operator-only command arrives mid-turn.

An operator's stop reached a running turn through the say inbox; the agent ran it, the plan read
complete, and the fresh-eyes review (ADR-0117) then judged that plan against the request the turn
had OPENED with, flagged a gap, and kept the stopped agent working. These tests pin the rule:
an operator-only command delivered mid-turn takes the review off that turn, on both turn paths;
an ordinary skill delivered the same way does not (the positive control); and the next turn is
reviewed again.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage

_DONE_PLAN = [{"title": "run the command", "status": "done"}, {"title": "report", "status": "done"}]


def _body(name: str) -> str:
    return (
        f"<command-message>{name} is running</command-message>\n"
        f"<command-name>/{name}</command-name>\n"
        "<command-args></command-args>\n\n"
        f"# {name.title()}\n\n## Phase 1: Act\n\nact\n\n## Phase 2: Report\n\nreport\n"
    )


class _Composed:
    def __init__(self, name: str) -> None:
        self.invoked = True
        self.name = name
        self.turn_text = _body(name)
        self.denied_reason = None
        self.error = None


async def _compose(name: str, args: str = "", *, fuzzy: bool = True) -> _Composed:
    return _Composed(name)


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


class _SayWhileRunning(Tool):
    """The operator sends a say while the turn is running a tool."""

    spec = ToolSpec(name="poke", description="Write a say into the workspace inbox.")

    def __init__(self, inbox: Path, text: str) -> None:
        self._inbox = inbox
        self._text = text

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        write_say(self._inbox, self._text)
        return ToolResult.ok(output="poked")


def _call(name: str, arguments: dict[str, Any] | None = None) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id="c", name=name, arguments=arguments or {})],
        usage=Usage(total_tokens=1),
    )


def _done(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


_JUDGE_OK = _done(
    json.dumps({"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}})
)


def _script() -> list[LLMResult]:
    """poke (the say lands), a finished plan of the model's own, the plan judge, the answer."""
    return [
        _call("poke"),
        _call("update_plan", {"tasks": _DONE_PLAN}),
        _JUDGE_OK,
        _done("The command ran and the run is stopped."),
    ]


def _loop(tmp_path: Path, say: str) -> tuple[AgentLoop, list[str]]:
    registry = ToolRegistry()
    registry.register(UpdatePlanTool())
    registry.register(_SayWhileRunning(say_path(tmp_path), say))
    session = Session(cwd=str(tmp_path), model="test/model")
    loop = AgentLoop(
        _Scripted(_script()),
        registry,
        session,
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=10,
        consume_say_inbox=True,
        compose_skill=_compose,
    )
    loop._user_only_skills = lambda: {"stop"}  # type: ignore[method-assign]
    reviewed: list[str] = []

    async def critic(request: str, claimed_result: str, *, record: str = "") -> tuple[bool, str]:
        reviewed.append(request)
        return True, ""

    loop._completion_critic = critic  # type: ignore[method-assign]
    return loop, reviewed


async def _run(loop: AgentLoop, streaming: bool, text: str) -> str:
    if streaming:
        events = [ev async for ev in loop.astream_turn(text)]
        return str(events[-1].stop_reason)
    return str((await loop.arun_turn(text)).stop_reason)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_an_operator_only_command_mid_turn_takes_the_review_off_the_turn(
    tmp_path: Path, streaming: bool
) -> None:
    loop, reviewed = _loop(tmp_path, "/stop")
    assert await _run(loop, streaming, "keep working on the queue") == "completed"
    assert reviewed == []
    assert not any(t.title.startswith("Reviewer flagged") for t in loop.session.task_network.tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_positive_control_an_ordinary_skill_mid_turn_is_still_reviewed(
    tmp_path: Path, streaming: bool
) -> None:
    loop, reviewed = _loop(tmp_path, "/probe")
    assert await _run(loop, streaming, "keep working on the queue") == "completed"
    assert reviewed == ["keep working on the queue"]


@pytest.mark.asyncio
async def test_the_next_turn_is_reviewed_again(tmp_path: Path) -> None:
    loop, reviewed = _loop(tmp_path, "/stop")
    assert await _run(loop, False, "keep working on the queue") == "completed"
    assert reviewed == []
    loop.provider = _Scripted(  # type: ignore[attr-defined]
        [_call("update_plan", {"tasks": _DONE_PLAN}), _JUDGE_OK, _done("Done.")]
    )
    assert await _run(loop, False, "one more thing") == "completed"
    assert reviewed == ["one more thing"]
