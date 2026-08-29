"""A skill invocation typed as text is the invocation, not an answer.

Measured 2026-08-28 (coach, zc-03, build 99bab59): the served ``/start`` finished its last
step and the model's next completion was the single line ``/boot``. The loop saw a text-only
completion, the plan gate pushed on, and the model called ``use_skill("aspirations")`` — the
whole boot (prime, hypothesis review, status report) skipped. Now a completion that IS one
``/<skill> [args]`` line naming a discovered skill is routed through ``use_skill``, the one
door skills take, in both twins. Prose that merely mentions a skill is left alone.

Also here: the per-iteration trace checkpoint — a runner's turn may never end, and a trace
that only lands at turn end never lands at all.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.config import load_settings
from zakcode.events import AgentEvent, AgentStatus
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tools.base import SkillLoad, ToolRegistry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.tools.builtins.use_skill import UseSkillTool

BOOT = (
    "# /boot\n\nBoot the agent.\n\n## Step 1: Status\n\nPrint status.\n\n"
    "## Step 2: Prime\n\nPrime.\n"
)


class _Resolver:
    def __init__(self, bodies: dict[str, str]) -> None:
        self._bodies = bodies

    def names(self) -> list[str]:
        return list(self._bodies)

    def body(self, name: str) -> str | None:
        return self._bodies.get(name)

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        if name in self._bodies:
            return SkillLoad(found=True, name=name, body=self._bodies[name])
        return SkillLoad(found=False, name=name)


class _Scripted(Provider):
    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self.calls = 0

    async def acomplete(self, messages: list[Message], **kw: Any) -> LLMResult:
        self.calls += 1
        return self._factory(self.calls)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return sum(len(getattr(b, "text", "") or "") for m in messages for b in m.blocks) // 4

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=32_768)


def _loop(provider: Provider, tmp_path: Path, *, trace_dir: str | None = None) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(UseSkillTool())
    registry.register(UpdatePlanTool())
    settings = load_settings(workspace_root=tmp_path)
    if trace_dir is not None:
        settings = settings.model_copy(update={"trace_dir": trace_dir})
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        settings=settings,
        workspace_root=tmp_path,
        max_iterations=10,
        skill_resolver=_Resolver({"boot": BOOT}),
    )


def _use_skill_results(loop: AgentLoop) -> list[ToolResultBlock]:
    return [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def test_a_slash_line_typed_as_text_runs_the_skill(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        return LLMResult(text="/boot") if n == 1 else LLMResult(text="booted")

    loop = _loop(_Scripted(script), tmp_path)
    result = asyncio.run(loop.arun_turn("start the agent"))
    assert result.stop_reason == "completed"
    (note,) = [e for e in loop._trace.events if e.data.get("kind") == "slash_text_routed"]
    assert note.data["skill"] == "boot"
    # The transcript pairs the synthesized use_skill call with its result, so the provider
    # sees a well-formed exchange, and the skill's sections became the plan.
    uses = [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolUseBlock)]
    assert uses and uses[0].name == "use_skill" and uses[0].input == {"name": "boot"}
    (res,) = _use_skill_results(loop)
    # Two short sections pack into one page (ADR-0088): the whole body arrives at once.
    assert res.tool_use_id == uses[0].id and "Print status." in res.output
    assert "Prime." in res.output and "— page 1 of" not in res.output
    assert [t.title for t in loop.session.task_network.tasks] == ["Step 1: Status", "Step 2: Prime"]


def test_args_after_the_slash_travel_with_it(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        return LLMResult(text="`/boot --recover --force`.") if n == 1 else LLMResult(text="ok")

    loop = _loop(_Scripted(script), tmp_path)
    asyncio.run(loop.arun_turn("go"))
    uses = [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolUseBlock)]
    assert uses and uses[0].input == {"name": "boot", "args": "--recover --force"}


def test_prose_that_mentions_a_skill_is_an_answer(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        return LLMResult(text="Next I would run /boot, but the state is wrong.")

    loop = _loop(_Scripted(script), tmp_path)
    asyncio.run(loop.arun_turn("what next?"))
    assert not [e for e in loop._trace.events if e.data.get("kind") == "slash_text_routed"]
    assert _use_skill_results(loop) == []


def test_an_unknown_slash_stays_text(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        return LLMResult(text="/nothing")

    loop = _loop(_Scripted(script), tmp_path)
    asyncio.run(loop.arun_turn("go"))
    assert _use_skill_results(loop) == []


def test_streaming_twin_routes_and_announces(tmp_path: Path) -> None:
    def script(n: int) -> LLMResult:
        return LLMResult(text="/boot") if n == 1 else LLMResult(text="booted")

    loop = _loop(_Scripted(script), tmp_path)

    async def run() -> list[AgentEvent]:
        return [ev async for ev in loop.astream_turn("start the agent")]

    events = asyncio.run(run())
    assert any(
        isinstance(ev, AgentStatus) and "'/boot' typed as text" in ev.message for ev in events
    )
    (res,) = _use_skill_results(loop)
    assert "Print status." in res.output and "Prime." in res.output


def test_the_trace_is_checkpointed_every_iteration(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    seen: list[bool] = []

    def script(n: int) -> LLMResult:
        if n == 1:
            return LLMResult(text="/boot")
        # Mid-turn: the previous iteration's checkpoint is already on disk.
        seen.append(any(trace_dir.rglob("turn_1.jsonl")))
        return LLMResult(text="booted")

    loop = _loop(_Scripted(script), tmp_path, trace_dir=str(trace_dir))
    asyncio.run(loop.arun_turn("start the agent"))
    # Side calls (the plan-quality check after seeding) share the provider and may run
    # before the batch boundary; the main-loop completion after it sees the checkpoint.
    assert seen and seen[-1] is True
    (dumped,) = list(trace_dir.rglob("turn_1.jsonl"))
    kinds = [json.loads(line)["kind"] for line in dumped.read_text().splitlines()]
    assert "stop" in kinds  # the final dump still carries the whole turn
