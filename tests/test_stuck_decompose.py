"""Decompose-on-stuck (ADR-0057): rung 1 adds investigative steps to the plan, not advice.

Field observation 2026-08-28 (coach on zc-03): every "recovering: no progress — nudging a
rethink" was the cue that the task needed MORE decomposition — the model needed steps on
its list, not a paragraph telling it to think differently. So the stuck evidence the
tracker already holds (which calls keep failing, whether the same result keeps being
re-measured) becomes primitive plan steps with done-conditions, spliced in ahead of the
stuck step, where the re-injected plan and the plan gate keep them in view.

Hermetic: scripted providers, in-memory tools, a tmp workspace.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentDone, AgentStatus, AgentTaskUpdate
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

PROBE_OUTPUT = (
    "---\n---\nNo mind refs in packed-refs\n---\n"
    "[runner-claim] acquire: HELD (backend=local) — another machine owns a live claim\n"
    "ACQUIRE_RC=4\n[exit code: 0]"
)


def _c(call_id: str, name: str, **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


class _BoomTool(Tool):
    spec = ToolSpec(name="boom", description="Always errors.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.error("kaboom: no such file")


class _ProbeTool(Tool):
    spec = ToolSpec(name="probe", description="Read-only; the same long output every time.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=PROBE_OUTPUT)


class _ScriptByCallProvider(Provider):
    def __init__(self, factory: Any) -> None:
        self._factory = factory
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._factory(self.calls)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _boom_forever() -> _ScriptByCallProvider:
    # A fresh arg every iteration: the exact-repeat doom guard never fires, the ladder does.
    return _ScriptByCallProvider(lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", n=n)]))


def _loop(provider: Provider, tmp_path: Path, *tools: Tool) -> AgentLoop:
    registry = ToolRegistry()
    for tool in tools or (_BoomTool(),):
        registry.register(tool)
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=15,
    )


def _rails(loop: AgentLoop) -> list[str]:
    return [m.text for m in loop.session.messages if m.role == "user" and m.text]


def _titles(loop: AgentLoop) -> list[str]:
    return [t.title for t in loop.session.task_network.tasks]


def _drain(loop: AgentLoop, text: str) -> list[Any]:
    async def run() -> list[Any]:
        return [ev async for ev in loop.astream_turn(text)]

    return asyncio.run(run())


# ── the network primitive ─────────────────────────────────────────────────────


def test_insert_before_splices_ahead_of_the_focus_and_demotes_it() -> None:
    network = TaskNetwork(tasks=[Task(title="A", status="in_progress"), Task(title="B")])
    network.normalize()
    anchor = network.current()
    assert anchor is not None and anchor.title == "A"
    probes = [Task(title="X"), Task(title="Y")]
    network.insert_before(anchor, probes)
    assert [t.title for t in network.tasks] == ["X", "Y", "A", "B"]
    assert [t.id for t in network.tasks] == ["1", "2", "3", "4"]  # re-numbered by position
    assert anchor.status == "pending"  # the stuck step no longer holds the focus
    current = network.current()
    assert current is probes[0]  # first new step is the current work, in document order
    assert network.contains(probes[0]) and network.contains(anchor)
    network.tasks = [Task(title="rewritten")]  # the model replaces the whole plan
    network.normalize()
    assert not network.contains(probes[0])


def test_insert_before_none_appends_at_the_top_level() -> None:
    network = TaskNetwork()
    network.insert_before(None, [Task(title="Investigate: something")])
    assert [t.id for t in network.tasks] == ["1"]
    assert network.current() is network.tasks[0]


# ── rung 1 in the loop (buffered twin) ────────────────────────────────────────


def test_stuck_rung_one_adds_investigative_steps_instead_of_advice(tmp_path: Path) -> None:
    loop = _loop(_boom_forever(), tmp_path)
    result = asyncio.run(loop.arun_turn("do the thing"))
    assert result.stop_reason == "stuck"  # the model ignores every rung; the ladder still ends
    titles = _titles(loop)
    # Varying arguments, same failure: the wrong-premise shape names the tool, not the args.
    assert titles[0] == "Investigate: why `boom` keeps failing across 3 attempts"
    assert titles[1].startswith("Decide: name the assumption")
    note = loop.session.task_network.tasks[0].note
    assert "arguments are not the problem" in note and "Done when" in note
    added = [r for r in _rails(loop) if "I added 2 investigative steps" in r]
    assert len(added) == 1
    assert "appear to be stuck" in added[0] and "(1, 2)" in added[0]
    assert added[0].startswith("[harness] Hint:")  # loop-injected guidance keeps its provenance
    transcript = "\n".join(_rails(loop))
    assert "harder than it looked" not in transcript  # the old advice suffix is gone
    assert "Stop and reconsider" not in transcript


def test_steps_go_ahead_of_the_step_the_model_is_stuck_on(tmp_path: Path) -> None:
    loop = _loop(_boom_forever(), tmp_path)
    loop.session.task_network = TaskNetwork(
        tasks=[Task(title="fix the build", status="in_progress"), Task(title="run the tests")]
    )
    loop.session.task_network.normalize()
    asyncio.run(loop.arun_turn("do the thing"))
    titles = _titles(loop)
    assert titles[0].startswith("Investigate:") and titles[1].startswith("Decide:")
    assert titles[2:] == ["fix the build", "run the tests"]  # investigate, decide, then retry
    assert loop.session.task_network.tasks[2].status == "pending"  # demoted from in_progress


def test_same_call_retried_names_the_call(tmp_path: Path) -> None:
    # A,B,A,B: each call repeats (the repeated-failure signal) without ever tripping the
    # exact-repeat doom guard, so the ladder — not the guard — handles it.
    paths = ["/nope/a", "/nope/b"]
    provider = _ScriptByCallProvider(
        lambda n: LLMResult(tool_calls=[_c(f"c{n}", "boom", path=paths[n % 2])])
    )
    loop = _loop(provider, tmp_path)
    asyncio.run(loop.arun_turn("do the thing"))
    titles = _titles(loop)
    assert titles[0] == "Investigate: why `boom` keeps failing with the same arguments"
    assert 'boom({"path": "/nope/' in loop.session.task_network.tasks[0].note


def test_no_plan_makes_the_investigation_the_plan_for_this_turn(tmp_path: Path) -> None:
    loop = _loop(_boom_forever(), tmp_path)
    assert loop.session.task_network.is_empty()
    asyncio.run(loop.arun_turn("do the thing"))
    network = loop.session.task_network
    assert len(network.tasks) == 2
    # The steps live for the turn that seeded them: still open at the end, they are retired
    # (cancelled, not deleted — the transcript stays honest) so they cannot haunt the next turn.
    assert all(t.status == "cancelled" for t in network.tasks)
    assert network.current() is None


class _RecoversOnStepBackProvider(Provider):
    """Fails until the step-back rail lands, then answers — never touching the plan."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        # The plan re-injection now follows every rail, so the rail is not the LAST message.
        recent = " ".join((m.text or "") for m in messages[-3:]).lower()
        if "take a step back" in recent:
            return LLMResult(text="Stepping back: the real path is elsewhere. Done.")
        return LLMResult(tool_calls=[_c(f"c{self.calls}", "boom", n=self.calls)])

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def test_open_investigation_steps_never_hold_a_recovered_turn(tmp_path: Path) -> None:
    # The model got unstuck another way (the step-back rail) and answered without marking
    # the harness's steps done. The plan gate must not send it back to do them.
    loop = _loop(_RecoversOnStepBackProvider(), tmp_path)
    result = asyncio.run(loop.arun_turn("fetch the drive notes"))
    assert result.stop_reason == "completed"
    assert result.iterations == 6
    assert not any("open step" in r for r in _rails(loop))  # no plan-gate nudge fired
    assert all(t.status == "cancelled" for t in loop.session.task_network.tasks)


def test_the_models_own_open_steps_still_hold_the_turn(tmp_path: Path) -> None:
    # The gate keeps its teeth for steps the MODEL committed to: only the harness's are skipped.
    loop = _loop(_RecoversOnStepBackProvider(), tmp_path)
    loop.session.task_network = TaskNetwork(tasks=[Task(title="write the summary")])
    loop.session.task_network.normalize()
    asyncio.run(loop.arun_turn("fetch the drive notes"))
    nudges = [r for r in _rails(loop) if "open step" in r]
    assert nudges and "write the summary" in nudges[0]


def test_repeated_outcome_gets_a_re_measurement_step(tmp_path: Path) -> None:
    # The coach shape (ADR-0038): the same probe output re-measured with a different comment
    # each time — nothing errors, so the evidence is the repeat itself, not a failing call.
    provider = _ScriptByCallProvider(
        lambda n: LLMResult(tool_calls=[_c(f"c{n}", "probe", command=f"# probe {n}\ngit refs")])
    )
    loop = _loop(provider, tmp_path, _ProbeTool())
    asyncio.run(loop.arun_turn("find the stale claim ref"))
    titles = _titles(loop)
    assert titles[0] == "Investigate: what the result you keep re-measuring already tells you"
    assert titles[1].startswith("Decide:")
    added = [r for r in _rails(loop) if "I added 2 investigative steps" in r]
    assert added and "observed the SAME tool result 3 times" in added[0]


def test_re_climb_points_back_at_open_steps_instead_of_adding_more(tmp_path: Path) -> None:
    # nudge@3 -> narrow@4 -> step-back@5 (streak resets) -> nudge again @8: the first batch is
    # still open (the model never marked anything done), so no second batch lands on top.
    loop = _loop(_boom_forever(), tmp_path)
    result = asyncio.run(loop.arun_turn("do the thing"))
    assert result.iterations == 10
    titles = _titles(loop)
    assert sum(t.startswith("Investigate:") for t in titles) == 1
    assert sum(t.startswith("Decide:") for t in titles) == 1
    rails = _rails(loop)
    assert sum("I added 2 investigative steps" in r for r in rails) == 1
    assert sum("are still open" in r for r in rails) == 1


# ── streaming twin ────────────────────────────────────────────────────────────


def test_streaming_twin_announces_the_steps_and_redraws_the_plan(tmp_path: Path) -> None:
    loop = _loop(_boom_forever(), tmp_path)
    events = _drain(loop, "do the thing")
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.stop_reason == "stuck"
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert any("added 2 investigative steps in the plan" in m for m in statuses)
    assert not any("nudging" in m for m in statuses)
    updates = [e for e in events if isinstance(e, AgentTaskUpdate)]
    assert updates and "Investigate:" in updates[0].plan  # the client can redraw the list
    assert _titles(loop)[0].startswith("Investigate:")
