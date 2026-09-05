"""A finished plan yields the VERDICT, not "plan finished" (ADR-0108).

Field report 2026-09-05: after the last plan step closed, the agent said "plan finished",
left the finished checklist on screen and never answered the question the user had asked.
Four seams line up to produce that: the update_plan result carries no hint once the plan
is complete; the next model call is handed the whole finished checklist as its
highest-salience message; no completion gate catches a bare status; the plan is dropped
only at the NEXT turn start. These tests pin the four fixes on both turn paths, with a
scripted provider (tier 1, no network).
"""

from __future__ import annotations

import json

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.events import AgentStatus
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage

BARE_STATUS = "The plan shows all steps are complete. No further action is needed."
VERDICT = "Answer: the flake is a stale cache; clearing it fixed the build and tests pass."


class _Scripted(Provider):
    """Returns a fixed sequence of completions and records the messages it was handed."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0
        self.seen: list[list] = []

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.seen.append(list(messages))
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _plan_call(tasks: list[dict]) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id="p1", name="update_plan", arguments={"tasks": tasks})],
        usage=Usage(total_tokens=1),
    )


def _judge_ok() -> LLMResult:
    # The always-on decomposition judge (ADR-0050) fires once per turn on the first
    # structural plan authoring; a strong scorecard keeps its critique silent.
    return LLMResult(
        text=json.dumps(
            {"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}
        ),
        usage=Usage(total_tokens=2),
    )


def _done(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _loop(provider: Provider) -> tuple[AgentLoop, Session]:
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(provider, default_registry(), session, max_iterations=20)
    return loop, session


def _completing_script(*completions: str) -> list[LLMResult]:
    """A plan whose every step is already terminal, the judge, then the given completions."""
    return [
        _plan_call([{"title": "A", "status": "done"}, {"title": "B", "status": "done"}]),
        _judge_ok(),
        *(_done(t) for t in completions),
    ]


def _harness_nudges(session: Session) -> list[str]:
    return [m.text for m in session.messages if m.role == "user" and "[harness]" in m.text]


# ── A. the update_plan tool hints at the verdict the moment the plan completes ──────────


@pytest.mark.asyncio
async def test_update_plan_hints_for_the_verdict_when_the_plan_completes(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    ctx.task_network = TaskNetwork()
    tool = UpdatePlanTool()

    open_plan = await tool.execute(
        {"tasks": [{"title": "A", "status": "done"}, {"title": "B", "status": "in_progress"}]}, ctx
    )
    assert open_plan.hint is not None and "<- current" in open_plan.hint  # the next-step rail

    complete = await tool.execute(
        {"tasks": [{"title": "A", "status": "done"}, {"title": "B", "status": "done"}]}, ctx
    )
    assert complete.hint is not None, "a completed plan must not leave the model with no hint"
    assert "original request" in complete.hint.lower()
    assert "conclusion" in complete.hint.lower()

    cleared = await tool.execute({"tasks": []}, ctx)
    assert cleared.hint is None  # nothing to say about an empty board


# ── B. the finished checklist is no longer re-injected; a one-line verdict prompt is ──────


@pytest.mark.asyncio
async def test_completed_plan_is_replaced_by_a_one_line_verdict_prompt() -> None:
    provider = _Scripted(_completing_script(VERDICT))
    loop, session = _loop(provider)
    result = await loop.arun_turn("why does the build flake?")
    assert result.stop_reason == "completed"

    # seen[0] authors the plan, seen[1] is the judge's scoring prompt, seen[2] is the model
    # call that produces the answer — the one that used to stare at the finished checklist.
    tail = [m.text for m in provider.seen[2]]
    assert not any("Current plan (" in t for t in tail), "finished checklist still re-injected"
    assert not any("Keep it current with the update_plan tool" in t for t in tail)
    assert any(
        "[plan]" in t and "complete" in t.lower() and "original request" in t.lower() for t in tail
    )
    # The network itself is intact until the next turn start (UIs, probes, the reset).
    assert session.task_network.is_complete()


# ── C. the plan-verdict rail: a bare status is asked once for the conclusion ─────────────


@pytest.mark.asyncio
async def test_bare_plan_status_is_nudged_once_then_the_verdict_completes() -> None:
    provider = _Scripted(_completing_script(BARE_STATUS, VERDICT))
    loop, session = _loop(provider)
    result = await loop.arun_turn("why does the build flake?")

    assert result.stop_reason == "completed"
    assert result.degraded is False
    assert provider.calls == 4  # plan, judge, bare status (nudged), verdict
    nudges = [t for t in _harness_nudges(session) if "original request" in t.lower()]
    assert len(nudges) == 1
    assert nudges[0].startswith("[harness] Hint:")


@pytest.mark.asyncio
async def test_verdict_nudge_fires_at_most_once_per_turn() -> None:
    provider = _Scripted(_completing_script(BARE_STATUS, "Plan finished."))
    loop, _ = _loop(provider)
    result = await loop.arun_turn("why does the build flake?")
    assert result.stop_reason == "completed"
    assert provider.calls == 4  # the second bare status ends the turn; no third ask


@pytest.mark.asyncio
async def test_a_real_conclusion_after_a_completed_plan_is_not_nudged() -> None:
    provider = _Scripted(_completing_script(VERDICT))
    loop, session = _loop(provider)
    result = await loop.arun_turn("why does the build flake?")
    assert result.stop_reason == "completed"
    assert provider.calls == 3
    assert not any("original request" in t.lower() for t in _harness_nudges(session))


@pytest.mark.asyncio
async def test_status_words_without_a_plan_are_not_nudged() -> None:
    provider = _Scripted([_done(BARE_STATUS)])
    loop, session = _loop(provider)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert provider.calls == 1
    assert _harness_nudges(session) == []


@pytest.mark.asyncio
async def test_streaming_path_nudges_the_bare_status_too() -> None:
    # The base Provider's astream wraps acomplete, so the same script drives the streaming
    # twin; the rail announces itself as an AgentStatus there.
    provider = _Scripted(_completing_script(BARE_STATUS, VERDICT))
    loop, session = _loop(provider)
    events = [ev async for ev in loop.astream_turn("why does the build flake?")]
    assert provider.calls == 4
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any("verdict" in s.lower() or "conclusion" in s.lower() for s in statuses)
    assert len([t for t in _harness_nudges(session) if "original request" in t.lower()]) == 1


def test_plan_status_matcher_is_surgical() -> None:
    from zakcode.agent.loop import _ends_on_plan_status

    incident = "The plan shows all 3 steps are complete. No further action is needed."  # ADR-0026
    assert _ends_on_plan_status(incident)
    assert _ends_on_plan_status("Plan finished.")
    assert _ends_on_plan_status("All steps are done.")
    assert _ends_on_plan_status("Every step of the plan has been completed.")
    assert not _ends_on_plan_status("I created the file and verified it; done.")
    assert not _ends_on_plan_status(VERDICT)
    assert not _ends_on_plan_status("The plan's second step revealed the bug: a stale cache.")
    # Only the TAIL is judged — an early status phrase followed by a real answer is fine.
    assert not _ends_on_plan_status("All steps are done. " + "Root cause: a stale cache. " * 30)
