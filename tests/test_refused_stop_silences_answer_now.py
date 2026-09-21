"""ADR-0205: once a turn-end hook has refused a stop taken on a FINISHED plan, that plan's
"answer now" line is not sent again until the plan changes.

The line (ADR-0108) is ephemeral and rides LAST. After a refused stop the hook has said what
happens next, and the line followed the hook's words on every request, saying the opposite:
carry on, then answer and stop. Measured on gpt-5.6-luna by the veto-door bench (registration
2, ``bench/results/veto-door-preregistration.log``): with the line, 18 of 116 rollouts stopped
a second time; without it, 0 of 116.

What these pin, on the WIRE and not only on a label (the scripted provider keeps which plan
reminder each request it was handed ended with): the line rides until the refusal and not after
it; the hook's words are then the last thing in the request; the silence is for exactly that
plan, so a plan that moves on and finishes again gets its line once more, even when it reads
the same and even when it moved on a resting call; a refusal with no finished plan silences
nothing; and the veto-stall fence (ADR-0187) still ends a model that never resumes after three
deliveries, with the line silent throughout.

Hermetic: scripted providers, an in-process hook, a fake composer. The hook's words are made
up for these tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import _VETO_STALL_THRESHOLD, AgentLoop, skill_reentry_in
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry
from zakcode.usage import Usage

MODEL = "fake/scripted"
#: A hook that wants more work and names no skill: the harness re-enters with its words.
PLAIN_REASON = "Not finished: the queue still holds work. Carry on with the next item."
#: A hook that names a skill re-entry, which the harness delivers (ADR-0187).
SKILL_REASON = "The turn ended early. Your FIRST action MUST be: Skill('cycle') with args='loop'."
#: The delivered skill, without sections: nothing is seeded into the plan, so the finished
#: plan stays exactly as it was and a text-only completion reaches the Stop-hook seam.
CYCLE_TURN = (
    "<command-message>cycle is running</command-message>\n"
    "<command-name>/cycle</command-name>\n"
    "<command-args>loop</command-args>\n\n"
    "# Cycle\n\nOpen the next cycle.\n"
)


def _text_of(message: Message) -> str:
    return "".join(str(getattr(block, "text", "") or "") for block in message.blocks)


def _reminder_kind(message: Message) -> str | None:
    """Which plan reminder a message IS, read from its own text and not from any label."""
    text = _text_of(message)
    if text.startswith("[plan] Plan complete"):
        return "plan_complete"
    return "plan" if text.startswith("[plan] ") else None


class _Scripted(Provider):
    """Canned completions in order (the last repeats). Keeps, per call, which plan reminder
    the request it was HANDED ended with, and that request's last message."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0
        self.ended_with: list[str | None] = []
        self.last_text: list[str] = []

    async def acomplete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any):  # type: ignore[override]
        self.ended_with.append(_reminder_kind(messages[-1]) if messages else None)
        self.last_text.append(_text_of(messages[-1]) if messages else "")
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]

    def model_id(self) -> str:
        return MODEL

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _plan(*statuses: str) -> LLMResult:
    tasks = [{"title": f"S{i}", "status": s, "note": "ok"} for i, s in enumerate(statuses)]
    call = ToolCall(id=f"p{len(statuses)}", name="update_plan", arguments={"tasks": tasks})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _say(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _judge_ok() -> LLMResult:
    scores = {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}
    return LLMResult(text=json.dumps({"scores": scores}), usage=Usage(total_tokens=2))


def _review_ok() -> LLMResult:
    return LLMResult(text='{"approved": true, "issues": ""}', usage=Usage(total_tokens=2))


class _Hook:
    """Refuses the stop with each scripted reason in turn, then lets the turn end."""

    def __init__(self, reasons: list[str]) -> None:
        self._reasons = list(reasons)
        self.consulted = 0

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        self.consulted += 1
        if not self._reasons:
            return None
        return TurnEndResult(vetoed=True, continuation_prompt=self._reasons.pop(0))


class _Composed:
    invoked, denied_reason, error = True, None, None

    def __init__(self, name: str) -> None:
        self.name, self.turn_text = name, CYCLE_TURN


async def _compose(name: str, args: str = "", *, fuzzy: bool = True, source: str = "command"):
    return _Composed(name)


def _loop(provider: Provider, tmp_path: Path, hook: _Hook) -> AgentLoop:
    loop = AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model=MODEL),
        workspace_root=tmp_path,
        max_iterations=30,
        turn_end_vetoable=True,
        compose_skill=_compose,
    )
    loop.hook_manager.register_turn_end(hook)
    return loop


async def _run(loop: AgentLoop, text: str, *, streamed: bool) -> str:
    if not streamed:
        return (await loop.arun_turn(text)).stop_reason
    async for _ in loop.astream_turn(text):
        pass
    return str(loop.session.last_stop_reason)


def _requests(loop: AgentLoop) -> list[tuple[list[str], list[str]]]:
    """``(rails, rails_silenced)`` of every main request of the turn, in order."""
    return [(e.data["rails"], e.data["rails_silenced"]) for e in loop._trace.of_kind("usage")]


def _main_calls(provider: _Scripted, script: list[LLMResult], side: set[int]) -> list[int]:
    """Indices of the calls that were main requests: every call but the side calls named."""
    return [i for i in range(provider.calls) if i not in side]


def _silenced_notes(loop: AgentLoop) -> int:
    notes = loop._trace.of_kind("intervention")
    return sum(1 for e in notes if e.data.get("kind") == "veto_plan_silenced")


# ── the rule ─────────────────────────────────────────────────�


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_line_rides_until_the_refusal_and_not_after_it(
    tmp_path: Path, streamed: bool
) -> None:
    script = [
        _plan("in_progress", "pending"),  # 0
        _judge_ok(),  # 1  side call
        _plan("done", "done"),  # 2
        _say("both steps are done: the answer"),  # 3  ended with the closing line, as always
        _review_ok(),  # 4  side call; then the stop, which the hook REFUSES
        _say("carrying on with the next item"),  # 5  the hook's words are last; the line is not
    ]
    provider, hook = _Scripted(script), _Hook([PLAIN_REASON])
    loop = _loop(provider, tmp_path, hook)
    assert await _run(loop, "two steps", streamed=streamed) == "completed"
    assert hook.consulted == 2  # refused once, then allowed
    assert [provider.ended_with[i] for i in (0, 2, 3, 5)] == [None, "plan", "plan_complete", None]
    assert PLAIN_REASON in provider.last_text[5]  # the hook has the last word
    assert _requests(loop) == [
        ([], []),
        (["plan"], []),
        (["plan_complete"], []),
        ([], ["plan_complete"]),  # not "no plan": kept silent, and said to be
    ]
    assert _silenced_notes(loop) == 1


@pytest.mark.asyncio
async def test_a_plan_that_moves_on_and_finishes_again_gets_its_line_again(tmp_path: Path) -> None:
    """Even when the second plan reads exactly like the refused one: same ids, titles and
    statuses, so the same signature. It is a new plan because the board was seen to move."""
    script = [
        _plan("in_progress"),  # 0
        _judge_ok(),  # 1  side
        _plan("done"),  # 2
        _say("done: the answer"),  # 3  line rides; review (4) then the refused stop
        _review_ok(),  # 4  side
        _plan("in_progress"),  # 5  silent; the model reopens the very same step
        _plan("done"),  # 6  the live checklist rides
        _say("done again: the answer"),  # 7  the closing line is owed once more
        _review_ok(),  # 8  side
    ]
    provider, hook = _Scripted(script), _Hook([PLAIN_REASON])
    loop = _loop(provider, tmp_path, hook)
    await loop.arun_turn("one step")
    main = [provider.ended_with[i] for i in (3, 5, 6, 7)]
    assert main == ["plan_complete", None, "plan", "plan_complete"]


@pytest.mark.asyncio
async def test_a_plan_that_moved_on_during_a_resting_call_is_still_seen_to_move(
    tmp_path: Path,
) -> None:
    """On a sparse-cache model the tail rests every other call (ADR-0193), and a resting call
    builds no reminder. The refused plan is reopened by the response to a request that carried
    a tail and closed again by the response to the RESTING one, so no reminder is ever built
    while it is open. It still finishes as a new plan."""
    script = [
        _plan("in_progress"),  # 0
        _judge_ok(),  # 1  side
        _plan("done"),  # 2
        _say("done: the answer"),  # 3  review (4), then the refused stop
        _review_ok(),  # 4  side
        _plan("in_progress"),  # 5  plan finished: never rests; silent. Response reopens it
        _plan("done"),  # 6  RESTS (call 5 carried the hook context). Response closes it
        _say("done again: the answer"),  # 7  finished: never rests; the line is owed
        _review_ok(),  # 8  side
    ]
    provider, hook = _Scripted(script), _Hook([PLAIN_REASON])
    loop = _loop(provider, tmp_path, hook)
    loop.session.tail_sparse_models.append(MODEL)
    loop.hook_manager.register_context(lambda payload: "recalled context")  # a tail without a plan
    await loop.arun_turn("one step")
    usage = loop._trace.of_kind("usage")
    assert [e.data["rails_rested"] for e in usage][-2:] == [True, False]  # call 6 rested
    assert provider.ended_with[7] == "plan_complete"


@pytest.mark.asyncio
async def test_a_refusal_with_no_finished_plan_silences_nothing(tmp_path: Path) -> None:
    provider, hook = _Scripted([_say("the answer"), _say("carrying on")]), _Hook([PLAIN_REASON])
    loop = _loop(provider, tmp_path, hook)
    assert (await loop.arun_turn("no plan")).stop_reason == "completed"
    assert hook.consulted == 2
    assert _requests(loop) == [([], []), ([], [])]
    assert _silenced_notes(loop) == 0


# ── the fence still stands ───────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_fence_still_ends_a_veto_stall_at_three(tmp_path: Path, streamed: bool) -> None:
    """The re-spiral, made deterministic: the plan is finished, the hook names a skill re-entry
    every time, and the model answers in words every time. Silencing the line must not unbound
    that: three deliveries are honoured and the fourth such veto ends the turn as
    ``veto_stall``, exactly as before, with the line silent on every re-entry request."""
    assert skill_reentry_in(SKILL_REASON) == ("cycle", "loop")  # the fence counts THESE vetoes
    spiral = [_say(f"Verdict: nothing left to do ({i})") for i in range(_VETO_STALL_THRESHOLD + 2)]
    script = [_plan("in_progress"), _judge_ok(), _plan("done"), _say("done"), _review_ok(), *spiral]
    provider = _Scripted(script)
    hook = _Hook([SKILL_REASON] * (_VETO_STALL_THRESHOLD + 3))
    loop = _loop(provider, tmp_path, hook)
    assert await _run(loop, "one step", streamed=streamed) == "veto_stall"
    assert hook.consulted == _VETO_STALL_THRESHOLD + 1  # consulted; its fourth veto refused
    reentries = _requests(loop)[3:]  # the requests after the first refused stop
    assert len(reentries) == _VETO_STALL_THRESHOLD
    assert all(request == ([], ["plan_complete"]) for request in reentries)
    assert "plan_complete" not in provider.ended_with[5:]  # and the wire agrees
