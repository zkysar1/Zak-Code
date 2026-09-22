"""ADR-0216: a wake-up turn that ends exactly as the last wake-up turn did cancels the sentinel
it re-armed, instead of letting the pair run again.

The line a fired sentinel hands over says "re-arm a wake-up FIRST, then re-enter the loop", and
it says that deliberately -- a net that fires while it is being replaced is a net with a hole.
But the model obeys it BEFORE it can discover whether there is anything to re-enter, so a loop
that cannot run arms its own next firing and the pair repeats. Measured 2026-09-22 on a served
Mind: six turns, ~3-5M tokens each, every one ending with the identical verdict that the agent
was IDLE and the loop would not start.

Neither existing guard could see it. The doom guard and the stuck ladder are both per-TURN state
(``stuck.py``: "Pure per-turn state ... the loop creates one tracker per turn"), built for a model
flailing inside one turn; these repeats are one turn apart, so no threshold could have been set
low enough. What is pinned here is therefore the new scope, not a new sensitivity.

THE POSITIVE CONTROL IS THE POINT OF THIS FILE, not a courtesy (guard-4166). This change's whole
effect is that something STOPS happening, and a suite that only asserts absence passes just as
well when the wake-up never worked at all. So every cancelling test has a twin that differs in
one thing only -- what the second turn SAID -- and asserts the net survives.

Hermetic: a scripted provider, no network, no clock dependence (the due time is passed in).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry
from zakcode.usage import Usage
from zakcode.wakeup import LOOP_SENTINEL, WakeupSlot, turn_fingerprint

MODEL = "fake/scripted"
#: The verdict sera repeated, shortened. Any fixed string does; that it is FIXED is the point.
VERDICT = "The loop will not start because the agent is IDLE. This is the designed behavior."
OTHER = "The loop started: the first goal is selected and the iteration is under way."


class _Scripted(Provider):
    """Hands back scripted results in order; the last one repeats."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any):  # type: ignore[override]
        index = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[index]

    def model_id(self) -> str:
        return MODEL

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


def _rearm(call_id: str) -> LLMResult:
    """The model's first act in a sentinel turn: re-arm, exactly as the fired line instructs."""
    call = ToolCall(
        id=call_id,
        name="schedule_wakeup",
        arguments={"prompt": LOOP_SENTINEL, "delaySeconds": 600},
    )
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


def _say(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        default_registry(),
        Session(cwd=str(tmp_path), model=MODEL),
        workspace_root=tmp_path,
        max_iterations=10,
    )


def _fire_sentinel(loop: AgentLoop) -> None:
    """Arm the sentinel and consume it as the REPL door would, through the public API only."""
    loop.wakeup_slot.arm(LOOP_SENTINEL, 60)
    prompt = loop.wakeup_slot.take_due_prompt(now=time.time() + 120)
    assert prompt == LOOP_SENTINEL
    assert loop.session.sentinel_turn_open is True


async def _run(loop: AgentLoop, text: str, *, streamed: bool) -> str:
    if not streamed:
        return (await loop.arun_turn(text)).stop_reason
    async for _ in loop.astream_turn(text):
        pass
    return str(loop.session.last_stop_reason)


def _held(loop: AgentLoop) -> str | None:
    held = loop.wakeup_slot.pending()
    return None if held is None else held.prompt


# ── the rule, through the real loop, at both turn-end twins ──────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_second_identical_wake_up_turn_cancels_the_net_it_armed(
    tmp_path: Path, streamed: bool
) -> None:
    provider = _Scripted([_rearm("w1"), _say(VERDICT), _rearm("w2"), _say(VERDICT)])
    loop = _loop(provider, tmp_path)

    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=streamed) == "completed"
    # The first sighting only records: the model re-armed and the net is held, as it should be.
    assert _held(loop) == LOOP_SENTINEL
    assert loop.session.last_sentinel_outcome != ""

    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=streamed) == "completed"
    # The second says the same thing, so the net it armed is gone and the pair cannot run again.
    assert _held(loop) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_second_wake_up_turn_that_ends_differently_keeps_the_net_it_armed(
    tmp_path: Path, streamed: bool
) -> None:
    """THE POSITIVE CONTROL. One character of difference from the test above: what turn 2 says."""
    provider = _Scripted([_rearm("w1"), _say(VERDICT), _rearm("w2"), _say(OTHER)])
    loop = _loop(provider, tmp_path)

    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=streamed) == "completed"
    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=streamed) == "completed"

    # A wake-up that re-entered something keeps its net: this is the deadman doing its job.
    # ONE assertion on purpose. An earlier draft also checked that the outcome had been
    # RECORDED, and the mutation proof showed why that is wrong for a control: removing either
    # turn-end call site broke the recording, so the control went red alongside the tests it is
    # supposed to stay green beside, and a suite where the control flips with everything else
    # cannot tell a working guard from a dead wake-up (guard-4166). The recording is a separate
    # claim and gets its own test below.
    assert _held(loop) == LOOP_SENTINEL


@pytest.mark.asyncio
async def test_a_wake_up_turn_that_ended_differently_is_recorded_for_the_next_one(
    tmp_path: Path,
) -> None:
    """The other half of the control: surviving is not enough, the turn must also be ON RECORD,
    or the sighting after it has nothing to be compared against."""
    provider = _Scripted([_rearm("w1"), _say(OTHER)])
    loop = _loop(provider, tmp_path)
    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=False) == "completed"
    assert loop.session.last_sentinel_outcome == turn_fingerprint("completed", OTHER)


@pytest.mark.asyncio
async def test_the_first_wake_up_turn_has_nothing_to_repeat_and_keeps_its_net(
    tmp_path: Path,
) -> None:
    provider = _Scripted([_rearm("w1"), _say(VERDICT)])
    loop = _loop(provider, tmp_path)
    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=False) == "completed"
    assert _held(loop) == LOOP_SENTINEL


@pytest.mark.asyncio
async def test_a_turn_a_person_opened_is_never_judged_however_exactly_it_repeats(
    tmp_path: Path,
) -> None:
    """Scope. A person asking the same question twice is a person, not a barren net."""
    provider = _Scripted([_rearm("w1"), _say(VERDICT), _rearm("w2"), _say(VERDICT)])
    loop = _loop(provider, tmp_path)

    assert await _run(loop, "what is the state?", streamed=False) == "completed"
    assert await _run(loop, "what is the state?", streamed=False) == "completed"

    assert _held(loop) == LOOP_SENTINEL
    assert loop.session.last_sentinel_outcome == ""


# ── the decision itself, unit-level ──────────────────────────────────────────────


def _slot(session: Session) -> WakeupSlot:
    return WakeupSlot(session)


def test_a_hooks_own_wake_up_is_not_this_guards_to_cancel() -> None:
    """A turn-end hook that armed its OWN prompt said something specific about when to come
    back. Only the sentinel is this guard's to drop -- the same precedence ``_arm_stall_net``
    already keeps, where the framework's net outranks the harness's."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    print_it = turn_fingerprint("completed", VERDICT)

    session.sentinel_turn_open = True
    assert slot.note_turn_end(print_it) is False  # first sighting: recorded only

    slot.arm("re-poll the reducer's claim", 600)  # a hook's own prompt, armed during turn 2
    session.sentinel_turn_open = True
    assert slot.note_turn_end(print_it) is True  # it DID repeat, and is reported as such
    held = slot.pending()
    assert held is not None and held.prompt == "re-poll the reducer's claim"


def test_the_same_words_under_a_different_stop_reason_are_not_a_repeat() -> None:
    """A turn that said the same thing but ended for a different reason ended differently."""
    assert turn_fingerprint("completed", VERDICT) != turn_fingerprint("stuck", VERDICT)
    # And a re-wrapped answer is still the same answer.
    assert turn_fingerprint("completed", VERDICT) == turn_fingerprint(
        "completed", VERDICT.replace(" ", "\n  ")
    )


def test_a_cancel_clears_the_record_so_the_next_sighting_starts_clean() -> None:
    """After a cancel the next sentinel is a fresh net, judged against nothing -- not against a
    turn that is now two nets old."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    fingerprint = turn_fingerprint("completed", VERDICT)

    session.sentinel_turn_open = True
    slot.note_turn_end(fingerprint)
    slot.arm(LOOP_SENTINEL, 600)
    session.sentinel_turn_open = True
    assert slot.note_turn_end(fingerprint) is True
    assert slot.pending() is None
    assert session.last_sentinel_outcome == ""

    slot.arm(LOOP_SENTINEL, 600)
    session.sentinel_turn_open = True
    assert slot.note_turn_end(fingerprint) is False  # a first sighting again
    assert slot.pending() is not None


def test_the_flag_is_cleared_even_when_the_turn_did_not_repeat() -> None:
    """A sentinel turn is judged once. Leaving the flag up would judge the NEXT turn -- a typed
    one -- against a wake-up's record, which is the scope error this guard must not make."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    session.sentinel_turn_open = True
    slot.note_turn_end(turn_fingerprint("completed", VERDICT))
    assert session.sentinel_turn_open is False
    assert slot.note_turn_end(turn_fingerprint("completed", VERDICT)) is False


def _unused(message: Message) -> None:  # pragma: no cover - import anchor for the type only
    assert message is not None
