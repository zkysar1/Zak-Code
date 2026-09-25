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
from zakcode.providers.base import Capabilities, LLMResult, Provider, ProviderError, ToolCall
from zakcode.session.store import Session
from zakcode.tools import default_registry
from zakcode.usage import Usage
from zakcode.wakeup import LOOP_SENTINEL, WakeupSlot, provider_hold_delay, turn_fingerprint

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


# ── the exception: a turn the PROVIDER failed is not a verdict on the loop (ADR-0250) ────
#
# Measured 2026-09-25 on three worker Bodies during a 12-hour pod outage: every sentinel turn
# ended identically -- the provider refused every call, the model never acted -- and on the
# second such turn the guard above cancelled the net, exactly as designed, and the Bodies sat
# at their prompts for four and a half hours after the pod came back. The tests below are the
# twins of the cancelling test at the top of this file; each differs in one thing only -- WHY
# the two turns ended identically -- and asserts the net survives and backs off.


class _Refusing(Provider):
    """A provider that is not there: every call is refused before any model work happens."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(self, messages: Any, *, system: Any = None, tools: Any = None, **kw: Any):  # type: ignore[override]
        self.calls += 1
        raise ProviderError("connection refused")

    def model_id(self) -> str:
        return MODEL

    def count_tokens(self, messages: Any, *, system: Any = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_wake_up_turn_the_provider_failed_keeps_its_net_and_backs_off(
    tmp_path: Path, streamed: bool
) -> None:
    """The twin of ``test_a_second_identical_wake_up_turn_cancels_the_net_it_armed``: two
    identical endings again, but the model never acted, so nothing was proved about the loop.
    Nothing else armed a net here (no hook, no re-arm call) -- the guard must arm it itself."""
    loop = _loop(_Refusing(), tmp_path)

    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=streamed) == "provider_error"
    held = loop.wakeup_slot.pending()
    assert held is not None and held.prompt == LOOP_SENTINEL and held.delay_seconds == 600
    assert loop.session.sentinel_provider_repeats == 1

    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=streamed) == "provider_error"
    held = loop.wakeup_slot.pending()
    assert held is not None and held.prompt == LOOP_SENTINEL and held.delay_seconds == 1200
    assert loop.session.sentinel_provider_repeats == 2
    # The record is KEPT, not cleared: the streak is what sizes the next hold.
    assert loop.session.last_sentinel_outcome == turn_fingerprint("provider_error", "")


@pytest.mark.asyncio
async def test_a_wake_up_turn_that_ran_again_resets_the_provider_backoff(tmp_path: Path) -> None:
    """When the provider is back, the first sentinel turn that actually runs ends the streak;
    the net it holds is the model's own re-arm, at the model's delay."""
    loop = _loop(_Refusing(), tmp_path)
    for _ in range(2):
        _fire_sentinel(loop)
        assert await _run(loop, "[harness] wake-up", streamed=False) == "provider_error"
    assert loop.session.sentinel_provider_repeats == 2

    loop.provider = _Scripted([_rearm("w1"), _say(OTHER)])
    _fire_sentinel(loop)
    assert await _run(loop, "[harness] wake-up", streamed=False) == "completed"
    assert loop.session.sentinel_provider_repeats == 0
    assert _held(loop) == LOOP_SENTINEL
    assert loop.session.last_sentinel_outcome == turn_fingerprint("completed", OTHER)


def test_a_veto_stall_over_provider_errors_holds_the_net_and_one_over_prose_cancels_it(
    tmp_path: Path,
) -> None:
    """The fence (ADR-0187) ends a turn ``veto_stall`` whichever ending the hook kept vetoing.
    Vetoed provider errors are the provider's failure and hold the net; vetoed prose is the
    model's -- ADR-0187's own case -- and the second identical one still cancels. Same
    stop reason, same empty text, same fingerprint; the cause is the only difference."""
    loop = _loop(_Scripted([_say(VERDICT)]), tmp_path)
    for expected_delay in (600, 1200):
        loop.wakeup_slot.arm(LOOP_SENTINEL, 600)  # what the stall fence armed
        loop.session.sentinel_turn_open = True
        loop._veto_stall_cause = "provider_error"
        loop._close_wake_repeat_guard("veto_stall", [])
        held = loop.wakeup_slot.pending()
        assert held is not None and held.prompt == LOOP_SENTINEL
        assert held.delay_seconds == expected_delay

    control = _loop(_Scripted([_say(VERDICT)]), tmp_path)
    for _ in range(2):
        control.wakeup_slot.arm(LOOP_SENTINEL, 600)
        control.session.sentinel_turn_open = True
        control._veto_stall_cause = "completed"
        control._close_wake_repeat_guard("veto_stall", [])
    assert control.wakeup_slot.pending() is None
    assert control.session.sentinel_provider_repeats == 0


def test_a_provider_failed_sentinel_turn_holds_the_net_at_a_growing_delay() -> None:
    """600, 1200, 2400, then the clamp -- a loop that cannot reach its provider is retried at
    most hourly, and the record is kept so the streak keeps counting."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    fingerprint = turn_fingerprint("provider_error", "")
    for expected in (600, 1200, 2400, 3600, 3600):
        slot.arm(LOOP_SENTINEL, 600)  # what a turn-end hook or the fence just armed
        session.sentinel_turn_open = True
        slot.note_turn_end(fingerprint, provider_failed=True)
        held = slot.pending()
        assert held is not None and held.prompt == LOOP_SENTINEL
        assert held.delay_seconds == expected
    assert session.last_sentinel_outcome == fingerprint
    assert session.sentinel_provider_repeats == 5
    assert session.sentinel_turn_open is False


def test_a_provider_failed_sentinel_turn_with_nothing_held_arms_a_net() -> None:
    """No hook, no fence, no model re-arm: the guard is the last party that can leave a net."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    session.sentinel_turn_open = True
    assert slot.pending() is None
    assert slot.note_turn_end(turn_fingerprint("provider_error", ""), provider_failed=True) is False
    held = slot.pending()
    assert held is not None and held.prompt == LOOP_SENTINEL and held.delay_seconds == 600


def test_a_hooks_own_wake_up_outranks_the_provider_hold() -> None:
    """The precedence ADR-0216 keeps holds here too: a hook that said when to come back is
    not overruled by the backoff, but the streak still counts for the next hold."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    slot.arm("re-poll the reducer's claim", 3600)
    session.sentinel_turn_open = True
    slot.note_turn_end(turn_fingerprint("provider_error", ""), provider_failed=True)
    held = slot.pending()
    assert held is not None and held.prompt == "re-poll the reducer's claim"
    assert held.delay_seconds == 3600
    assert session.sentinel_provider_repeats == 1


def test_a_sentinel_turn_that_ran_resets_the_provider_streak_at_the_slot() -> None:
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    for _ in range(2):
        session.sentinel_turn_open = True
        slot.note_turn_end(turn_fingerprint("provider_error", ""), provider_failed=True)
    assert session.sentinel_provider_repeats == 2
    session.sentinel_turn_open = True
    assert slot.note_turn_end(turn_fingerprint("completed", OTHER)) is False
    assert session.sentinel_provider_repeats == 0
    assert session.last_sentinel_outcome == turn_fingerprint("completed", OTHER)


def test_the_door_puts_an_unfired_sentinel_back_and_lifts_the_turn_mark() -> None:
    """Taking the sentinel marks the turn it opens; when the door holds it instead, no turn
    opens, and the mark must go too -- or the next turn a person types is judged as one."""
    session = Session(cwd=".", model=MODEL)
    slot = _slot(session)
    slot.arm(LOOP_SENTINEL, 60)
    assert slot.take_due_prompt(now=time.time() + 120) == LOOP_SENTINEL
    assert session.sentinel_turn_open is True
    held = slot.hold_unfired_sentinel()
    assert session.sentinel_turn_open is False
    assert held.prompt == LOOP_SENTINEL and held.delay_seconds == 600
    assert slot.pending() is held
    assert session.sentinel_provider_repeats == 1


def test_the_provider_hold_schedule_doubles_to_the_clamp() -> None:
    assert [provider_hold_delay(n) for n in range(1, 7)] == [600, 1200, 2400, 3600, 3600, 3600]


def _unused(message: Message) -> None:  # pragma: no cover - import anchor for the type only
    assert message is not None
