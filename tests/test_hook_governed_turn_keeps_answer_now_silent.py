"""ADR-0208: once a turn-end hook's honoured refusal has NAMED a skill re-entry, no plan that
finishes later in that turn is sent its "answer now" line.

ADR-0205 keeps the line silent for the one plan a stop was refused on. A perpetual loop finishes
a NEW plan on every pass, so one pass later the line rode again and asked for the closing answer
the hook refuses. Measured in a served loop on gpt-5.6-luna with the line drawn by lot inside the
run (sample 7, ``bench/results/served-luna-preregistration.log``): sent, 12 of 13 finished plans
ended in a stop the model made in words; kept silent, 5 of 13.

This is a NARROWING change (something stops being sent), so half of what is pinned here are
CONTROLS that must hold with or without it: the line still rides before any refusal, the open
checklist still rides in a governed turn, a refusal in plain words governs nothing, and a new
turn starts ungoverned. ``bench``-free and hermetic: scripted providers, an in-process hook, a
fake composer, and the helpers of the ADR-0205 tests, whose scenario this one continues. Every
claim about what was sent is read off the WIRE (the last message of the request the scripted
provider was handed), and the per-request trace label is checked against it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_refused_stop_silences_answer_now import (
    PLAIN_REASON,
    SKILL_REASON,
    _Hook,
    _judge_ok,
    _loop,
    _plan,
    _requests,
    _review_ok,
    _run,
    _say,
    _Scripted,
)
from zakcode.agent.loop import _VETO_STALL_THRESHOLD, AgentLoop, skill_reentry_in
from zakcode.providers.base import LLMResult


def _two_plans_one_refusal() -> list[LLMResult]:
    """A plan finishes, the stop taken on it is refused, and the model finishes ANOTHER plan in
    the same turn. Main requests are calls 0, 2, 3, 5, 6 and 7; 1 and 4 are side calls (the plan
    judge, and the one review a turn gets), so the script is walked exactly and ends at 7."""
    return [
        _plan("in_progress"),  # 0
        _judge_ok(),  # 1  side
        _plan("done"),  # 2  the open checklist rides
        _say("done: the answer"),  # 3  the line rides: no refusal yet. Then the refused stop
        _review_ok(),  # 4  side
        _plan("in_progress"),  # 5  silent (ADR-0205: exactly the refused plan). Reopens it
        _plan("done"),  # 6  the open checklist rides again. Finishes a NEW plan
        _say("done again: the answer"),  # 7  ADR-0208 decides what THIS request ends with
    ]


def _governed_notes(loop: AgentLoop) -> list[str]:
    """The skill named by each ``turn_end_governed`` note of the trace, in order."""
    notes = loop._trace.of_kind("intervention")
    return [str(e.data.get("skill")) for e in notes if e.data.get("kind") == "turn_end_governed"]


# ── the rule ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_plan_finished_after_a_skill_naming_refusal_stays_silent(
    tmp_path: Path, streamed: bool
) -> None:
    assert skill_reentry_in(SKILL_REASON) == ("cycle", "loop")  # the refusal NAMES a re-entry
    provider, hook = _Scripted(_two_plans_one_refusal()), _Hook([SKILL_REASON])
    loop = _loop(provider, tmp_path, hook)
    assert await _run(loop, "one step", streamed=streamed) == "completed"
    assert hook.consulted == 2  # refused once, then allowed
    # On the wire. CONTROLS first: before any refusal the line rides (3), and in the governed
    # turn the OPEN checklist still rides (6). The rule: the second finished plan is silent (7).
    assert [provider.ended_with[i] for i in (2, 3, 5, 6, 7)] == [
        "plan",
        "plan_complete",
        None,
        "plan",
        None,
    ]
    # And the label says the same of each request, in the same words as ADR-0205's.
    assert _requests(loop) == [
        ([], []),
        (["plan"], []),
        (["plan_complete"], []),
        ([], ["plan_complete"]),  # silent because this plan's stop was refused (ADR-0205)
        (["plan"], []),
        ([], ["plan_complete"]),  # silent because the hook governs the turn's end (ADR-0208)
    ]
    assert _governed_notes(loop) == ["cycle"]  # said once, by the skill's name


@pytest.mark.asyncio
async def test_the_governed_note_is_written_once_however_often_the_hook_refuses(
    tmp_path: Path,
) -> None:
    script = [*_two_plans_one_refusal(), _say("carrying on"), _say("and on")]
    provider, hook = _Scripted(script), _Hook([SKILL_REASON, SKILL_REASON])
    loop = _loop(provider, tmp_path, hook)
    assert (await loop.arun_turn("one step")).stop_reason == "completed"
    assert hook.consulted == 3  # refused twice, then allowed
    assert _governed_notes(loop) == ["cycle"]
    assert "plan_complete" not in provider.ended_with[5:]  # nothing after the first refusal


# ── the scope: what does NOT govern ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_refusal_in_plain_words_governs_nothing(tmp_path: Path) -> None:
    """The same turn with a hook that names no skill. ADR-0205 still silences the refused plan,
    and the plan that finishes afterwards is owed its line: a plain refusal answers one stop and
    says nothing about the next. A control for ADR-0208 and the scope it must not outgrow."""
    assert skill_reentry_in(PLAIN_REASON) is None
    provider, hook = _Scripted(_two_plans_one_refusal()), _Hook([PLAIN_REASON])
    loop = _loop(provider, tmp_path, hook)
    assert (await loop.arun_turn("one step")).stop_reason == "completed"
    assert [provider.ended_with[i] for i in (3, 5, 6, 7)] == [
        "plan_complete",
        None,
        "plan",
        "plan_complete",
    ]
    assert _governed_notes(loop) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_new_turn_starts_ungoverned(tmp_path: Path, streamed: bool) -> None:
    """The hook governed the FIRST turn's end. The second turn owes its first finished plan the
    line again: whether a hook governs is a fact about one turn, learned inside it. Both turn
    paths, because each starts a turn in its own code."""
    first = _two_plans_one_refusal()
    second = [
        _plan("in_progress"),  # 8
        _judge_ok(),  # 9  side
        _plan("done"),  # 10
        _say("done: the second turn's answer"),  # 11  the line is owed, and rides
        _review_ok(),  # 12  side
    ]
    provider, hook = _Scripted([*first, *second]), _Hook([SKILL_REASON])
    loop = _loop(provider, tmp_path, hook)
    assert await _run(loop, "one step", streamed=streamed) == "completed"
    assert provider.ended_with[7] is None  # governed: silent, as above
    assert _governed_notes(loop) == ["cycle"]
    assert await _run(loop, "another step", streamed=streamed) == "completed"
    assert provider.calls == len(first) + len(second)  # the script was walked as written
    assert provider.ended_with[11] == "plan_complete"
    assert _governed_notes(loop) == []  # the trace is this turn's, and it was never governed


# ── the fence still stands ───────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_fence_still_ends_a_respiral_in_a_governed_turn(
    tmp_path: Path, streamed: bool
) -> None:
    """The re-spiral with a NEW plan finished after the first refusal, which is the shape a
    served loop makes: the hook names the re-entry every time and the model answers in words
    every time. Keeping the later plan's line silent must not unbound that. Three deliveries are
    honoured and the fourth such veto ends the turn as ``veto_stall`` (ADR-0187), and no request
    after the first refusal carries the line."""
    spiral = [_say(f"Verdict: nothing left to do ({i})") for i in range(_VETO_STALL_THRESHOLD + 2)]
    script = [*_two_plans_one_refusal(), *spiral]
    provider = _Scripted(script)
    hook = _Hook([SKILL_REASON] * (_VETO_STALL_THRESHOLD + 3))
    loop = _loop(provider, tmp_path, hook)
    assert await _run(loop, "one step", streamed=streamed) == "veto_stall"
    assert hook.consulted == _VETO_STALL_THRESHOLD + 1  # consulted; its fourth veto refused
    assert provider.ended_with[3] == "plan_complete"  # CONTROL: the first plan had its line
    assert "plan_complete" not in provider.ended_with[5:]
    assert _governed_notes(loop) == ["cycle"]
