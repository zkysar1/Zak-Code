"""ADR-0419: no fresh-eyes plan review in a session whose turn-end hook has named its loop.

The review (ADR-0117) reads the model's last answer against the turn's request and, on a gap,
seeds a plan step that the plan gate then holds the turn for. A session whose hook has named its
loop (``Session.loop_skill``, ADR-0187) has no person's request waiting on that answer (ADR-0392):
its turns are the loop's passes and its turn end is the hook's. The review is skipped there; every
other session keeps it.

Hermetic: the scripted provider, the in-process hook and the fake composer of the ADR-0205
scenario. A review is one provider call, so the call count says whether it ran: a review that ran
on a script that has run out reads the last completion again, fails open on the unparseable
verdict and lets the turn end, one call later.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_refused_stop_silences_answer_now import (
    SKILL_REASON,
    _Hook,
    _judge_ok,
    _loop,
    _plan,
    _review_ok,
    _run,
    _say,
    _Scripted,
)


def _reviews(loop) -> int:
    events = loop._trace.of_kind("intervention")
    return sum(1 for e in events if e.data.get("kind") == "plan_review")


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_session_whose_hook_named_its_loop_finishes_its_plan_unreviewed(
    tmp_path: Path, streamed: bool
) -> None:
    provider = _Scripted([_plan("in_progress"), _judge_ok(), _plan("done"), _say("pass done")])
    loop = _loop(provider, tmp_path, _Hook([]))
    loop.session.loop_skill = "cycle loop"  # named by an earlier turn's hook (ADR-0187)

    assert await _run(loop, "next pass", streamed=streamed) == "completed"

    assert provider.calls == 4  # plan, judge, plan, answer: no review call
    assert _reviews(loop) == 0
    assert not [t for t in loop.session.task_network.leaves() if t.title.startswith("Reviewer")]


@pytest.mark.asyncio
@pytest.mark.parametrize("named", ["", "   "], ids=["no-loop", "blank-name"])
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_a_session_with_no_loop_named_is_still_reviewed(
    tmp_path: Path, streamed: bool, named: str
) -> None:
    provider = _Scripted(
        [_plan("in_progress"), _judge_ok(), _plan("done"), _say("the answer"), _review_ok()]
    )
    loop = _loop(provider, tmp_path, _Hook([]))
    loop.session.loop_skill = named

    assert await _run(loop, "one step", streamed=streamed) == "completed"

    assert provider.calls == 5  # plan, judge, plan, answer, review
    assert _reviews(loop) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_once_the_hook_names_the_loop_the_next_turns_finish_unreviewed(
    tmp_path: Path, streamed: bool
) -> None:
    # Turn 1 is ADR-0208's scenario: the first finish is reviewed (no loop is named yet), then
    # the stop taken on it is refused with a reason naming Skill('cycle') args='loop'. Turn 2 is
    # the loop's next pass in the same session: its finished plan goes to the hook unreviewed.
    provider = _Scripted(
        [
            _plan("in_progress"),  # 0
            _judge_ok(),  # 1  side
            _plan("done"),  # 2
            _say("done: the answer"),  # 3  reviewed, then refused: the harness delivers /cycle
            _review_ok(),  # 4  side: the review
            _plan("in_progress"),  # 5
            _plan("done"),  # 6
            _say("done again"),  # 7  the turn's one review is spent; the hook lets it end
            _plan("in_progress"),  # 8  turn 2
            _judge_ok(),  # 9  side
            _plan("done"),  # 10
            _say("pass done"),  # 11 no review: the loop is named
        ]
    )
    loop = _loop(provider, tmp_path, _Hook([SKILL_REASON]))

    assert await _run(loop, "one step", streamed=streamed) == "completed"
    assert loop.session.loop_skill == "cycle loop"
    assert provider.calls == 8 and _reviews(loop) == 1

    assert await _run(loop, "next pass", streamed=streamed) == "completed"
    assert provider.calls == 12 and _reviews(loop) == 0  # the trace is per turn
