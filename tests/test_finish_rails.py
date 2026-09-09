"""ADR-0117: a finish is not a finish while the answer defers the ask or the plan hides a gap.

Two serene turns (gemini-2.5-flash, 2026-09-08): one ended a 50-iteration plan on "which will
enable further debugging in a future session"; the next ended one iteration in on "I will now
re-attempt to debug …". The deferral rail and the widened intent gate are deterministic and
always on; the fresh-eyes plan review is the judged complement — the last gate before a finish
— and seeds a flagged gap as a plan step the plan gate then holds the turn for.
"""

from __future__ import annotations

import pytest

from zakcode.agent.loop import _DEFERRAL_NUDGE, AgentLoop, _defers_work
from zakcode.config import Settings
from zakcode.events import AgentStatus
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.usage import Usage

DEFERRAL = (
    "Auth works and google-drive-search finds files. The google-drive-list skill still reports "
    "no files; the environment now allows its execution, which will enable further debugging "
    "in a future session."
)
ANSWER = "The list call sends parent_id='' which the API treats as no parent: fixed, 12 files."
REVIEW_PHRASE = "An independent reviewer read your answer against the request and your plan record"


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


def _plan(tasks: list[dict]) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id="p", name="update_plan", arguments={"tasks": tasks})],
        usage=Usage(total_tokens=1),
    )


def _judge_ok() -> LLMResult:
    import json

    return LLMResult(
        text=json.dumps(
            {"scores": {"coverage": 0.9, "granularity": 0.9, "ordering": 0.9, "soundness": 0.9}}
        ),
        usage=Usage(total_tokens=2),
    )


def _done(text: str) -> LLMResult:
    return LLMResult(text=text, tool_calls=[], usage=Usage(total_tokens=1))


def _review(approved: bool, issues: str = "") -> LLMResult:
    return LLMResult(
        text=f'{{"approved": {"true" if approved else "false"}, "issues": "{issues}"}}',
        usage=Usage(total_tokens=2),
    )


def _loop(provider: Provider, **settings) -> tuple[AgentLoop, Session]:
    session = Session(cwd="/tmp", model="test/model")
    loop = AgentLoop(
        provider, default_registry(), session, settings=Settings(**settings), max_iterations=20
    )
    return loop, session


def _nudges(session: Session, phrase: str) -> list[str]:
    return [m.text for m in session.messages if m.role == "user" and phrase in m.text]


_DONE_PLAN = [{"title": "check auth", "status": "done"}, {"title": "list files", "status": "done"}]


# ── the deferral rail ─────────────────────────────────────────────────────────────────


def test_deferral_matcher_reads_the_tail_for_postponed_work() -> None:
    for text in (
        DEFERRAL,
        "The red flag from google-drive-list is still present, but now debuggable.",
        "This remains unresolved.",
        "Further debugging is needed.",
        "I left the second part for a follow-up.",
        "The rest can be addressed later.",
    ):
        assert _defers_work(text), text
    for text in (
        ANSWER,
        "All done: the route works and the tests pass.",
        "I will present the results in the next section.",
        "I left the file for another reviewer to read.",
    ):
        assert not _defers_work(text), text


@pytest.mark.asyncio
async def test_a_conclusion_that_defers_the_ask_is_asked_once_for_the_work() -> None:
    provider = _Scripted(
        [_plan(_DONE_PLAN), _judge_ok(), _done(DEFERRAL), _done(ANSWER), _review(True)]
    )
    loop, session = _loop(provider)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed" and result.degraded is False
    nudges = _nudges(session, "leaves part of the request for later")
    assert len(nudges) == 1 and nudges[0].startswith("[harness] Hint:")
    assert "update_plan" in _DEFERRAL_NUDGE and "blocked" in _DEFERRAL_NUDGE
    assert provider.calls == 5  # plan, judge, deferral (nudged), answer, plan review


@pytest.mark.asyncio
async def test_deferral_rail_fires_once_then_the_turn_ends() -> None:
    # The second completion defers again in DIFFERENT words (a verbatim repeat is the
    # broken-record guard's business, ADR-0026): the rail has had its say and the turn ends.
    again = "As I said, the remaining google-drive-list problem is left for a follow-up."
    provider = _Scripted(
        [_plan(_DONE_PLAN), _judge_ok(), _done(DEFERRAL), _done(again), _review(True)]
    )
    loop, session = _loop(provider)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed"
    assert len(_nudges(session, "leaves part of the request for later")) == 1
    assert provider.calls == 5


@pytest.mark.asyncio
async def test_deferral_rail_needs_no_plan() -> None:
    provider = _Scripted([_done(DEFERRAL), _done(ANSWER)])
    loop, session = _loop(provider)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed"
    assert len(_nudges(session, "leaves part of the request for later")) == 1
    assert provider.calls == 2


# ── the fresh-eyes plan review ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_flagged_gap_becomes_a_plan_step_and_is_done_before_the_finish() -> None:
    gap = "the list call still returns no files"
    provider = _Scripted(
        [
            _plan(_DONE_PLAN),
            _judge_ok(),
            _done("Auth works; the list is empty."),
            _review(False, gap),
            _plan(
                _DONE_PLAN
                + [{"title": f"Reviewer flagged: {gap}", "status": "done", "outcome": "fixed"}]
            ),
            _done(ANSWER),
        ]
    )
    loop, session = _loop(provider)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed" and result.degraded is False
    assert result.open_steps == 0
    nudges = _nudges(session, REVIEW_PHRASE)
    assert len(nudges) == 1 and gap in nudges[0]
    kinds = [(e.kind, e.detail) for e in session.task_network.log]
    assert any(k == "seeded" and "fresh-eyes review" in d for k, d in kinds)
    seeded = [t for t in session.task_network.leaves() if t.title.startswith("Reviewer flagged")]
    assert seeded and seeded[0].origin == "harness" and seeded[0].status == "done"
    assert provider.calls == 6  # plan, judge, answer, review (gap), plan, answer — one review


@pytest.mark.asyncio
async def test_an_ignored_gap_is_held_by_the_plan_gate_and_the_finish_is_degraded() -> None:
    provider = _Scripted(
        [
            _plan(_DONE_PLAN),
            _judge_ok(),
            _done("Auth works; the list is empty."),
            _review(False, "the list call still returns no files"),
            _done("I consider this done."),
        ]
    )
    loop, session = _loop(provider)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed"
    assert result.degraded is True and result.open_steps == 1  # the seeded step stayed open
    assert len(_nudges(session, "Your plan still has 1 open step")) == 2  # the nudge cap
    assert provider.calls == 7  # plan, judge, answer, review, done ×3 (two nudges, then cap)


@pytest.mark.asyncio
async def test_plan_review_is_off_when_disabled_and_skips_an_anchor_only_board() -> None:
    provider = _Scripted([_plan(_DONE_PLAN), _judge_ok(), _done(ANSWER)])
    loop, _ = _loop(provider, plan_review=False)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed"
    assert provider.calls == 3  # no judge call for the review


@pytest.mark.asyncio
async def test_plan_review_fails_open_on_an_unparseable_verdict() -> None:
    provider = _Scripted([_plan(_DONE_PLAN), _judge_ok(), _done(ANSWER), _done("looks fine to me")])
    loop, session = _loop(provider)
    result = await loop.arun_turn("are there .tar.gz files in the drive?")
    assert result.stop_reason == "completed" and result.degraded is False
    assert _nudges(session, REVIEW_PHRASE) == []


@pytest.mark.asyncio
async def test_plan_review_holds_on_the_streaming_path() -> None:
    gap = "the list call still returns no files"
    provider = _Scripted(
        [
            _plan(_DONE_PLAN),
            _judge_ok(),
            _done("Auth works; the list is empty."),
            _review(False, gap),
            _plan(
                _DONE_PLAN
                + [{"title": f"Reviewer flagged: {gap}", "status": "done", "outcome": "fixed"}]
            ),
            _done(ANSWER),
        ]
    )
    loop, session = _loop(provider)
    events = [ev async for ev in loop.astream_turn("are there .tar.gz files in the drive?")]
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any("fresh eyes on the finished plan" in s for s in statuses)
    done = events[-1]
    assert done.stop_reason == "completed" and done.open_steps == 0
    assert len(_nudges(session, REVIEW_PHRASE)) == 1
