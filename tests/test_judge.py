"""Tests for the LLM-as-judge primitives (quality engine, increment 1).

Hermetic: a scripted in-memory provider returns canned verdict JSON, so the binary/pairwise/vote/
tournament logic is exercised deterministically with no network. The fan-out helpers (vote_binary,
best_of) gather concurrent judge calls; the scripted provider has no await suspension point, so the
calls resolve in order and the aggregation is deterministic.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import best_of, binary_judge, pairwise_judge, vote_binary, vote_pairwise
from zakcode.usage import Usage


class _Provider(Provider):
    """Returns a FIXED verdict text, or cycles a LIST of texts (one per call, in call order)."""

    def __init__(self, text: str | list[str]) -> None:
        self._list = text if isinstance(text, list) else None
        self._fixed = text if isinstance(text, str) else None
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        i = self.calls
        self.calls += 1
        text = self._fixed if self._fixed is not None else self._list[i % len(self._list)]
        return LLMResult(text=text, usage=Usage(total_tokens=2, cost_usd=0.001))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/test"


def _v(approved: bool, issues: str = "") -> str:
    return json.dumps({"approved": approved, "issues": issues})


def _p(winner: str, reason: str = "") -> str:
    return json.dumps({"winner": winner, "reason": reason})


class _RankedJudge(Provider):
    """A CONTENT-AWARE pairwise judge for best_of's both-orderings debias: prefers whichever
    candidate ranks EARLIER in ``order`` (earlier = better). It parses candidate_a / candidate_b
    out of the prompt, so it returns a CONSISTENT verdict when best_of swaps a/b -- a fixed-verdict
    mock would look position-biased and every pair would tie."""

    def __init__(self, order: list[str]) -> None:
        self._rank = {name: i for i, name in enumerate(order)}
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        payload = messages[-1].blocks[0].text
        a = payload.split("<candidate_a>\n", 1)[1].split("\n</candidate_a>", 1)[0]
        b = payload.split("<candidate_b>\n", 1)[1].split("\n</candidate_b>", 1)[0]
        winner = "a" if self._rank[a] < self._rank[b] else "b"
        return LLMResult(text=_p(winner), usage=Usage(total_tokens=2, cost_usd=0.001))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/ranked"


# ── binary_judge ─────────────────────────────────────────────────────────────


async def test_binary_judge_reject_carries_issues() -> None:
    v, u = await binary_judge(
        _Provider(_v(False, "missing Y")), criteria="do X and Y", artifact="X"
    )
    assert v.approved is False and "missing Y" in v.issues
    assert u.total_tokens == 2


async def test_binary_judge_approve() -> None:
    v, _ = await binary_judge(_Provider(_v(True)), criteria="c", artifact="a")
    assert v.approved is True and v.issues == ""


async def test_binary_judge_fails_open_on_non_json() -> None:
    # A judge that can't return a valid verdict ABSTAINS by approving — it never traps the caller.
    v, u = await binary_judge(_Provider("sure, looks fine to me"), criteria="c", artifact="a")
    assert v.approved is True
    assert u.total_tokens == 0  # the fail path reports no spend


# ── pairwise_judge ───────────────────────────────────────────────────────────


async def test_pairwise_judge_picks_winner() -> None:
    v, _ = await pairwise_judge(_Provider(_p("a", "more complete")), criteria="c", a="AA", b="BB")
    assert v.winner == "a" and "complete" in v.reason


async def test_pairwise_judge_invalid_winner_is_tie() -> None:
    v, _ = await pairwise_judge(_Provider('{"winner": "neither"}'), criteria="c", a="x", b="y")
    assert v.winner == "tie"


# ── vote_binary (N-judge majority) ───────────────────────────────────────────


async def test_vote_binary_strict_majority_rejects_and_unions_issues() -> None:
    prov = _Provider([_v(True), _v(False, "no tests"), _v(False, "no docs")])
    v, u = await vote_binary(prov, criteria="c", artifact="a", n=3)
    assert v.approved is False  # 1 approve, 2 reject → majority rejects
    assert "no tests" in v.issues and "no docs" in v.issues  # dissenters' issues unioned
    assert prov.calls == 3
    assert u.total_tokens == 6  # 3 judges x 2 tokens — spend is summed


async def test_vote_binary_majority_approves() -> None:
    prov = _Provider([_v(True), _v(True), _v(False, "minor")])
    v, _ = await vote_binary(prov, criteria="c", artifact="a", n=3)
    assert v.approved is True  # 2 of 3 approve


# ── vote_pairwise (N-judge majority winner) ──────────────────────────────────


async def test_vote_pairwise_majority_winner() -> None:
    prov = _Provider([_p("a", "clearer"), _p("a"), _p("b")])  # 2 a, 1 b → majority a
    v, u = await vote_pairwise(prov, criteria="c", a="A", b="B", n=3)
    assert v.winner == "a" and v.reason == "clearer"  # reason from a winning-side judge
    assert prov.calls == 3 and u.total_tokens == 6  # 3 judges x 2 tokens — spend summed


async def test_vote_pairwise_ties_on_even_split() -> None:
    prov = _Provider([_p("a"), _p("b"), _p("tie")])  # 1 a, 1 b, 1 abstain → no majority → tie
    v, _ = await vote_pairwise(prov, criteria="c", a="A", b="B", n=3)
    assert v.winner == "tie"


# ── best_of (pairwise round-robin) ───────────────────────────────────────────


async def test_best_of_round_robin_picks_winner() -> None:
    # A content-aware judge ranks C best → C wins every pair (consistently both ways) → index 2.
    # The position-bias debias judges each pair in BOTH orderings, so 3 pairs → 6 calls.
    prov = _RankedJudge(order=["C", "A", "B"])
    best, _ = await best_of(prov, criteria="c", candidates=["A", "B", "C"])
    assert best == 2
    assert prov.calls == 6  # 3 unordered pairs x 2 orderings (debias)


async def test_best_of_ties_break_to_earliest_index() -> None:
    # Every pair is a tie → all candidates score equally → the earliest index wins (stable).
    prov = _Provider(_p("tie"))
    best, _ = await best_of(prov, criteria="c", candidates=["A", "B", "C"])
    assert best == 0


async def test_best_of_single_candidate_makes_no_calls() -> None:
    prov = _Provider(_p("a"))
    best, u = await best_of(prov, criteria="c", candidates=["only"])
    assert best == 0 and prov.calls == 0 and u.total_tokens == 0


async def test_best_of_votes_polls_a_panel_per_pair() -> None:
    # 2 candidates → 1 pair; votes=3 → a 3-voter panel per ordering, and the debias judges the pair
    # BOTH ways → 1 pair x 2 orderings x 3 votes = 6 calls. A content-aware judge ranks B best → B.
    prov = _RankedJudge(order=["B", "A"])
    best, _ = await best_of(prov, criteria="c", candidates=["A", "B"], votes=3)
    assert best == 1 and prov.calls == 6
