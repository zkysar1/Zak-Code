"""Tests for the best-of-N orchestrator (quality engine, increment 2).

Hermetic: a scripted judge provider returns canned pairwise verdicts, and a diverse generator thunk
yields a distinct candidate per call (counter-based, no await suspension → deterministic under the
concurrent fan-out). Covers selection, n=1 (no judging), dropping failed attempts, all-fail, and
usage accumulation.
"""

from __future__ import annotations

import json
from typing import Any

from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.quality import best_of_n
from zakcode.usage import Usage


class _Judge(Provider):
    """A scripted judge: returns a FIXED pairwise verdict JSON for every acomplete call."""

    def __init__(self, winner: str = "b") -> None:
        self._text = json.dumps({"winner": winner})
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, usage=Usage(total_tokens=1, cost_usd=0.0005))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "judge/test"


def _gen(candidates: list[str], *, fail_on: int | None = None):
    """A diverse generator thunk: returns ``candidates[i]`` (+ usage) on the i-th call; the call at
    index ``fail_on`` raises (a flaky attempt)."""
    state = {"i": 0}

    async def gen() -> tuple[str, Usage]:
        i = state["i"]
        state["i"] += 1
        if i == fail_on:
            raise RuntimeError("attempt failed")
        return candidates[i], Usage(total_tokens=5, cost_usd=0.002)

    return gen


def _gen_always_fail():
    async def gen() -> tuple[str, Usage]:
        raise RuntimeError("always fails")

    return gen


async def test_best_of_n_selects_the_winner() -> None:
    # The judge always picks "b" (the later candidate), so C wins the round-robin (A=0, B=1, C=2).
    judge = _Judge(winner="b")
    best, idx, usage = await best_of_n(judge, criteria="c", generate=_gen(["A", "B", "C"]), n=3)
    assert best == "C" and idx == 2
    assert judge.calls == 3  # round-robin of 3 candidates = 3 pairwise comparisons
    assert usage.total_tokens == 3 * 5 + 3 * 1  # 3 generations + 3 judge calls


async def test_best_of_n_n1_generates_once_without_judging() -> None:
    judge = _Judge()
    best, idx, usage = await best_of_n(judge, criteria="c", generate=_gen(["only"]), n=1)
    assert best == "only" and idx == 0
    assert judge.calls == 0  # a single candidate needs no selection
    assert usage.total_tokens == 5


async def test_best_of_n_drops_a_failed_attempt() -> None:
    # 3 attempts; the middle one raises → survivors [A, C] are judged. A flaky attempt is tolerated.
    judge = _Judge(winner="a")
    best, idx, usage = await best_of_n(
        judge, criteria="c", generate=_gen(["A", "B", "C"], fail_on=1), n=3
    )
    assert best == "A" and idx == 0  # judge picks "a"; A is survivor 0
    assert judge.calls == 1  # 2 survivors → 1 pairwise
    assert usage.total_tokens == 2 * 5 + 1  # 2 successful generations + 1 judge


async def test_best_of_n_all_attempts_fail_returns_empty() -> None:
    judge = _Judge()
    best, idx, usage = await best_of_n(judge, criteria="c", generate=_gen_always_fail(), n=3)
    assert best == "" and idx == 0
    assert judge.calls == 0  # nothing survived to judge
    assert usage.total_tokens == 0
