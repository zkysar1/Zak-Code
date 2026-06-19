"""Hybrid candidate selection: oracle-FILTER, then judge-RANK the survivors.

The measured fix for best-of-N's selection gap. On `04-todo-cli` the small-model fan-out *generated*
a passing solution but the LLM judge, reading source, *selected* a failing one — a passing candidate
was thrown away. The split the data argues for:

* an **oracle** answers the deterministic *"does it actually work?"* (run the candidate's tests,
  compile it, lint it) — exactly what a judge reading source guesses wrong;
* a **judge** answers *"is it good?"* (design, completeness — what no oracle can see).

So :func:`select_best` keeps the candidates the oracle passes, then judge-ranks the survivors.
Oracle for "works", judge for "good". The oracle is OPTIONAL and INJECTED (this module never runs a
subprocess) — the caller supplies "does candidate i work" however it likes (the agent loop: run the
recipe/verify gate; the bench: the held-out ``verify.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from zakcode.providers.base import Provider
from zakcode.quality.judge import best_of
from zakcode.usage import Usage


async def select_best(
    judge_provider: Provider,
    *,
    criteria: str,
    candidates: list[str],
    oracle: Callable[[int], Awaitable[bool]] | None = None,
    votes: int = 1,
) -> tuple[int, Usage]:
    """The index of the best candidate: oracle-FILTER then judge-RANK.

    ``oracle`` (optional) is an async ``index -> bool`` — *"does candidate ``i`` actually work?"*.
    When given, candidates that pass are kept and the rest discarded BEFORE judging; the judge then
    ranks only the survivors on the quality an oracle can't see, via a pairwise tournament
    (:func:`~zakcode.quality.judge.best_of`). With NO oracle — or if NONE pass — the judge ranks all
    candidates (best effort). ``votes`` is forwarded to that tournament — a majority panel per
    comparison (:func:`~zakcode.quality.judge.vote_pairwise`). FAIL-SAFE: an oracle that raises
    for a candidate is treated as not-passing, so a flaky check never derails selection. Returns
    ``(index, usage)`` into the original list; ``≤ 1`` candidate returns ``0`` with no calls.
    """
    if len(candidates) <= 1:
        return 0, Usage()

    survivors = list(range(len(candidates)))
    if oracle is not None:

        async def _passes(i: int) -> bool:
            try:
                return bool(await oracle(i))
            except Exception:  # noqa: BLE001 — a flaky oracle is "not passing", never fatal
                return False

        results = await asyncio.gather(*(_passes(i) for i in survivors))
        passing = [i for i, ok in zip(survivors, results, strict=True) if ok]
        if passing:  # keep only working candidates; if NONE work, fall through to rank them all
            survivors = passing

    if len(survivors) == 1:
        return survivors[0], Usage()

    sub_index, usage = await best_of(
        judge_provider,
        criteria=criteria,
        candidates=[candidates[i] for i in survivors],
        votes=votes,
    )
    return survivors[sub_index], usage


__all__ = ["select_best"]
