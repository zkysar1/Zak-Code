"""Best-of-N: fan out N independent attempts at a generation, then judge them to keep the best.

The product form of the small-model bet — N cheap attempts at a hard generation, then the judge
substrate (a pairwise tournament, :func:`~zakcode.quality.judge.best_of`) picks the winner. The
gamble is that several diverse small-model tries plus a selector beat one (small *or* large) try.

Generation and selection are DECOUPLED: the caller supplies an async ``generate`` thunk (a model
call, a sub-agent run — anything that yields a candidate + its :class:`~zakcode.usage.Usage`) and
the ``criteria``; this orchestrates the fan-out and the selection. For best-of-N to actually help,
the thunk must produce DIVERSE candidates (sample at a non-zero temperature, or vary the prompt) —
N identical tries select nothing. Synthesis (merging the best parts of several candidates,
graph-of-thoughts style) is a future option; this first cut SELECTS one winner.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from zakcode.providers.base import Provider
from zakcode.quality.judge import best_of
from zakcode.usage import Usage


async def best_of_n(
    judge_provider: Provider,
    *,
    criteria: str,
    generate: Callable[[], Awaitable[tuple[str, Usage]]],
    n: int = 3,
) -> tuple[str, int, Usage]:
    """Generate ``n`` candidates concurrently via ``generate`` and return the best by a pairwise
    tournament against ``criteria``.

    ``generate`` is an async thunk returning ``(candidate_text, usage)``; it is called ``n`` times
    concurrently. Returns ``(best_text, best_index, total_usage)`` — ``best_index`` is into the
    surviving candidates (in generation order), useful for tracing which attempt won.

    Bounds + safety: ``n`` is clamped to ≥ 1 (``n=1`` generates once and returns it, no judging). A
    ``generate`` call that RAISES drops just that candidate (best-of-N tolerates a flaky attempt);
    if EVERY attempt fails, returns ``("", 0, usage)``. The judge itself is fail-safe (see
    :func:`~zakcode.quality.judge.best_of`), so selection never raises either.
    """
    n = max(1, n)

    async def _safe_generate() -> tuple[str, Usage] | None:
        try:
            return await generate()
        except Exception:  # noqa: BLE001 — one flaky attempt must not sink the whole fan-out
            return None

    produced = await asyncio.gather(*(_safe_generate() for _ in range(n)))

    candidates: list[str] = []
    usage = Usage()
    for item in produced:
        if item is None:
            continue
        text, gen_usage = item
        candidates.append(text)
        usage = usage + gen_usage

    if not candidates:
        return "", 0, usage
    if len(candidates) == 1:
        return candidates[0], 0, usage

    index, judge_usage = await best_of(judge_provider, criteria=criteria, candidates=candidates)
    return candidates[index], index, usage + judge_usage


__all__ = ["best_of_n"]
