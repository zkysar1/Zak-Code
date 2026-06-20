"""Best-of-N attempts: run up to N independent tries, keep the FIRST that VERIFIES (seam B's core).

The product form of seam B — when a turn STALLS, fan out fresh attempts and adopt the first one that
passes an oracle (the verifier). ORACLE-FIRST and LAZY: a verified attempt is objectively a pass, so
there is nothing to rank — stop at the first success. This is the small-model bet on exactly the
tasks the suite measurement said it pays off on (hard-but-solvable: one try stalls, but N tries
contain a passer).

Generation, verification, and adoption are DECOUPLED: the caller injects ``run`` (make an attempt
"handle") and ``verify`` (does it pass?). So this stays pure orchestration: vendor-agnostic,
side-effect-free, and unit-testable. The *hard* part of seam B (how to isolate an attempt and adopt
its result in a live workspace) lives in the caller, deliberately, where it can be done carefully.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

H = TypeVar("H")


async def best_attempt(
    n: int,
    *,
    run: Callable[[int], Awaitable[H]],
    verify: Callable[[H], Awaitable[bool]],
) -> H | None:
    """Run up to ``n`` attempts via ``run`` (async ``index -> handle``); return the FIRST whose
    ``verify(handle)`` is True, or ``None`` if none verify.

    LAZY: it stops at the first verified attempt — a verified attempt IS a pass, so running the rest
    would only burn budget. FAIL-SAFE: a ``run`` or ``verify`` that raises drops just that attempt
    (best-of-N tolerates a flaky try), never the whole fan-out. ``n`` clamps to ≥ 0 (``0`` runs
    nothing and returns ``None``).
    """
    for i in range(max(0, n)):
        try:
            handle = await run(i)
        except Exception:  # noqa: BLE001 — a flaky attempt is dropped, not fatal
            continue
        try:
            verified = await verify(handle)
        except Exception:  # noqa: BLE001 — an unverifiable attempt counts as not-passing
            verified = False
        if verified:
            return handle
    return None


__all__ = ["best_attempt"]
