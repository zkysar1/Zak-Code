"""The shared iteration budget for a turn and its sub-agents (M4).

A single :class:`IterationBudget` instance is shared by a parent agent loop and
every sub-agent it spawns. This is the mechanism that keeps delegation from
multiplying cost: the *total* number of model iterations across the whole
delegation tree is bounded by one pool, not by per-agent caps that would compound
(a parent of 50 spawning 32 children of 50 each would be 1,600 iterations — a
shared budget makes it 50, total).

Design (deliberately small and synchronous):

* **Shared-instance, lazy consume.** The *same* budget object is handed to the
  parent loop and to every child loop (see :class:`~zakcode.agent.subagent.SubAgentRunner`).
  Each loop iteration draws one unit via :meth:`try_consume`; when the pool is
  empty the loop stops with ``stop_reason="max_iterations"``. The agent loop runs
  on a single asyncio event loop and the counter mutators contain no ``await``, so
  check-then-deduct is atomic with respect to concurrently-``gather``ed siblings:
  the tree can never collectively exceed ``total``. (Fairness is *not* guaranteed —
  a greedy child can consume more of the shared pool than a sibling; only the total
  is bounded, which is the cost-safety property that matters.)
* **Child cap.** :meth:`register_child` counts each spawned sub-agent and refuses
  past :attr:`max_children`, so a single turn cannot fan out without bound. Nesting
  depth is enforced elsewhere — structurally — by building child loops with no
  spawner (one level only); the budget itself does not model depth.

This module has no dependencies on the loop, providers, or tools — it is a pure
value object, unit-tested in isolation.
"""

from __future__ import annotations

#: Default ceiling on how many sub-agents may be spawned from one shared budget.
DEFAULT_MAX_CHILDREN = 32


class BudgetExhausted(Exception):
    """Raised by :meth:`IterationBudget.consume` when no allowance remains."""


class ChildLimitExceeded(Exception):
    """Raised by :meth:`IterationBudget.register_child` when the cap is reached."""


class IterationBudget:
    """A shared count of model iterations for a whole delegation tree."""

    def __init__(self, total: int, *, max_children: int = DEFAULT_MAX_CHILDREN) -> None:
        if total < 0:
            raise ValueError("budget total must be >= 0")
        if max_children < 0:
            raise ValueError("max_children must be >= 0")
        self._total = total
        self._consumed = 0
        self._max_children = max_children
        self._children_spawned = 0

    # ── reporting ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """The size of the shared pool (constant for the budget's lifetime)."""
        return self._total

    @property
    def consumed(self) -> int:
        """Iterations consumed across the whole tree."""
        return self._consumed

    @property
    def remaining(self) -> int:
        """Iterations still available to consume (never negative)."""
        return max(0, self._total - self._consumed)

    @property
    def max_children(self) -> int:
        return self._max_children

    @property
    def children_spawned(self) -> int:
        """How many children have been registered against this budget so far."""
        return self._children_spawned

    # ── consuming ────────────────────────────────────────────────────────────

    def try_consume(self, n: int = 1) -> bool:
        """Consume ``n`` iterations if the pool can cover them.

        Returns ``True`` and deducts ``n`` when ``n <= remaining``; otherwise
        leaves the budget untouched and returns ``False`` (the caller's signal to
        stop with ``stop_reason="max_iterations"``). The check-then-deduct is
        ``await``-free, so it is atomic across concurrently-scheduled siblings.
        """
        if n < 0:
            raise ValueError("cannot consume a negative amount")
        if n > self.remaining:
            return False
        self._consumed += n
        return True

    def consume(self, n: int = 1) -> None:
        """Consume ``n`` iterations, raising :class:`BudgetExhausted` if it can't."""
        if not self.try_consume(n):
            raise BudgetExhausted(
                f"need {n} iteration(s) but only {self.remaining} remain of {self._total}"
            )

    # ── child accounting ─────────────────────────────────────────────────────

    def can_spawn_child(self) -> bool:
        """Whether another child may be registered (cap not yet reached)."""
        return self._children_spawned < self._max_children

    def register_child(self) -> None:
        """Count one spawned child, raising :class:`ChildLimitExceeded` past the cap."""
        if not self.can_spawn_child():
            raise ChildLimitExceeded(
                f"sub-agent cap reached ({self._max_children} children already spawned)"
            )
        self._children_spawned += 1


__all__ = [
    "IterationBudget",
    "BudgetExhausted",
    "ChildLimitExceeded",
    "DEFAULT_MAX_CHILDREN",
]
