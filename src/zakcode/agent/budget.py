"""The shared, refundable iteration budget for a turn and its sub-agents (M4).

A single :class:`IterationBudget` instance is shared by a parent agent loop and
every sub-agent it spawns. This is the mechanism that keeps delegation from
multiplying cost: the *total* number of model iterations across the whole
delegation tree is bounded by one pool, not by per-agent caps that would compound
(a parent of 50 spawning 32 children of 50 each is 1,600 iterations — a budget
makes it 50, shared).

Design (deliberately small and synchronous):

* The agent loop runs on a single asyncio event loop, so there is no real thread
  contention; the counters are guarded only against logical misuse (consuming more
  than remains, refunding more than was taken), not against parallel mutation.
* **Reserve / refund.** When a parent spawns a child it :meth:`reserve` s a chunk
  up front so concurrently-running siblings cannot each independently overrun the
  shared pool. Whatever the child does not use is :meth:`refund` ed when it
  returns, so a later sequential sibling is not starved by an over-generous
  reservation. (Lazy per-iteration :meth:`try_consume` is also offered for the
  non-delegating path.)
* **Child cap & depth.** A default cap on how many children may be spawned from
  one budget, and a maximum nesting depth (one level by default: sub-agents do not
  themselves spawn sub-agents). Both are advisory limits the orchestrator checks;
  this object only counts and reports them.

This module has no dependencies on the loop, providers, or tools — it is a pure
value object so it can be frozen and unit-tested in isolation before anything
wires it in.
"""

from __future__ import annotations

#: Default ceiling on how many sub-agents may be spawned from one shared budget.
DEFAULT_MAX_CHILDREN = 32

#: Default maximum delegation nesting depth. ``1`` means the top-level agent may
#: spawn sub-agents, but those sub-agents may not spawn their own.
DEFAULT_MAX_DEPTH = 1


class BudgetExhausted(Exception):
    """Raised by :meth:`IterationBudget.consume` when no allowance remains."""


class ChildLimitExceeded(Exception):
    """Raised by :meth:`IterationBudget.register_child` when the cap is reached."""


class IterationBudget:
    """A shared, refundable count of model iterations for a delegation tree."""

    def __init__(
        self,
        total: int,
        *,
        max_children: int = DEFAULT_MAX_CHILDREN,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        if total < 0:
            raise ValueError("budget total must be >= 0")
        if max_children < 0:
            raise ValueError("max_children must be >= 0")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        self._total = total
        self._consumed = 0
        self._max_children = max_children
        self._max_depth = max_depth
        self._children_spawned = 0

    # ── reporting ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """The size of the shared pool (constant for the budget's lifetime)."""
        return self._total

    @property
    def consumed(self) -> int:
        """Iterations consumed (or currently reserved) across the whole tree."""
        return self._consumed

    @property
    def remaining(self) -> int:
        """Iterations still available to consume or reserve (never negative)."""
        return max(0, self._total - self._consumed)

    @property
    def max_children(self) -> int:
        return self._max_children

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def children_spawned(self) -> int:
        """How many children have been registered against this budget so far."""
        return self._children_spawned

    # ── consuming ────────────────────────────────────────────────────────────

    def try_consume(self, n: int = 1) -> bool:
        """Consume ``n`` iterations if the pool can cover them.

        Returns ``True`` and deducts ``n`` when ``n <= remaining``; otherwise
        leaves the budget untouched and returns ``False`` (the caller's signal to
        stop with ``stop_reason="max_iterations"``). The lazy, per-iteration path
        for a loop that is not pre-reserving a block.
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

    # ── reserve / refund (the delegation path) ───────────────────────────────

    def reserve(self, n: int) -> int:
        """Reserve up to ``n`` iterations up front, returning how many were granted.

        Grants ``min(n, remaining)`` so a reservation never overdraws the pool —
        the caller must honor the returned (possibly smaller) amount as the child's
        ceiling. Reserved iterations count as consumed until :meth:`refund` ed, so
        concurrent siblings each see the pool shrink immediately and cannot all
        independently claim the same headroom.
        """
        if n < 0:
            raise ValueError("cannot reserve a negative amount")
        granted = min(n, self.remaining)
        self._consumed += granted
        return granted

    def refund(self, n: int) -> None:
        """Return ``n`` previously-consumed/reserved iterations to the pool.

        Clamped so ``consumed`` never goes below zero (refunding more than was
        taken is a no-op past zero rather than an error, keeping the accounting
        robust against an over-eager caller).
        """
        if n < 0:
            raise ValueError("cannot refund a negative amount")
        self._consumed = max(0, self._consumed - n)

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

    def child_budget(self, *, depth: int) -> IterationBudget:
        """Derive a child budget that shares this pool's remaining allowance.

        The child gets the *same* total/remaining view (it draws from the shared
        pool via its own reservations) but with depth-aware limits: a child one
        level deeper may spawn its own children only while ``depth < max_depth``.
        At or past the depth limit the child's ``max_children`` is ``0``, so the
        one-level-nesting rule is enforced structurally, not by ad-hoc checks.

        Note: this returns a *view-like* sibling object that shares neither counter
        storage nor identity with the parent — true pool sharing happens when the
        orchestrator reserves from the parent and hands the child a derived ceiling.
        Callers that want a single shared counter should pass the *same* budget
        instance to the child instead and rely on reserve/refund.
        """
        allow_grandchildren = depth < self._max_depth
        return IterationBudget(
            self.remaining,
            max_children=self._max_children if allow_grandchildren else 0,
            max_depth=self._max_depth,
        )


__all__ = [
    "IterationBudget",
    "BudgetExhausted",
    "ChildLimitExceeded",
    "DEFAULT_MAX_CHILDREN",
    "DEFAULT_MAX_DEPTH",
]
