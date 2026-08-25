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
    """A shared count of model iterations — and optional cost/token ceilings — for a whole
    delegation tree.

    The iteration count is the original bound. ``max_cost_usd`` / ``max_tokens`` (parity #4)
    are optional ceilings on the *cumulative* spend across the whole tree: each loop folds a
    completed call's actuals via :meth:`add_usage`, and the loop stops with
    ``stop_reason="budget_exhausted"`` once :meth:`over_budget` is True. Both are
    POST-completion cumulative checks (total spent so far), not per-call input gating —
    they bound runaway spend on long delegation trees and large-context models. ``None``
    (the default) = that ceiling is unbounded, preserving the iteration-only behavior.
    """

    def __init__(
        self,
        total: int,
        *,
        max_children: int = DEFAULT_MAX_CHILDREN,
        max_cost_usd: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if total < 0:
            raise ValueError("budget total must be >= 0")
        if max_children < 0:
            raise ValueError("max_children must be >= 0")
        if max_cost_usd is not None and max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")
        if max_tokens is not None and max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        self._total = total
        self._consumed = 0
        self._max_children = max_children
        self._children_spawned = 0
        self._max_cost_usd = max_cost_usd
        self._max_tokens = max_tokens
        self._cost_spent = 0.0
        self._tokens_spent = 0

    # ── reporting ────────────────────────────────────────────────────────────

    @property
    def total(self) -> int:
        """The size of the shared pool (constant for the budget's lifetime). 0 = unlimited."""
        return self._total

    @property
    def unlimited(self) -> bool:
        """True when the pool has no iteration ceiling (``total == 0``) — the cost/token
        ceilings and the doom-loop detector are then the only bounds."""
        return self._total == 0

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

    @property
    def cost_spent(self) -> float:
        """Cumulative USD cost folded in across the whole tree."""
        return self._cost_spent

    @property
    def tokens_spent(self) -> int:
        """Cumulative total tokens folded in across the whole tree."""
        return self._tokens_spent

    # ── cost / token ceilings (parity #4) ─────────────────────────────────────

    def add_usage(self, cost_usd: float = 0.0, total_tokens: int = 0) -> None:
        """Fold one completed call's actuals into the shared cost/token totals.

        Called by each loop after a model call returns. ``await``-free, so atomic against
        concurrently-scheduled siblings (like :meth:`try_consume`). Negative inputs are
        clamped to 0 so a junk usage record can never *reduce* the running total.
        """
        self._cost_spent += max(0.0, cost_usd)
        self._tokens_spent += max(0, total_tokens)

    def cost_exhausted(self) -> bool:
        """True once cumulative cost has reached the ``max_cost_usd`` ceiling (if set)."""
        return self._max_cost_usd is not None and self._cost_spent >= self._max_cost_usd

    def tokens_exhausted(self) -> bool:
        """True once cumulative tokens have reached the ``max_tokens`` ceiling (if set)."""
        return self._max_tokens is not None and self._tokens_spent >= self._max_tokens

    def over_budget(self) -> bool:
        """True if either the cost or token ceiling has been crossed — the loop's signal to
        stop with ``stop_reason="budget_exhausted"``. False when neither ceiling is set."""
        return self.cost_exhausted() or self.tokens_exhausted()

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
        if not self.unlimited and n > self.remaining:
            return False
        self._consumed += n
        return True

    def consume(self, n: int = 1) -> None:
        """Consume ``n`` iterations, raising :class:`BudgetExhausted` if it can't."""
        if not self.try_consume(n):
            raise BudgetExhausted(
                f"need {n} iteration(s) but only {self.remaining} remain of {self._total}"
            )

    def refund(self, n: int = 1) -> int:
        """Return up to ``n`` consumed iterations to the shared pool.

        Used when an iteration did no real work (an empty model completion, or a
        tool batch that was entirely permission-denied / hook-vetoed), so wasted
        iterations do not deplete a *shared* delegation budget at the expense of
        siblings. Refunds only the shared pool — the per-turn ``max_iterations`` cap
        is unaffected, so refunding can never turn a turn into an unbounded loop.
        Capped at :attr:`consumed` (never goes negative); returns the amount actually
        refunded. ``await``-free, so it is atomic against concurrently-scheduled
        siblings, like :meth:`try_consume`.
        """
        if n < 0:
            raise ValueError("cannot refund a negative amount")
        refunded = min(n, self._consumed)
        self._consumed -= refunded
        return refunded

    def reset(self) -> None:
        """Start a new top-level turn-tree: zero the consumed iterations, the cumulative
        spend, and the children count.

        The module docstring has always said this budget is "for a turn and its
        sub-agents", but nothing ever reset it — the pool was drained across an
        Agent's LIFETIME, so one long turn that hit ``max_iterations`` wedged every
        later turn into an instant ``max_iterations`` stop at 0 iterations (field
        report 2026-08-25: "(say) continue → stopped early — max iterations · 0
        iterations"). Called by the Agent's turn entry points only — sub-agent loops
        share the object mid-tree and must never reset it.
        """
        self._consumed = 0
        self._cost_spent = 0.0
        self._tokens_spent = 0
        self._children_spawned = 0

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
