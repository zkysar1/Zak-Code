"""Token/cost accounting value objects.

``Usage`` is a neutral value object produced by the provider layer and stored inline on
each persisted message, so cumulative cost is reconstructable on resume without a side
file (see ``docs/ARCHITECTURE.md`` — Sessions & persistence). Kept in its own module so
both ``providers`` and ``session`` can import it without a circular dependency.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token counts and cost for a single model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        """Combine two usage records (for accumulating a session total)."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class UsageTracker(BaseModel):
    """Running total of usage across a session."""

    total: Usage = Field(default_factory=Usage)

    def add(self, usage: Usage) -> Usage:
        """Fold ``usage`` into the running total and return the new total."""
        self.total = self.total + usage
        return self.total


__all__ = ["Usage", "UsageTracker"]
