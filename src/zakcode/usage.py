"""Token/cost accounting value objects.

``Usage`` is a neutral value object produced by the provider layer and stored inline on
each persisted message, so cumulative cost is reconstructable on resume without a side
file (see ``docs/ARCHITECTURE.md`` — Sessions & persistence). Kept in its own module so
both ``providers`` and ``session`` can import it without a circular dependency.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token counts and cost for a single model call.

    ``cache_read_tokens`` / ``cache_creation_tokens`` break out the prompt-cache portion
    of ``prompt_tokens`` when the backend reports it (Anthropic prompt caching, OpenAI
    automatic caching). They are a *subset view* for visibility — ``prompt_tokens`` and
    ``cost_usd`` already reflect the (discounted) cached billing — so a cache hit shows as
    a high ``cache_read_tokens`` against a low effective cost. Both default 0, so a
    persisted message from before this field existed loads forward-compatibly.

    ``reasoning_tokens`` is the same kind of subset view on the other side of the ledger: the
    portion of ``completion_tokens`` a reasoning model spent thinking rather than answering.
    It is the only per-call record of how much reasoning a call actually did — the requested
    ``reasoning_effort`` is a knob on the way out and is not echoed back by any provider — so
    this is what makes a depth setting observable after the fact. Defaults 0 like the others.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    #: Prompt tokens served from the cache (a read hit) — Anthropic ``cache_read_input_tokens``
    #: / OpenAI ``prompt_tokens_details.cached_tokens``. A subset of ``prompt_tokens``.
    cache_read_tokens: int = 0
    #: Prompt tokens written to the cache this call (Anthropic ``cache_creation_input_tokens``).
    cache_creation_tokens: int = 0
    #: Completion tokens spent on reasoning rather than the visible answer — OpenAI
    #: ``completion_tokens_details.reasoning_tokens`` (chat) /
    #: ``output_tokens_details.reasoning_tokens`` (Responses). A subset of ``completion_tokens``,
    #: billed at the output rate. 0 when the model does no reasoning, when the depth was
    #: ``none``, or when the backend does not report it.
    reasoning_tokens: int = 0
    #: The litellm model string this call ran on, for per-model cost attribution (e.g. the
    #: ``/cost`` breakdown under zakpick, where a session spans several models). Empty for older
    #: persisted records and for aggregate totals (a sum across models has no single model).
    model: str = ""
    #: True when ``cost_usd`` could NOT be determined for at least one call folded in here —
    #: the model is absent from litellm's price map, so the call was priced at nothing because
    #: nothing is known, not because it was free. Without this flag those two facts are the same
    #: ``0.0`` and a cost ceiling reads "no spend" on a lane it cannot price at all. The price is
    #: deliberately never guessed (``test_extract_cost_unknown_model_stays_zero``); the
    #: uncertainty is carried instead, so :class:`~zakcode.agent.budget.IterationBudget` can tell
    #: an unenforceable ceiling from an unspent one and no reader mistakes the 0.0 for a
    #: measurement.
    cost_unpriced: bool = False
    #: The side call this record came from (``"summarizer"``: a compaction's calls, ADR-0241),
    #: or empty for a call of the main conversation. A side call has no reply of its own, so
    #: the per-turn pairing of ``zakcode throughput`` (ADR-0104) leaves it out. Empty for
    #: older persisted records and for aggregate totals.
    side_call: str = ""

    def __add__(self, other: Usage) -> Usage:
        """Combine two usage records (for accumulating a session total).

        ``model`` survives only when both operands share it — a sum across different models is a
        mixed total with no single model, so it collapses to empty; ``side_call`` follows the same
        rule. ``cost_unpriced`` is sticky under addition: a total containing one unpriceable call
        is itself an underestimate, so
        the flag must survive into the aggregate or the session total silently launders it back
        into a clean-looking number (the rb-7829 shape — a missingness flag that sibling fields
        and aggregates do not consult).
        """
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            model=self.model if self.model == other.model else "",
            cost_unpriced=self.cost_unpriced or other.cost_unpriced,
            side_call=self.side_call if self.side_call == other.side_call else "",
        )


class UsageTracker(BaseModel):
    """Running total of usage across a session."""

    total: Usage = Field(default_factory=Usage)

    def add(self, usage: Usage) -> Usage:
        """Fold ``usage`` into the running total and return the new total."""
        self.total = self.total + usage
        return self.total


__all__ = ["Usage", "UsageTracker"]
