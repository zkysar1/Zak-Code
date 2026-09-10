"""Context compaction (M8): keep long sessions within the model's context window.

When a conversation approaches the model's context window, the :class:`Compactor`
replaces the older turns with a single summary message and keeps the most recent
turns verbatim, so the agent can continue without losing the plot.

Design notes:

* **Real token thresholds** — the trigger uses the provider's ``count_tokens`` (a
  real tokenizer), not a character heuristic. If the provider doesn't declare a
  context window, a conservative fallback is used.
* **Tool-pair safety** — the boundary between "old" (summarized) and "recent" (kept)
  is never placed such that a ``tool`` result is separated from the assistant message
  that requested it. We walk the boundary backwards past any leading ``tool`` message.
* **A tail budget, not only a tail count** (ADR-0132) — the kept tail is a number of
  messages, but a message can be a tool result the size of a quarter window, so the
  tail is also held to a token budget: over it, the OLDEST long tool outputs in the
  tail are replaced by elision stubs (the model has already acted on them), and the
  newest message is never touched — it is the result the model has not read yet.
* **Idempotent** — a prior summary is folded into the new one, so re-compaction never
  accumulates a stack of summaries; there is always exactly one leading summary.
* **Injected dependencies** — ``count_tokens`` and ``summarize`` are passed in, so the
  compactor is pure and unit-testable without a live provider.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from zakcode.messages import Message, ToolResultBlock

#: Prefix that marks a system message as a compaction summary (used to detect and
#: fold a prior summary on re-compaction).
SUMMARY_MARKER = "[Conversation summary]"

#: Appended to a summary so the model knows to resume rather than restart.
CONTINUATION_NOTE = (
    "\n\n(The text above is an automated summary of earlier conversation, compacted to "
    "stay within the context window. Continue the task from this point.)"
)

#: A tool output longer than this is dropped by the model-free compaction (ADR-0083).
ELIDE_MIN_CHARS = 2000
#: Prefix of the stub that replaces an elided tool output — how a stub is recognised.
ELISION_MARKER = "[tool output elided at compaction"


def elision_note(chars: int) -> str:
    """The stub left in place of a dropped tool output."""
    return f"{ELISION_MARKER} — {chars:,} characters dropped; re-run the tool if you need it]"


#: A function that counts tokens for a list of messages (e.g. ``provider.count_tokens``).
CountTokens = Callable[[list[Message]], int]
#: An async function that summarizes a list of messages into prose.
Summarize = Callable[[list[Message]], Awaitable[str]]


class CompactionConfig(BaseModel):
    """Tunables for when and how aggressively to compact."""

    #: Compact once token usage exceeds this fraction of the context window.
    threshold_fraction: float = 0.8
    #: Number of most-recent messages to always keep verbatim.
    preserve_recent: int = 6
    #: The kept tail may hold at most this fraction of the context window (ADR-0132); past
    #: it, its oldest long tool outputs are elided. Applies only when the caller passes the
    #: window and a token counter to :meth:`Compactor.compact` / :meth:`Compactor.elide`.
    tail_budget_fraction: float = 0.25


class CompactionResult(BaseModel):
    """Outcome of a compaction attempt."""

    compacted: bool
    messages: list[Message]
    summary: str | None = None
    summarized_count: int = 0
    #: Long tool outputs elided INSIDE the kept tail to fit its token budget (ADR-0132).
    tail_elided: int = 0


def is_summary(message: Message) -> bool:
    """True if ``message`` is a compaction summary produced by this module."""
    return message.role == "system" and message.text.startswith(SUMMARY_MARKER)


def _elidable(block: object) -> bool:
    return (
        isinstance(block, ToolResultBlock)
        and len(block.output) > ELIDE_MIN_CHARS
        and not block.output.startswith(ELISION_MARKER)
    )


def _elide_message(message: Message) -> tuple[Message, int]:
    """``message`` with every long tool output replaced by a stub, and how many were."""
    blocks = list(message.blocks)
    dropped = 0
    for i, block in enumerate(blocks):
        if _elidable(block):
            assert isinstance(block, ToolResultBlock)
            blocks[i] = block.model_copy(update={"output": elision_note(len(block.output))})
            dropped += 1
    if not dropped:
        return message, 0
    return message.model_copy(update={"blocks": blocks}), dropped


def trim_tail(
    recent: list[Message], *, budget: int, count_tokens: CountTokens
) -> tuple[list[Message], int]:
    """Elide the oldest long tool outputs in ``recent`` until it fits ``budget`` tokens.

    The kept tail is chosen by COUNT, and a count says nothing about size: measured
    2026-09-10 (coach, zc-03, 131k window), two clamped grep results in an eight-message
    tail survived a compaction at ~45k tokens, the prompt came back down only to 75k, and
    six calls later the session compacted again — three summarizer calls on ~100k
    prompts in one turn. This bounds the tail's SUM the way the seam clamp bounds one
    result. Oldest first, because the model has already acted on those; the LAST message
    is never touched — at the per-call check it is the tool result the model is about to
    read for the first time, and an elided stub there loses it before it was ever seen.
    Stops when nothing older is left to elide, even if still over budget.
    """
    out = list(recent)
    elided = 0
    while count_tokens(out) > budget:
        oldest = next(
            (i for i, m in enumerate(out[:-1]) if any(_elidable(b) for b in m.blocks)), None
        )
        if oldest is None:
            break
        out[oldest], dropped = _elide_message(out[oldest])
        elided += dropped
    return out, elided


class Compactor:
    """Decides when to compact a message list and performs the compaction."""

    def __init__(self, config: CompactionConfig | None = None) -> None:
        self.config = config or CompactionConfig()

    def should_compact(
        self,
        messages: list[Message],
        *,
        context_window: int | None,
        count_tokens: CountTokens,
    ) -> bool:
        """True if ``messages`` exceed the configured fraction of the context window.

        Raises when the window is unknown: there is no honest threshold without one, and a
        stand-in number is how a 131k pod once compacted against 8,192 (ADR-0066).
        """
        if not context_window or context_window <= 0:
            raise ValueError("compaction needs the model's context window, and none is known")
        threshold = int(context_window * self.config.threshold_fraction)
        return count_tokens(messages) > threshold

    def _split_index(self, messages: list[Message], keep: int | None = None) -> int:
        """Index where the preserved "recent" tail begins.

        Returns 0 if there is nothing old enough to summarize. Never returns an index
        whose message is a ``tool`` result (that must stay paired with its assistant
        request), so the boundary is walked backwards past any leading tool message.
        """
        n = len(messages)
        if keep is None:
            keep = self.config.preserve_recent
        if n <= keep:
            return 0
        idx = n - keep
        while idx > 0 and messages[idx].role == "tool":
            idx -= 1
        return idx

    def _tail_budget(self, context_window: int | None) -> int | None:
        if not context_window or context_window <= 0:
            return None
        return int(context_window * self.config.tail_budget_fraction)

    async def compact(
        self,
        messages: list[Message],
        *,
        summarize: Summarize,
        context_window: int | None = None,
        count_tokens: CountTokens | None = None,
    ) -> CompactionResult:
        """Replace older messages with one summary; keep the recent tail verbatim.

        A prior leading summary is folded into the new summary (idempotent). Returns
        ``compacted=False`` when there is nothing worth compacting. Given the model's
        ``context_window`` and a ``count_tokens``, the kept tail is also held to its token
        budget (ADR-0132, :func:`trim_tail`); without them it is kept as it stands.
        """
        idx = self._split_index(messages)
        if idx == 0:
            return CompactionResult(compacted=False, messages=list(messages))
        old = messages[:idx]
        recent = messages[idx:]
        # Nothing new to fold: avoid pointlessly re-summarizing a lone prior summary.
        if len(old) == 1 and is_summary(old[0]):
            return CompactionResult(compacted=False, messages=list(messages))

        summary_text = await summarize(old)
        summary_message = Message.system(f"{SUMMARY_MARKER}\n{summary_text}{CONTINUATION_NOTE}")
        budget = self._tail_budget(context_window)
        tail_elided = 0
        if budget is not None and count_tokens is not None:
            recent, tail_elided = trim_tail(recent, budget=budget, count_tokens=count_tokens)
        return CompactionResult(
            compacted=True,
            messages=[summary_message, *recent],
            summary=summary_text,
            summarized_count=len(old),
            tail_elided=tail_elided,
        )

    def elide(
        self,
        messages: list[Message],
        *,
        keep_recent: int | None = None,
        context_window: int | None = None,
        count_tokens: CountTokens | None = None,
    ) -> CompactionResult:
        """Model-free compaction (ADR-0083): replace every long tool output with a stub.

        The fallback for when the summarizer cannot run — it needs no model, so it cannot
        fail the way a summarize call can (a provider error, an overflow of its own, a
        busy pod). Tool outputs are the bulk of any long transcript and the one part the
        model can regenerate by re-running the tool; the conversation's own words stay.
        ``keep_recent`` defaults to the preserved tail, which is left verbatim — or, given
        ``context_window`` and ``count_tokens``, held to its token budget like
        :meth:`compact`'s (ADR-0132); ``0`` reaches the tail too — for a transcript whose
        LAST tool result is what overflows the window. Idempotent: a stub is never elided
        again. ``summarized_count`` is how many outputs were dropped from the old region,
        ``tail_elided`` how many from the tail; ``compacted`` is False when nothing was
        long enough.
        """
        keep = self.config.preserve_recent if keep_recent is None else keep_recent
        end = self._split_index(messages, keep) if keep > 0 else len(messages)
        out: list[Message] = []
        dropped = 0
        for index, message in enumerate(messages):
            if index < end:
                message, count = _elide_message(message)
                dropped += count
            out.append(message)
        tail_elided = 0
        budget = self._tail_budget(context_window)
        if end < len(out) and budget is not None and count_tokens is not None:
            tail, tail_elided = trim_tail(out[end:], budget=budget, count_tokens=count_tokens)
            out[end:] = tail
        return CompactionResult(
            compacted=dropped > 0 or tail_elided > 0,
            messages=out,
            summarized_count=dropped,
            tail_elided=tail_elided,
        )
