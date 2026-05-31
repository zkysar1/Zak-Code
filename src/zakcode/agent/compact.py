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
* **Idempotent** — a prior summary is folded into the new one, so re-compaction never
  accumulates a stack of summaries; there is always exactly one leading summary.
* **Injected dependencies** — ``count_tokens`` and ``summarize`` are passed in, so the
  compactor is pure and unit-testable without a live provider.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from zakcode.messages import Message

#: Prefix that marks a system message as a compaction summary (used to detect and
#: fold a prior summary on re-compaction).
SUMMARY_MARKER = "[Conversation summary]"

#: Appended to a summary so the model knows to resume rather than restart.
CONTINUATION_NOTE = (
    "\n\n(The text above is an automated summary of earlier conversation, compacted to "
    "stay within the context window. Continue the task from this point.)"
)

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
    #: Fallback context window when the provider doesn't declare one.
    fallback_context_window: int = 8192


class CompactionResult(BaseModel):
    """Outcome of a compaction attempt."""

    compacted: bool
    messages: list[Message]
    summary: str | None = None
    summarized_count: int = 0


def is_summary(message: Message) -> bool:
    """True if ``message`` is a compaction summary produced by this module."""
    return message.role == "system" and message.text.startswith(SUMMARY_MARKER)


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
        """True if ``messages`` exceed the configured fraction of the context window."""
        window = context_window or self.config.fallback_context_window
        if window <= 0:
            return False
        threshold = int(window * self.config.threshold_fraction)
        return count_tokens(messages) > threshold

    def _split_index(self, messages: list[Message]) -> int:
        """Index where the preserved "recent" tail begins.

        Returns 0 if there is nothing old enough to summarize. Never returns an index
        whose message is a ``tool`` result (that must stay paired with its assistant
        request), so the boundary is walked backwards past any leading tool message.
        """
        n = len(messages)
        keep = self.config.preserve_recent
        if n <= keep:
            return 0
        idx = n - keep
        while idx > 0 and messages[idx].role == "tool":
            idx -= 1
        return idx

    async def compact(self, messages: list[Message], *, summarize: Summarize) -> CompactionResult:
        """Replace older messages with one summary; keep the recent tail verbatim.

        A prior leading summary is folded into the new summary (idempotent). Returns
        ``compacted=False`` when there is nothing worth compacting.
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
        return CompactionResult(
            compacted=True,
            messages=[summary_message, *recent],
            summary=summary_text,
            summarized_count=len(old),
        )
