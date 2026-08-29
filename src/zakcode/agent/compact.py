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

    def elide(self, messages: list[Message], *, keep_recent: int | None = None) -> CompactionResult:
        """Model-free compaction (ADR-0083): replace every long tool output with a stub.

        The fallback for when the summarizer cannot run — it needs no model, so it cannot
        fail the way a summarize call can (a provider error, an overflow of its own, a
        busy pod). Tool outputs are the bulk of any long transcript and the one part the
        model can regenerate by re-running the tool; the conversation's own words stay.
        ``keep_recent`` defaults to the preserved tail, which is left verbatim; ``0``
        reaches the tail too — for a transcript whose LAST tool result is what overflows
        the window. Idempotent: a stub is never elided again. ``summarized_count`` is how
        many outputs were dropped; ``compacted`` is False when nothing was long enough.
        """
        keep = self.config.preserve_recent if keep_recent is None else keep_recent
        end = self._split_index(messages, keep) if keep > 0 else len(messages)
        out: list[Message] = []
        dropped = 0
        for index, message in enumerate(messages):
            blocks = list(message.blocks)
            changed = False
            if index < end:
                for i, block in enumerate(blocks):
                    if (
                        isinstance(block, ToolResultBlock)
                        and len(block.output) > ELIDE_MIN_CHARS
                        and not block.output.startswith(ELISION_MARKER)
                    ):
                        blocks[i] = block.model_copy(
                            update={"output": elision_note(len(block.output))}
                        )
                        changed = True
                        dropped += 1
            out.append(message.model_copy(update={"blocks": blocks}) if changed else message)
        return CompactionResult(compacted=dropped > 0, messages=out, summarized_count=dropped)
