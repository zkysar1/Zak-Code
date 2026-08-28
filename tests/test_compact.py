"""Tests for the M8 Compactor (pure, with injected count_tokens / summarize)."""

from __future__ import annotations

import asyncio

import pytest

from zakcode.agent.compact import (
    CONTINUATION_NOTE,
    SUMMARY_MARKER,
    CompactionConfig,
    Compactor,
    is_summary,
)
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock


async def _summarize(messages: list[Message]) -> str:
    return f"summary of {len(messages)} messages"


def _convo(n: int) -> list[Message]:
    """n simple alternating user/assistant messages."""
    out: list[Message] = []
    for i in range(n):
        if i % 2 == 0:
            out.append(Message.user(f"user {i}"))
        else:
            out.append(Message.assistant_text(f"assistant {i}"))
    return out


# ── should_compact ────────────────────────────────────────────────────────────


def test_should_compact_over_threshold() -> None:
    c = Compactor(CompactionConfig(threshold_fraction=0.8))
    msgs = _convo(4)
    assert c.should_compact(msgs, context_window=1000, count_tokens=lambda m: 801) is True
    assert c.should_compact(msgs, context_window=1000, count_tokens=lambda m: 800) is False


def test_should_compact_refuses_an_unknown_window() -> None:
    # ADR-0066: there is no fallback window. A compactor asked to size a threshold for a
    # model nobody knows the window of must say so, not run on a stand-in number.
    c = Compactor(CompactionConfig(threshold_fraction=0.5))
    with pytest.raises(ValueError, match="context window"):
        c.should_compact(_convo(2), context_window=None, count_tokens=lambda m: 60)
    with pytest.raises(ValueError, match="context window"):
        c.should_compact(_convo(2), context_window=0, count_tokens=lambda m: 60)


# ── compact: basic behavior ─────────────────────────────────────────────────────


def test_compact_preserves_recent_and_summarizes_old() -> None:
    async def scenario() -> None:
        c = Compactor(CompactionConfig(preserve_recent=4))
        msgs = _convo(10)
        result = await c.compact(msgs, summarize=_summarize)
        assert result.compacted is True
        # one summary + 4 preserved
        assert len(result.messages) == 5
        assert is_summary(result.messages[0])
        assert result.summarized_count == 6
        # the 4 preserved are the original last 4, unchanged
        assert result.messages[1:] == msgs[-4:]

    asyncio.run(scenario())


def test_summary_message_shape() -> None:
    async def scenario() -> None:
        c = Compactor(CompactionConfig(preserve_recent=2))
        result = await c.compact(_convo(6), summarize=_summarize)
        head = result.messages[0]
        assert head.role == "system"
        assert head.text.startswith(SUMMARY_MARKER)
        assert CONTINUATION_NOTE.strip() in head.text
        assert "summary of 4 messages" in head.text

    asyncio.run(scenario())


def test_no_compaction_when_short() -> None:
    async def scenario() -> None:
        c = Compactor(CompactionConfig(preserve_recent=6))
        msgs = _convo(5)
        result = await c.compact(msgs, summarize=_summarize)
        assert result.compacted is False
        assert result.messages == msgs

    asyncio.run(scenario())


# ── tool-pair safety ─────────────────────────────────────────────────────────────


def test_never_splits_tool_pair() -> None:
    async def scenario() -> None:
        # assistant requests a tool, then a tool result follows. With a small
        # preserve_recent the naive boundary would start the tail at the tool result
        # (orphaning it); the compactor must walk back so the pair stays together.
        msgs = [
            Message.user("do a thing"),
            Message.assistant_text("calling tool"),
            Message(role="assistant", blocks=[ToolUseBlock(id="t1", name="x", input={})]),
            Message.tool_results([ToolResultBlock(tool_use_id="t1", output="ok")]),
        ]
        c = Compactor(CompactionConfig(preserve_recent=1))
        result = await c.compact(msgs, summarize=_summarize)
        assert result.compacted is True
        # the tail must NOT begin with a tool message
        assert result.messages[1].role != "tool"

    asyncio.run(scenario())


# ── idempotent re-compaction ─────────────────────────────────────────────────────


def test_recompaction_folds_prior_summary() -> None:
    async def scenario() -> None:
        c = Compactor(CompactionConfig(preserve_recent=2))
        first = await c.compact(_convo(10), summarize=_summarize)
        assert first.compacted is True
        # grow the convo, then compact again
        grown = [*first.messages, Message.user("u"), Message.assistant_text("a")]
        second = await c.compact(grown, summarize=_summarize)
        assert second.compacted is True
        # exactly one summary message, at the head (no stacking)
        summaries = [m for m in second.messages if is_summary(m)]
        assert len(summaries) == 1
        assert is_summary(second.messages[0])

    asyncio.run(scenario())


def test_lone_summary_is_not_recompacted() -> None:
    async def scenario() -> None:
        # [summary, recent...] where only the summary is "old" -> nothing new to fold.
        c = Compactor(CompactionConfig(preserve_recent=2))
        first = await c.compact(_convo(8), summarize=_summarize)
        # first.messages == [summary, m6, m7]; recompacting with preserve_recent=2
        # leaves old=[summary] only -> no-op.
        second = await c.compact(first.messages, summarize=_summarize)
        assert second.compacted is False

    asyncio.run(scenario())
