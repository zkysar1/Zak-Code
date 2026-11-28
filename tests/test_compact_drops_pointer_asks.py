"""A compaction drops the lookups a request does not carry (ADR-0423).

ADR-0400 sends a completion that only asked read_rule for rules the model holds, with its pointer
answers, as one harness line, and the session keeps both messages. A compaction read the session,
so the summarizer read a run of such asks as work worth recording, and the kept tail could open
the next window on one. Projection-level tests pin that exactly the pairs a request does not
carry are dropped and that everything else is kept; loop-level tests pin that the summarizer is
never handed a dropped pair and that the installed history holds none, with the work beside them
summarized and kept as before.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.test_compact_loop import _history, _loop, _SummarizerProvider
from tests.test_pointer_ask_hidden import _ask
from zakcode.agent.compact import CompactionConfig, Compactor
from zakcode.agent.loop import _drop_pointer_asks, _hide_pointer_asks
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock


def _work(n: int) -> list[Message]:
    """A completion that ran another tool, and its answer."""
    call = Message(role="assistant", blocks=[ToolUseBlock(id=f"w{n}", name="probe", input={})])
    return [call, Message.tool_results([ToolResultBlock(tool_use_id=f"w{n}", output=f"ok {n}")])]


def _rule_asks(messages: list[Message]) -> list[str]:
    return [
        b.id
        for m in messages
        for b in m.blocks
        if isinstance(b, ToolUseBlock) and b.name == "read_rule"
    ]


# ── the projection ──────────────────────────────────────────────────────────────


def test_a_lookup_answered_only_by_pointers_is_dropped_whole() -> None:
    head = Message.user("do the thing")
    first, second = _work(1), _work(2)
    history = [head, *first, *_ask(1, "a-rule"), *second, *_ask(2, "a-rule", "b-rule")]
    kept = _drop_pointer_asks(history)
    assert kept == [head, *first, *second]
    assert all(a is b for a, b in zip(kept, [head, *first, *second], strict=True))
    assert len(history) == 9  # the history itself is untouched


def test_with_nothing_to_drop_the_history_itself_comes_back() -> None:
    history = [Message.user("go"), *_work(1)]
    assert _drop_pointer_asks(history) is history
    assert _drop_pointer_asks([]) == []


@pytest.mark.parametrize(
    "exchange",
    [
        _ask(1, "a-rule", answers=["text"]),  # a first read, or refresh=true: the text is news
        _ask(1, "a-rule", answers=["error"]),  # an error is news
        _ask(1, "a-rule", answers=["refused"]),  # ADR-0399's refusal says why it was not run
        _ask(1, "a-rule", "b-rule", answers=["pointer", "text"]),  # one answer was news
        _ask(1, "a-rule", other="probe"),  # another tool ran beside it
        _ask(1, "a-rule", text="Checking the rule first."),  # the model said something
    ],
    ids=["first-read", "error", "refused", "pointer-and-text", "beside-other-tool", "with-text"],
)
def test_an_exchange_that_carried_anything_is_kept(exchange: list[Message]) -> None:
    history = [Message.user("go"), *exchange]
    assert _drop_pointer_asks(history) is history


def test_only_a_whole_pair_is_dropped() -> None:
    """A pointer answer without its call, a call without its answer: both stay, so a call is
    never left without its result."""
    call, answer = _ask(1, "a-rule")
    for history in ([Message.user("go"), answer], [Message.user("go"), call]):
        assert _drop_pointer_asks(history) is history


def test_it_drops_the_pairs_a_request_shows_as_a_line_and_nothing_else() -> None:
    """One test for both (ADR-0400): what a request replaces with a harness line is what a
    compaction drops."""
    history = [
        Message.user("go"),
        *_ask(1, "a-rule"),
        *_work(1),
        *_ask(2, "a-rule", answers=["text"]),
        *_ask(3, "b-rule"),
        *_ask(4, "a-rule", other="probe"),
        Message.assistant_text("done"),
    ]
    sent = [m for m in _hide_pointer_asks(history) if not m.text.startswith("[harness] ")]
    assert _drop_pointer_asks(history) == sent
    assert len(sent) == len(history) - 4


# ── the compaction ──────────────────────────────────────────────────────────────


def _compacted(tmp_path: Path, history: list[Message]) -> tuple[str, list[Message]]:
    """Compact ``history``; return the transcript the summarizer was handed and the history
    the loop installed."""
    provider = _SummarizerProvider(["the summary"], tokens=100_000)
    loop = _loop(provider, tmp_path, compactor=Compactor(CompactionConfig()))
    loop.session.messages.extend(history)
    assert asyncio.run(loop.compact_now(trigger="auto")) is True
    (sent,) = provider.seen[0]
    return sent.text, list(loop.session.messages)


def test_the_summarizer_never_reads_a_lookup_that_brought_back_nothing(tmp_path: Path) -> None:
    history = [*_history(3)]
    for n in range(1, 6):
        history += [*_ask(n, "a-rule"), *_work(n)]
    history += _history(3)
    transcript, _ = _compacted(tmp_path, history)
    assert "question 0" in transcript and "(called probe" in transcript  # the work is summarized
    assert "read_rule" not in transcript
    assert "[already read; ask" not in transcript


def test_the_kept_tail_holds_no_such_lookup(tmp_path: Path) -> None:
    """The lookup-loop fence (ADR-0422) compacts in the middle of a run of lookups: the window
    after it opens on the work before the run, not on the run."""
    history = [*_history(4), *_work(1), *_work(2)]
    for n in range(1, 8):
        history += _ask(n, "a-rule")
    _, installed = _compacted(tmp_path, history)
    assert _rule_asks(installed) == []
    outputs = [b.output for m in installed for b in m.blocks if isinstance(b, ToolResultBlock)]
    assert outputs[-2:] == ["ok 1", "ok 2"]


def test_a_lookup_that_brought_something_back_is_summarized_as_before(tmp_path: Path) -> None:
    history = [*_history(3), *_ask(1, "a-rule", answers=["text"]), *_history(3)]
    transcript, _ = _compacted(tmp_path, history)
    assert "(called read_rule" in transcript and "# a-rule\nDo it." in transcript
