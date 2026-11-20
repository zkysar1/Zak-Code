"""A completion that only asked read_rule for rules the model holds is not sent back (ADR-0400).

The session keeps the call and its answer. What the model is sent has one harness line in
their place, built from the exchange alone, so every request renders the same line at the same
place and a cached prefix holds. A first read, an error, a refusal, a batch with another tool
and a call with text beside it are sent as they are. What was measured is recorded in ADR-0400.

Projection-level tests pin what is replaced and what is not, and that the projection only ever
grows at its end; loop-level tests pin, on BOTH turn paths, that the provider is not handed the
exchange while the session still holds it, and that ADR-0399's bound still counts it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_lookup_only_bound import _first_reads, _kinds, _rule_call, _rule_results, _turn
from tests.test_loop_planning import _done, _judge_ok, _plan_call, _review_ok
from tests.test_plan_only_bound import _probe_call, _steps
from zakcode.agent.loop import _hide_pointer_asks
from zakcode.messages import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from zakcode.tools.base import RULE_POINTER

_POINTER = "[already read; ask 1] Rule 'a-rule' is in your context from an earlier read_rule call"


def _ask(
    n: int,
    *names: str,
    answers: list[str] | None = None,
    text: str = "",
    thinking: str = "",
    other: str | None = None,
) -> list[Message]:
    """An assistant completion asking read_rule for ``names`` and the tool message answering it.
    ``answers`` per call: ``pointer`` (the default), ``text``, ``error`` or ``refused``."""
    uses = [
        ToolUseBlock(id=f"r{n}-{i}", name="read_rule", input={"name": name})
        for i, name in enumerate(names)
    ]
    if other is not None:
        uses.append(ToolUseBlock(id=f"o{n}", name=other, input={}))
    blocks: list[Any] = []
    if thinking:
        blocks.append(ThinkingBlock(text=thinking))
    if text:
        blocks.append(TextBlock(text=text))
    blocks.extend(uses)
    kinds = answers or ["pointer"] * len(uses)
    results = []
    for use, kind in zip(uses, kinds, strict=True):
        if kind == "pointer":
            results.append(
                ToolResultBlock(tool_use_id=use.id, output=_POINTER, data={RULE_POINTER: True})
            )
        elif kind == "text":
            results.append(ToolResultBlock(tool_use_id=use.id, output="# a-rule\nDo it."))
        elif kind == "error":  # flagged too: an error is news whatever else it carries
            results.append(
                ToolResultBlock(
                    tool_use_id=use.id, output="failed", is_error=True, data={RULE_POINTER: True}
                )
            )
        else:  # refused, as ADR-0399's bound answers it
            results.append(
                ToolResultBlock(
                    tool_use_id=use.id,
                    output="Not executed: this is response 3 in a row that only asks read_rule",
                    is_error=True,
                    data={"plan_only_bound": True},
                )
            )
    return [Message(role="assistant", blocks=blocks), Message.tool_results(results)]


def _wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Each message whole, its time stamp included: a line stands for its exchange alone."""
    return [m.model_dump() for m in messages]


# ── the projection ──────────────────────────────────────────────────────────────


def test_a_lookup_answered_only_by_pointers_is_sent_as_one_harness_line() -> None:
    head = Message.user("do the thing")
    work = [Message(role="assistant", blocks=[ToolUseBlock(id="w1", name="probe", input={})])]
    work.append(Message.tool_results([ToolResultBlock(tool_use_id="w1", output="ok")]))
    history = [head, *work, *_ask(1, "a-rule"), Message.assistant_text("done")]
    sent = _hide_pointer_asks(history)
    assert len(sent) == len(history) - 1
    assert sent[0] is head and sent[1] is work[0] and sent[2] is work[1] and sent[-1] is history[-1]
    line = sent[3]
    assert line.role == "user" and line.created_at == history[3].created_at
    assert line.text.startswith("[harness] ")
    assert "You asked read_rule for a-rule: it is already in your context" in line.text
    assert "Apply it to the step you are on and take its next action." in line.text
    assert not [b for m in sent for b in m.blocks if isinstance(b, ToolUseBlock) and b.id == "r1-0"]
    assert len(history) == 6  # the history itself is untouched


def test_two_rules_asked_together_are_named_in_one_line() -> None:
    sent = _hide_pointer_asks([Message.user("go"), *_ask(1, "a-rule", "b-rule")])
    assert len(sent) == 2
    assert (
        "You asked read_rule for a-rule, b-rule: they are already in your context" in sent[1].text
    )
    assert "Apply them to the step you are on" in sent[1].text
    twice = _hide_pointer_asks([Message.user("go"), *_ask(2, "a-rule", "a-rule")])
    assert "You asked read_rule for a-rule: it is already in your context" in twice[1].text


def test_a_thinking_block_goes_with_the_completion_it_belongs_to() -> None:
    sent = _hide_pointer_asks([Message.user("go"), *_ask(1, "a-rule", thinking="let me check")])
    assert len(sent) == 2 and not [b for b in sent[1].blocks if isinstance(b, ThinkingBlock)]


def test_with_nothing_to_hide_the_history_itself_is_sent() -> None:
    history = [Message.user("go"), *_ask(1, "a-rule", answers=["text"])]
    assert _hide_pointer_asks(history) is history
    assert _hide_pointer_asks([]) == []


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
def test_an_exchange_that_carried_anything_is_sent_as_it_is(exchange: list[Message]) -> None:
    history = [Message.user("go"), *exchange]
    assert _hide_pointer_asks(history) is history


def test_a_pointer_answer_without_its_call_is_sent_as_it_is() -> None:
    """Only a whole pair is replaced: a tool message whose call is not the one before it, a call
    whose answer is not the next message, and an answer that lacks a result all stay, so a call
    is never sent without its result."""
    call, answer = _ask(1, "a-rule")
    stray = [Message.user("go"), answer]
    assert _hide_pointer_asks(stray) is stray
    unanswered = [Message.user("go"), call]
    assert _hide_pointer_asks(unanswered) is unanswered
    mismatched = _ask(2, "a-rule")[1]
    split = [Message.user("go"), call, mismatched]
    assert _hide_pointer_asks(split) is split
    both, answers = _ask(3, "a-rule", "b-rule")
    first = [b for b in answers.blocks if isinstance(b, ToolResultBlock)][:1]
    partial = [Message.user("go"), both, Message.tool_results(first)]
    assert _hide_pointer_asks(partial) is partial


def test_the_projection_only_ever_grows_at_its_end() -> None:
    """Built from each exchange alone: what was sent for a history is the start of what is sent
    once more messages follow, whatever they are, so the provider's cached prefix holds."""
    history: list[Message] = [Message.user("go")]
    tails = [
        _ask(1, "a-rule"),
        [Message(role="assistant", blocks=[ToolUseBlock(id="w1", name="probe", input={})])],
        [Message.tool_results([ToolResultBlock(tool_use_id="w1", output="ok")])],
        _ask(2, "b-rule"),
        _ask(3, "a-rule", answers=["text"]),
        _ask(4, "a-rule", "b-rule"),
        [Message.assistant_text("done")],
    ]
    before = _wire(_hide_pointer_asks(history))
    for tail in tails:
        history = [*history, *tail]
        after = _wire(_hide_pointer_asks(history))
        assert after[: len(before)] == before
        before = after


# ── the loop, both turn paths ───────────────────────────────────────────────────


def _sent_tool_ids(request: list[Message]) -> set[str]:
    return {b.id for m in request for b in m.blocks if isinstance(b, ToolUseBlock)}


def _first_request_with(seen: list[list[Message]], call_id: str) -> list[Message]:
    """The first request that carried the tool call ``call_id``: the main call that handed its
    answer back."""
    return next(request for request in seen if call_id in _sent_tool_ids(request))


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_model_is_not_handed_its_held_rule_asks_but_the_session_keeps_them(
    streamed: bool,
) -> None:
    script = [_plan_call(_steps("a", "b")), _judge_ok(), *_first_reads()]
    script += [_probe_call(1), _rule_call(3, "a-rule"), _rule_call(4, "b-rule"), _probe_call(2)]
    script += [_done(), _review_ok()]
    session, loop, probe, _stop = await _turn(script, streamed)
    assert probe.runs == 2
    # The record: three first reads and two pointers, as before this change.
    assert _kinds(_rule_results(session)) == ["text"] * 3 + ["pointer"] * 2
    held = {b.id for m in session.messages for b in m.blocks if isinstance(b, ToolUseBlock)}
    assert {"rule-3", "rule-4"} <= held
    # The request that handed back the second probe: neither held ask's call, one line for each,
    # and the first reads, which carried the texts, still in it.
    request = _first_request_with(loop.provider.seen, "probe-2")
    ids = _sent_tool_ids(request)
    assert {"rule-3", "rule-4"}.isdisjoint(ids)
    assert {"rule-0", "rule-1", "rule-2", "probe-1", "probe-2"} <= ids
    text = "\n".join(m.text for m in request)
    assert "You asked read_rule for a-rule: it is already in your context" in text
    assert "You asked read_rule for b-rule: it is already in your context" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True], ids=["buffered", "streaming"])
async def test_the_lookup_bound_still_counts_the_asks_it_does_not_send(streamed: bool) -> None:
    """ADR-0399 reads the calls and answers, not the request: a cycle through held rules is still
    refused from its third completion and stops at the fifth refusal, and the refusals, which
    say why the calls did not run, are sent."""
    rules = ("a-rule", "b-rule", "c-rule")
    script = [_plan_call(_steps("a", "b")), _judge_ok(), *_first_reads()]
    script += [_rule_call(n, rules[n % 3]) for n in range(3, 40)]
    script += [_done(), _review_ok()]
    session, loop, _probe, stop_reason = await _turn(script, streamed)
    assert stop_reason == "stuck"
    assert _kinds(_rule_results(session)) == ["text"] * 3 + ["pointer"] * 2 + ["refused"] * 5
    ids = _sent_tool_ids(_first_request_with(loop.provider.seen, "rule-8"))
    assert {"rule-3", "rule-4"}.isdisjoint(ids)
    assert {"rule-0", "rule-1", "rule-2", "rule-5", "rule-6", "rule-7", "rule-8"} <= ids
