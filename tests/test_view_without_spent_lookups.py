"""ADR-0396: a rule lookup that sent no text leaves the request once the model has gone on past it.

Measured 2026-11-19 on ten worker sessions of a host framework, from transcript metadata only:
877 of 1,213 repeat answers came in runs where the model asked a rule again between real steps,
and every earlier ask stayed in the conversation for it to follow. The view a provider call sends
drops those asks; the session keeps them, so the ask count still reads them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.agent.loop import _LOOKUP_UNEXECUTED, AgentLoop, _without_spent_lookups
from zakcode.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import Rule, RuleRegistry
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.tools.builtins.read_rule import ALREADY_READ, JUST_ASKED, PINNED, ReadRuleTool
from zakcode.usage import Usage

TEXT = "# a-rule\nDo the a thing.\n[end of rule 'a-rule']"
POINTER = ALREADY_READ.format(n=1, name="a-rule", step="", summary=" Its summary: A")
SPENT = {
    "pointer": POINTER,
    "just_asked": JUST_ASKED.format(n=2, name="a-rule", step=""),
    "pinned": PINNED.format(n=1, name="a-rule", step=""),
    "lookup_loop": _LOOKUP_UNEXECUTED.format(names="a-rule"),
}


def _ask(call_id: str, *, text: str = "") -> Message:
    blocks: list = [TextBlock(text=text)] if text else []
    blocks.append(ToolUseBlock(id=call_id, name="read_rule", input={"name": "a-rule"}))
    return Message(role="assistant", blocks=blocks)


def _work(call_id: str) -> Message:
    return Message(role="assistant", blocks=[ToolUseBlock(id=call_id, name="noop", input={})])


def _answer(call_id: str, output: str, *, is_error: bool = False) -> Message:
    return Message.tool_results(
        [ToolResultBlock(tool_use_id=call_id, output=output, is_error=is_error)]
    )


def _outputs(messages: list[Message]) -> list[str]:
    return [b.output for m in messages for b in m.blocks if isinstance(b, ToolResultBlock)]


# ── the view ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("answer", sorted(SPENT))
def test_an_ask_that_sent_no_text_leaves_once_the_model_goes_on(answer: str) -> None:
    messages = [
        Message.user("go"),
        _ask("c1"),
        _answer("c1", TEXT),
        _work("w1"),
        _answer("w1", "ok"),
        _ask("c2", text="Checking the rule again."),
        _answer("c2", SPENT[answer]),
        _work("w2"),
        _answer("w2", "ok"),
    ]
    view = _without_spent_lookups(messages)
    assert view == [*messages[:5], *messages[7:]]
    assert _outputs(view) == [TEXT, "ok", "ok"]  # the rule's text stays
    assert len(messages) == 9 and _outputs(messages)[2] == SPENT[answer]  # stored, untouched


def test_positive_control_the_asks_since_the_last_step_stay() -> None:
    """The model sees the answers to its latest calls: nothing goes before it has moved on."""
    messages = [
        Message.user("go"),
        _ask("c1"),
        _answer("c1", TEXT),
        _work("w1"),
        _answer("w1", "ok"),
        _ask("c2"),
        _answer("c2", POINTER),
        _ask("c3"),
        _answer("c3", POINTER),
    ]
    assert _without_spent_lookups(messages) is messages


def test_a_rule_text_and_an_error_stay() -> None:
    """Only an answer that sent nothing goes: the text, and a not-found error that named other
    rules, told the model something."""
    messages = [
        Message.user("go"),
        _ask("c1"),
        _answer("c1", TEXT),
        _ask("c2"),
        _answer("c2", "No rule named 'zzz'. Closest: 'a-rule'.", is_error=True),
        _work("w1"),
        _answer("w1", "ok"),
    ]
    assert _without_spent_lookups(messages) is messages


def test_another_tools_output_that_reads_like_a_pointer_stays() -> None:
    messages = [
        Message.user("go"),
        _work("w1"),
        _answer("w1", POINTER),
        _work("w2"),
        _answer("w2", "ok"),
    ]
    assert _without_spent_lookups(messages) is messages


def test_a_mixed_turn_keeps_its_other_calls_and_the_stored_turn_is_untouched() -> None:
    mixed = Message(
        role="assistant",
        blocks=[
            TextBlock(text="Rule, then the edit."),
            ToolUseBlock(id="c2", name="read_rule", input={"name": "a-rule"}),
            ToolUseBlock(id="w2", name="noop", input={}),
        ],
    )
    results = Message.tool_results(
        [ToolResultBlock(tool_use_id="c2", output=POINTER), ToolResultBlock(tool_use_id="w2")]
    )
    messages = [Message.user("go"), _ask("c1"), _answer("c1", TEXT), mixed, results]
    messages += [_work("w3"), _answer("w3", "ok")]
    view = _without_spent_lookups(messages)
    assert [type(b).__name__ for b in view[3].blocks] == ["TextBlock", "ToolUseBlock"]
    assert view[3].tool_uses[0].id == "w2"
    assert [b.tool_use_id for b in view[4].blocks] == ["w2"]
    assert len(mixed.blocks) == 3 and len(results.blocks) == 2  # copies, not edits
    assert view[3].created_at == mixed.created_at


def test_call_ids_that_repeat_across_turns_pair_by_position() -> None:
    """A provider can number its calls per turn: of three asks all called ``c0``, the first sent
    the text, the second a pointer and the third the text again (refresh), and only the second
    goes."""
    messages = [
        Message.user("go"),
        _ask("c0"),
        _answer("c0", TEXT),
        _work("w0"),
        _answer("w0", "ok"),
        _ask("c0"),
        _answer("c0", POINTER),
        _work("w0"),
        _answer("w0", "ok"),
        _ask("c0"),
        _answer("c0", TEXT),
        _work("w0"),
        _answer("w0", "ok"),
    ]
    assert _without_spent_lookups(messages) == [*messages[:5], *messages[7:]]


def test_a_text_answer_is_going_on_and_the_nudge_stays() -> None:
    nudge = Message.user("You have asked read_rule for a-rule several times in a row.")
    messages = [
        Message.user("go"),
        _ask("c1"),
        _answer("c1", SPENT["lookup_loop"]),
        nudge,
        Message.assistant_text("Done for now."),
        Message.user("continue"),
    ]
    assert _without_spent_lookups(messages) == [messages[0], nudge, *messages[4:]]


def test_every_call_in_the_view_keeps_its_answer() -> None:
    messages: list[Message] = [Message.user("go"), _ask("t"), _answer("t", TEXT)]
    for i in range(6):
        messages += [_ask(f"p{i}"), _answer(f"p{i}", POINTER)]
        messages += [_work(f"w{i}"), _answer(f"w{i}", "ok")]
    view = _without_spent_lookups(messages)
    assert _outputs(view) == [TEXT] + ["ok"] * 6
    for i, message in enumerate(view):
        if message.role != "assistant":
            continue
        assert i + 1 < len(view) and view[i + 1].role == "tool"
        assert {u.id for u in message.tool_uses} == {
            b.tool_use_id for b in view[i + 1].blocks if isinstance(b, ToolResultBlock)
        }


# ── through the loop, both turn paths ─────────────────────────────────────────


class _Noop(Tool):
    spec = ToolSpec(name="noop", description="does nothing")

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok("ok")


class _Recording(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.sent: list[list[str]] = []

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        self.sent.append(_outputs(list(messages)))
        return self._results[min(len(self.sent) - 1, len(self._results) - 1)]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=32768)


def _call(call_id: str, name: str) -> LLMResult:
    args = {"name": "a-rule"} if name == "read_rule" else {}
    return LLMResult(
        text="", tool_calls=[ToolCall(id=call_id, name=name, arguments=args)], usage=Usage()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_model_stops_seeing_its_spent_asks_and_the_count_goes_on(
    tmp_path: Path, streaming: bool
) -> None:
    registry = RuleRegistry()
    registry.add(Rule("a-rule", "# A\nDo the a thing.", Path("/x/a-rule.md")))
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    tools.register(_Noop())
    provider = _Recording(
        [
            _call("c1", "read_rule"),  # 1: the text
            _call("w1", "noop"),
            _call("c2", "read_rule"),  # 3: ask 1
            _call("w2", "noop"),  # 4: sees ask 1, its latest
            _call("c3", "read_rule"),  # 5: ask 2, counted from the stored conversation
            _call("w3", "noop"),
            LLMResult(text="done", usage=Usage()),
        ]
    )
    loop = AgentLoop(provider, tools, Session(cwd=str(tmp_path), model="t"), rule_registry=registry)
    if streaming:
        _ = [ev async for ev in loop.astream_turn("go")]
    else:
        await loop.arun_turn("go")
    stored = _outputs(loop.session.messages)
    pointers = [o[:21] for o in stored if o.startswith("[already")]
    assert pointers == ["[already read; ask 1]", "[already read; ask 2]"]
    assert len(provider.sent) == 7
    fourth, fifth, last = provider.sent[3], provider.sent[4], provider.sent[6]
    assert any(o.startswith("[already read; ask 1]") for o in fourth)
    assert not any(o.startswith("[already read") for o in fifth)
    assert not any(o.startswith("[already read") for o in last)
    assert sum(o.startswith("# a-rule\n") for o in last) == 1  # the text stays
