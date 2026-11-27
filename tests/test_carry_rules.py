"""ADR-0386: a compaction carries the rule texts the session read, right after the summary.

Measured 2026-11-17 on ten worker sessions of a host framework: under ``lean_rules`` a session
fetches a rule's text with ``read_rule``, a compaction summarizes the region that holds it, and the
session fetches it again; one made 81 ``read_rule`` calls of its 113 after a compaction. The loop
now counts every ask per rule and, when a compaction installs a summary, puts the texts of the
most-asked rules right after it, within a budget, where the pointer's check (ADR-0372) sees them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.compact import (
    CARRIED_RULES_MARKER,
    SUMMARY_MARKER,
    CompactionConfig,
    Compactor,
    is_carried_rules,
    is_summary,
)
from zakcode.agent.loop import _CARRY_RULES_FRACTION, AgentLoop
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import Rule, RuleRegistry
from zakcode.session.store import Session, TurnEnd
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.read_rule import ReadRuleTool, rule_text
from zakcode.usage import Usage


def _registry() -> RuleRegistry:
    reg = RuleRegistry()
    reg.add(Rule("alpha-rule", "# Alpha\nAlways do the alpha thing.", Path("/x/alpha-rule.md")))
    reg.add(Rule("beta-rule", "# Beta\nNever do the beta thing.", Path("/x/beta-rule.md")))
    reg.add(Rule("gamma-rule", "# Gamma\nCheck the gamma gauge first.", Path("/x/gamma-rule.md")))
    reg.add(
        Rule("pinned-rule", "# Pinned\nAlways on.", Path("/x/pinned-rule.md"), always_apply=True)
    )
    # Sized for the budget tests: about 330 tokens each at the tail budget's 3 characters a token.
    reg.add(Rule("mid-one", "# Mid one\n" + "The first middle rule.\n" * 43, Path("/x/mid-one.md")))
    reg.add(Rule("mid-two", "# Mid two\n" + "The other middle rule.\n" * 43, Path("/x/mid-two.md")))
    reg.add(Rule("big-rule", "# Big\n" + "A rule far over the budget.\n" * 300, Path("/x/big.md")))
    return reg


class _Scripted(Provider):
    """Canned results in order; records the messages of every call, the summarizer's too."""

    def __init__(self, results: list[LLMResult], *, window: int = 131_072) -> None:
        self.results = results
        self.window = window
        self.seen: list[list[Message]] = []

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.seen.append(list(messages))
        return self.results[min(len(self.seen), len(self.results)) - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=self.window)


def _read(call_id: str, name: str) -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id=call_id, name="read_rule", arguments={"name": name})],
        usage=Usage(total_tokens=1),
    )


_DONE = LLMResult(text="done", usage=Usage(total_tokens=1))


def _loop(tmp_path: Path, provider: _Scripted) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    return AgentLoop(
        provider,
        tools,
        Session(cwd=str(tmp_path), model="t"),
        workspace_root=tmp_path,
        rule_registry=_registry(),
        compactor=Compactor(CompactionConfig(preserve_recent=2)),
    )


def _filler(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out.append(Message.user(f"question {i}"))
        out.append(Message.assistant_text(f"answer {i}"))
    return out


async def _turn(loop: AgentLoop, text: str, *, streaming: bool = False) -> None:
    if streaming:
        async for _event in loop.astream_turn(text):
            pass
    else:
        await loop.arun_turn(text)


async def _read_then_compact(
    tmp_path: Path, *names: str, streaming: bool = False, window: int = 131_072
) -> AgentLoop:
    """A turn that reads ``names`` in order, filler after it so every read is in the summarized
    region, then a compaction that installs a summary."""
    provider = _Scripted([*(_read(f"r{i}", n) for i, n in enumerate(names)), _DONE], window=window)
    loop = _loop(tmp_path, provider)
    await _turn(loop, "go", streaming=streaming)
    loop.session.messages.extend(_filler(2))
    provider.results, provider.seen = [LLMResult(text="the summary")], []
    assert await loop.compact_now(trigger="auto") is True
    return loop


def _text(name: str) -> str:
    rule = _registry().get(name)
    assert rule is not None
    text = rule_text(name, rule)
    assert text is not None
    return text


def _carried(loop: AgentLoop) -> list[Message]:
    return [m for m in loop.session.messages if is_carried_rules(m)]


def _outputs(loop: AgentLoop) -> list[str]:
    return [
        block.output
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]


def _boundaries(loop: AgentLoop) -> list[dict[str, Any]]:
    path = loop._cc_transcript_path()
    assert path is not None
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [row["compactMetadata"] for row in rows if row.get("subtype") == "compact_boundary"]


# ── the carry ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_a_compaction_carries_the_rules_the_session_read(
    tmp_path: Path, streaming: bool
) -> None:
    loop = await _read_then_compact(tmp_path, "alpha-rule", "beta-rule", streaming=streaming)
    messages = loop.session.messages
    assert is_summary(messages[0])
    (carried,) = _carried(loop)
    assert messages[1] is carried  # right after the summary, before the kept tail
    assert carried.role == "system" and carried.text.startswith(CARRIED_RULES_MARKER)
    assert _text("alpha-rule") in carried.text and _text("beta-rule") in carried.text
    assert [m.text for m in messages[2:]] == ["question 1", "answer 1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_an_ask_for_a_carried_rule_gets_the_text_then_the_pointer(
    tmp_path: Path, streaming: bool
) -> None:
    """Only the carried message holds the text, so the first ask sends it as a tool result
    (ADR-0417); the ask after it gets the pointer (ADR-0379)."""
    loop = await _read_then_compact(tmp_path, "beta-rule")
    provider = loop.provider
    assert isinstance(provider, _Scripted)
    provider.results = [_read("r8", "beta-rule"), _read("r9", "beta-rule"), _DONE]
    provider.seen = []
    await _turn(loop, "again", streaming=streaming)
    text, pointer = _outputs(loop)
    assert text == _text("beta-rule")
    assert pointer.startswith("[already read; ask 1] Rule 'beta-rule' ")


@pytest.mark.asyncio
async def test_a_session_that_read_no_rule_compacts_as_before(tmp_path: Path) -> None:
    provider = _Scripted([_DONE])
    loop = _loop(tmp_path, provider)
    await loop.arun_turn("go")
    loop.session.messages.extend(_filler(2))
    tail = loop.session.messages[-2:]
    provider.results, provider.seen = [LLMResult(text="the summary")], []
    assert await loop.compact_now(trigger="auto") is True
    messages = loop.session.messages
    assert len(messages) == 3 and is_summary(messages[0])
    assert messages[1:] == tail
    assert "carriedRules" not in _boundaries(loop)[0]


@pytest.mark.asyncio
async def test_the_boundary_row_says_how_many_rules_were_carried(tmp_path: Path) -> None:
    loop = await _read_then_compact(tmp_path, "alpha-rule", "beta-rule")
    (carried,) = _carried(loop)
    (row,) = _boundaries(loop)
    assert row["carriedRules"] == 2
    assert row["carriedRulesChars"] == len(carried.text)
    assert row["postMessages"] == len(loop.session.messages)


@pytest.mark.asyncio
async def test_the_summarizer_never_reads_the_carried_texts_and_one_carry_replaces_the_last(
    tmp_path: Path,
) -> None:
    loop = await _read_then_compact(tmp_path, "alpha-rule", "beta-rule")
    provider = loop.provider
    assert isinstance(provider, _Scripted)
    loop.session.messages.extend(_filler(3))
    provider.results, provider.seen = [LLMResult(text="the second summary")], []
    assert await loop.compact_now(trigger="auto") is True
    asked = "\n".join(m.text for call in provider.seen for m in call)
    assert "Always do the alpha thing." not in asked and "Never do the beta thing." not in asked
    messages = loop.session.messages
    (carried,) = _carried(loop)
    assert messages[1] is carried
    assert _text("alpha-rule") in carried.text and _text("beta-rule") in carried.text


def test_a_summary_and_its_carried_rules_alone_are_nothing_to_compact() -> None:
    async def never(_old: list[Message]) -> str:
        raise AssertionError("nothing new to fold, so nothing to summarize")

    messages = [
        Message.system(f"{SUMMARY_MARKER}\nthe summary"),
        Message.system(f"{CARRIED_RULES_MARKER} texts"),
        Message.user("question"),
        Message.assistant_text("answer"),
    ]
    compactor = Compactor(CompactionConfig(preserve_recent=2))
    result = asyncio.run(compactor.compact(messages, summarize=never))
    assert result.compacted is False


# ── which rules, in what order, within what budget ───────────────────────────────────


def _carry_for(tmp_path: Path, reads: dict[str, list[int]], *, window: int) -> Message | None:
    loop = _loop(tmp_path, _Scripted([_DONE], window=window))
    loop.session.rule_reads = {name: list(entry) for name, entry in reads.items()}
    return loop._carried_rules_message([Message.system(f"{SUMMARY_MARKER}\nthe summary")])


def _window(tokens: int) -> int:
    """The context window whose carry budget is ``tokens``."""
    return int(tokens / _CARRY_RULES_FRACTION) + 1


def test_the_most_asked_rule_goes_first_then_the_most_recent(tmp_path: Path) -> None:
    carried = _carry_for(
        tmp_path,
        {"alpha-rule": [1, 5], "beta-rule": [2, 1], "gamma-rule": [1, 9]},
        window=131_072,
    )
    assert carried is not None
    order = [carried.text.index(_text(n)) for n in ("beta-rule", "gamma-rule", "alpha-rule")]
    assert order == sorted(order)


def test_the_budget_keeps_the_most_asked_rule_and_drops_the_other(tmp_path: Path) -> None:
    # Room for the head and one middle rule (~330 tokens each), not for two.
    window = _window(600)
    first = _carry_for(tmp_path, {"mid-one": [3, 1], "mid-two": [1, 2]}, window=window)
    assert first is not None
    assert _text("mid-one") in first.text and _text("mid-two") not in first.text
    swapped = _carry_for(tmp_path, {"mid-one": [1, 1], "mid-two": [3, 2]}, window=window)
    assert swapped is not None
    assert _text("mid-two") in swapped.text and _text("mid-one") not in swapped.text


def test_a_rule_over_the_budget_is_skipped_and_a_smaller_one_after_it_still_goes(
    tmp_path: Path,
) -> None:
    carried = _carry_for(tmp_path, {"big-rule": [9, 2], "alpha-rule": [1, 1]}, window=_window(600))
    assert carried is not None
    assert _text("alpha-rule") in carried.text and "A rule far over the budget." not in carried.text


def test_pinned_rules_and_rules_still_in_the_tail_are_not_carried(tmp_path: Path) -> None:
    loop = _loop(tmp_path, _Scripted([_DONE]))
    loop.session.rule_reads = {"pinned-rule": [5, 2], "alpha-rule": [3, 1]}
    kept = Message.tool_results([ToolResultBlock(tool_use_id="t1", output=_text("alpha-rule"))])
    head = Message.system(f"{SUMMARY_MARKER}\nthe summary")
    assert loop._carried_rules_message([head, kept]) is None
    carried = loop._carried_rules_message([head])
    assert carried is not None
    assert _text("alpha-rule") in carried.text and "Always on." not in carried.text


def test_no_reads_carry_nothing_and_a_rule_gone_from_the_registry_is_passed_over(
    tmp_path: Path,
) -> None:
    assert _carry_for(tmp_path, {}, window=131_072) is None
    carried = _carry_for(tmp_path, {"no-such-rule": [4, 1], "alpha-rule": [1, 2]}, window=131_072)
    assert carried is not None
    assert _text("alpha-rule") in carried.text and "no-such-rule" not in carried.text


# ── the count ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_counts_every_ask_and_the_latest(tmp_path: Path, streaming: bool) -> None:
    asks = [_read("r1", "alpha-rule"), _read("r2", "beta-rule"), _read("r3", "alpha-rule")]
    loop = _loop(tmp_path, _Scripted([*asks, _DONE]))
    await _turn(loop, "go", streaming=streaming)
    assert loop.session.rule_reads == {"alpha-rule": [2, 3], "beta-rule": [1, 2]}


def test_the_tool_notes_found_rules_by_their_own_name_and_nothing_else() -> None:
    noted: list[str] = []
    reg = _registry()
    reg.add(Rule("empty-rule", " ", Path("/x/empty-rule.md")))
    seen = ToolContext(workspace_root=Path("/tmp"), rule_registry=reg, note_rule_read=noted.append)
    pointed = ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=reg,
        note_rule_read=noted.append,
        in_context=lambda _text: True,
    )
    tool = ReadRuleTool()
    asyncio.run(tool.execute({"name": "ALPHA-RULE"}, seen))  # the text, by another spelling
    asyncio.run(tool.execute({"name": "beta-rule"}, pointed))  # a pointer is an ask too
    asyncio.run(tool.execute({"name": "no-such-rule"}, seen))
    asyncio.run(tool.execute({"name": "empty-rule"}, seen))
    assert noted == ["alpha-rule", "beta-rule"]


def test_the_count_is_kept_with_the_session(tmp_path: Path) -> None:
    session = Session(cwd=str(tmp_path), model="t", rule_reads={"alpha-rule": [2, 7]})
    again = Session.model_validate_json(session.model_dump_json())
    assert again.rule_reads == {"alpha-rule": [2, 7]}
    assert Session(cwd=str(tmp_path), model="t").rule_reads == {}


# ── turn ends stay in their place ────────────────────────────────────────────────────


def test_a_turn_end_just_before_the_kept_tail_goes_when_rules_were_carried() -> None:
    """Ten messages, the first five summarized, rules carried: the kept five follow the summary
    and the carried message. The end after the fifth message belonged to a turn told only in the
    summary; the end after the sixth moves to after the summary, the carry and that message."""
    session = Session(cwd="/w", model="t")
    session.messages[:] = [Message.user(f"m{i}") for i in range(10)]
    session.turn_ends = [TurnEnd(after=a, stop_reason="stuck") for a in (5, 6)]
    session.messages[:] = [
        Message.system(f"{SUMMARY_MARKER}\nthe summary"),
        Message.system(f"{CARRIED_RULES_MARKER} texts"),
        *session.messages[5:],
    ]
    session.shift_turn_ends(10, summarized=True, head=2)
    assert [e.after for e in session.turn_ends] == [3]


@pytest.mark.asyncio
async def test_the_loop_counts_the_carried_rules_ahead_of_the_kept_tail(tmp_path: Path) -> None:
    """Through the loop: the end of a turn told only in the summary goes, and an end inside the
    kept tail stays after the summary, the carried rules and the message before it."""
    provider = _Scripted([_read("r0", "alpha-rule"), _DONE])
    loop = _loop(tmp_path, provider)
    await loop.arun_turn("go")
    loop.session.messages.extend(_filler(2))
    n = len(loop.session.messages)  # the kept tail is the last two
    loop.session.turn_ends = [TurnEnd(after=a, stop_reason="stuck") for a in (n - 2, n - 1)]
    provider.results, provider.seen = [LLMResult(text="the summary")], []
    assert await loop.compact_now(trigger="auto") is True
    assert len(_carried(loop)) == 1
    assert [e.after for e in loop.session.turn_ends] == [3]
