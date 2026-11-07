"""Tests for the ``read_rule`` tool — Vinheim Lever A chunk 2.

The tool is the retrieval half of ``lean_rules``: ``render_index()`` names every rule without
its body, and this is how the model fetches one. So the tests pin BOTH halves — the tool
returns the right body, and the index actually points at the tool.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import Rule, RuleRegistry, discover_rules
from zakcode.session.store import Session
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.read_rule import ALREADY_READ, MAX_RULE_BODY_CHARS, ReadRuleTool
from zakcode.usage import Usage


def _registry() -> RuleRegistry:
    reg = RuleRegistry()
    reg.add(Rule("alpha-rule", "# Alpha\nAlways do the alpha thing.", Path("/x/alpha-rule.md")))
    reg.add(Rule("beta-rule", "# Beta\nNever do the beta thing.", Path("/x/beta-rule.md")))
    return reg


def _ctx(registry: object | None) -> ToolContext:
    return ToolContext(workspace_root=Path("/tmp"), rule_registry=registry)


def _run(args: dict, registry: object | None):
    return asyncio.run(ReadRuleTool().execute(args, _ctx(registry)))


def test_returns_the_named_rule_body() -> None:
    res = _run({"name": "alpha-rule"}, _registry())
    assert not res.is_error
    assert "Always do the alpha thing." in res.output
    # Only the requested rule — the whole point is NOT paying for every body.
    assert "beta thing" not in res.output


def test_name_is_case_insensitive() -> None:
    """A capitalisation slip while copying a name out of prose is not a missing rule."""
    res = _run({"name": "Alpha-Rule"}, _registry())
    assert not res.is_error
    assert "Always do the alpha thing." in res.output


def test_unknown_rule_lists_the_available_names() -> None:
    res = _run({"name": "nope"}, _registry())
    assert res.is_error
    assert "no rule named" in res.output
    # The fix must be actionable: name what IS available, or the model cannot recover.
    assert "alpha-rule" in (res.fix or "") and "beta-rule" in (res.fix or "")


def test_missing_registry_is_a_clean_error_not_a_crash() -> None:
    res = _run({"name": "alpha-rule"}, None)
    assert res.is_error
    assert "not enabled" in res.output


def test_blank_name_rejected() -> None:
    for bad in ({"name": "   "}, {"name": ""}, {}):
        res = _run(bad, _registry())
        assert res.is_error
        assert "required" in res.output


def test_empty_body_is_an_error_not_an_empty_success() -> None:
    """A stub rule must not read as a successful empty answer."""
    reg = RuleRegistry()
    reg.add(Rule("stub", "   \n  ", Path("/x/stub.md")))
    res = _run({"name": "stub"}, reg)
    assert res.is_error
    assert "empty" in res.output


def test_oversized_body_is_truncated_and_says_so() -> None:
    reg = RuleRegistry()
    reg.add(Rule("big", "x" * (MAX_RULE_BODY_CHARS + 5000), Path("/x/big.md")))
    res = _run({"name": "big"}, reg)
    assert not res.is_error
    assert "truncated" in res.output
    # Bounded: the announcement adds a little, but nowhere near the original overflow.
    assert len(res.output) < MAX_RULE_BODY_CHARS + 200


def test_tool_is_read_only_and_parallel_safe() -> None:
    """It reads an in-memory registry and no model-supplied path, so it must never prompt."""
    spec = ReadRuleTool.spec
    assert spec.name == "read_rule"
    assert spec.concurrency.value == "read_only_safe"


def test_index_points_at_the_tool_on_a_real_rules_tree(tmp_path: Path) -> None:
    """The retrieval path must be self-documenting: the index names the tool that fetches.

    Second verification outcome. Without this the index tells the model a rule
    exists but not how to read it cheaply, and the lean path silently falls back to
    generic file reads keyed on a path.
    """
    rules_dir = tmp_path / ".claude" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "sample-rule.md").write_text("# Sample\nBODY-MARKER\n", encoding="utf-8")
    reg, errors = discover_rules(tmp_path)
    assert not errors
    index = reg.render_index()
    assert "read_rule" in index
    assert "sample-rule" in index
    # The index is an INDEX: it names rules, it does not inline their bodies.
    assert "BODY-MARKER" not in index
    # And the body IS reachable through the tool it advertises — the two halves agree.
    res = _run({"name": "sample-rule"}, reg)
    assert not res.is_error and "BODY-MARKER" in res.output


# --- ADR-0372: a repeat of a rule still in the conversation is answered with a pointer ---------


def _ctx_seeing(*outputs: str) -> ToolContext:
    def held(text: str) -> bool:
        return any(text in output for output in outputs)

    return ToolContext(workspace_root=Path("/tmp"), rule_registry=_registry(), in_context=held)


def _body(name: str) -> str:
    return asyncio.run(ReadRuleTool().execute({"name": name}, _ctx(_registry()))).output


def test_a_repeat_of_a_rule_still_in_context_is_a_pointer() -> None:
    res = asyncio.run(
        ReadRuleTool().execute({"name": "alpha-rule"}, _ctx_seeing(_body("alpha-rule")))
    )
    assert not res.is_error
    assert res.output == ALREADY_READ.format(name="alpha-rule")
    assert "Always do the alpha thing." not in res.output


def test_a_pointer_is_not_verbatim_and_a_body_is() -> None:
    held = _ctx_seeing(_body("alpha-rule"))
    assert not asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, held)).verbatim
    assert asyncio.run(ReadRuleTool().execute({"name": "beta-rule"}, held)).verbatim


def test_another_rule_still_gets_its_body() -> None:
    res = asyncio.run(
        ReadRuleTool().execute({"name": "beta-rule"}, _ctx_seeing(_body("alpha-rule")))
    )
    assert "Never do the beta thing." in res.output


def test_a_bare_context_always_delivers() -> None:
    """No seam (a sub-tool, a test double): never claim the text is in context."""
    for _ in range(2):
        assert "Always do the alpha thing." in _body("alpha-rule")


def test_a_rule_edited_since_it_was_read_is_delivered_again() -> None:
    """The check compares the text, so a changed rule is not 'unchanged'."""
    stale = "# alpha-rule\n# Alpha\nAn older wording."
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, _ctx_seeing(stale)))
    assert "Always do the alpha thing." in res.output


def _read_after_edit(content: str) -> str:
    """Read alpha-rule as ``content`` while its full text from an earlier read is in context."""
    held = _body("alpha-rule")
    edited = RuleRegistry()
    edited.add(Rule("alpha-rule", content, Path("/x/alpha-rule.md")))
    ctx = ToolContext(
        workspace_root=Path("/tmp"), rule_registry=edited, in_context=lambda text: text in held
    )
    return asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, ctx)).output


def test_a_rule_whose_last_lines_were_deleted_is_delivered_again() -> None:
    """The shorter text was a prefix of the older one, so the check read it as unchanged while the
    model still held the deleted lines. The closing line makes the text end where the rule does."""
    assert _read_after_edit("# Alpha") == "# alpha-rule\n# Alpha\n[end of rule 'alpha-rule']"


def test_a_rule_whose_first_lines_were_deleted_is_delivered_again() -> None:
    text = "# alpha-rule\nAlways do the alpha thing.\n[end of rule 'alpha-rule']"
    assert _read_after_edit("Always do the alpha thing.") == text


def test_a_body_ends_with_its_closing_line() -> None:
    assert _body("alpha-rule").endswith("\nAlways do the alpha thing.\n[end of rule 'alpha-rule']")


def test_a_read_after_the_pointer_gets_the_body() -> None:
    """Once per rule: a session that asks again after the pointer wants the text."""
    seen = _ctx_seeing(_body("alpha-rule"), ALREADY_READ.format(name="alpha-rule"))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert "Always do the alpha thing." in res.output
    assert res.verbatim


def test_another_rules_pointer_does_not_count() -> None:
    seen = _ctx_seeing(_body("alpha-rule"), ALREADY_READ.format(name="beta-rule"))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == ALREADY_READ.format(name="alpha-rule")


class _Scripted(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, **kw) -> LLMResult:
        i = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[i]

    def count_tokens(self, messages, *, system=None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


def _read(call_id: str, name: str = "alpha-rule") -> LLMResult:
    return LLMResult(
        text="",
        tool_calls=[ToolCall(id=call_id, name="read_rule", arguments={"name": name})],
        usage=Usage(total_tokens=1),
    )


def _loop(tmp_path: Path, *ids: str) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(ReadRuleTool())
    results = [_read(i) for i in ids] + [LLMResult(text="done", usage=Usage(total_tokens=1))]
    return AgentLoop(
        _Scripted(results),
        registry,
        Session(cwd=str(tmp_path), model="t"),
        rule_registry=_registry(),
    )


def _outputs(loop: AgentLoop) -> list[str]:
    return [
        block.output
        for message in loop.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_answers_the_second_read_with_the_pointer(
    tmp_path: Path, streaming: bool
) -> None:
    loop = _loop(tmp_path, "r1", "r2")
    if streaming:
        async for _event in loop.astream_turn("go"):
            pass
    else:
        await loop.arun_turn("go")
    first, second = _outputs(loop)
    assert "Always do the alpha thing." in first
    assert second == ALREADY_READ.format(name="alpha-rule")


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_gives_the_pointer_once_then_the_body(
    tmp_path: Path, streaming: bool
) -> None:
    """The trial's loop: after one body every ask got the pointer, and the session kept asking.

    Four identical reads in a row, as the trial saw them; the doom-loop guard declines one of
    them, so the assertions count pointers and read the last answer instead of fixing positions.
    """
    loop = _loop(tmp_path, "r1", "r2", "r3", "r4")
    if streaming:
        async for _event in loop.astream_turn("go"):
            pass
    else:
        await loop.arun_turn("go")
    outputs = _outputs(loop)
    pointer = ALREADY_READ.format(name="alpha-rule")
    assert "Always do the alpha thing." in outputs[0]
    assert outputs.count(pointer) == 1
    assert "Always do the alpha thing." in outputs[-1]


@pytest.mark.asyncio
async def test_a_compaction_that_took_the_body_delivers_it_again(tmp_path: Path) -> None:
    """The ADR-0080 incident: a pointer to text the compaction removed is a false pointer."""
    loop = _loop(tmp_path, "r1", "r2")
    await loop.arun_turn("go")
    loop._adopt_compacted([Message.user("summary")], summary="summary", trigger="auto")
    loop.provider._results = [_read("r3"), LLMResult(text="done", usage=Usage(total_tokens=1))]
    loop.provider.calls = 0
    await loop.arun_turn("again")
    assert "Always do the alpha thing." in _outputs(loop)[-1]


@pytest.mark.asyncio
async def test_an_elided_body_is_delivered_again(tmp_path: Path) -> None:
    """An elision keeps the block and shortens its output: the words are gone, so is the pointer."""
    loop = _loop(tmp_path, "r1")
    await loop.arun_turn("go")
    for message in loop.session.messages:
        for block in message.blocks:
            if isinstance(block, ToolResultBlock):
                block.output = "[elided]"
    loop.provider._results = [_read("r2"), LLMResult(text="done", usage=Usage(total_tokens=1))]
    loop.provider.calls = 0
    await loop.arun_turn("again")
    assert "Always do the alpha thing." in _outputs(loop)[-1]


@pytest.mark.asyncio
async def test_a_body_kept_in_the_tail_still_counts(tmp_path: Path) -> None:
    """A read in the kept tail survives a compaction, so the pointer stays true."""
    loop = _loop(tmp_path, "r1")
    await loop.arun_turn("go")
    kept = list(loop.session.messages)
    loop._adopt_compacted([Message.user("summary"), *kept], summary="summary", trigger="auto")
    loop.provider._results = [_read("r2"), LLMResult(text="done", usage=Usage(total_tokens=1))]
    loop.provider.calls = 0
    await loop.arun_turn("again")
    assert _outputs(loop)[-1] == ALREADY_READ.format(name="alpha-rule")


_LONG = "\n".join(f"Step {i}: do the alpha thing." for i in range(20))


def _loop_reading(
    tmp_path: Path, content: str, *ids: str, session: Session | None = None
) -> AgentLoop:
    """A loop whose alpha-rule reads ``content``. Given ``session``, a new loop over that
    conversation, as a restart, a resume or a served turn builds one after a rule was edited."""
    rules = RuleRegistry()
    rules.add(Rule("alpha-rule", content, Path("/x/alpha-rule.md")))
    tools = ToolRegistry()
    tools.register(ReadRuleTool())
    results = [_read(i) for i in ids] + [LLMResult(text="done", usage=Usage(total_tokens=1))]
    return AgentLoop(
        _Scripted(results),
        tools,
        session or Session(cwd=str(tmp_path), model="t"),
        rule_registry=rules,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "edited",
    [
        _LONG.replace("Step 19: do the alpha", "Step 19: do the beta"),
        _LONG.rsplit("\n", 1)[0],
        _LONG.replace("Step 3:", "step 3:"),
    ],
    ids=["edited near its end", "last line deleted", "letter case only"],
)
async def test_a_rule_edited_between_two_loops_is_delivered_again(
    tmp_path: Path, edited: str
) -> None:
    first = _loop_reading(tmp_path, _LONG, "r1")
    await first.arun_turn("go")
    again = _loop_reading(tmp_path, edited, "r2", session=first.session)
    await again.arun_turn("again")
    assert _outputs(again)[-1] == f"# alpha-rule\n{edited}\n[end of rule 'alpha-rule']"


@pytest.mark.asyncio
async def test_an_unchanged_rule_in_a_new_loop_is_a_pointer(tmp_path: Path) -> None:
    first = _loop_reading(tmp_path, _LONG, "r1")
    await first.arun_turn("go")
    again = _loop_reading(tmp_path, _LONG, "r2", session=first.session)
    await again.arun_turn("again")
    assert _outputs(again)[-1] == ALREADY_READ.format(name="alpha-rule")


@pytest.mark.asyncio
async def test_a_body_with_a_hook_note_after_it_still_counts(tmp_path: Path) -> None:
    """A PostToolUse hook's message rides after the result as a ``[hook]`` line, so the stored
    output holds the rule's text and more."""
    loop = _loop(tmp_path, "r1")
    await loop.arun_turn("go")
    for message in loop.session.messages:
        for block in message.blocks:
            if isinstance(block, ToolResultBlock):
                block.output += "\n[hook] checked by the workspace's PostToolUse hook"
    loop.provider._results = [_read("r2"), LLMResult(text="done", usage=Usage(total_tokens=1))]
    loop.provider.calls = 0
    await loop.arun_turn("again")
    assert _outputs(loop)[-1] == ALREADY_READ.format(name="alpha-rule")


@pytest.mark.asyncio
async def test_a_rule_quoted_outside_a_tool_result_is_still_delivered(tmp_path: Path) -> None:
    """A summary or a message that quotes a rule is not a read of it; only a tool result is."""
    loop = _loop(tmp_path, "r1")
    text = (await ReadRuleTool().execute({"name": "alpha-rule"}, _ctx(_registry()))).output
    loop.session.messages.extend([Message.system(f"Summary.\n{text}"), Message.user(text)])
    await loop.arun_turn("go")
    assert "Always do the alpha thing." in _outputs(loop)[-1]
