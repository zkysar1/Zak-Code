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
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import ToolContext, ToolRegistry
from zakcode.tools.builtins.read_rule import (
    ALREADY_READ,
    ASKED_AGAIN,
    MAX_RULE_BODY_CHARS,
    ReadRuleTool,
)
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


def _named_registry(*names: str) -> RuleRegistry:
    reg = RuleRegistry()
    for rule_name in names:
        reg.add(Rule(rule_name, f"# {rule_name}\nBody of {rule_name}.", Path(f"/x/{rule_name}.md")))
    return reg


_NAMES = ("commit-style", "review-before-merge", "test-first")


def test_near_miss_name_leads_with_the_closest_rule() -> None:
    """A name a few letters off a real rule is answered with that rule first (ADR-0380), and the
    full list still follows, so a wrong lead costs nothing."""
    res = _run({"name": "reviewed-before-merging"}, _named_registry(*_NAMES))
    assert res.is_error
    assert "no rule named 'reviewed-before-merging'" in res.output
    fix = res.fix or ""
    assert fix.startswith("Did you mean 'review-before-merge'? Use one of the available rules:")
    assert all(rule_name in fix for rule_name in _NAMES)


def test_far_name_gets_no_suggestion() -> None:
    res = _run({"name": "database-migrations"}, _named_registry(*_NAMES))
    assert res.is_error
    assert (res.fix or "").startswith("Use one of the available rules:")
    assert "Did you mean" not in (res.fix or "")


def test_shared_connective_words_alone_are_not_a_suggestion() -> None:
    """'read-before-edit' scores 0.63 against 'review-before-merge' only through the shared
    '-before-': a neighbour at the tool registry's 0.6, noise at the rule line (ADR-0380)."""
    res = _run({"name": "read-before-edit"}, _named_registry(*_NAMES))
    assert "Did you mean" not in (res.fix or "")


def test_two_close_rules_are_both_named_closest_first() -> None:
    reg = _named_registry("release-checklist", "release-checklists", "commit-style")
    res = _run({"name": "release-check-list"}, reg)
    assert (res.fix or "").startswith("Did you mean 'release-checklist' or 'release-checklists'?")


def test_closest_rule_ignores_case_and_answers_the_registered_spelling() -> None:
    """Both sides are compared lowercased: a shouted guess still finds the rule, and a rule
    registered in capitals is offered in its own spelling."""
    res = _run({"name": "REVIEWED-BEFORE-MERGE"}, _named_registry(*_NAMES))
    assert (res.fix or "").startswith("Did you mean 'review-before-merge'?")
    res = _run(
        {"name": "reviewed-before-merge"}, _named_registry("REVIEW-BEFORE-MERGE", "test-first")
    )
    assert (res.fix or "").startswith("Did you mean 'REVIEW-BEFORE-MERGE'?")


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


def test_the_index_and_the_tool_say_to_read_each_rule_once() -> None:
    """ADR-0379: both told the model to read a rule when it looks relevant to the current step,
    and a rule summarised as a trigger looks relevant after every step."""
    assert "read each rule once" in ReadRuleTool.spec.description
    assert "read each rule once" in _registry().render_index()


def test_the_index_and_the_tool_tie_a_read_to_the_first_time_not_to_each_step() -> None:
    """ADR-0381: both still opened with "relevant to the current step", and in worker sessions a
    read came 4 s to 2 min after each plan update, with real work between."""
    for text in (ReadRuleTool.spec.description, _registry().render_index()):
        assert "the first time a rule" in text
        assert "at a new plan step too" in text
        assert "relevant to the current step" not in text


# --- ADR-0372, ADR-0379: every repeat of a rule still in the conversation gets a pointer ---------


def _ctx_seeing(*outputs: str, plan: TaskNetwork | None = None) -> ToolContext:
    def held(text: str) -> bool:
        return any(text in output for output in outputs)

    return ToolContext(
        workspace_root=Path("/tmp"), rule_registry=_registry(), in_context=held, task_network=plan
    )


def _body(name: str) -> str:
    return asyncio.run(ReadRuleTool().execute({"name": name}, _ctx(_registry()))).output


def _pointer(name: str = "alpha-rule", n: int = 1, summary: str = "Alpha", step: str = "") -> str:
    return ALREADY_READ.format(n=n, name=name, step=step, summary=f" Its summary: {summary}")


def test_a_repeat_of_a_rule_still_in_context_is_a_pointer() -> None:
    res = asyncio.run(
        ReadRuleTool().execute({"name": "alpha-rule"}, _ctx_seeing(_body("alpha-rule")))
    )
    assert not res.is_error
    assert res.output == _pointer()
    assert res.output.startswith("[already read; ask 1] Rule 'alpha-rule' ")
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


def test_a_read_after_the_pointer_gets_the_next_pointer_not_the_body() -> None:
    """ADR-0372 answered the ask after its pointer with the body, and the body was asked for
    again and again (one session: three rules nine times each in 21 minutes). Each ask now gets
    the next number, read back from the pointers still in the conversation."""
    seen = _ctx_seeing(_body("alpha-rule"), _pointer(n=1), _pointer(n=2))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == _pointer(n=3)
    assert "Always do the alpha thing." not in res.output
    assert not res.verbatim


def test_refresh_returns_the_body_while_it_is_in_context() -> None:
    """The deliberate way back to the text: refresh=true, never a plain repeat."""
    seen = _ctx_seeing(_body("alpha-rule"), _pointer(n=1))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule", "refresh": True}, seen))
    assert res.output == _body("alpha-rule")
    assert res.verbatim


def test_the_pointer_names_the_plan_step_to_go_on_with() -> None:
    plan = TaskNetwork(tasks=[Task(title="Write the tests")])
    seen = _ctx_seeing(_body("alpha-rule"), plan=plan)
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == _pointer(step=" to your current step, 'Write the tests'")


def test_another_rules_pointer_does_not_count() -> None:
    seen = _ctx_seeing(_body("alpha-rule"), _pointer("beta-rule", summary="Beta"))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == _pointer()


# --- ADR-0381: an ask right after the rule's own pointer gets the text --------------------------


def _ctx_after(last: str, *outputs: str) -> ToolContext:
    """A context holding ``outputs`` and, as its most recent tool result, ``last``."""
    held = (*outputs, last)

    def seen(text: str) -> bool:
        return any(text in output for output in held)

    return ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=_registry(),
        in_context=seen,
        last_tool_output=lambda: last,
    )


def _again(name: str = "alpha-rule") -> str:
    return ASKED_AGAIN.format(name=name) + "\n" + _body(name)


def _ask(ctx: ToolContext, name: str = "alpha-rule", **args: object):
    return asyncio.run(ReadRuleTool().execute({"name": name, **args}, ctx))


def test_an_ask_right_after_its_pointer_gets_the_text() -> None:
    """ADR-0379 answered it with the next pointer, and in a trial two sessions asked on until the
    exact-repeat guard refused them."""
    res = _ask(_ctx_after(_pointer(n=1), _body("alpha-rule")))
    assert res.output == _again()
    assert res.verbatim
    assert not res.is_error


def test_an_ask_right_after_a_later_pointer_gets_the_text_too() -> None:
    res = _ask(_ctx_after(_pointer(n=3), _body("alpha-rule"), _pointer(n=1), _pointer(n=2)))
    assert res.output == _again()


def test_an_ask_by_another_spelling_right_after_its_pointer_gets_the_text() -> None:
    assert _ask(_ctx_after(_pointer(n=1), _body("alpha-rule")), "ALPHA-RULE").output == _again()


def test_an_ask_right_after_the_text_gets_a_pointer() -> None:
    assert _ask(_ctx_after(_body("alpha-rule"))).output == _pointer()


def test_an_ask_after_other_work_gets_the_next_pointer() -> None:
    seen = _ctx_after("3 matches in 2 files", _body("alpha-rule"), _pointer(n=1))
    assert _ask(seen).output == _pointer(n=2)


def test_an_ask_right_after_another_rules_pointer_gets_a_pointer() -> None:
    last = _pointer("beta-rule", summary="Beta")
    assert _ask(_ctx_after(last, _body("alpha-rule"), _body("beta-rule"))).output == _pointer()


def test_the_ask_after_the_text_sent_again_gets_the_next_pointer() -> None:
    seen = _ctx_after(_again(), _body("alpha-rule"), _pointer(n=1))
    assert _ask(seen).output == _pointer(n=2)


def test_a_pointer_quoted_inside_another_result_does_not_count() -> None:
    """The check reads where the last result BEGINS: a pointer quoted in a log line is not one."""
    seen = _ctx_after("log: " + _pointer(n=1), _body("alpha-rule"), _pointer(n=1))
    assert _ask(seen).output == _pointer(n=2)


def test_refresh_right_after_its_pointer_is_the_plain_text() -> None:
    res = _ask(_ctx_after(_pointer(n=1), _body("alpha-rule")), refresh=True)
    assert res.output == _body("alpha-rule")


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
    assert second == _pointer()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_numbers_every_repeat_and_sends_the_body_once(
    tmp_path: Path, streaming: bool
) -> None:
    """Four identical reads in a row. ADR-0372 sent the pointer once and the body again after it;
    every repeat now gets the next number, and the body is sent once. The doom-loop guard
    declines one of the reads, so the assertions count bodies and read the last answer instead of
    fixing positions."""
    loop = _loop(tmp_path, "r1", "r2", "r3", "r4")
    if streaming:
        async for _event in loop.astream_turn("go"):
            pass
    else:
        await loop.arun_turn("go")
    outputs = _outputs(loop)
    assert "Always do the alpha thing." in outputs[0]
    assert sum("Always do the alpha thing." in output for output in outputs) == 1
    assert _pointer(n=1) in outputs
    assert outputs[-1] == _pointer(n=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_sends_the_text_for_an_ask_right_after_its_pointer(
    tmp_path: Path, streaming: bool
) -> None:
    """ADR-0381 through both turn paths: alpha, beta, alpha (a pointer: beta's text came between),
    then alpha again at once (its pointer did not land: the text, with the lead)."""
    registry = ToolRegistry()
    registry.register(ReadRuleTool())
    asks = [("r1", "alpha-rule"), ("r2", "beta-rule"), ("r3", "alpha-rule"), ("r4", "alpha-rule")]
    loop = AgentLoop(
        _Scripted(
            [_read(i, n) for i, n in asks] + [LLMResult(text="done", usage=Usage(total_tokens=1))]
        ),
        registry,
        Session(cwd=str(tmp_path), model="t"),
        rule_registry=_registry(),
    )
    if streaming:
        async for _event in loop.astream_turn("go"):
            pass
    else:
        await loop.arun_turn("go")
    outputs = _outputs(loop)
    assert len(outputs) == 4
    assert outputs[2] == _pointer()
    body = (await ReadRuleTool().execute({"name": "alpha-rule"}, _ctx(_registry()))).output
    assert outputs[3] == ASKED_AGAIN.format(name="alpha-rule") + "\n" + body


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
    assert _outputs(loop)[-1] == _pointer()


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
    assert _outputs(again)[-1] == _pointer(summary="Step 0: do the alpha thing.")


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
    assert _outputs(loop)[-1] == _pointer()


@pytest.mark.asyncio
async def test_a_rule_quoted_outside_a_tool_result_is_still_delivered(tmp_path: Path) -> None:
    """A summary or a message that quotes a rule is not a read of it; only a tool result is."""
    loop = _loop(tmp_path, "r1")
    text = (await ReadRuleTool().execute({"name": "alpha-rule"}, _ctx(_registry()))).output
    loop.session.messages.extend([Message.system(f"Summary.\n{text}"), Message.user(text)])
    await loop.arun_turn("go")
    assert "Always do the alpha thing." in _outputs(loop)[-1]
