"""Tests for the ``read_rule`` tool — Vinheim Lever A chunk 2.

The tool is the retrieval half of ``lean_rules``: ``render_index()`` names every rule without
its body, and this is how the model fetches one. So the tests pin BOTH halves — the tool
returns the right body, and the index actually points at the tool.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.agent.loop import AgentLoop, _hide_pointer_asks
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.rules import Rule, RuleRegistry, discover_rules
from zakcode.session.store import Session
from zakcode.tasks import Task, TaskNetwork
from zakcode.tools.base import RULE_POINTER, SkillLoad, ToolContext, ToolRegistry, ToolResult
from zakcode.tools.builtins.read_rule import (
    ALREADY_READ,
    JUST_ASKED,
    MAX_RULE_BODY_CHARS,
    REFRESH_HEAD,
    REFRESHED,
    ReadRuleTool,
    rule_text,
    would_point,
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


def test_unknown_rule_says_where_the_rules_and_other_documents_are() -> None:
    """ADR-0424: the fix says where the rules are listed and how another document is reached,
    and no longer lists every rule after "Use one of the available rules"."""
    res = _run({"name": "nope"}, _registry())
    assert res.is_error
    assert "no rule named 'nope'" in res.output
    fix = res.fix or ""
    assert fix == (
        "Your system prompt lists the rules by name. For another document, find a file named "
        "'nope.md' with your file tools and read it."
    )
    assert "alpha-rule" not in fix and "Use one of the available rules" not in fix


class _Skills:
    def names(self) -> list[str]:
        return ["verify-unit", "work-loop"]

    def body(self, name: str) -> str | None:
        return None

    async def load(self, name: str, *, query: str = "", args: str = "") -> SkillLoad:
        raise NotImplementedError  # pragma: no cover — read_rule never loads a skill


def _run_with_skills(args: dict, registry: object) -> ToolResult:
    ctx = ToolContext(workspace_root=Path("/tmp"), rule_registry=registry, skill_resolver=_Skills())
    return asyncio.run(ReadRuleTool().execute(args, ctx))


def test_a_skill_asked_for_as_a_rule_is_named_as_a_skill() -> None:
    """ADR-0424: measured on worker sessions, most names read_rule did not know were skills or
    other documents; the 'Skill' call is named the way the shell and the tool registry name it."""
    res = _run_with_skills({"name": "Work-Loop"}, _registry())
    assert res.is_error
    assert "no rule named 'Work-Loop'" in res.output
    assert (
        res.fix == """'Work-Loop' is a skill, not a rule: load it with Skill(skill="work-loop")."""
    )


def test_a_name_that_is_neither_a_rule_nor_a_skill_is_not_called_a_skill() -> None:
    res = _run_with_skills({"name": "goal-schemas"}, _registry())
    assert res.fix == (
        "Your system prompt lists the rules by name. For another document, find a file named "
        "'goal-schemas.md' with your file tools and read it."
    )


def _named_registry(*names: str) -> RuleRegistry:
    reg = RuleRegistry()
    for rule_name in names:
        reg.add(Rule(rule_name, f"# {rule_name}\nBody of {rule_name}.", Path(f"/x/{rule_name}.md")))
    return reg


_NAMES = ("commit-style", "review-before-merge", "test-first")


def test_near_miss_name_leads_with_the_closest_rule() -> None:
    """A name a few letters off a real rule is answered with that rule first (ADR-0380), and the
    answer says where the rules are listed (ADR-0424), so a wrong lead costs nothing."""
    res = _run({"name": "reviewed-before-merging"}, _named_registry(*_NAMES))
    assert res.is_error
    assert "no rule named 'reviewed-before-merging'" in res.output
    fix = res.fix or ""
    assert fix.startswith(
        "Did you mean 'review-before-merge'? Your system prompt lists the rules by name."
    )


def test_far_name_gets_no_suggestion() -> None:
    res = _run({"name": "database-migrations"}, _named_registry(*_NAMES))
    assert res.is_error
    assert (res.fix or "").startswith("Your system prompt lists the rules by name.")
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
    """The deliberate way back to the text: refresh=true, never a plain repeat. A text the model
    holds comes back marked, so the next refresh can see this copy (ADR-0413)."""
    seen = _ctx_seeing(_body("alpha-rule"), _pointer(n=1))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule", "refresh": True}, seen))
    assert res.output == _refreshed("alpha-rule")
    assert res.verbatim
    assert not (res.data or {}).get(RULE_POINTER)


# --- ADR-0413: one refresh while the text is held, then a pointer to that copy -----------------


def _text(name: str = "alpha-rule") -> str:
    """The text a first read sends, rendered without running the tool, so a test already inside
    an event loop can build it."""
    text = rule_text(name, _registry().get(name))
    assert text is not None
    return text


def _refreshed(name: str = "alpha-rule") -> str:
    return REFRESH_HEAD.format(name=name) + _text(name)


def _refreshed_pointer(n: int = 1, name: str = "alpha-rule", step: str = "") -> str:
    return REFRESHED.format(n=n, name=name, step=step)


def _refresh(ctx: ToolContext, name: str = "alpha-rule"):
    return asyncio.run(ReadRuleTool().execute({"name": name, "refresh": True}, ctx))


def test_a_refresh_while_the_last_refreshs_copy_is_held_gets_a_pointer_not_the_text() -> None:
    """Measured: 138 of 180 refreshes answered with the text came after an earlier refresh of the
    same rule with no compaction between, 646,912 characters."""
    res = _refresh(_ctx_seeing(_body("alpha-rule"), _refreshed()))
    assert res.output == _refreshed_pointer()
    assert "Always do the alpha thing." not in res.output
    assert not res.verbatim
    assert not res.is_error
    assert (res.data or {}).get(RULE_POINTER) is True


def test_the_count_goes_on_through_the_refreshed_pointer() -> None:
    seen = _ctx_seeing(_body("alpha-rule"), _pointer(n=1), _refreshed(), _refreshed_pointer(n=2))
    assert _refresh(seen).output == _refreshed_pointer(n=3)


def test_the_refreshed_pointer_names_the_plan_step() -> None:
    plan = TaskNetwork(tasks=[Task(title="Write the tests")])
    seen = _ctx_seeing(_body("alpha-rule"), _refreshed(), plan=plan)
    step = " to your current step, 'Write the tests'"
    assert _refresh(seen).output == _refreshed_pointer(step=step)


def test_a_refresh_of_a_text_not_held_is_the_plain_text() -> None:
    """Nothing to mark: the text goes out as a first read does, and the next refresh is answered."""
    assert _refresh(_ctx_seeing("3 matches in 2 files")).output == _body("alpha-rule")


def test_a_refresh_after_its_copy_is_gone_sends_the_text_again() -> None:
    """An elision or a compaction took the refreshed copy: what is held is the text alone."""
    assert _refresh(_ctx_seeing(_body("alpha-rule"), _pointer(n=1))).output == _refreshed()


def test_a_plain_ask_after_a_refreshed_copy_gets_the_pointer() -> None:
    seen = _ctx_seeing(_refreshed())
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == _pointer()


def test_another_rules_refreshed_copy_does_not_count() -> None:
    seen = _ctx_seeing(_body("alpha-rule"), _body("beta-rule"), _refreshed("beta-rule"))
    assert _refresh(seen).output == _refreshed()


def test_a_refreshed_copy_of_an_older_wording_does_not_count() -> None:
    """The check is the whole string the refresh sends, so an edit since then is sent."""
    stale = REFRESH_HEAD.format(name="alpha-rule") + "# alpha-rule\n# Alpha\nAn older wording."
    assert _refresh(_ctx_seeing(_body("alpha-rule"), stale)).output == _refreshed()


def test_would_point_says_the_second_refresh_is_a_pointer_and_the_first_is_not() -> None:
    args = {"name": "alpha-rule", "refresh": True}
    assert would_point(args, _ctx_seeing(_body("alpha-rule"), _refreshed()))
    assert not would_point(args, _ctx_seeing(_body("alpha-rule")))


def test_the_pointer_names_the_plan_step_to_go_on_with() -> None:
    plan = TaskNetwork(tasks=[Task(title="Write the tests")])
    seen = _ctx_seeing(_body("alpha-rule"), plan=plan)
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == _pointer(step=" to your current step, 'Write the tests'")


def test_another_rules_pointer_does_not_count() -> None:
    seen = _ctx_seeing(_body("alpha-rule"), _pointer("beta-rule", summary="Beta"))
    res = asyncio.run(ReadRuleTool().execute({"name": "alpha-rule"}, seen))
    assert res.output == _pointer()


# --- ADR-0381, ADR-0387: an ask right after the rule's own pointer gets the firmer line ---------


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


def _just(name: str = "alpha-rule", n: int = 2, step: str = "") -> str:
    return JUST_ASKED.format(n=n, name=name, step=step)


def _ask(ctx: ToolContext, name: str = "alpha-rule", **args: object):
    return asyncio.run(ReadRuleTool().execute({"name": name, **args}, ctx))


def test_an_ask_right_after_its_pointer_gets_the_firmer_line_not_the_text() -> None:
    """ADR-0381 sent the text again here, and after 4 of 7 such texts the same rule was asked for
    again at once (ADR-0387). The line says where the text is, and the count goes on."""
    res = _ask(_ctx_after(_pointer(n=1), _body("alpha-rule")))
    assert res.output == _just(n=2)
    assert "Always do the alpha thing." not in res.output
    assert not res.verbatim
    assert not res.is_error


def test_an_ask_right_after_a_later_pointer_gets_the_firmer_line_too() -> None:
    res = _ask(_ctx_after(_pointer(n=3), _body("alpha-rule"), _pointer(n=1), _pointer(n=2)))
    assert res.output == _just(n=4)


def test_an_ask_by_another_spelling_right_after_its_pointer_gets_the_firmer_line() -> None:
    assert _ask(_ctx_after(_pointer(n=1), _body("alpha-rule")), "ALPHA-RULE").output == _just()


def test_an_ask_right_after_the_text_gets_a_pointer() -> None:
    assert _ask(_ctx_after(_body("alpha-rule"))).output == _pointer()


def test_an_ask_after_other_work_gets_the_next_pointer() -> None:
    seen = _ctx_after("3 matches in 2 files", _body("alpha-rule"), _pointer(n=1))
    assert _ask(seen).output == _pointer(n=2)


def test_an_ask_right_after_another_rules_pointer_gets_a_pointer() -> None:
    last = _pointer("beta-rule", summary="Beta")
    assert _ask(_ctx_after(last, _body("alpha-rule"), _body("beta-rule"))).output == _pointer()


def test_an_ask_right_after_the_firmer_line_gets_it_again_with_the_next_number() -> None:
    seen = _ctx_after(_just(n=2), _body("alpha-rule"), _pointer(n=1))
    assert _ask(seen).output == _just(n=3)


def test_a_pointer_quoted_inside_another_result_does_not_count() -> None:
    """The check reads where the last result BEGINS: a pointer quoted in a log line is not one."""
    seen = _ctx_after("log: " + _pointer(n=1), _body("alpha-rule"), _pointer(n=1))
    assert _ask(seen).output == _pointer(n=2)


def test_refresh_right_after_its_pointer_is_the_text_not_the_firmer_line() -> None:
    res = _ask(_ctx_after(_pointer(n=1), _body("alpha-rule")), refresh=True)
    assert res.output == _refreshed()


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
async def test_the_loop_answers_an_ask_right_after_its_pointer_with_the_firmer_line(
    tmp_path: Path, streaming: bool
) -> None:
    """ADR-0387 through both turn paths: alpha, beta, alpha (a pointer: beta's text came between),
    then alpha again at once (the firmer line, and alpha's text was sent once)."""
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
    assert outputs[3] == _just(n=2)
    assert sum("Always do the alpha thing." in output for output in outputs) == 1


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


def _refresh_call(call_id: str, name: str = "alpha-rule") -> LLMResult:
    call = ToolCall(id=call_id, name="read_rule", arguments={"name": name, "refresh": True})
    return LLMResult(text="", tool_calls=[call], usage=Usage(total_tokens=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_the_loop_sends_a_held_text_on_one_refresh_and_points_at_it_after(
    tmp_path: Path, streaming: bool
) -> None:
    """ADR-0413 through both turn paths, on the loop's own conversation check: a read, then three
    refreshes of the same rule, with another rule read between the last two so the doom guard's
    repeat test does not take one. The text goes out twice: the read and the first refresh."""
    registry = ToolRegistry()
    registry.register(ReadRuleTool())
    script = [_read("r1"), _refresh_call("f1"), _refresh_call("f2"), _read("r2", "beta-rule")]
    script += [_refresh_call("f3"), LLMResult(text="done", usage=Usage(total_tokens=1))]
    loop = AgentLoop(
        _Scripted(script),
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
    assert len(outputs) == 5
    assert outputs[0] == _text()
    assert outputs[1] == _refreshed()
    assert outputs[2] == _refreshed_pointer()
    assert outputs[4] == _refreshed_pointer(n=2)
    assert sum("Always do the alpha thing." in output for output in outputs) == 2
    # A completion answered by that pointer alone is not sent again in the request (ADR-0400).
    sent = [
        block.output
        for message in _hide_pointer_asks(loop.session.messages)
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert sent == [_text(), _refreshed(), _text("beta-rule")]


@pytest.mark.asyncio
async def test_a_refresh_after_its_copy_was_elided_is_answered_with_the_text(
    tmp_path: Path,
) -> None:
    """An elision shortens the refreshed copy's output: the copy is gone, so is the pointer."""
    loop = _loop(tmp_path, "r1")
    await loop.arun_turn("go")
    loop.provider._results = [
        _refresh_call("f1"),
        LLMResult(text="done", usage=Usage(total_tokens=1)),
    ]
    loop.provider.calls = 0
    await loop.arun_turn("again")
    assert _outputs(loop)[-1] == _refreshed()
    for message in loop.session.messages:
        for block in message.blocks:
            if isinstance(block, ToolResultBlock) and block.output.startswith("[sent again"):
                block.output = "[elided]"
    loop.provider._results = [
        _refresh_call("f2"),
        LLMResult(text="done", usage=Usage(total_tokens=1)),
    ]
    loop.provider.calls = 0
    await loop.arun_turn("once more")
    assert _outputs(loop)[-1] == _refreshed()


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
