"""ADR-0417: a rule carried across a compaction is sent as a tool result on its first ask.

After a compaction the rules the session read sit in one system message (ADR-0386). No tool
result holds them, yet the pointer's check (ADR-0372) counted that message, so the first ask
after the compaction got a pointer to text that no read_rule call had put in front of the model.
Measured on eleven unattended worker sessions: 245 of 357 read_rule asks were answered with a
pointer, and one session asked for one short rule 13 times in nine minutes. The first ask for a
rule held only by the carried message now gets its text; later asks get the pointer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.test_carry_rules import _DONE, _outputs, _read, _read_then_compact, _Scripted, _text
from zakcode.agent.compact import CARRIED_RULES_MARKER
from zakcode.rules import Rule, RuleRegistry
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.read_rule import ReadRuleTool, rule_text, would_point

_BODY = "# Beta\nNever do the beta thing."


def _registry() -> RuleRegistry:
    reg = RuleRegistry()
    reg.add(Rule("beta-rule", _BODY, Path("/x/beta-rule.md")))
    return reg


def _text_of(reg: RuleRegistry) -> str:
    text = rule_text("beta-rule", reg.get("beta-rule"))
    assert text is not None
    return text


def _ctx(
    reg: RuleRegistry, *, carried: str = "", results: tuple[str, ...] = (), narrow: bool = True
) -> ToolContext:
    """A context whose conversation holds ``carried`` in a system message and ``results`` in tool
    results, as the loop's two predicates read it."""
    in_results = lambda text: any(text in output for output in results)  # noqa: E731
    return ToolContext(
        workspace_root=Path("/tmp"),
        rule_registry=reg,
        in_context=lambda text: in_results(text) or text in carried,
        in_tool_results=in_results if narrow else None,
        last_tool_output=lambda: results[-1] if results else None,
    )


def _ask(ctx: ToolContext, **args: object):
    return asyncio.run(ReadRuleTool().execute({"name": "beta-rule", **args}, ctx))


def test_a_rule_held_only_by_the_carried_message_gets_its_text_then_the_pointer() -> None:
    reg = _registry()
    text = _text_of(reg)
    carried = f"{CARRIED_RULES_MARKER}\n{text}"
    first = _ask(_ctx(reg, carried=carried))
    assert first.output == text
    assert first.verbatim
    again = _ask(_ctx(reg, carried=carried, results=(first.output,)))
    assert again.output.startswith("[already read; ask 1] Rule 'beta-rule' ")


def test_a_tool_result_holding_the_text_still_gets_the_pointer() -> None:
    reg = _registry()
    res = _ask(_ctx(reg, results=(_text_of(reg),)))
    assert res.output.startswith("[already read; ask 1] Rule 'beta-rule' ")


def test_without_in_tool_results_the_carried_message_still_counts() -> None:
    reg = _registry()
    carried = f"{CARRIED_RULES_MARKER}\n{_text_of(reg)}"
    res = _ask(_ctx(reg, carried=carried, narrow=False))
    assert res.output.startswith("[already read; ask 1] Rule 'beta-rule' ")


def test_refresh_returns_the_text_in_every_state() -> None:
    reg = _registry()
    text = _text_of(reg)
    # A refresh of a held text may open with the marker ADR-0413 puts on it; the text follows.
    assert _ask(_ctx(reg, results=(text,)), refresh=True).output.endswith(text)
    assert _ask(_ctx(reg, carried=text), refresh=True).output.endswith(text)


def test_would_point_agrees_with_execute_before_and_after_the_text_is_sent() -> None:
    reg = _registry()
    text = _text_of(reg)
    carried_only = _ctx(reg, carried=text)
    assert would_point({"name": "beta-rule"}, carried_only) is False
    assert not _ask(carried_only).output.startswith("[already read")
    sent = _ctx(reg, carried=text, results=(text,))
    assert would_point({"name": "beta-rule"}, sent) is True
    assert _ask(sent).output.startswith("[already read")


@pytest.mark.asyncio
async def test_through_the_loop_the_first_ask_after_a_compaction_sends_the_text(
    tmp_path: Path,
) -> None:
    loop = await _read_then_compact(tmp_path, "beta-rule")
    provider = loop.provider
    assert isinstance(provider, _Scripted)
    provider.results = [_read("r8", "beta-rule"), _read("r9", "beta-rule"), _DONE]
    provider.seen = []
    await loop.arun_turn("again")
    text, pointer = _outputs(loop)
    assert text == _text("beta-rule")
    assert pointer.startswith("[already read; ask 1] Rule 'beta-rule' ")
