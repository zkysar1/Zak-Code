"""Tests for the ``read_rule`` tool — Vinheim Lever A chunk 2 (g-016-82).

The tool is the retrieval half of ``lean_rules``: ``render_index()`` names every rule without
its body, and this is how the model fetches one. So the tests pin BOTH halves — the tool
returns the right body, and the index actually points at the tool.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from zakcode.rules import Rule, RuleRegistry, discover_rules
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.read_rule import MAX_RULE_BODY_CHARS, ReadRuleTool


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

    Second verification outcome of g-016-82. Without this the index tells the model a rule
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
