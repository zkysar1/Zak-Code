"""Tests for the model-facing ``read_rule`` tool (Vinheim Lever A Chunk 2).

Two layers: the tool in isolation against a real :class:`RuleRegistry` (its result/error
contract), and the real wiring on the ``Agent`` — that ``read_rule`` is registered ONLY in
lean-rules mode (full-render mode already has every body in-prompt, so the default tool
surface stays unchanged).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.rules import Rule, RuleRegistry
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.read_rule import ReadRuleTool


def _registry(*rules: Rule) -> RuleRegistry:
    reg = RuleRegistry()
    for r in rules:
        reg.add(r)
    return reg


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── the tool in isolation ────────────────────────────────────────────────────────


async def test_read_rule_returns_body_by_name(tmp_path: Path) -> None:
    reg = _registry(Rule("style", "USE TABS NOT SPACES", Path("style.md")))
    res = await ReadRuleTool(reg).execute({"name": "style"}, _ctx(tmp_path))
    assert res.is_error is False
    assert res.output == "USE TABS NOT SPACES"  # the body IS the tool result
    assert res.data == {"rule": "style"}
    assert res.hint  # nudges the model to apply it


async def test_read_rule_resolves_out_of_sandbox_path(tmp_path: Path) -> None:
    # The whole point: a bundled/user rule lives OUTSIDE the workspace; the registry holds its
    # content in memory, so read_rule returns the body regardless of where the file is.
    outside = Path("/etc/zakcode/bundled/security.md")  # not under tmp_path (the sandbox root)
    reg = _registry(Rule("security", "NEVER LOG SECRETS", outside))
    res = await ReadRuleTool(reg).execute({"name": "security"}, _ctx(tmp_path))
    assert res.is_error is False
    assert res.output == "NEVER LOG SECRETS"


async def test_read_rule_unknown_lists_available(tmp_path: Path) -> None:
    reg = _registry(
        Rule("alpha", "a", Path("alpha.md")), Rule("beta", "b", Path("beta.md"))
    )
    res = await ReadRuleTool(reg).execute({"name": "gamma"}, _ctx(tmp_path))
    assert res.is_error is True
    assert "no rule named 'gamma'" in res.output
    assert res.fix and "alpha, beta" in res.fix  # tells the model the real options


async def test_read_rule_empty_registry_reports_none(tmp_path: Path) -> None:
    res = await ReadRuleTool(_registry()).execute({"name": "x"}, _ctx(tmp_path))
    assert res.is_error is True
    assert "(none discovered)" in res.fix


async def test_read_rule_requires_a_name(tmp_path: Path) -> None:
    reg = _registry(Rule("alpha", "a", Path("alpha.md")))
    for args in ({}, {"name": ""}, {"name": "   "}, {"name": 5}):
        res = await ReadRuleTool(reg).execute(args, _ctx(tmp_path))  # type: ignore[arg-type]
        assert res.is_error is True and "'name' is required" in res.output


# ── facade wiring: lean registers it, full / no-rules omit it ─────────────────────


def _agent(tmp_path: Path, **kw: object) -> Agent:
    _write(tmp_path / ".zakcode" / "rules" / "team.md", "ALWAYS_RUN_TESTS")
    return Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        default_model="scripted/test",
        workspace_root=str(tmp_path),
        **kw,  # type: ignore[arg-type]
    )


def test_agent_lean_rules_registers_read_rule_tool(tmp_path: Path) -> None:
    agent = _agent(tmp_path, enable_rules=True, lean_rules=True)
    assert agent.registry.get("read_rule") is not None  # on-demand reader is wired in lean mode


def test_agent_full_rules_omits_read_rule_tool(tmp_path: Path) -> None:
    # Full-render mode has every body in-prompt already → no read_rule (tool surface unchanged).
    agent = _agent(tmp_path, enable_rules=True)
    assert agent.registry.get("read_rule") is None


def test_agent_no_rules_omits_read_rule_tool(tmp_path: Path) -> None:
    agent = _agent(tmp_path)  # enable_rules defaults False
    assert agent.registry.get("read_rule") is None


# ── config wiring: Settings.lean_rules / ZAKCODE_LEAN_RULES (Chunk 3) ─────────────


def test_agent_setting_lean_rules_registers_read_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Vinheim Lever A wiring: with NO explicit lean_rules kwarg, the Agent defers to
    # Settings.lean_rules (ZAKCODE_LEAN_RULES) — so a served MIND or dev loop flips the lever
    # via config alone (same precedence as enable_output_style). enable_rules stays on; lean
    # only changes HOW the rules render, so read_rule must be wired in.
    monkeypatch.setenv("ZAKCODE_LEAN_RULES", "true")
    agent = _agent(tmp_path, enable_rules=True)  # note: no lean_rules= kwarg
    assert agent.registry.get("read_rule") is not None


def test_agent_explicit_lean_rules_false_overrides_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Precedence: an explicit lean_rules=False from the host WINS over ZAKCODE_LEAN_RULES=true
    # (None defers to the setting, a real bool does not) — so a caller can force full-render
    # even when the env enables lean. Mirrors enable_output_style's explicit-wins contract.
    monkeypatch.setenv("ZAKCODE_LEAN_RULES", "true")
    agent = _agent(tmp_path, enable_rules=True, lean_rules=False)
    assert agent.registry.get("read_rule") is None
