"""Tests for the always-on rules surface (zakcode.rules) and its prompt integration."""

from __future__ import annotations

from pathlib import Path

from zakcode.agent import DYNAMIC_BOUNDARY, SystemPromptBuilder
from zakcode.config import load_settings
from zakcode.rules import (
    MAX_RULE_FILE_CHARS,
    MAX_RULES_TOTAL_CHARS,
    RuleRegistry,
    discover_rule_dir,
    discover_rules,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── discovery ────────────────────────────────────────────────────────────────


def test_discover_flat_md_files(tmp_path: Path) -> None:
    _write(tmp_path / "style.md", "Use tabs, not spaces.")
    _write(tmp_path / "naming.md", "snake_case for functions.")
    _write(tmp_path / "notes.txt", "ignored (not .md)")
    rules, errors = discover_rule_dir(tmp_path)
    assert errors == {}
    names = sorted(r.name for r in rules)
    assert names == ["naming", "style"]  # stem is the name; sorted, .txt skipped


def test_frontmatter_name_overrides_stem(tmp_path: Path) -> None:
    _write(
        tmp_path / "f.md",
        "---\nname: http-conventions\ndescription: where HTTP lives\n---\nUse api/ for HTTP.",
    )
    rules, _errors = discover_rule_dir(tmp_path)
    assert rules[0].name == "http-conventions"
    assert rules[0].description == "where HTTP lives"
    assert rules[0].content == "Use api/ for HTTP."  # frontmatter stripped from body


def test_empty_rule_is_recorded_as_error(tmp_path: Path) -> None:
    _write(tmp_path / "blank.md", "   \n  ")
    rules, errors = discover_rule_dir(tmp_path)
    assert rules == []
    assert "blank.md" in errors


def test_missing_dir_yields_empty(tmp_path: Path) -> None:
    rules, errors = discover_rule_dir(tmp_path / "nope")
    assert rules == [] and errors == {}


def test_discover_rules_project_overrides_by_name(tmp_path: Path) -> None:
    # A project .zakcode/rules rule shadows a same-named .claude/rules rule? No —
    # .claude/rules is discovered AFTER .zakcode/rules, so it wins. Assert order.
    _write(tmp_path / ".zakcode" / "rules" / "x.md", "ZAKCODE version")
    _write(tmp_path / ".claude" / "rules" / "x.md", "CLAUDE version")
    registry, _errors = discover_rules(tmp_path)
    assert registry.get("x") is not None
    assert registry.get("x").content == "CLAUDE version"  # later source wins


def test_claude_rules_dir_is_discovered(tmp_path: Path) -> None:
    _write(tmp_path / ".claude" / "rules" / "cm.md", "CLAUDE_MIND_RULE_MARKER")
    registry, _errors = discover_rules(tmp_path)
    assert "CLAUDE_MIND_RULE_MARKER" in registry.render()


# ── render + caps ────────────────────────────────────────────────────────────


def test_render_includes_names_and_content(tmp_path: Path) -> None:
    registry, _ = discover_rules(tmp_path)  # empty
    assert registry.render() == ""

    _write(tmp_path / ".zakcode" / "rules" / "a.md", "RULE_A_BODY")
    registry, _ = discover_rules(tmp_path)
    rendered = registry.render()
    assert "Project rules" in rendered
    assert "## a" in rendered
    assert "RULE_A_BODY" in rendered


def test_render_caps_per_rule(tmp_path: Path) -> None:
    _write(tmp_path / ".zakcode" / "rules" / "big.md", "z" * (MAX_RULE_FILE_CHARS * 2))
    registry, _ = discover_rules(tmp_path)
    rendered = registry.render()
    # The body is truncated to the per-rule cap (plus header/name overhead).
    assert rendered.count("z") == MAX_RULE_FILE_CHARS


def test_render_caps_total(tmp_path: Path) -> None:
    for i in range(10):
        _write(
            tmp_path / ".zakcode" / "rules" / f"r{i}.md",
            f"rule{i}-" + "y" * MAX_RULE_FILE_CHARS,
        )
    registry, _ = discover_rules(tmp_path)
    rendered = registry.render()
    assert len(rendered) <= MAX_RULES_TOTAL_CHARS + 200  # header + omission note slack
    assert "omitted" in rendered  # the budget note appears


# ── render_index (lean: Vinheim Lever A) ─────────────────────────────────────


def test_render_index_empty_is_blank(tmp_path: Path) -> None:
    registry, _ = discover_rules(tmp_path)  # empty
    assert registry.render_index() == ""


def test_render_index_lists_names_and_frontmatter_summaries(tmp_path: Path) -> None:
    _write(
        tmp_path / ".zakcode" / "rules" / "http.md",
        "---\nname: http-conventions\ndescription: where HTTP lives\n---\n" + ("BODY " * 200),
    )
    registry, _ = discover_rules(tmp_path)
    index = registry.render_index()
    assert "Project rules" in index
    assert "- http-conventions: where HTTP lives" in index  # name + frontmatter description
    assert "http.md" in index  # the on-disk path for on-demand reads


def test_render_index_summary_falls_back_to_first_body_line(tmp_path: Path) -> None:
    _write(tmp_path / ".zakcode" / "rules" / "style.md", "# Tabs over spaces\n\nLong body...")
    registry, _ = discover_rules(tmp_path)
    index = registry.render_index()
    # No frontmatter description -> first non-empty body line, heading marker stripped.
    assert "- style: Tabs over spaces" in index


def test_render_index_omits_full_bodies(tmp_path: Path) -> None:
    # The whole point: a long body is NOT folded into the index (only name + summary + path).
    _write(
        tmp_path / ".zakcode" / "rules" / "big.md",
        "---\ndescription: short summary\n---\n" + ("UNIQUE_BODY_TOKEN " * 500),
    )
    registry, _ = discover_rules(tmp_path)
    index = registry.render_index()
    assert "UNIQUE_BODY_TOKEN" not in index  # body excluded from the lean index
    assert "short summary" in index


def test_render_index_is_far_smaller_than_full_render(tmp_path: Path) -> None:
    # Many full-size rules: the index must be dramatically smaller than the full render.
    for i in range(12):
        _write(
            tmp_path / ".zakcode" / "rules" / f"r{i}.md",
            f"---\ndescription: rule {i} summary\n---\n" + ("z" * MAX_RULE_FILE_CHARS),
        )
    registry, _ = discover_rules(tmp_path)
    full = registry.render()
    index = registry.render_index()
    assert index  # non-empty
    assert len(index) < len(full) // 10  # at least a 10x reduction on the cached prefix


def test_render_index_summary_is_capped(tmp_path: Path) -> None:
    from zakcode.rules import _INDEX_SUMMARY_CHARS, Rule

    reg = RuleRegistry()
    reg.add(Rule("r", "body", Path("r.md"), description="d" * 500))
    index = reg.render_index()
    assert "..." in index  # the over-long summary is truncated
    # The rendered summary token never exceeds the cap (137 chars + the "..." ellipsis).
    assert "d" * _INDEX_SUMMARY_CHARS not in index


def test_render_index_bounded_with_many_rules() -> None:
    from zakcode.rules import Rule

    reg = RuleRegistry()
    for i in range(5000):
        reg.add(Rule(f"r{i}", "x", Path(f"r{i}.md"), description="d" * 200))
    index = reg.render_index()
    assert len(index) <= MAX_RULES_TOTAL_CHARS  # genuinely bounded
    assert "omitted" in index  # the budget note appears


def test_agent_lean_rules_renders_index_not_bodies(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _write(
        tmp_path / ".zakcode" / "rules" / "team.md",
        "---\ndescription: TEAM_RULE_SUMMARY\n---\nALWAYS_RUN_TESTS_BODY_MARKER",
    )
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_rules=True,
        lean_rules=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    prompt = agent.loop.prompt_builder.build(agent.settings)
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    stable = prompt[:boundary_at]
    assert "TEAM_RULE_SUMMARY" in stable  # the compact index summary lands in the cached tier
    assert "ALWAYS_RUN_TESTS_BODY_MARKER" not in prompt  # the full body is NOT injected


def test_agent_lean_rules_off_is_unchanged_full_render(tmp_path: Path) -> None:
    # Default (lean_rules=False): the full body still renders — byte-for-byte unchanged.
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _write(
        tmp_path / ".zakcode" / "rules" / "team.md",
        "---\ndescription: TEAM_RULE_SUMMARY\n---\nALWAYS_RUN_TESTS_BODY_MARKER",
    )
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_rules=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    prompt = agent.loop.prompt_builder.build(agent.settings)
    stable = prompt[: prompt.index(DYNAMIC_BOUNDARY)]
    assert "ALWAYS_RUN_TESTS_BODY_MARKER" in stable  # full body present when lean is off


# ── prompt-builder integration ───────────────────────────────────────────────


def test_rules_land_in_stable_tier(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    rules_text = "Project rules:\n\n## x\nRULE_MARKER_77"
    prompt = SystemPromptBuilder(rules=rules_text).build(settings)
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    # Rules are static guidance → cacheable STABLE tier, above the boundary.
    assert "RULE_MARKER_77" in prompt[:boundary_at]


def test_no_rules_omits_section(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)
    assert "Project rules" not in prompt


# ── facade wiring ────────────────────────────────────────────────────────────


def test_agent_enable_rules_renders_into_prompt(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _write(tmp_path / ".zakcode" / "rules" / "team.md", "ALWAYS_RUN_TESTS_MARKER")
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_rules=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    assert agent.rule_registry is not None
    prompt = agent.loop.prompt_builder.build(agent.settings)
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    assert "ALWAYS_RUN_TESTS_MARKER" in prompt[:boundary_at]


def test_registry_add_no_clobber() -> None:
    from zakcode.rules import Rule

    reg = RuleRegistry()
    assert reg.add(Rule("x", "first", Path("x.md"))) is True
    assert reg.add(Rule("x", "second", Path("x.md"))) is False  # no clobber
    assert reg.get("x").content == "first"
    assert reg.add(Rule("x", "third", Path("x.md")), replace=True) is True
    assert reg.get("x").content == "third"


def test_render_total_is_bounded_with_many_tiny_rules() -> None:
    # Many tiny rules: the un-counted separators/header must not blow the budget.
    from zakcode.rules import Rule

    reg = RuleRegistry()
    for i in range(5000):
        reg.add(Rule(f"r{i}", "x", Path(f"r{i}.md")))
    rendered = reg.render()
    assert len(rendered) <= MAX_RULES_TOTAL_CHARS  # genuinely bounded (docstring guarantee)
    assert "omitted" in rendered


# ── facade: skills + rules combined, injected builder, sub-agent inheritance ──


def test_agent_enable_skills_and_rules_both_in_stable(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _write(
        tmp_path / ".zakcode" / "skills" / "helper" / "SKILL.md",
        "---\nname: skillmarker\ndescription: SKILL_DESC_MARKER\n---\nbody",
    )
    _write(tmp_path / ".zakcode" / "rules" / "x.md", "RULE_MARKER_AB")
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_skills=True,
        enable_rules=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    assert agent.skill_registry is not None and agent.rule_registry is not None
    prompt = agent.loop.prompt_builder.build(agent.settings)
    stable = prompt[: prompt.index(DYNAMIC_BOUNDARY)]
    assert "skillmarker" in stable  # skills catalog
    assert "RULE_MARKER_AB" in stable  # rules


def test_injected_prompt_builder_is_populated_with_rules(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _write(tmp_path / ".zakcode" / "rules" / "x.md", "INJECTED_BUILDER_RULE")
    pb = SystemPromptBuilder()
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        prompt_builder=pb,
        enable_rules=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    # enable_rules is not a silent no-op: the injected builder's empty rules slot is filled.
    assert agent.loop.prompt_builder is pb
    assert pb.rules is not None and "INJECTED_BUILDER_RULE" in pb.rules


def test_subagent_runner_carries_rules_into_child_prompt() -> None:
    from zakcode.agent.budget import IterationBudget
    from zakcode.agent.subagent import GENERAL_PURPOSE, SubAgentRunner
    from zakcode.evals.harness import ScriptedProvider, reply
    from zakcode.tools import default_registry

    settings = load_settings()
    runner = SubAgentRunner(
        provider=ScriptedProvider(script=[reply("ok")]),
        registry=default_registry(),
        settings=settings,
        budget=IterationBudget(10),
        rules="Project rules:\n\n## x\nSUBAGENT_RULE_MARKER",
    )
    child_prompt = runner.prompt_builder_for(GENERAL_PURPOSE).build(settings)
    assert "SUBAGENT_RULE_MARKER" in child_prompt[: child_prompt.index(DYNAMIC_BOUNDARY)]


def test_agent_subagents_inherit_parent_rules(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    _write(tmp_path / ".zakcode" / "rules" / "x.md", "DELEGATED_RULE_MARKER")
    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_rules=True,
        enable_subagents=True,
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    # The runner behind the task spawner carries the parent's rendered rules.
    runner = agent.loop.spawner._runner  # type: ignore[union-attr]
    assert runner.rules is not None and "DELEGATED_RULE_MARKER" in runner.rules
