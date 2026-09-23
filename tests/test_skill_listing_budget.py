"""The skill listing budget (ADR-0222).

Claude Code bounds the skill listing it shows the model: 1% of the context window at 4
characters per token, each description cut at 1,536 characters, and over budget the least-used
skills lose their descriptions first while every name stays. zakcode rendered every description
unconditionally, so a Mind's 146-skill catalogue cost 85,714 characters (about 20k tokens) on
every call, against 7,882 under the budget, with every skill still named.

What these tests pin: a catalogue that fits renders exactly as before (bench prompts and small
workspaces do not move); over budget every line stays, in registration order, descriptions go
to the most-used skills in Claude Code's eviction order, a note says how many are names only;
the usage record that ranks them (skills/usage.py) never fails a load; and the Agent counts a
model's or an operator's choice of a skill, never the harness re-entering its own loop.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

import zakcode
from tests.conftest import StubProvider
from zakcode.config import zakcode_home
from zakcode.skills import (
    MAX_LISTING_DESC_CHARS,
    MIN_LISTING_BUDGET_CHARS,
    Skill,
    SkillRegistry,
    listing_budget_chars,
    parse_frontmatter,
)
from zakcode.skills.usage import load_skill_usage, record_skill_use, usage_path

NOTE = "listed by name only, to keep this list within its budget"


def _registry(*skills: tuple[str, str]) -> SkillRegistry:
    """A registry of ``(name, description)`` skills, in the order given."""
    reg = SkillRegistry()
    for name, desc in skills:
        fm, _ = parse_frontmatter(f"---\nname: {name}\ndescription: {desc}\n---\nbody of {name}\n")
        reg.add(Skill(fm, Path(name) / "SKILL.md"))
    return reg


def _described(catalog: str) -> list[str]:
    """The names whose line carries a description, in the order they are listed."""
    return [
        line.split('"')[1]
        for line in catalog.splitlines()
        if line.startswith('- Skill(skill="') and " — " in line
    ]


def _listed(catalog: str) -> list[str]:
    """Every listed skill's name, in order."""
    return [
        line.split('"')[1] for line in catalog.splitlines() if line.startswith('- Skill(skill="')
    ]


#: Four skills whose descriptions are 200 characters each, in registration order a, b, c, d.
FOUR = tuple((name, name * 200) for name in "abcd")


def _budget_for(reg: SkillRegistry, descriptions: int) -> int:
    """A budget with room for exactly ``descriptions`` of FOUR's descriptions."""
    names_only = reg.render_catalog(budget_chars=1)
    # Each admitted description adds " — " plus its 200 characters to its line.
    return len(names_only) + descriptions * (len(" — ") + 200)


# ── the budget ───────────────────────────────────────────────────────────────────────────


def test_the_budget_is_claude_codes_formula_with_its_standard_window_as_the_floor() -> None:
    assert listing_budget_chars(None) is None
    assert listing_budget_chars(0) is None
    # 200,000 x 4 x 1% = 8,000: Claude Code's budget on its standard window, and the floor.
    assert listing_budget_chars(200_000) == 8_000 == MIN_LISTING_BUDGET_CHARS
    assert listing_budget_chars(131_072) == 8_000  # 5,242 by the formula alone
    assert listing_budget_chars(1_000_000) == 40_000


def test_a_catalogue_that_fits_renders_exactly_as_an_unbounded_one() -> None:
    reg = _registry(*FOUR)
    unbounded = reg.render_catalog()
    assert reg.render_catalog(budget_chars=len(unbounded), usage={"d": 9}) == unbounded
    assert NOTE not in unbounded


def test_over_budget_every_skill_keeps_its_line_in_registration_order() -> None:
    reg = _registry(*FOUR)
    catalog = reg.render_catalog(budget_chars=_budget_for(reg, 2), usage={"d": 5, "c": 3})
    assert _listed(catalog) == ["a", "b", "c", "d"]
    assert _described(catalog) == ["c", "d"]  # the two most used, still in registration order
    assert len(catalog) <= _budget_for(reg, 2)
    assert f"2 of these 4 skills are {NOTE}" in catalog


def test_descriptions_go_to_the_most_used_skills_and_ties_keep_registration_order() -> None:
    reg = _registry(*FOUR)
    budget = _budget_for(reg, 3)
    assert _described(reg.render_catalog(budget_chars=budget, usage={"d": 1})) == ["a", "b", "d"]
    # No usage at all (a fresh home, or stable_prompt_identity): registration order decides.
    assert _described(reg.render_catalog(budget_chars=budget, usage={})) == ["a", "b", "c"]


def test_the_first_description_that_does_not_fit_ends_the_list() -> None:
    # Claude Code evicts the least used first, so a skill never keeps a description that a
    # more-used skill lost: the long description of the most-used skill does not fit, and the
    # short one of a less-used skill is not admitted in its place.
    reg = _registry(("long", "L" * 1_000), ("short", "S" * 10), ("top", "T" * 10))
    names_only = len(reg.render_catalog(budget_chars=1))
    budget = names_only + len(" — ") + 10 + len(" — ") + 500  # "top", then not "long"
    catalog = reg.render_catalog(budget_chars=budget, usage={"top": 3, "long": 2, "short": 1})
    assert _described(catalog) == ["top"]
    assert f"2 of these 3 skills are {NOTE}" in catalog


def test_every_name_stays_when_the_names_alone_exceed_the_budget() -> None:
    reg = _registry(*FOUR)
    catalog = reg.render_catalog(budget_chars=10, usage={"a": 1})
    assert _listed(catalog) == ["a", "b", "c", "d"]
    assert _described(catalog) == []
    assert f"4 of these 4 skills are {NOTE}" in catalog


def test_a_skill_with_no_description_is_not_counted_as_listed_by_name_only() -> None:
    reg = _registry(*FOUR, ("bare", ""))
    catalog = reg.render_catalog(budget_chars=_budget_for(reg, 1), usage={"a": 1})
    assert _described(catalog) == ["a"]
    # b, c and d lost theirs; "bare" never had one.
    assert f"3 of these 5 skills are {NOTE}" in catalog


def test_the_budget_keeps_the_user_only_and_provenance_lines() -> None:
    reg = _registry(*FOUR)
    fm, _ = parse_frontmatter(
        "---\nname: start\ndescription: operator only\ndisable-model-invocation: true\n---\nb\n"
    )
    reg.add(Skill(fm, Path("start/SKILL.md")))
    catalog = reg.render_catalog(budget_chars=1)
    assert "User-only commands (/start)" in catalog
    assert "When a user message BEGINS with a <command-name> block" in catalog
    assert 'Skill(skill="start")' not in catalog


def test_a_description_past_claude_codes_cap_is_cut() -> None:
    reg = _registry(("wordy", "w" * (MAX_LISTING_DESC_CHARS + 500)))
    [line] = [x for x in reg.render_catalog().splitlines() if x.startswith("- Skill(")]
    desc = line.split(" — ", 1)[1]
    assert len(desc) == MAX_LISTING_DESC_CHARS
    assert desc.endswith("…")


# ── the usage record ─────────────────────────────────────────────────────────────────────


def test_the_usage_record_lives_in_the_zakcode_home() -> None:
    assert usage_path() == zakcode_home() / "skill-usage.json"


def test_no_usage_file_reads_as_no_usage(tmp_path: Path) -> None:
    assert load_skill_usage(tmp_path / "absent.json") == {}


def test_an_unusable_usage_file_reads_as_no_usage(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "usage.json"
    path.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="zakcode.skills.usage"):
        assert load_skill_usage(path) == {}
    assert "unreadable" in caplog.text
    path.write_text("[1, 2]", encoding="utf-8")
    assert load_skill_usage(path) == {}


def test_malformed_entries_cost_themselves_not_the_record(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text(
        json.dumps({"ok": 3, "negative": -1, "flag": True, "text": "7", "real": 2.5}),
        encoding="utf-8",
    )
    assert load_skill_usage(path) == {"ok": 3}


def test_recording_a_use_counts_it(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "usage.json"
    record_skill_use("a", path)
    record_skill_use("a", path)
    record_skill_use("b", path)
    assert load_skill_usage(path) == {"a": 2, "b": 1}
    assert [p.name for p in path.parent.iterdir()] == ["usage.json"]  # no temp file left


def test_a_failed_write_keeps_the_old_counts_and_raises_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "home"
    blocker.write_text("a file where the directory should be", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="zakcode.skills.usage"):
        record_skill_use("a", blocker / "usage.json")  # mkdir fails under a file
    assert "could not record a use of skill 'a'" in caplog.text


def test_concurrent_uses_in_one_process_are_all_counted(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"

    def use() -> None:
        for _ in range(25):
            record_skill_use("a", path)

    threads = [threading.Thread(target=use) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert load_skill_usage(path) == {"a": 200}


# ── the Agent ────────────────────────────────────────────────────────────────────────────


def _seed_skills(workspace: Path, count: int) -> list[str]:
    """``count`` skills in the workspace's .claude/skills, each with a 300-character
    description, so a catalogue of 40 is well past the 8,000-character floor."""
    names = [f"skill-{i:02d}" for i in range(count)]
    for name in names:
        skill_dir = workspace / ".claude" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {'x' * 300}\n---\nDo the {name} thing.\n",
            encoding="utf-8",
        )
    return names


def _model_names(agent: zakcode.Agent) -> list[str]:
    """Every skill the model may run, in registration order (bundled skills first)."""
    assert agent.skill_registry is not None
    return [name for name, _ in agent.skill_registry.model_catalog()]


def _catalogue(agent: zakcode.Agent) -> str:
    text = agent.loop.prompt_builder.extra_instructions or ""
    assert "Available skills" in text  # anti-vacuity: this IS the rendered catalogue
    return text


def test_an_agent_with_a_known_window_budgets_its_catalogue(tmp_path: Path) -> None:
    names = _seed_skills(tmp_path, 40)
    agent = zakcode.Agent(
        default_model="openai/zds-qwen3.8-27b",
        workspace_root=tmp_path,
        context_window=131072,
        enable_skills=True,
    )
    catalogue = _catalogue(agent)
    assert set(names) <= set(_listed(catalogue))
    assert _listed(catalogue) == _model_names(agent)  # every name, bundled skills included
    assert NOTE in catalogue
    assert len(catalogue) <= MIN_LISTING_BUDGET_CHARS


def test_an_agent_without_a_known_window_renders_every_description(tmp_path: Path) -> None:
    # An injected provider resolves no window (the constructor refuses to start any OTHER
    # agent without one, ADR-0066), so nothing bounds the catalogue: the behaviour every
    # embedding host that brings its own provider already had.
    _seed_skills(tmp_path, 40)
    agent = zakcode.Agent(workspace_root=tmp_path, enable_skills=True, provider=StubProvider())
    assert agent.context_windows == {}
    assert _described(_catalogue(agent)) == _model_names(agent)


def test_usage_ranks_the_catalogue_unless_the_prompt_identity_is_pinned(tmp_path: Path) -> None:
    names = _seed_skills(tmp_path, 40)
    usage_path().parent.mkdir(parents=True, exist_ok=True)
    usage_path().write_text(json.dumps({"skill-39": 12}), encoding="utf-8")
    kwargs = {
        "default_model": "openai/zds-qwen3.8-27b",
        "workspace_root": tmp_path,
        "context_window": 131072,
        "enable_skills": True,
    }
    agent = zakcode.Agent(**kwargs)  # type: ignore[arg-type]
    everyone = _model_names(agent)
    assert everyone[-1] == "skill-39"  # registered last, so no tie-break could admit it
    ranked = _described(_catalogue(agent))
    assert ranked[-1] == "skill-39" and ranked[:-1] == everyone[: len(ranked) - 1]
    # ADR-0157: two runs of the same work must see a byte-identical prompt, so the counts
    # (which move between runs) are not read at all.
    pinned = _described(
        _catalogue(zakcode.Agent(stable_prompt_identity=True, **kwargs))  # type: ignore[arg-type]
    )
    assert pinned == everyone[: len(pinned)] and "skill-39" not in pinned
    assert names[0] in pinned  # anti-vacuity: the pinned catalogue does describe skills


@pytest.mark.asyncio
async def test_a_model_call_and_an_operator_slash_count_and_a_harness_reentry_does_not(
    tmp_path: Path,
) -> None:
    _seed_skills(tmp_path, 1)
    agent = zakcode.Agent(workspace_root=tmp_path, enable_skills=True)
    for source in ("tool", "command", "harness"):
        load = await agent._load_skill_body("skill-00", source=source)  # type: ignore[arg-type]
        assert load.body is not None and "Do the skill-00 thing." in load.body
        agent._begin_skill_turn()  # a new turn, so the next load is not a same-turn pointer
    assert load_skill_usage() == {"skill-00": 2}
