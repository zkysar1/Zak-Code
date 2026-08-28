"""Tests for runtime skill authoring (save_skill + the save_skill tool) and
.claude/skills discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.skills import (
    SkillError,
    discover_skill_dir,
    discover_skills,
    parse_frontmatter,
    project_skills_dir,
    save_skill,
)
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.save_skill import SaveSkillTool

# ── save_skill serialization round-trip ───────────────────────────────────────


def test_save_skill_round_trips(tmp_path: Path) -> None:
    path = save_skill(
        "release-checklist",
        "Walks the release steps; use when cutting a release.",
        "1. run tests\n2. tag\n3. publish",
        skills_dir=tmp_path,
        allowed_tools=["bash", "read_file"],
    )
    assert path.exists()
    fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm.name == "release-checklist"
    assert "cutting a release" in fm.description
    assert fm.allowed_tools == ["bash", "read_file"]
    assert "publish" in body

    # And it is discoverable as a normal skill.
    skills, errors = discover_skill_dir(tmp_path)
    assert errors == {}
    assert any(s.name == "release-checklist" for s in skills)


def test_save_skill_rejects_unsafe_name(tmp_path: Path) -> None:
    for bad in ["../escape", "has/slash", "Has Space", "UPPER", ""]:
        with pytest.raises(SkillError):
            save_skill(bad, "d", "body", skills_dir=tmp_path)
    # Nothing escaped the skills dir.
    assert not (tmp_path.parent / "escape").exists()


def test_save_skill_empty_body_rejected(tmp_path: Path) -> None:
    with pytest.raises(SkillError):
        save_skill("ok-name", "d", "   ", skills_dir=tmp_path)


def test_save_skill_description_newlines_survive_round_trip(tmp_path: Path) -> None:
    # CRLF / lone-CR in a description must not truncate it on the next read (the
    # parser splits on CR, LF, and CRLF), so they are collapsed to a single line.
    for desc in ["line one\r\nline two", "a\rb", "x\ny"]:
        path = save_skill("rt", desc, "body", skills_dir=tmp_path, overwrite=True)
        fm, _body = parse_frontmatter(path.read_text(encoding="utf-8"))
        assert "\n" not in fm.description and "\r" not in fm.description
        # No content was lost to truncation.
        for token in desc.replace("\r", " ").replace("\n", " ").split():
            assert token in fm.description


def test_save_skill_no_clobber_without_overwrite(tmp_path: Path) -> None:
    save_skill("dup", "d", "first", skills_dir=tmp_path)
    with pytest.raises(SkillError):
        save_skill("dup", "d", "second", skills_dir=tmp_path)
    # overwrite=True replaces.
    save_skill("dup", "d", "second", skills_dir=tmp_path, overwrite=True)
    _fm, body = parse_frontmatter((tmp_path / "dup" / "SKILL.md").read_text(encoding="utf-8"))
    assert body == "second"


# ── .claude/skills discovery ──────────────────────────────────────────────────


def test_claude_skills_dir_is_discovered(tmp_path: Path) -> None:
    cm = tmp_path / ".claude" / "skills" / "cm-skill"
    cm.mkdir(parents=True)
    (cm / "SKILL.md").write_text(
        "---\nname: cm-skill\ndescription: from claude dir\n---\nbody", encoding="utf-8"
    )
    registry, _errors = discover_skills(tmp_path)
    assert registry.get("cm-skill") is not None


def test_claude_skills_override_zakcode_by_name(tmp_path: Path) -> None:
    for root, marker in ((".zakcode", "ZAK"), (".claude", "CLAUDE")):
        d = tmp_path / root / "skills" / "x"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"---\nname: x\ndescription: {marker}\n---\nbody", encoding="utf-8"
        )
    registry, _errors = discover_skills(tmp_path)
    # .claude is discovered after .zakcode, so it wins on a name clash.
    assert registry.get("x").description == "CLAUDE"


# ── the save_skill tool ───────────────────────────────────────────────────────


async def test_save_skill_tool_authors_a_discoverable_skill(tmp_path: Path) -> None:
    skills_dir = project_skills_dir(tmp_path)
    tool = SaveSkillTool(skills_dir)
    ctx = ToolContext(workspace_root=tmp_path)

    res = await tool.execute(
        {"name": "greet-user", "description": "say hello", "body": "Say hi to the user."}, ctx
    )
    assert not res.is_error
    assert res.data is not None and res.data["name"] == "greet-user"

    # A fresh discovery (i.e. "next session") finds it.
    registry, _errors = discover_skills(tmp_path)
    assert registry.get("greet-user") is not None


async def test_save_skill_tool_rejects_bad_name(tmp_path: Path) -> None:
    tool = SaveSkillTool(project_skills_dir(tmp_path))
    res = await tool.execute(
        {"name": "../evil", "description": "d", "body": "b"},
        ToolContext(workspace_root=tmp_path),
    )
    assert res.is_error


async def test_save_skill_tool_requires_body(tmp_path: Path) -> None:
    tool = SaveSkillTool(project_skills_dir(tmp_path))
    res = await tool.execute(
        {"name": "ok-name", "description": "d", "body": ""},
        ToolContext(workspace_root=tmp_path),
    )
    assert res.is_error


# ── facade wiring ─────────────────────────────────────────────────────────────


def test_agent_enable_skills_registers_save_skill(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        enable_skills=True,
        default_model="scripted/test",
        context_window=8192,
        workspace_root=str(tmp_path),
    )
    assert agent.registry.get("save_skill") is not None
