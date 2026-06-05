"""Tests for the skills core (M7-1): frontmatter parse, lazy body, discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.skills import (
    Skill,
    SkillError,
    SkillRegistry,
    discover_skill_dir,
    discover_skills,
    parse_frontmatter,
)

_SKILL = """\
---
name: commit-helper
description: Draft a conventional commit message.
allowed-tools: [read_file, bash]
version: 1.2.0
---
# Commit helper

Read the diff, then write a conventional-commit subject + body.
"""


def _write_skill(root: Path, name: str, text: str = _SKILL) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")


# ── frontmatter parsing ──────────────────────────────────────────────────────────


def test_parse_frontmatter_fields_and_body() -> None:
    fm, body = parse_frontmatter(_SKILL)
    assert fm.name == "commit-helper"
    assert fm.description == "Draft a conventional commit message."
    assert fm.allowed_tools == ["read_file", "bash"]
    assert fm.version == "1.2.0"
    assert body.startswith("# Commit helper")
    assert "conventional-commit" in body


def test_parse_frontmatter_requires_fence() -> None:
    with pytest.raises(SkillError):
        parse_frontmatter("no frontmatter here\n")


def test_parse_frontmatter_requires_close() -> None:
    with pytest.raises(SkillError):
        parse_frontmatter("---\nname: x\n(no close)\n")


def test_parse_frontmatter_requires_name() -> None:
    with pytest.raises(SkillError):
        parse_frontmatter("---\ndescription: no name\n---\nbody\n")


def test_parse_frontmatter_comma_list() -> None:
    fm, _ = parse_frontmatter("---\nname: s\nallowed-tools: read_file, grep\n---\nb\n")
    assert fm.allowed_tools == ["read_file", "grep"]


# ── Skill lazy body ──────────────────────────────────────────────────────────────


def test_body_is_lazy(tmp_path: Path) -> None:
    _write_skill(tmp_path, "s")
    skills, _ = discover_skill_dir(tmp_path)
    skill = skills[0]
    assert isinstance(skill, Skill)
    assert skill.body_loaded is False  # discovery did NOT read the body
    body = skill.body()
    assert "conventional-commit" in body
    assert skill.body_loaded is True  # now cached


def test_directory_points_at_skill_dir(tmp_path: Path) -> None:
    _write_skill(tmp_path, "s")
    skills, _ = discover_skill_dir(tmp_path)
    assert skills[0].directory == tmp_path / "s"


# ── discovery ────────────────────────────────────────────────────────────────────


def test_discover_dir(tmp_path: Path) -> None:
    _write_skill(tmp_path, "a")
    _write_skill(tmp_path, "b")
    skills, errors = discover_skill_dir(tmp_path)
    assert {s.name for s in skills} == {"commit-helper"}  # both have same name in fixture
    assert errors == {}


def test_discover_missing_dir(tmp_path: Path) -> None:
    skills, errors = discover_skill_dir(tmp_path / "nope")
    assert skills == [] and errors == {}


def test_malformed_skill_is_recorded(tmp_path: Path) -> None:
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("no frontmatter", encoding="utf-8")
    skills, errors = discover_skill_dir(tmp_path)
    assert skills == []
    assert "bad" in errors


def test_missing_skill_md_is_recorded(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    skills, errors = discover_skill_dir(tmp_path)
    assert skills == []
    assert "empty" in errors


# ── registry + multi-source discovery ────────────────────────────────────────────


def test_registry_catalog_and_get() -> None:
    reg = SkillRegistry()
    fm1, _ = parse_frontmatter("---\nname: a\ndescription: alpha\n---\nbody a\n")
    fm2, _ = parse_frontmatter("---\nname: b\ndescription: beta\n---\nbody b\n")
    reg.add(Skill(fm1, Path("a/SKILL.md")))
    reg.add(Skill(fm2, Path("b/SKILL.md")))
    assert reg.names() == ["a", "b"]
    assert reg.catalog() == [("a", "alpha"), ("b", "beta")]
    assert "a: alpha" in reg.render_catalog()
    assert reg.get("a") is not None
    assert reg.get("missing") is None


def test_registry_add_no_clobber() -> None:
    reg = SkillRegistry()
    fm, _ = parse_frontmatter("---\nname: a\ndescription: first\n---\nx\n")
    fm2, _ = parse_frontmatter("---\nname: a\ndescription: second\n---\ny\n")
    assert reg.add(Skill(fm, Path("a"))) is True
    assert reg.add(Skill(fm2, Path("a"))) is False  # clash, not replaced
    assert reg.get("a").description == "first"
    assert reg.add(Skill(fm2, Path("a")), replace=True) is True
    assert reg.get("a").description == "second"


def test_render_catalog_empty() -> None:
    assert SkillRegistry().render_catalog() == ""


def test_discover_skills_project_overrides(tmp_path: Path) -> None:
    # A project skill shadows a same-named user/bundled one (project discovered last).
    proj = tmp_path / ".zakcode" / "skills"
    _write_skill(proj, "x", "---\nname: shared\ndescription: project version\n---\nbody\n")
    registry, errors = discover_skills(tmp_path)
    skill = registry.get("shared")
    assert skill is not None
    assert skill.description == "project version"


# ── extra_skill_dirs (--skill-dir) ──────────────────────────────────────────────


def test_discover_skills_extra_dirs(tmp_path: Path) -> None:
    """Extra skill dirs are scanned and their skills appear in the registry."""
    ext = tmp_path / "external-skills"
    _write_skill(ext, "ext", "---\nname: ext-tool\ndescription: from external dir\n---\nbody\n")
    registry, errors = discover_skills(tmp_path, extra_skill_dirs=[ext])
    skill = registry.get("ext-tool")
    assert skill is not None
    assert skill.description == "from external dir"
    assert errors == {}


def test_discover_skills_extra_dir_shadows_project(tmp_path: Path) -> None:
    """An extra skill dir is scanned AFTER project dirs, so it shadows same-named skills."""
    proj = tmp_path / ".zakcode" / "skills"
    _write_skill(proj, "s", "---\nname: shared\ndescription: project version\n---\nbody\n")
    ext = tmp_path / "external-skills"
    _write_skill(ext, "s", "---\nname: shared\ndescription: external version\n---\nbody\n")
    registry, _ = discover_skills(tmp_path, extra_skill_dirs=[ext])
    skill = registry.get("shared")
    assert skill is not None
    assert skill.description == "external version"


def test_discover_skills_extra_dir_missing_is_harmless(tmp_path: Path) -> None:
    """Passing a nonexistent extra skill dir produces no errors and no crash."""
    registry, errors = discover_skills(tmp_path, extra_skill_dirs=[tmp_path / "does-not-exist"])
    # Should succeed with no skills from the missing dir (bundled may still be present).
    assert errors == {}


def test_discover_skills_multiple_extra_dirs(tmp_path: Path) -> None:
    """Multiple extra dirs are scanned in order; later ones shadow earlier."""
    ext1 = tmp_path / "skills-a"
    ext2 = tmp_path / "skills-b"
    _write_skill(ext1, "s", "---\nname: clash\ndescription: first\n---\nbody\n")
    _write_skill(ext2, "s", "---\nname: clash\ndescription: second\n---\nbody\n")
    registry, _ = discover_skills(tmp_path, extra_skill_dirs=[ext1, ext2])
    assert registry.get("clash").description == "second"
