"""Tests for the skills core (M7-1): frontmatter parse, lazy body, discovery."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zakcode.skills import (
    Skill,
    SkillError,
    SkillRegistry,
    discover_skill_dir,
    discover_skills,
    parse_frontmatter,
    save_skill,
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


def test_parse_frontmatter_block_style_lists() -> None:
    # YAML block sequences are the MAJORITY spelling in real Claude-Mind skills (measured
    # 2026-08-20: 60 of 78 in a live Mind used `key:` + `- item` lines) and previously
    # parsed as an empty string — silently dropping triggers routing and the cognitive
    # metadata the extras-preservation promise (audit P1-2) exists to keep.
    fm, body = parse_frontmatter(
        "---\n"
        "name: start\n"
        "description: control skill.\n"
        "allowed-tools:\n"
        "  - read_file\n"
        "  - bash\n"
        "triggers:\n"
        '  - "/start"\n'
        "companion_scripts:\n"
        "  - session-state-get.sh\n"
        "  - session-mode-get.sh\n"
        "minimum_mode: any\n"
        "---\n"
        "Body.\n"
    )
    assert fm.allowed_tools == ["read_file", "bash"]  # typed field takes the block form too
    assert fm.extras["triggers"] == ["/start"]
    assert fm.extras["companion_scripts"] == ["session-state-get.sh", "session-mode-get.sh"]
    assert fm.extras["minimum_mode"] == "any"  # scalar AFTER a block is not swallowed
    assert body == "Body."


def test_parse_frontmatter_bare_empty_key_stays_empty() -> None:
    # A bare `key:` with NO dash lines after it keeps the pre-existing empty-string
    # behavior — the block lookahead only fires when items actually follow.
    fm, _ = parse_frontmatter("---\nname: a\nprevious_revision_id:\nversion: 1.0.0\n---\nx\n")
    assert fm.extras["previous_revision_id"] == ""
    assert fm.version == "1.0.0"


def test_parse_frontmatter_block_list_of_mappings_survives_as_strings() -> None:
    # A `- name: x` item (list-of-maps, seen in Mind `parameters:` blocks) is kept as the
    # string "name: x" — imperfect but strictly better than vanishing, and the non-dash
    # continuation lines fall through to the ordinary key parse exactly as before.
    fm, _ = parse_frontmatter("---\nname: a\nparameters:\n  - name: agent\n---\nx\n")
    assert fm.extras["parameters"] == ["name: agent"]


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


# ADR-0245: an edited SKILL.md is read again, as Claude Code does within a session. A Mind's
# Body runs for hours while its loop merges framework updates, and a body cached for the
# life of the process kept serving it the old text.


def _rewrite(path: Path, text: str, *, mtime_ns: int) -> None:
    """Replace a SKILL.md's text and set its mtime, so each test controls which half of the
    file's (mtime, size) stamp moves."""
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_a_skill_edited_to_a_new_size_is_read_again(tmp_path: Path) -> None:
    _write_skill(tmp_path, "s")
    skill = discover_skill_dir(tmp_path)[0][0]
    path = tmp_path / "s" / "SKILL.md"
    assert "conventional-commit" in skill.body()
    # Same mtime, different size: only the size half of the stamp moves.
    _rewrite(path, _SKILL.replace("+ body", "only"), mtime_ns=path.stat().st_mtime_ns)
    body = skill.body()
    assert "subject only" in body
    assert "+ body" not in body


def test_a_skill_edited_to_the_same_size_is_read_again(tmp_path: Path) -> None:
    _write_skill(tmp_path, "s")
    skill = discover_skill_dir(tmp_path)[0][0]
    path = tmp_path / "s" / "SKILL.md"
    assert "Read the diff" in skill.body()
    edited = _SKILL.replace("Read the diff", "Load the diff")
    assert len(edited) == len(_SKILL)  # the size half of the stamp cannot see this edit
    _rewrite(path, edited, mtime_ns=path.stat().st_mtime_ns + 1_000_000_000)
    assert "Load the diff" in skill.body()


def test_an_unchanged_skill_is_read_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_skill(tmp_path, "s")
    skill = discover_skill_dir(tmp_path)[0][0]
    reads: list[Path] = []
    real_read = Path.read_text

    def counting_read(self: Path, *args: object, **kwargs: object) -> str:
        reads.append(self)
        return real_read(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", counting_read)
    first = skill.body()
    assert skill.body() == first
    assert skill.body() == first
    assert len(reads) == 1


def test_a_skill_whose_file_vanished_is_not_served_from_the_cache(tmp_path: Path) -> None:
    _write_skill(tmp_path, "s")
    skill = discover_skill_dir(tmp_path)[0][0]
    assert "conventional-commit" in skill.body()
    (tmp_path / "s" / "SKILL.md").unlink()
    with pytest.raises(OSError):
        skill.body()


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
    cat = reg.render_catalog()
    # Skills render as the exact `use_skill(...)` call, NOT a bare "name: desc" line
    # (which small models mistake for a tool). See the render_catalog anti-confusion fix.
    assert 'Skill(skill="a")' in cat
    assert "alpha" in cat
    assert "a: alpha" not in cat
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


def test_render_catalog_states_user_provenance_contract() -> None:
    # The catalog block is where the model learns what a <command-name> frame MEANS: a human
    # typed that slash in the terminal, so "user-invocable only" rules are satisfied. Without
    # this sentence a rule-following model refuses the operator's own keystroke (live
    # 2026-08-19: a Mind's /start declined as "user-only command" — typed by the user).
    reg = SkillRegistry()
    fm, _ = parse_frontmatter("---\nname: a\ndescription: alpha\n---\nbody a\n")
    reg.add(Skill(fm, Path("a/SKILL.md")))
    cat = reg.render_catalog()
    assert "BEGINS with a <command-name> block" in cat
    assert "BY THE USER" in cat


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


# ── bundled skills ──────────────────────────────────────────────────────────────


def test_bundled_research_skill_is_discoverable_and_well_formed() -> None:
    # The shipped `research` playbook lives in src/zakcode/skills/bundled and must parse, declare
    # the web tools, and tell the model to fan out search then fetch then synthesize.
    import zakcode.skills as skills_mod

    bundled = Path(skills_mod.__file__).parent / "bundled"
    found, errors = discover_skill_dir(bundled)
    assert errors == {}
    research = next((s for s in found if s.name == "research"), None)
    assert research is not None, f"bundled skills found: {[s.name for s in found]}"
    assert set(research.frontmatter.allowed_tools) >= {"web_search", "web_fetch"}
    body = research.body().lower()
    assert "web_search" in body and "web_fetch" in body
    assert "parallel" in body and "synthe" in body  # the playbook's shape


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


# ── audit3 #2: skills discovery / save_skill stay inside the skills tree ──────


def _make_dir_link(link: Path, target: Path) -> bool:
    """Create a directory junction (Windows, no admin) or symlink (POSIX). False if unable."""
    import os
    import subprocess
    import sys

    if sys.platform == "win32":
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return r.returncode == 0 and link.exists()
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        return False


def test_discover_skips_skill_dir_escaping_root_via_link(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "SKILL.md").write_text("---\nname: evil\ndescription: x\n---\nstolen\n")
    if not _make_dir_link(skills_dir / "evil", outside):
        pytest.skip("cannot create a directory junction/symlink in this environment")
    skills, errors = discover_skill_dir(skills_dir)
    assert skills == []  # the out-of-tree skill is NOT pulled into the catalog/prompt
    assert "evil" in errors


def test_save_skill_refuses_out_of_tree_link(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if not _make_dir_link(skills_dir / "evil", outside):
        pytest.skip("cannot create a directory junction/symlink in this environment")
    with pytest.raises(SkillError):
        save_skill("evil", "d", "body text", skills_dir=skills_dir, overwrite=True)
    assert not (outside / "SKILL.md").exists()  # nothing written through the link


def test_render_catalog_maps_claude_codes_tool_names() -> None:
    # ADR-0187: a framework written for Claude Code says "Skill('aspirations') with
    # args='loop'" and "ScheduleWakeup(...)" in its hook reasons and script output; a small
    # model could not map those onto use_skill / schedule_wakeup on its own (measured
    # 2026-09-17: hours of text against exactly that reason). One static line says how.
    reg = SkillRegistry()
    fm, _ = parse_frontmatter("---\nname: a\ndescription: alpha\n---\nbody a\n")
    reg.add(Skill(fm, Path("a/SKILL.md")))
    cat = reg.render_catalog()
    # ADR-0190: the tool IS named Skill, so the catalog states the imperative's meaning in the
    # tool's own name and carries no translation table.
    assert "`Skill(<name>) with args='<args>'` means exactly this call" in cat
    assert "use_skill" not in cat and "schedule_wakeup" not in cat
