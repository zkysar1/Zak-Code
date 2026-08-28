"""Tests for the ordered system-prompt builder and project-context discovery."""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.agent import DYNAMIC_BOUNDARY, SystemPromptBuilder
from zakcode.agent.prompt import (
    MAX_CONTEXT_FILE_CHARS,
    MAX_CONTEXT_TOTAL_CHARS,
    discover_context,
)
from zakcode.config import PermissionTier, load_settings
from zakcode.tools import default_registry
from zakcode.tools.base import ConcurrencyClass, ToolSpec

# ── structure ──────────────────────────────────────────────────────────────────


def test_prompt_has_identity_and_boundary(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)
    assert "You are Zak Code, a vendor-agnostic AI coding assistant" in prompt
    assert DYNAMIC_BOUNDARY in prompt


def test_prompt_defines_the_harness_provenance_tags(tmp_path: Path) -> None:
    # ADR-0021: injected nudges arrive as user-role messages tagged [harness]; the prompt
    # must define the tag family once so no model attributes them to the human — and must
    # carry the no-apology instruction that keeps a nudged model acting instead of
    # apologizing.
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)
    assert "[harness], [hook], [plan], [plan critique], [verified], or" in prompt
    assert "[unverified]" in prompt
    assert "never attribute them to the user" in prompt
    assert "[user message — arrived mid-task]" in prompt  # ADR-0051: the one user-authored tag
    assert "do not apologize" in prompt


def test_operator_identity_replaces_default(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder(identity="You are Vinheim, a friendly guide.").build(settings)
    stable = prompt[: prompt.index(DYNAMIC_BOUNDARY)]
    assert "You are Vinheim, a friendly guide." in stable
    assert stable.lstrip().startswith("You are Vinheim")  # identity leads the stable tier
    assert "You are Zak Code, a vendor-agnostic AI coding assistant" not in prompt


def test_identity_none_is_byte_for_byte_default(tmp_path: Path) -> None:
    # Cache-stability guard: identity=None must reproduce the default prompt exactly.
    settings = load_settings(workspace_root=tmp_path, default_model="openai/gpt-4o")
    assert SystemPromptBuilder(identity=None).build(settings) == SystemPromptBuilder().build(
        settings
    )


def test_stable_precedes_boundary_and_context_follows(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path, default_model="openai/gpt-4o")
    prompt = SystemPromptBuilder().build(settings)

    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    stable, context = prompt[:boundary_at], prompt[boundary_at:]

    # Stable identity/safety live above the boundary.
    assert "You are Zak Code" in stable
    assert "untrusted" in stable.lower()
    assert "exfiltrate" in stable.lower()

    # Environment (and any context files) live below the boundary.
    assert "Environment:" in context
    assert str(tmp_path) in context
    assert "openai/gpt-4o" in context
    assert "Environment:" not in stable


def test_environment_section_names_the_shell(tmp_path: Path, monkeypatch) -> None:
    # The Shell line tells the model what it's driving up front; on Windows it steers
    # toward powershell to dodge the cmd.exe bash-quoting trap, on POSIX it names /bin/sh.
    from zakcode.agent import prompt as prompt_mod

    settings = load_settings(workspace_root=tmp_path, default_model="openai/gpt-4o")

    monkeypatch.setattr(prompt_mod.platform, "system", lambda: "Windows")
    win = SystemPromptBuilder().build(settings)
    assert "- Shell:" in win
    assert "powershell" in win.lower() and "cmd.exe" in win.lower()

    monkeypatch.setattr(prompt_mod.platform, "system", lambda: "Linux")
    posix = SystemPromptBuilder().build(settings)
    assert "- Shell:" in posix
    assert "/bin/sh" in posix


def test_planning_guidance_names_primitiveness_criteria(tmp_path: Path) -> None:
    # The decomposition stopping-rule must name the two criteria a single-action floor alone
    # omits: a checkable done-condition, and no approach decision still hidden in the step
    # (convergent across the /decompose + Ayoai-Mind HTN surveys; keeps weak models from
    # stopping at vague, half-decided steps). Pinned in BOTH the system prompt and the
    # update_plan tool description — the model reads the latter exactly when it fills 'subtasks'.
    prompt = SystemPromptBuilder().build(load_settings(workspace_root=tmp_path)).lower()
    assert "done-condition" in prompt  # clear completion
    assert "figure out how" in prompt  # no hidden approach decision

    desc = (default_registry().get("update_plan").spec.description or "").lower()
    assert "done-condition" in desc
    assert "figure out how" in desc


def test_step_note_is_the_checkable_done_condition(tmp_path: Path) -> None:
    # verification-as-schema, the lean way: the EXISTING `note` field IS the per-step
    # done-condition (a new `done_when` field would just duplicate it). The planning guidance
    # tells the model to record the done-condition in `note`, and the update_plan schema frames
    # the `note` field itself as a checkable acceptance criterion.
    prompt = SystemPromptBuilder().build(load_settings(workspace_root=tmp_path)).lower()
    assert "completion stays checkable" in prompt  # guidance: record the done-condition in note

    params = json.dumps(default_registry().get("update_plan").spec.parameters).lower()
    assert "done-condition" in params  # the note field schema frames itself as the done-condition


def test_tool_specs_are_summarized(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    tools = [
        ToolSpec(
            name="read_file",
            description="Read a file from the workspace.\nSecond line ignored.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file.",
            parameters={
                "type": "object",
                "properties": {"path": {}, "content": {}},
                "required": ["path", "content"],
            },
            required_permission=PermissionTier.WORKSPACE_WRITE,
            concurrency=ConcurrencyClass.PATH_SCOPED,
        ),
    ]
    prompt = SystemPromptBuilder().build(settings, tools=tools)

    assert "Available tools" in prompt
    # Required args render as a signature; only the first description line is kept.
    assert "read_file(path): Read a file from the workspace." in prompt
    assert "write_file(path, content): Create or overwrite a file." in prompt
    assert "Second line ignored." not in prompt
    # Grouped by what the tool does (permission tier), least-privileged first.
    assert "Inspect (read-only)" in prompt
    assert "Edit (writes to the workspace)" in prompt
    assert prompt.index("read_file(path)") < prompt.index("write_file(path, content)")


def test_tool_without_required_args_renders_bare_name(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    tools = [ToolSpec(name="ping", description="Health check.")]  # no required params
    prompt = SystemPromptBuilder().build(settings, tools=tools)
    assert "- ping: Health check." in prompt  # no empty parentheses


def test_no_tools_omits_tool_section(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings, tools=[])
    assert "Available tools" not in prompt


def test_extra_context_lands_in_dynamic_section(tmp_path: Path) -> None:
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings, extra_context="PROJECT_FACT_XYZ")
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    assert "PROJECT_FACT_XYZ" in prompt[boundary_at:]


# ── context-file discovery ─────────────────────────────────────────────────────


def test_zak_md_is_discovered_and_appears_in_prompt(tmp_path: Path) -> None:
    (tmp_path / "ZAK.md").write_text("Use tabs, not spaces. MEMORY_MARKER_42", encoding="utf-8")
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)

    assert "MEMORY_MARKER_42" in prompt
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    assert "MEMORY_MARKER_42" in prompt[boundary_at:]  # context files are in the dynamic tier


def test_discover_context_walks_ancestor_chain_root_to_cwd(tmp_path: Path) -> None:
    parent = tmp_path
    child = tmp_path / "sub"
    child.mkdir()
    (parent / ".git").mkdir()  # parent is the project (VCS) root, so the chain includes it
    (parent / "ZAK.md").write_text("PARENT_RULES", encoding="utf-8")
    (child / "ZAK.md").write_text("CHILD_RULES", encoding="utf-8")

    discovered = discover_context(child)
    contents = [c for _, c in discovered]

    assert "PARENT_RULES" in contents
    assert "CHILD_RULES" in contents
    # Outermost (project root) first, deepest (cwd) last.
    assert contents.index("PARENT_RULES") < contents.index("CHILD_RULES")


def test_discover_context_dedupes_identical_content(tmp_path: Path) -> None:
    parent = tmp_path
    child = tmp_path / "sub"
    child.mkdir()
    (parent / ".git").mkdir()  # project root → both dirs in the chain
    same = "IDENTICAL_MEMORY_BODY"
    (parent / "ZAK.md").write_text(same, encoding="utf-8")
    (child / "ZAK.md").write_text(same, encoding="utf-8")

    discovered = discover_context(child)
    bodies = [c for _, c in discovered]
    assert bodies.count(same) == 1  # kept once, at its shallowest occurrence
    assert discovered[0][0] == parent / "ZAK.md"


def test_discover_context_caps_per_file(tmp_path: Path) -> None:
    (tmp_path / "ZAK.md").write_text("x" * (MAX_CONTEXT_FILE_CHARS * 2), encoding="utf-8")
    discovered = discover_context(tmp_path)
    assert len(discovered) == 1
    assert len(discovered[0][1]) == MAX_CONTEXT_FILE_CHARS


def test_discover_context_caps_total(tmp_path: Path) -> None:
    # Build a deep chain of distinct file-cap-sized files far exceeding the total budget.
    (tmp_path / ".git").mkdir()  # project root at the top so the whole chain is in-project
    current = tmp_path
    for i in range(10):
        body = f"block{i}-" + "y" * MAX_CONTEXT_FILE_CHARS
        (current / "ZAK.md").write_text(body, encoding="utf-8")
        current = current / "d"
        current.mkdir()

    discovered = discover_context(current.parent)
    total = sum(len(c) for _, c in discovered)
    # The cap must actually ENGAGE: exactly floor(total/file) files survive (the rest dropped),
    # filling the budget to the brim — not a vacuous "<= cap" that also passes for 0 or 1 file.
    expected_files = MAX_CONTEXT_TOTAL_CHARS // MAX_CONTEXT_FILE_CHARS
    assert len(discovered) == expected_files  # 4 of the 10, the rest excluded
    assert total == MAX_CONTEXT_TOTAL_CHARS  # filled exactly to the cap


def test_discover_context_total_cap_truncates_the_straddling_file(tmp_path: Path) -> None:
    # Exercise the partial-slice branch: a file that straddles the total cap is truncated to the
    # remaining budget (not dropped whole, not added whole). Distinct fill chars avoid dedup and
    # give exact, prefix-free lengths.
    (tmp_path / ".git").mkdir()
    fills = ["a", "b", "c", "d", "e"]
    sizes = [MAX_CONTEXT_FILE_CHARS, MAX_CONTEXT_FILE_CHARS, MAX_CONTEXT_FILE_CHARS, 3424, 9000]
    current = tmp_path
    dirs = []
    for _ in sizes:
        dirs.append(current)
        current = current / "d"
        current.mkdir()
    for d, fill, size in zip(dirs, fills, sizes, strict=True):
        (d / "ZAK.md").write_text(fill * size, encoding="utf-8")

    discovered = discover_context(dirs[-1])  # workspace = the deepest dir; chain = all five
    total = sum(len(c) for _, c in discovered)
    assert total == MAX_CONTEXT_TOTAL_CHARS  # filled exactly to the brim …
    # 3*8192 + 3424 = 28000 consumed; the straddling 5th file is sliced to the remaining 4768.
    assert len(discovered[-1][1]) == MAX_CONTEXT_TOTAL_CHARS - (3 * MAX_CONTEXT_FILE_CHARS + 3424)


def test_discover_context_skips_empty_and_missing(tmp_path: Path) -> None:
    (tmp_path / "ZAK.md").write_text("   \n  ", encoding="utf-8")  # whitespace only
    assert discover_context(tmp_path) == []

    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()
    assert discover_context(empty_dir) == []


# ── AGENTS.md / CLAUDE.md agent guides (Codex/Claude-Code compatibility) ──────────


def test_agents_md_is_discovered_and_in_prompt(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Project rule: AGENTS_MARKER_7", encoding="utf-8")
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)
    assert "AGENTS_MARKER_7" in prompt


def test_claude_md_is_discovered(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("CLAUDE_MARKER_9", encoding="utf-8")
    bodies = [c for _, c in discover_context(tmp_path)]
    assert any("CLAUDE_MARKER_9" in b for b in bodies)


def test_both_agents_and_claude_are_loaded(tmp_path: Path) -> None:
    # This repo's exact shape: an AGENTS.md pointer + a canonical CLAUDE.md. BOTH must load, so a
    # "pick one" rule that chose AGENTS.md first would not silently drop the real guide.
    (tmp_path / "AGENTS.md").write_text("See CLAUDE.md. POINTER_MARK", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("the real guide CANONICAL_MARK", encoding="utf-8")
    bodies = [c for _, c in discover_context(tmp_path)]
    assert any("POINTER_MARK" in b for b in bodies)
    assert any("CANONICAL_MARK" in b for b in bodies)


def test_identical_guides_are_deduped(tmp_path: Path) -> None:
    # A repo that copies the same text into AGENTS.md and CLAUDE.md should not double the context.
    same = "ONE_SHARED_GUIDE_BODY"
    (tmp_path / "AGENTS.md").write_text(same, encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(same, encoding="utf-8")
    bodies = [c for _, c in discover_context(tmp_path)]
    assert bodies.count(same) == 1


# ── README in context (workspace-root only, toggleable) ──────────────────────────


def test_readme_is_loaded_at_workspace_root(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\nREADME_MARKER_3", encoding="utf-8")
    paths = [p.name for p, _ in discover_context(tmp_path)]
    assert "README.md" in paths


def test_readme_is_not_pulled_from_an_ancestor(tmp_path: Path) -> None:
    # A README is a project-ROOT doc — even when an ancestor IS in the (project) chain and its
    # GUIDE is read, its README must not leak into a deeper workspace.
    (tmp_path / ".git").mkdir()  # tmp_path is the project root → it's in the chain for `child`
    (tmp_path / "AGENTS.md").write_text("ANCESTOR_GUIDE", encoding="utf-8")
    (tmp_path / "README.md").write_text("ANCESTOR_README", encoding="utf-8")
    child = tmp_path / "project"
    child.mkdir()
    (child / "AGENTS.md").write_text("child guide", encoding="utf-8")
    bodies = [c for _, c in discover_context(child)]
    assert any("ANCESTOR_GUIDE" in b for b in bodies)  # the ancestor IS scanned for guides …
    assert all("ANCESTOR_README" not in b for b in bodies)  # … but NOT for its README


def test_readme_excluded_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("README_OFF_MARKER", encoding="utf-8")
    paths = [p.name for p, _ in discover_context(tmp_path, include_readme=False)]
    assert "README.md" not in paths


def test_readme_is_ordered_after_the_guides(tmp_path: Path) -> None:
    # Guides come first so a large README can never crowd them out of the total budget.
    (tmp_path / "AGENTS.md").write_text("GUIDE_BODY", encoding="utf-8")
    (tmp_path / "README.md").write_text("README_BODY", encoding="utf-8")
    names = [p.name for p, _ in discover_context(tmp_path)]
    assert names.index("AGENTS.md") < names.index("README.md")


def test_context_include_readme_defaults_on(tmp_path: Path) -> None:
    assert load_settings(workspace_root=tmp_path).context_include_readme is True


# ── project-boundary scoping + robustness (review fixes) ─────────────────────────


def test_discover_stops_at_the_project_root(tmp_path: Path) -> None:
    # A guide ABOVE the project (VCS) root must NOT be folded into the trusted tier — the threat
    # is a planted/stray ~/CLAUDE.md or a shared-machine parent-dir guide.
    (tmp_path / "CLAUDE.md").write_text("OUTSIDE_THE_REPO", encoding="utf-8")  # above the repo
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()  # the project boundary
    (repo / "CLAUDE.md").write_text("INSIDE_THE_REPO", encoding="utf-8")
    bodies = [c for _, c in discover_context(repo)]
    assert any("INSIDE_THE_REPO" in b for b in bodies)
    assert all("OUTSIDE_THE_REPO" not in b for b in bodies)  # the ascent stopped at .git


def test_discover_outside_a_repo_scans_only_the_workspace_root(tmp_path: Path) -> None:
    # No VCS boundary anywhere → never ascend into out-of-workspace dirs; scan only the root.
    (tmp_path / "AGENTS.md").write_text("PARENT_GUIDE", encoding="utf-8")
    child = tmp_path / "work"
    child.mkdir()
    (child / "AGENTS.md").write_text("WORKSPACE_GUIDE", encoding="utf-8")
    bodies = [c for _, c in discover_context(child)]
    assert any("WORKSPACE_GUIDE" in b for b in bodies)
    assert all("PARENT_GUIDE" not in b for b in bodies)


def test_discover_survives_a_non_utf8_file(tmp_path: Path) -> None:
    # A non-UTF-8 guide/README must be skipped, NOT crash the whole prompt build (UnicodeDecodeError
    # is a ValueError, not an OSError). A valid sibling still loads.
    (tmp_path / "AGENTS.md").write_bytes(b"\xff\xfe\x00bad utf-16 bytes")  # invalid UTF-8
    (tmp_path / "CLAUDE.md").write_text("VALID_GUIDE_MARK", encoding="utf-8")
    discovered = discover_context(tmp_path)  # must not raise
    bodies = [c for _, c in discovered]
    assert any("VALID_GUIDE_MARK" in b for b in bodies)  # the readable sibling survives
    assert all("bad utf-16" not in b for b in bodies)  # the bad file was skipped


def test_build_prompt_survives_a_non_utf8_context_file(tmp_path: Path) -> None:
    # End-to-end: the full prompt build must not raise on a non-UTF-8 context file.
    (tmp_path / "README.md").write_bytes(b"\xff\xfe garbage")
    (tmp_path / "AGENTS.md").write_text("GUIDE_OK_42", encoding="utf-8")
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)  # no crash
    assert "GUIDE_OK_42" in prompt
