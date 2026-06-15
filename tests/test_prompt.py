"""Tests for the ordered system-prompt builder and ZAK.md memory discovery."""

from __future__ import annotations

from pathlib import Path

from zakcode.agent import DYNAMIC_BOUNDARY, SystemPromptBuilder
from zakcode.agent.prompt import (
    MAX_MEMORY_FILE_CHARS,
    MAX_MEMORY_TOTAL_CHARS,
    discover_memory,
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

    # Environment (and any memory) live below the boundary.
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


# ── memory discovery ───────────────────────────────────────────────────────────


def test_zak_md_is_discovered_and_appears_in_prompt(tmp_path: Path) -> None:
    (tmp_path / "ZAK.md").write_text("Use tabs, not spaces. MEMORY_MARKER_42", encoding="utf-8")
    settings = load_settings(workspace_root=tmp_path)
    prompt = SystemPromptBuilder().build(settings)

    assert "MEMORY_MARKER_42" in prompt
    boundary_at = prompt.index(DYNAMIC_BOUNDARY)
    assert "MEMORY_MARKER_42" in prompt[boundary_at:]  # memory is dynamic context


def test_discover_memory_walks_ancestor_chain_root_to_cwd(tmp_path: Path) -> None:
    parent = tmp_path
    child = tmp_path / "sub"
    child.mkdir()
    (parent / "ZAK.md").write_text("PARENT_RULES", encoding="utf-8")
    (child / "ZAK.md").write_text("CHILD_RULES", encoding="utf-8")

    discovered = discover_memory(child)
    contents = [c for _, c in discovered]

    assert "PARENT_RULES" in contents
    assert "CHILD_RULES" in contents
    # Outermost (ancestor) first, deepest (cwd) last.
    assert contents.index("PARENT_RULES") < contents.index("CHILD_RULES")


def test_discover_memory_dedupes_identical_content(tmp_path: Path) -> None:
    parent = tmp_path
    child = tmp_path / "sub"
    child.mkdir()
    same = "IDENTICAL_MEMORY_BODY"
    (parent / "ZAK.md").write_text(same, encoding="utf-8")
    (child / "ZAK.md").write_text(same, encoding="utf-8")

    discovered = discover_memory(child)
    bodies = [c for _, c in discovered]
    assert bodies.count(same) == 1  # kept once, at its shallowest occurrence
    assert discovered[0][0] == parent / "ZAK.md"


def test_discover_memory_caps_per_file(tmp_path: Path) -> None:
    (tmp_path / "ZAK.md").write_text("x" * (MAX_MEMORY_FILE_CHARS * 2), encoding="utf-8")
    discovered = discover_memory(tmp_path)
    assert len(discovered) == 1
    assert len(discovered[0][1]) == MAX_MEMORY_FILE_CHARS


def test_discover_memory_caps_total(tmp_path: Path) -> None:
    # Build a deep chain of distinct large files exceeding the total budget.
    current = tmp_path
    for i in range(10):
        (current / "ZAK.md").write_text(
            f"block{i}-" + "y" * MAX_MEMORY_FILE_CHARS, encoding="utf-8"
        )
        current = current / "d"
        current.mkdir()

    discovered = discover_memory(current.parent)
    total = sum(len(c) for _, c in discovered)
    assert total <= MAX_MEMORY_TOTAL_CHARS


def test_discover_memory_skips_empty_and_missing(tmp_path: Path) -> None:
    (tmp_path / "ZAK.md").write_text("   \n  ", encoding="utf-8")  # whitespace only
    assert discover_memory(tmp_path) == []

    empty_dir = tmp_path / "nothing"
    empty_dir.mkdir()
    assert discover_memory(empty_dir) == []
