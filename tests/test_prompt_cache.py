"""The prompt-cache invariant: the STABLE tier is byte-identical regardless of the
per-session (dynamic) facts, so a provider's cached prefix is never invalidated."""

from __future__ import annotations

from pathlib import Path

from zakcode.agent import DYNAMIC_BOUNDARY, SystemPromptBuilder
from zakcode.config import load_settings
from zakcode.tools.base import ToolSpec

_TOOLS = [ToolSpec(name="read_file", description="Read a file.")]


def _stable(prompt: str) -> str:
    return prompt[: prompt.index(DYNAMIC_BOUNDARY)]


def test_stable_tier_is_identical_across_dynamic_changes(tmp_path: Path) -> None:
    # Same builder (same identity/tools/rules/skills), different dynamic facts.
    builder = SystemPromptBuilder(
        rules="Project rules:\n\n## x\nALWAYS_TABS",
        extra_instructions="Available skills:\n- helper: does things",
    )
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    s1 = builder.build(load_settings(workspace_root=a, default_model="openai/gpt-4o"), tools=_TOOLS)
    s2 = builder.build(load_settings(workspace_root=b, default_model="ollama/llama3"), tools=_TOOLS)

    # The cacheable prefix must be byte-identical despite different cwd + model...
    assert _stable(s1) == _stable(s2)
    # ...and the rules/skills DO live in that stable prefix (cache-safe placement).
    assert "ALWAYS_TABS" in _stable(s1)
    assert "helper: does things" in _stable(s1)
    # ...while the dynamic tier genuinely differs (proving the test is meaningful).
    assert "openai/gpt-4o" in s1 and "openai/gpt-4o" not in s2


def test_build_is_deterministic(tmp_path: Path) -> None:
    builder = SystemPromptBuilder(rules="## r\nbody")
    settings = load_settings(workspace_root=tmp_path)
    assert builder.build(settings, tools=_TOOLS) == builder.build(settings, tools=_TOOLS)
