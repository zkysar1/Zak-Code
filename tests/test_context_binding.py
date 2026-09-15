"""Binding framing of the project context (ADR-0179): the folded guides open with a line that says
the rules below bind every file written in the project and that a rule beats the model's usual
approach. The per-file folds are untouched; only the block's opening changes."""

from __future__ import annotations

from pathlib import Path

from zakcode.agent.prompt import SystemPromptBuilder, discover_context

BINDING = "The guides and conventions are BINDING for every file you write in this project"


def test_context_block_opens_with_the_binding_line(tmp_path: Path) -> None:
    guide = "# Guide\n\n## Rules\n\nAssert deltas, never absolutes.\n"
    (tmp_path / "CLAUDE.md").write_text(guide, encoding="utf-8")
    rendered = SystemPromptBuilder._render_context(discover_context(tmp_path))
    assert rendered.startswith("Project context (")
    assert BINDING in rendered
    assert "the rule wins" in rendered
    assert "Assert deltas, never absolutes." in rendered  # the fold itself is unchanged


def test_no_context_files_means_no_block(tmp_path: Path) -> None:
    assert SystemPromptBuilder._render_context(discover_context(tmp_path)) == ""


def test_readme_alone_is_framed_as_orientation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Project\n\nHow to run it.\n", encoding="utf-8")
    rendered = SystemPromptBuilder._render_context(discover_context(tmp_path, include_readme=True))
    assert "The README is orientation." in rendered
    assert "How to run it." in rendered
