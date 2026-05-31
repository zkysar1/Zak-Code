"""Tests for the exact-string edit tool (:class:`EditFileTool`)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.edit import EditFileTool


def _run(coro):
    """Run an async coroutine to completion (tests are synchronous)."""
    return asyncio.run(coro)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


def _temp_files(directory: Path) -> list[Path]:
    """Leftover atomic-write temp files, if any."""
    return list(directory.glob(".zaktmp-*"))


def test_unique_replace(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("hello world", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "a.txt", "old_string": "world", "new_string": "there"},
            _ctx(tmp_path),
        )
    )

    assert not result.is_error
    assert result.data == {"path": str(target.resolve()), "replacements": 1}
    assert target.read_text(encoding="utf-8") == "hello there"
    assert _temp_files(tmp_path) == []


def test_replace_all_multi(tmp_path: Path) -> None:
    target = tmp_path / "b.txt"
    target.write_text("x x x", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "b.txt", "old_string": "x", "new_string": "y", "replace_all": True},
            _ctx(tmp_path),
        )
    )

    assert not result.is_error
    assert result.data["replacements"] == 3
    assert target.read_text(encoding="utf-8") == "y y y"
    assert _temp_files(tmp_path) == []


def test_not_found_is_error(tmp_path: Path) -> None:
    target = tmp_path / "c.txt"
    target.write_text("hello", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "c.txt", "old_string": "absent", "new_string": "z"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error
    assert "not found" in result.output
    assert target.read_text(encoding="utf-8") == "hello"


def test_ambiguous_match_reports_count(tmp_path: Path) -> None:
    target = tmp_path / "d.txt"
    target.write_text("dup dup dup", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "d.txt", "old_string": "dup", "new_string": "z"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error
    assert "3 occurrences" in result.output
    assert "replace_all" in result.output
    # File is untouched on an ambiguous match.
    assert target.read_text(encoding="utf-8") == "dup dup dup"


def test_identical_old_and_new_is_error(tmp_path: Path) -> None:
    target = tmp_path / "e.txt"
    target.write_text("same", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "e.txt", "old_string": "same", "new_string": "same"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error
    assert "identical" in result.output


def test_empty_old_string_is_error(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("content", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "f.txt", "old_string": "", "new_string": "new"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error
    assert "write_file" in result.output
    assert target.read_text(encoding="utf-8") == "content"


def test_path_escape_is_error(tmp_path: Path) -> None:
    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "../escape.txt", "old_string": "a", "new_string": "b"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error
    assert "outside the workspace" in result.output


def test_missing_file_is_error(tmp_path: Path) -> None:
    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "nope.txt", "old_string": "a", "new_string": "b"},
            _ctx(tmp_path),
        )
    )

    assert result.is_error
    assert "File not found" in result.output


def test_content_round_trips_exactly(tmp_path: Path) -> None:
    target = tmp_path / "g.txt"
    original = "line1\nline2\r\n\ttab\nend"  # mixed newlines + tab preserved
    target.write_bytes(original.encode("utf-8"))

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "g.txt", "old_string": "line2", "new_string": "LINE2"},
            _ctx(tmp_path),
        )
    )

    assert not result.is_error
    expected = original.replace("line2", "LINE2")
    assert target.read_bytes().decode("utf-8") == expected
    assert _temp_files(tmp_path) == []


def test_no_temp_file_left_after_success(tmp_path: Path) -> None:
    target = tmp_path / "h.txt"
    target.write_text("alpha beta", encoding="utf-8")

    tool = EditFileTool()
    result = _run(
        tool.execute(
            {"path": "h.txt", "old_string": "beta", "new_string": "gamma"},
            _ctx(tmp_path),
        )
    )

    assert not result.is_error
    assert _temp_files(tmp_path) == []


def test_default_registry_exposes_edit_alongside_originals(tmp_path: Path) -> None:
    registry = default_registry()

    target = tmp_path / "i.txt"
    target.write_text("greet name", encoding="utf-8")

    # Both the canonical name and the alias must resolve and execute.
    result_canonical = _run(
        registry.execute(
            "edit_file",
            {"path": "i.txt", "old_string": "greet", "new_string": "hi"},
            _ctx(tmp_path),
        )
    )
    assert not result_canonical.is_error
    assert target.read_text(encoding="utf-8") == "hi name"

    result_alias = _run(
        registry.execute(
            "edit",
            {"path": "i.txt", "old_string": "name", "new_string": "world"},
            _ctx(tmp_path),
        )
    )
    assert not result_alias.is_error
    assert target.read_text(encoding="utf-8") == "hi world"

    # The original six tools must still be registered.
    for name in ("read_file", "write_file", "list_dir", "glob", "grep", "bash"):
        result = _run(registry.execute(name, {}, _ctx(tmp_path)))
        # We only care that the tool resolves (executes without an
        # "unknown tool" failure); a missing arg yielding ToolResult.error is fine.
        assert not _is_unknown_tool(result)


def _is_unknown_tool(result: object) -> bool:
    """Heuristic: did the registry fail to find the tool by name?"""
    output = getattr(result, "output", "") or ""
    lowered = output.lower()
    return "unknown tool" in lowered or "not registered" in lowered or "no such tool" in lowered


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
