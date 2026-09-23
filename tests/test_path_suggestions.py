"""A not-found answer names what the workspace DOES have under that name (ADR-0040).

Field transcript 2026-08-27: the model looked for ``google-drive-list`` by path, got
``File not found``, marked the step blocked and asked the operator for the path — twice.
"you can't grep it?" → seven hits on the first search. The tools now do that search
themselves and put the answer inside the failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from zakcode.tools.base import ToolContext, ToolResult
from zakcode.tools.builtins._suggest import literal_stem, suggest
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.read_file import ReadFileTool


def _workspace(tmp_path: Path) -> ToolContext:
    skill = tmp_path / ".zakcode" / "skills" / "google-drive-list"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: google-drive-list\n---\nList files.\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text(
        "# notes\nrun python3 google-drive-list to list the drive\n"
    )
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "gdrive_list.py").write_text("print('x')\n")
    (tmp_path / "unrelated.txt").write_text("nothing here\n")
    return ToolContext(workspace_root=tmp_path)


def _run(tool: Any, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    return asyncio.run(tool.execute(args, ctx))


def test_read_file_not_found_lists_paths_by_name_and_by_content(tmp_path: Path) -> None:
    ctx = _workspace(tmp_path)
    result = _run(ReadFileTool(), {"path": "google-drive-list"}, ctx)
    assert result.is_error
    assert result.output.startswith("File not found: google-drive-list\n")
    assert "Closest paths by name:" in result.output
    assert ".zakcode/skills/google-drive-list" in result.output
    assert "Files whose content mentions 'google-drive-list':" in result.output
    assert "docs/notes.md:2" in result.output
    assert result.fix is not None and "not the workspace" in result.fix
    assert result.data is not None and result.data["suggestions"]["by_name"]


def test_read_file_not_found_with_nothing_close_says_search_first(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    result = _run(ReadFileTool(), {"path": "zzz-missing.py"}, ToolContext(workspace_root=tmp_path))
    assert result.is_error and result.output == "File not found: zzz-missing.py"
    assert result.fix is not None and 'Grep(pattern="zzz\\-missing\\.py")' in result.fix
    assert result.data == {"suggestions": {"by_name": [], "by_content": []}}


def test_glob_no_match_suggests_from_the_pattern_literal(tmp_path: Path) -> None:
    ctx = _workspace(tmp_path)
    result = _run(GlobTool(), {"pattern": "**/google-drive-list.py"}, ctx)
    assert not result.is_error
    assert result.output.startswith("(no matches)\n")
    assert ".zakcode/skills/google-drive-list" in result.output
    assert result.data is not None and result.data["count"] == 0
    assert result.hint is not None and "not the workspace" in result.hint


def test_glob_extension_only_pattern_stays_a_plain_empty_listing(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n")
    result = _run(GlobTool(), {"pattern": "*.py"}, ToolContext(workspace_root=tmp_path))
    assert result.output == "(no matches)" and result.hint is None


def test_list_dir_and_grep_not_found_suggest_too(tmp_path: Path) -> None:
    ctx = _workspace(tmp_path)
    listed = _run(ListDirTool(), {"path": "skills/google-drive-list"}, ctx)
    assert listed.is_error and ".zakcode/skills/google-drive-list" in listed.output
    grepped = _run(GrepTool(), {"pattern": "x", "path": "gdrive"}, ctx)
    assert grepped.is_error and grepped.output.startswith("Path not found: gdrive\n")
    assert "tools/gdrive_list.py" in grepped.output


def test_suggest_skips_ignored_dirs_and_never_raises(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "google-drive-list").write_text("hook\n")
    (tmp_path / "node_modules" / "google-drive-list").mkdir(parents=True)
    by_name, by_content = suggest("google-drive-list", tmp_path)
    assert by_name == [] and by_content == []
    assert suggest("", tmp_path) == ([], [])
    assert suggest("x", tmp_path) == ([], [])
    assert suggest("google-drive-list", tmp_path / "does-not-exist") == ([], [])


def test_an_extra_root_under_an_ignored_dir_is_searched_by_its_own_rules(tmp_path: Path) -> None:
    """ADR-0229. A Mind keeps its world under ``.mind-data/``, which the checkout's
    .gitignore excludes, and declares it as an extra root. Field 2026-09-23 (a 27B worker):
    ``Read world/program.md`` came back with six look-alike names from the checkout and never
    the file itself, because the extra root was judged by the primary root's ignore rules."""
    (tmp_path / ".gitignore").write_text(".mind-data/\nbuild/\n")
    world = tmp_path / ".mind-data" / "world"
    world.mkdir(parents=True)
    (world / "program.md").write_text("# The Program\n")
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "old-program.md").write_text("x\n")
    # Positive control (guard-4166): what the PRIMARY root ignores stays hidden.
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "program.md").write_text("generated\n")

    by_name, _ = suggest("world/program.md", tmp_path, [world])
    assert by_name[0] == ".mind-data/world/program.md"
    assert "notes/old-program.md" in by_name
    assert "build/program.md" not in by_name
    # Undeclared, the ignored directory stays ignored.
    by_name, _ = suggest("world/program.md", tmp_path)
    assert by_name == ["notes/old-program.md"]

    ctx = ToolContext(workspace_root=tmp_path, extra_workspace_roots=[world])
    result = _run(ReadFileTool(), {"path": "world/program.md"}, ctx)
    assert result.is_error
    assert result.output.splitlines()[:3] == [
        "File not found: world/program.md",
        "Closest paths by name:",
        "  .mind-data/world/program.md",
    ]


def test_literal_stem() -> None:
    assert literal_stem("**/google-drive-list*") == "google-drive-list"
    assert literal_stem("src/**/test_*.py") == "test_"
    assert literal_stem("*.py") == ""
    assert literal_stem("**/*") == ""
