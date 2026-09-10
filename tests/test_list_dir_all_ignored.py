"""A directory whose every entry is ignored is listed anyway (ADR-0129).

Field 2026-09-10, a Mind workspace on a local 35B model: the Mind's world lives under a
gitignored root (``.mind-data/``). Three ``list_dir`` calls on its knowledge tree came back as
ONE line — ``[... 5 ignored entries hidden; include_ignored=true to show ...]`` — the model
skimmed that note four times, the no-progress rail fired, and five of the turn's twelve
iterations went to a directory the tool could have simply shown. Claude Code's listing shows
everything. The hygiene for MIXED directories (node_modules beside source) is kept; only the
all-hidden case, the one shape that reads as "nothing here", changes.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.list_dir import ListDirTool


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text("data/\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print(1)\n", encoding="utf-8")
    tree = tmp_path / "data" / "world" / "tree"
    tree.mkdir(parents=True)
    (tree / "strategy.md").write_text("# strategy\n", encoding="utf-8")
    (tree / "execution").mkdir()
    (tree / ".git").mkdir()  # the always-hidden class stays hidden even here
    return tmp_path


async def test_an_all_ignored_directory_is_listed_tagged_instead_of_hidden(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    result = await ListDirTool().execute(
        {"path": "data/world/tree"}, ToolContext(workspace_root=ws)
    )
    assert not result.is_error
    lines = result.output.splitlines()
    assert lines[:2] == ["execution/", "strategy.md"]
    assert ".git/" not in lines  # the always-hidden class is not resurrected
    assert "all 2 entries here are ignored" in lines[-1] and "shown anyway" in lines[-1]
    assert result.data is not None
    assert result.data["entries"] == ["execution", "strategy.md"]
    assert result.data["all_ignored"] is True and result.data["ignored"] == 3


async def test_a_mixed_directory_still_hides_its_ignored_entries_behind_a_count(
    tmp_path: Path,
) -> None:
    ws = _workspace(tmp_path)
    result = await ListDirTool().execute({}, ToolContext(workspace_root=ws))
    assert not result.is_error
    assert "src/" in result.output and "data/" not in result.output.replace("[", "")
    assert "1 ignored entries hidden; include_ignored=true" in result.output
    assert result.data is not None and result.data["all_ignored"] is False


async def test_include_ignored_is_unchanged(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    result = await ListDirTool().execute(
        {"path": "data/world/tree", "include_ignored": True}, ToolContext(workspace_root=ws)
    )
    assert not result.is_error
    assert result.output.splitlines()[:2] == ["execution/", "strategy.md"]
    assert result.data is not None and result.data["all_ignored"] is False
