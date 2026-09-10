"""A match line is clipped to a window around the match; the whole output is capped (ADR-0130).

Field 2026-09-10, a Mind workspace on a local 35B model: the Mind's stores are JSONL, one
record per line, and a record can carry a whole knowledge article. Three searches over the
agent's directory returned 18 matches each of ~10 KB — 900 transcript lines per call — the
model re-ran near-identical searches three times and the no-progress rail fired. Claude Code's
grep never hands the model a 10 KB line either. The match is still shown, with the dropped
lengths marked, and read_file is the way to the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.grep import _MAX_LINE_CHARS, _MAX_OUTPUT_CHARS, GrepTool


def _body(row: str, path: Path, line_no: int = 1) -> str:
    """The text after ``<path>:<line>:`` — split on the known prefix, never on ":" (a Windows
    path carries its own colon)."""
    prefix = f"{path}:{line_no}:"
    assert row.startswith(prefix), row[:120]
    return row[len(prefix) :]


def _store(tmp_path: Path, records: int = 1, filler: int = 5000) -> Path:
    rows = [
        json.dumps({"id": f"rec-{i:03d}", "title": "Yahoo league data", "body": "x" * filler})
        for i in range(records)
    ]
    (tmp_path / "records.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text(
        "the Yahoo league id lives in prep-tasks\n", encoding="utf-8"
    )
    return tmp_path


async def test_a_long_match_line_is_clipped_around_the_match(tmp_path: Path) -> None:
    ws = _store(tmp_path)
    result = await GrepTool().execute({"pattern": "Yahoo league"}, ToolContext(workspace_root=ws))
    assert not result.is_error
    rows = result.output.splitlines()
    long_row = next(r for r in rows if "records.jsonl" in r)
    short_row = next(r for r in rows if "notes.md" in r)
    assert "Yahoo league data" in long_row
    body = _body(long_row, ws / "records.jsonl")
    assert len(body) <= _MAX_LINE_CHARS + 60  # the window plus the tail marker
    assert "chars; read_file the line for the rest]" in long_row
    assert short_row.endswith("the Yahoo league id lives in prep-tasks")  # short lines untouched
    assert result.data is not None and result.data["count"] == 2 and result.data["capped"] == 0


async def test_the_window_keeps_the_match_when_it_sits_deep_in_the_line(tmp_path: Path) -> None:
    line = "a" * 4000 + " NEEDLE-HERE " + "b" * 4000
    (tmp_path / "wide.txt").write_text(line + "\n", encoding="utf-8")
    result = await GrepTool().execute(
        {"pattern": "NEEDLE-HERE"}, ToolContext(workspace_root=tmp_path)
    )
    row = result.output.splitlines()[0]
    assert "NEEDLE-HERE" in row and _body(row, tmp_path / "wide.txt").startswith("[… +")


async def test_the_whole_output_is_capped_with_a_count(tmp_path: Path) -> None:
    ws = _store(tmp_path, records=400, filler=400)  # 400 rows x ~300 chars > the cap
    result = await GrepTool().execute(
        {"pattern": "Yahoo league", "glob": "*.jsonl"}, ToolContext(workspace_root=ws)
    )
    assert not result.is_error
    assert len(result.output) <= _MAX_OUTPUT_CHARS + 200
    assert "output capped:" in result.output and "matches not shown" in result.output
    assert result.data is not None
    assert result.data["count"] == 400 and result.data["capped"] > 0
