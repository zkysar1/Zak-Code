"""Tests for the substrate-polish pass: #4 idempotency-as-contract and #5 dense output.

Hermetic: tmp-dir workspaces via ToolContext, an in-process SqliteMemoryProvider for remember.
The principle under test: a repeated call that finds the desired state already in place returns
a benign no-op (not an error), and any output that is capped says so EXPLICITLY (never a silent
cut that reads to a small model as "that's everything").
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zakcode.memory.sqlite_store import SqliteMemoryProvider
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins import grep as grep_mod
from zakcode.tools.builtins import list_dir as list_dir_mod
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.memory import RememberTool
from zakcode.tools.builtins.read_file import ReadFileTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


# ── #4 idempotency ──────────────────────────────────────────────────────────────


async def test_edit_already_applied_is_benign_noop(tmp_path: Path, ctx: ToolContext) -> None:
    # The file already holds the post-edit state: old_string absent, a DISTINCTIVE new_string
    # already present (>= _MIN_NOOP_LEN chars), so a retry is a benign no-op, not an error.
    (tmp_path / "f.txt").write_text("timeout = 60\n", encoding="utf-8")
    res = await EditFileTool().execute(
        {"path": "f.txt", "old_string": "timeout = 30", "new_string": "timeout = 60"}, ctx
    )
    assert res.is_error is False  # NOT an error — the desired edit is already in place
    assert res.data and res.data.get("already_applied") is True
    assert res.data.get("replacements") == 0
    assert "already applied" in res.output.lower()


async def test_edit_short_coincidental_new_string_still_errors(
    tmp_path: Path, ctx: ToolContext
) -> None:
    # old_string absent (wrong target) but a SHORT new_string ("beta") coincidentally appears
    # elsewhere. The no-op must NOT fire (it would mask a genuine wrong-target edit) — the
    # recoverable error fires instead, and the file is left unchanged. (review: coincidental sub)
    original = "alpha\nbeta\n"
    (tmp_path / "f.txt").write_text(original, encoding="utf-8")
    res = await EditFileTool().execute(
        {"path": "f.txt", "old_string": "gamma", "new_string": "beta"}, ctx
    )
    assert res.is_error is True
    assert res.fix and "re-read" in res.fix.lower()
    assert (tmp_path / "f.txt").read_text(encoding="utf-8") == original  # untouched


async def test_edit_genuine_miss_still_errors(tmp_path: Path, ctx: ToolContext) -> None:
    # old_string absent AND new_string absent -> a real miss (typo / wrong file), still an error.
    (tmp_path / "f.txt").write_text("abc\n", encoding="utf-8")
    res = await EditFileTool().execute(
        {"path": "f.txt", "old_string": "xyz", "new_string": "qqq"}, ctx
    )
    assert res.is_error is True
    assert res.fix and "re-read" in res.fix.lower()


async def test_edit_empty_new_string_does_not_falsely_noop(
    tmp_path: Path, ctx: ToolContext
) -> None:
    # A pure deletion (new_string="") can't be told apart from a wrong target, so a missing
    # old_string must still error rather than masquerade as "already applied".
    (tmp_path / "f.txt").write_text("abc\n", encoding="utf-8")
    res = await EditFileTool().execute(
        {"path": "f.txt", "old_string": "zzz", "new_string": ""}, ctx
    )
    assert res.is_error is True


async def test_remember_same_text_is_idempotent(tmp_path: Path, ctx: ToolContext) -> None:
    mem = SqliteMemoryProvider(str(tmp_path / "m.db"))
    tool = RememberTool(mem)
    first = await tool.execute({"text": "my favorite color is blue"}, ctx)
    assert first.is_error is False and mem.count() == 1
    second = await tool.execute({"text": "my favorite color is blue"}, ctx)
    assert second.is_error is False
    assert second.data and second.data.get("duplicate") is True
    assert mem.count() == 1  # no duplicate row created
    # A genuinely different note still stores.
    third = await tool.execute({"text": "my dog is named Mochi"}, ctx)
    assert third.is_error is False and mem.count() == 2


# ── #5 dense output: explicit, actionable truncation markers ─────────────────────


async def test_read_file_partial_slice_has_continuation_marker(
    tmp_path: Path, ctx: ToolContext
) -> None:
    (tmp_path / "big.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8")
    res = await ReadFileTool().execute({"path": "big.txt", "limit": 3}, ctx)
    assert res.is_error is False
    assert "line1" in res.output and "line3" in res.output
    assert "line4" not in res.output  # only 3 lines returned
    # The slice stopped before EOF -> an explicit, actionable marker, not a silent cut.
    assert "of 10" in res.output and "offset=4" in res.output


async def test_read_large_file_slice_uses_real_total_not_byte_window(
    tmp_path: Path, ctx: ToolContext
) -> None:
    # review HIGH: line accounting must be off the FULL file, not the byte-capped text. A
    # >100KB file read with an in-range offset must report the REAL total and never falsely
    # claim "beyond the end of the file" just because the 100KB byte window was exceeded.
    content = "".join(f"line{i}\n" for i in range(1, 20001))  # ~155KB, 20000 lines
    (tmp_path / "huge.txt").write_text(content, encoding="utf-8")
    res = await ReadFileTool().execute({"path": "huge.txt", "offset": 15000, "limit": 5}, ctx)
    assert res.is_error is False
    assert "line15000" in res.output
    assert "beyond the end" not in res.output  # the offset is valid in the real file
    assert "of 20000" in res.output  # real total, not the truncated byte-window count
    assert res.data and res.data.get("total_lines") == 20000


async def test_read_file_slice_to_eof_has_no_marker(tmp_path: Path, ctx: ToolContext) -> None:
    (tmp_path / "big.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8")
    res = await ReadFileTool().execute({"path": "big.txt", "offset": 9, "limit": 5}, ctx)
    assert res.is_error is False
    assert "use offset=" not in res.output  # reached EOF, nothing more to page


async def test_list_dir_soft_cap_has_marker(
    tmp_path: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(list_dir_mod, "_MAX_ENTRIES", 2)
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    res = await ListDirTool().execute({"path": "."}, ctx)
    assert res.is_error is False
    assert "more entries" in res.output
    assert res.data and res.data.get("truncated") is True
    assert res.data.get("count") == 3  # structured data still carries the full count


async def test_grep_oversized_file_has_partial_scan_marker(
    tmp_path: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grep_mod, "_MAX_FILE_BYTES", 16)
    (tmp_path / "big.log").write_text("NEEDLE here\n" + "x" * 200, encoding="utf-8")
    res = await GrepTool().execute({"pattern": "NEEDLE", "path": "."}, ctx)
    assert res.is_error is False
    assert "scanned" in res.output  # explicit partial-scan marker
    assert res.data and res.data.get("files_partially_scanned", 0) >= 1
