"""Tests for the PDF tools (read_pdf / create_pdf).

Exercises the real backends (pdfplumber + reportlab) end-to-end: a create→read round-trip,
genuine multi-page pagination of long text (the thing a naive single-`drawString` writer gets
wrong), the error paths (missing file, wrong suffix, malformed `pages`), the page selector,
markup escaping, and workspace path scoping. Async tests run under pytest-asyncio auto mode.
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from zakcode.tools.base import ToolContext, ToolResult
from zakcode.tools.builtins.pdf import CreatePdfTool, ReadPdfTool


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace_root=tmp_path)


async def _create(tmp_path: Path, name: str, text: str, title: str | None = None) -> ToolResult:
    args: dict[str, object] = {"path": name, "text": text}
    if title is not None:
        args["title"] = title
    return await CreatePdfTool().execute(args, _ctx(tmp_path))


async def _read(tmp_path: Path, name: str, pages: str | None = None) -> ToolResult:
    args: dict[str, object] = {"path": name}
    if pages is not None:
        args["pages"] = pages
    return await ReadPdfTool().execute(args, _ctx(tmp_path))


async def test_create_then_read_roundtrip(tmp_path: Path) -> None:
    r = await _create(tmp_path, "doc.pdf", "Hello world.\n\nSecond paragraph.", title="My Title")
    assert not r.is_error
    assert (tmp_path / "doc.pdf").is_file()
    with pdfplumber.open(str(tmp_path / "doc.pdf")) as pdf:  # a genuinely valid PDF
        assert len(pdf.pages) >= 1
    rr = await _read(tmp_path, "doc.pdf")
    assert not rr.is_error
    assert "--- page 1 ---" in rr.output
    for fragment in ("Hello world", "Second paragraph", "My Title"):
        assert fragment in rr.output


async def test_create_long_text_paginates(tmp_path: Path) -> None:
    # Long content must FLOW across multiple pages — the bug a single drawString has (it would
    # clip everything past the first line). The page count is read back from the real file.
    filler = "This sentence is repeated to fill space across many pages. "
    long_text = "\n\n".join(f"Paragraph {i}. " + filler * 20 for i in range(40))
    r = await _create(tmp_path, "long.pdf", long_text)
    assert not r.is_error
    assert isinstance(r.data, dict) and r.data["pages"] and r.data["pages"] >= 2
    rr = await _read(tmp_path, "long.pdf")
    assert not rr.is_error
    assert "Paragraph 0." in rr.output and "Paragraph 39." in rr.output  # nothing lost off-page
    assert rr.output.count("--- page ") >= 2


async def test_read_missing_file_is_error(tmp_path: Path) -> None:
    r = await _read(tmp_path, "nope.pdf")
    assert r.is_error and "File not found" in r.output


async def test_read_non_pdf_is_error(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hi", encoding="utf-8")
    r = await _read(tmp_path, "notes.txt")
    assert r.is_error and ".pdf" in r.output


async def test_read_malformed_pages_is_clean_error_not_crash(tmp_path: Path) -> None:
    await _create(tmp_path, "d.pdf", "x")
    r = await _read(tmp_path, "d.pdf", pages="abc")
    assert r.is_error and "pages" in r.output  # a clean error, never an exception


async def test_read_page_selector(tmp_path: Path) -> None:
    long_text = "\n\n".join(f"MARK{i} " + "word " * 250 for i in range(12))
    await _create(tmp_path, "multi.pdf", long_text)
    full = await _read(tmp_path, "multi.pdf")
    assert isinstance(full.data, dict) and full.data["page_count"] >= 2
    r = await _read(tmp_path, "multi.pdf", pages="1")
    assert not r.is_error
    assert "--- page 1 ---" in r.output and "--- page 2 ---" not in r.output


async def test_create_escapes_markup(tmp_path: Path) -> None:
    # reportlab Paragraph parses a mini-markup; special chars must be escaped, not break it or
    # double-escape (no literal "&amp;" survives to the extracted text).
    r = await _create(tmp_path, "x.pdf", "a < b & c > d")
    assert not r.is_error
    rr = await _read(tmp_path, "x.pdf")
    assert not rr.is_error
    assert "&" in rr.output and "&amp;" not in rr.output


async def test_paths_are_workspace_scoped(tmp_path: Path) -> None:
    assert (await _read(tmp_path, "../escape.pdf")).is_error
    assert (await _create(tmp_path, "../escape.pdf", "x")).is_error


async def test_create_appends_pdf_suffix(tmp_path: Path) -> None:
    r = await _create(tmp_path, "noext", "hello")
    assert not r.is_error
    assert (tmp_path / "noext.pdf").is_file()


async def test_create_rejects_wrong_suffix(tmp_path: Path) -> None:
    r = await _create(tmp_path, "doc.txt", "hello")
    assert r.is_error and ".pdf" in r.output
