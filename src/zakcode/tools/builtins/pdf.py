"""Read text from PDFs and create simple text PDFs.

``read_pdf`` extracts a PDF's text page-by-page (the path for feeding research papers and
reports to the model); ``create_pdf`` writes a text PDF that FLOWS across as many pages as the
content needs (long text wraps and paginates — never overflows a single line). Both follow the
office/image tool conventions: workspace-scoped paths (``resolve_path``), never-raise handlers,
atomic writes, and a downloadable :class:`~zakcode.artifacts.ArtifactRef` for the created file.

Backends are permissively licensed (MIT/BSD): ``pdfplumber`` (on ``pdfminer.six``) for
extraction — it orders text by layout, which reads multi-column papers far better than a naive
stream dump — and ``reportlab`` for generation.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

import pdfplumber
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from zakcode.artifacts import ArtifactError, artifact_from_path
from zakcode.config import PermissionTier
from zakcode.tools.base import ConcurrencyClass, Tool, ToolContext, ToolResult, ToolSpec
from zakcode.tools.builtins._safety import PathEscapeError, resolve_path

#: A PDF larger than this is refused (research papers are big, but this bounds memory/time).
_MAX_PDF_READ_BYTES = 100 * 1024 * 1024
#: Extracted text returned per call is capped here, with a clear note + the `pages` hint to
#: read the rest — a long paper can exceed a model's useful context otherwise.
_MAX_READ_OUTPUT = 200_000


def _parse_pages(spec: object, page_count: int) -> tuple[list[int], str | None]:
    """Parse a 1-based ``pages`` selector ("1-5", "3", "1,4-6") into sorted 0-based indices.

    Returns ``(indices, None)`` on success or ``([], error)`` on a malformed selector — so a
    bad value is a clean tool error, never an exception. Out-of-range pages are dropped silently
    (asking for pages 1-99 of a 10-page doc reads the 10 that exist).
    """
    if spec is None:
        return list(range(page_count)), None
    if not isinstance(spec, str):
        return [], "'pages' must be a string like '1-5' or '3'"
    indices: list[int] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
            else:
                start = end = int(part)
        except ValueError:
            return [], f"invalid 'pages' value {spec!r}: use 1-based numbers like '1-5' or '3'"
        if start < 1 or end < start:
            return [], f"invalid 'pages' range {part!r}: use 1-based ascending ranges (e.g. 2-5)"
        for n in range(start, end + 1):
            if 1 <= n <= page_count and (n - 1) not in indices:
                indices.append(n - 1)
    return sorted(indices), None


def _target_pdf_path(path: object, ctx: ToolContext) -> tuple[str, Path] | str:
    """Resolve a write path to ``(display_path, resolved)`` within the workspace, or an error str.

    Appends ``.pdf`` when the path has no suffix; rejects a different suffix; rejects a directory
    or an outside-workspace path (``resolve_path``).
    """
    if not isinstance(path, str) or not path:
        return "'path' is required and must be a string."
    candidate = Path(path)
    if candidate.suffix.lower() == ".pdf":
        final = path
    elif candidate.suffix:
        return "path must end with .pdf"
    else:
        final = f"{path}.pdf"
    try:
        resolved = resolve_path(final, ctx.workspace_root, ctx.extra_workspace_roots)
    except PathEscapeError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 — handlers must never raise
        return f"Failed to resolve path {path!r}: {exc}"
    if resolved.exists() and resolved.is_dir():
        return f"Path is a directory, not a file: {final}"
    return final, resolved


class ReadPdfTool(Tool):
    """Extract a PDF's text, page by page, for the model to read."""

    spec = ToolSpec(
        name="read_pdf",
        description=(
            "Read a PDF file and return its extracted text, one page at a time with "
            "`--- page N ---` markers. Use this to read research papers, reports, or any PDF. "
            "Optionally restrict to a 1-based page selection with `pages` (e.g. '1-5', '3', or "
            "'1,4-6'). Scanned/image-only PDFs yield no text (OCR is not performed)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file, within the workspace.",
                },
                "pages": {
                    "type": "string",
                    "description": "Optional 1-based page selection, e.g. '1-5', '3', or "
                    "'1,4-6'. Default: all pages.",
                },
            },
            "required": ["path"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")
        try:
            resolved = resolve_path(path, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 — handlers must never raise
            return ToolResult.error(f"Failed to resolve path {path!r}: {exc}")
        if not resolved.is_file():
            return ToolResult.error(f"File not found: {path}")
        if resolved.suffix.lower() != ".pdf":
            return ToolResult.error("path must end with .pdf")
        size = resolved.stat().st_size
        if size > _MAX_PDF_READ_BYTES:
            return ToolResult.error(f"PDF is too large ({size} bytes; max {_MAX_PDF_READ_BYTES}).")

        try:
            with pdfplumber.open(str(resolved)) as pdf:
                page_count = len(pdf.pages)
                indices, err = _parse_pages(args.get("pages"), page_count)
                if err is not None:
                    return ToolResult.error(err)
                parts = [
                    f"--- page {i + 1} ---\n{(pdf.pages[i].extract_text() or '').strip()}"
                    for i in indices
                ]
        except Exception as exc:  # noqa: BLE001 — a corrupt/encrypted PDF is a result, not a crash
            return ToolResult.error(
                f"Could not read PDF {path!r}: {type(exc).__name__}: {exc}",
                fix="Check the file is a valid, non-encrypted PDF.",
            )

        out = "\n\n".join(parts)
        truncated = len(out) > _MAX_READ_OUTPUT
        if truncated:
            hidden = len(out) - _MAX_READ_OUTPUT
            out = (
                out[:_MAX_READ_OUTPUT]
                + f"\n\n[... extracted text truncated at {_MAX_READ_OUTPUT} chars; {hidden} more "
                "hidden — read specific pages with the `pages` argument ...]"
            )
        data = {
            "path": path,
            "page_count": page_count,
            "pages_read": [i + 1 for i in indices],
            "truncated": truncated,
        }
        if not any(p.split("\n", 1)[-1].strip() for p in parts):
            return ToolResult.ok(
                f"Read {len(indices)} page(s) of {path}, but found no extractable text — the PDF "
                "is likely scanned images (OCR is not supported).",
                data=data,
            )
        return ToolResult.ok(out, data=data)


class CreatePdfTool(Tool):
    """Write a text PDF, flowing the content across as many pages as it needs."""

    spec = ToolSpec(
        name="create_pdf",
        description=(
            "Create a PDF file from text. The text is wrapped and FLOWED across as many pages as "
            "needed (it never overflows a single line). Separate paragraphs with a blank line; "
            "single newlines within a paragraph are kept as line breaks. Optional `title` is "
            "rendered at the top."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write the .pdf to, within the workspace.",
                },
                "text": {
                    "type": "string",
                    "description": "The document text. Blank lines separate paragraphs.",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title rendered at the top of the document.",
                },
            },
            "required": ["path", "text"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.PATH_SCOPED,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        text = args.get("text")
        if not isinstance(text, str):
            return ToolResult.error("'text' is required and must be a string.")
        prepared = _target_pdf_path(args.get("path"), ctx)
        if isinstance(prepared, str):
            return ToolResult.error(prepared)
        final, resolved = prepared

        parent = resolved.parent
        if parent.exists() and not parent.is_dir():
            return ToolResult.error(f"Parent path is not a directory: {parent}")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult.error(f"Could not create parent directory for {final}: {exc}")

        styles = getSampleStyleSheet()
        story: list[Any] = []
        title = args.get("title")
        if isinstance(title, str) and title.strip():
            story.append(Paragraph(_xml_escape(title.strip()), styles["Title"]))
            story.append(Spacer(1, 12))
        for block in text.split("\n\n"):
            block = block.strip("\n")
            if block.strip():
                # reportlab Paragraph parses a mini-markup, so escape, then keep intra-paragraph
                # newlines as explicit line breaks.
                story.append(
                    Paragraph(_xml_escape(block).replace("\n", "<br/>"), styles["BodyText"])
                )
                story.append(Spacer(1, 6))
        if not story:
            story.append(Paragraph("(empty document)", styles["BodyText"]))

        fd, tmp_name = tempfile.mkstemp(suffix=".pdf", dir=str(parent))
        os.close(fd)
        try:
            SimpleDocTemplate(tmp_name, pagesize=LETTER).build(story)
        except Exception as exc:  # noqa: BLE001 — generation failure is a result, not a crash
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            return ToolResult.error(f"Failed to create PDF: {type(exc).__name__}: {exc}")
        os.replace(tmp_name, resolved)

        # Read back the page count — confirms the file is a valid PDF and reports real pagination.
        pages = None
        with contextlib.suppress(Exception), pdfplumber.open(str(resolved)) as pdf:
            pages = len(pdf.pages)

        artifacts = []
        with contextlib.suppress(ArtifactError, OSError):
            artifacts.append(
                artifact_from_path(
                    resolved, workspace_root=ctx.workspace_root, created_by_tool="create_pdf"
                )
            )
        data: dict[str, Any] = {"path": final, "pages": pages}
        if artifacts:
            data["artifact_id"] = artifacts[0].id
        suffix = f" ({pages} page{'s' if pages != 1 else ''})" if pages else ""
        return ToolResult.ok(f"Created PDF at {final}{suffix}.", data=data, artifacts=artifacts)


__all__ = ["ReadPdfTool", "CreatePdfTool"]
