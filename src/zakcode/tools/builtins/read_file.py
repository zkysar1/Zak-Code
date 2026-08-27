"""Read a text file from within the workspace, optionally sliced by line."""

from __future__ import annotations

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._safety import PathEscapeError, resolve_path
from zakcode.tools.builtins._suggest import not_found_fix, render, suggest

# Maximum number of bytes we will read before truncating.
_MAX_BYTES = 100 * 1024


class ReadFileTool(Tool):
    """Read a UTF-8 text file inside the workspace, with an optional line slice."""

    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a text file within the workspace. Optionally provide a 1-based "
            "line 'offset' and a 'limit' to read only a slice of lines. Output is "
            "capped at roughly 100KB with a truncation note."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to the workspace root).",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from.",
                    "minimum": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read.",
                    "minimum": 1,
                },
            },
            "required": ["path"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Read the requested file, returning its (optionally sliced) text."""
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")

        offset = args.get("offset")
        limit = args.get("limit")
        # ``bool`` is an ``int`` subclass; reject it explicitly so True/False are
        # not silently treated as 1/0.
        if offset is not None and (not isinstance(offset, int) or isinstance(offset, bool)):
            return ToolResult.error("'offset' must be an integer.")
        if offset is not None and offset < 1:
            return ToolResult.error("'offset' must be >= 1.")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool)):
            return ToolResult.error("'limit' must be an integer.")
        if limit is not None and limit < 1:
            return ToolResult.error("'limit' must be >= 1.")

        try:
            resolved = resolve_path(path, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {path!r}: {exc}")

        try:
            if not resolved.exists():
                # A not-found answer is about ONE path (ADR-0040): say what the workspace
                # DOES have under that name, so the model reads the right file instead of
                # asking the operator for a path a single grep would have found.
                by_name, by_content = suggest(path, ctx.workspace_root, ctx.extra_workspace_roots)
                extra = render(path, by_name, by_content)
                return ToolResult.error(
                    f"File not found: {path}" + (f"\n{extra}" if extra else ""),
                    fix=not_found_fix(path, bool(by_name or by_content)),
                    data={"suggestions": {"by_name": by_name, "by_content": by_content}},
                )
            if resolved.is_dir():
                return ToolResult.error(f"Path is a directory, not a file: {path}")

            try:
                raw = resolved.read_bytes()
            except PermissionError as exc:
                return ToolResult.error(f"Permission denied reading {path}: {exc}")
            except OSError as exc:
                return ToolResult.error(f"Could not read {path}: {exc}")

            offset_past_eof = False
            slice_note: str | None = None
            total_lines: int | None = None
            byte_truncated = False

            if offset is not None or limit is not None:
                # Line-account against the FULL file (already in memory via read_bytes), NOT the
                # byte-capped text — otherwise `total` undercounts and a valid offset past the
                # 100KB window is falsely reported "beyond EOF". The byte cap governs only how
                # much we RETURN. (review: HIGH false-EOF + MEDIUM undercount)
                # errors='replace' keeps non-UTF-8 content from crashing us (U+FFFD).
                lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)
                total_lines = len(lines)
                start = (offset - 1) if isinstance(offset, int) else 0
                end = (start + limit) if isinstance(limit, int) else total_lines
                if start >= total_lines:
                    offset_past_eof = True
                    text = ""
                else:
                    shown_end = min(end, total_lines)
                    text = "".join(lines[start:shown_end])
                    # Explicit, actionable continuation marker when the slice stops before the
                    # REAL EOF — without it a partial read reads as "that's the whole file".
                    if shown_end < total_lines:
                        slice_note = (
                            f"[... showed lines {start + 1}-{shown_end} of {total_lines}; "
                            f"use offset={shown_end + 1} to read more ...]"
                        )
                # A slice with a huge limit could still be enormous; cap the RETURNED bytes.
                encoded = text.encode("utf-8")
                if len(encoded) > _MAX_BYTES:
                    text = encoded[:_MAX_BYTES].decode("utf-8", errors="ignore")
                    byte_truncated = True
                    # The cap may have cut the slice SHORTER than ``shown_end`` (or below EOF for
                    # a slice that reached it). Rebase the continuation marker on the lines
                    # ACTUALLY returned (complete = newline-terminated), so its offset can never
                    # skip the un-returned tail between the byte cutoff and shown_end. (review)
                    shown_lines = text.count("\n")
                    real_end = start + shown_lines
                    if shown_lines >= 1 and real_end < total_lines:
                        slice_note = (
                            f"[... showed lines {start + 1}-{real_end} of {total_lines}; "
                            f"use offset={real_end + 1} to read more ...]"
                        )
                    else:
                        # Not even one whole line fit (a single line > 100KB): no honest line
                        # offset to give — the byte-truncation note alone explains the cut.
                        slice_note = None
            else:
                # Whole-file read: cap the returned bytes directly off raw (no full re-encode).
                if len(raw) > _MAX_BYTES:
                    raw = raw[:_MAX_BYTES]
                    byte_truncated = True
                text = raw.decode("utf-8", errors="replace")

            notes: list[str] = []
            if offset_past_eof:
                notes.append("[... offset is beyond the end of the file; no lines returned ...]")
            if slice_note:
                notes.append(slice_note)
            if byte_truncated:
                notes.append(
                    "[... output truncated at 100KB; re-read with offset/limit to page "
                    "through the rest ...]"
                )
            if notes:
                text = (text + "\n\n" if text else "") + "\n".join(notes)

            return ToolResult.ok(
                text,
                data={
                    "path": str(resolved),
                    # `truncated` = the output is incomplete for ANY reason (byte cap OR a slice
                    # that stopped before EOF), so a data-only client sees the same signal the
                    # model reads in the markers. `byte_truncated` keeps the precise cap meaning.
                    "truncated": byte_truncated or slice_note is not None,
                    "byte_truncated": byte_truncated,
                    "total_lines": total_lines,
                },
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to read {path!r}: {exc}")
