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
from zakcode.tools.builtins._safety import PathEscapeError, resolve_in_workspace

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
            resolved = resolve_in_workspace(path, ctx.workspace_root)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {path!r}: {exc}")

        try:
            if not resolved.exists():
                return ToolResult.error(f"File not found: {path}")
            if resolved.is_dir():
                return ToolResult.error(f"Path is a directory, not a file: {path}")

            try:
                raw = resolved.read_bytes()
            except PermissionError as exc:
                return ToolResult.error(f"Permission denied reading {path}: {exc}")
            except OSError as exc:
                return ToolResult.error(f"Could not read {path}: {exc}")

            truncated = False
            if len(raw) > _MAX_BYTES:
                raw = raw[:_MAX_BYTES]
                truncated = True

            # errors='replace' keeps binary / non-UTF-8 content from crashing us;
            # undecodable bytes become U+FFFD rather than raising.
            text = raw.decode("utf-8", errors="replace")

            offset_past_eof = False
            if offset is not None or limit is not None:
                lines = text.splitlines(keepends=True)
                start = (offset - 1) if isinstance(offset, int) else 0
                end = (start + limit) if isinstance(limit, int) else len(lines)
                if start >= len(lines):
                    offset_past_eof = True
                    text = ""
                else:
                    text = "".join(lines[start:end])

            notes: list[str] = []
            if offset_past_eof:
                notes.append("[... offset is beyond the end of the file; no lines returned ...]")
            if truncated:
                notes.append("[... output truncated at 100KB ...]")
            if notes:
                text = (text + "\n\n" if text else "") + "\n".join(notes)

            return ToolResult.ok(text, data={"path": str(resolved), "truncated": truncated})
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to read {path!r}: {exc}")
