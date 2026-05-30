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

            raw = resolved.read_bytes()
            truncated = False
            if len(raw) > _MAX_BYTES:
                raw = raw[:_MAX_BYTES]
                truncated = True

            text = raw.decode("utf-8", errors="replace")

            if offset is not None or limit is not None:
                lines = text.splitlines(keepends=True)
                start = (offset - 1) if isinstance(offset, int) and offset > 0 else 0
                end = (start + limit) if isinstance(limit, int) and limit > 0 else len(lines)
                text = "" if start > len(lines) else "".join(lines[start:end])

            if truncated:
                text += "\n\n[... output truncated at 100KB ...]"

            return ToolResult.ok(text, data={"path": str(resolved), "truncated": truncated})
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to read {path!r}: {exc}")
