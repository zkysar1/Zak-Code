"""List directory entries within the workspace."""

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

#: Soft cap on entries rendered into the model-facing output; beyond this an explicit marker
#: points the model at glob. ``data["entries"]`` still carries the full list for clients.
_MAX_ENTRIES = 1000


class ListDirTool(Tool):
    """List the entries of a directory inside the workspace."""

    spec = ToolSpec(
        name="list_dir",
        description=(
            "List the entries of a directory within the workspace. Directories are "
            "suffixed with '/'. Defaults to the workspace root."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to list (absolute or relative to the workspace "
                        "root). Defaults to the workspace root."
                    ),
                },
            },
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """List entries of the requested directory."""
        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return ToolResult.error("'path' must be a string.")
        target = path if path else "."

        try:
            resolved = resolve_path(target, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {target!r}: {exc}")

        try:
            if not resolved.exists():
                return ToolResult.error(f"Directory not found: {target}")
            if not resolved.is_dir():
                return ToolResult.error(f"Path is not a directory: {target}")

            try:
                children = sorted(resolved.iterdir(), key=lambda p: p.name)
            except PermissionError as exc:
                return ToolResult.error(f"Permission denied listing {target}: {exc}")

            entries: list[str] = []
            names: list[str] = []
            for entry in children:
                if entry.is_dir():
                    entries.append(f"{entry.name}/")
                else:
                    entries.append(entry.name)
                names.append(entry.name)

            # Soft cap with an explicit marker so a huge directory cannot flood the model's
            # context one-entry-per-line with no signal that it was capped. (#5 dense output)
            shown = entries
            truncated = len(entries) > _MAX_ENTRIES
            if truncated:
                hidden = len(entries) - _MAX_ENTRIES
                shown = entries[:_MAX_ENTRIES] + [
                    f"[... {hidden} more entries; use glob with a pattern to narrow ...]"
                ]
            output = "\n".join(shown) if shown else "(empty directory)"
            return ToolResult.ok(
                output,
                data={
                    "path": str(resolved),
                    "count": len(names),
                    "entries": names,
                    "truncated": truncated,
                },
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to list {target!r}: {exc}")
