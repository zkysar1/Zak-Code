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
from zakcode.tools.builtins._safety import PathEscapeError, resolve_in_workspace


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
            resolved = resolve_in_workspace(target, ctx.workspace_root)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {target!r}: {exc}")

        try:
            if not resolved.exists():
                return ToolResult.error(f"Directory not found: {target}")
            if not resolved.is_dir():
                return ToolResult.error(f"Path is not a directory: {target}")

            entries: list[str] = []
            names: list[str] = []
            for entry in sorted(resolved.iterdir(), key=lambda p: p.name):
                if entry.is_dir():
                    entries.append(f"{entry.name}/")
                else:
                    entries.append(entry.name)
                names.append(entry.name)

            output = "\n".join(entries) if entries else "(empty directory)"
            return ToolResult.ok(
                output,
                data={"path": str(resolved), "count": len(names), "entries": names},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to list {target!r}: {exc}")
