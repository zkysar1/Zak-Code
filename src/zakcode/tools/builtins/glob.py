"""Glob for files within the workspace using pathlib patterns."""

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

# Maximum number of matches to return.
_MAX_RESULTS = 1000


class GlobTool(Tool):
    """Match files by glob pattern within the workspace."""

    spec = ToolSpec(
        name="glob",
        description=(
            "Find files matching a glob pattern within the workspace. Patterns "
            "containing '**' search recursively. Results are sorted and capped."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '*.py' or '**/*.txt'.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Base directory to search (absolute or relative to the "
                        "workspace root). Defaults to the workspace root."
                    ),
                },
            },
            "required": ["pattern"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Return sorted, workspace-relative paths matching ``pattern``."""
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return ToolResult.error("'pattern' is required and must be a string.")

        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return ToolResult.error("'path' must be a string.")
        base = path if path else "."

        try:
            resolved_base = resolve_in_workspace(base, ctx.workspace_root)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {base!r}: {exc}")

        try:
            if not resolved_base.is_dir():
                return ToolResult.error(f"Base path is not a directory: {base}")

            matcher = resolved_base.rglob if "**" in pattern else resolved_base.glob
            # Strip leading recursive marker for rglob, which already recurses.
            effective = pattern.replace("**/", "", 1) if "**" in pattern else pattern

            matches = sorted(str(p) for p in matcher(effective))
            total = len(matches)
            truncated = total > _MAX_RESULTS
            if truncated:
                matches = matches[:_MAX_RESULTS]

            if not matches:
                return ToolResult.ok("(no matches)", data={"count": 0, "matches": []})

            output = "\n".join(matches)
            if truncated:
                output += f"\n\n[... {total - _MAX_RESULTS} more matches truncated ...]"

            return ToolResult.ok(
                output,
                data={"count": total, "matches": matches, "truncated": truncated},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Glob failed for pattern {pattern!r}: {exc}")
