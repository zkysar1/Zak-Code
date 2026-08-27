"""Glob for files within the workspace using pathlib patterns."""

from __future__ import annotations

from pathlib import Path, PurePath

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._ignore import load_ignore
from zakcode.tools.builtins._safety import PathEscapeError, resolve_path
from zakcode.tools.builtins._suggest import literal_stem, not_found_fix, render, suggest

# Maximum number of matches to return.
_MAX_RESULTS = 1000


class GlobTool(Tool):
    """Match files by glob pattern within the workspace."""

    spec = ToolSpec(
        name="glob",
        description=(
            "Find files matching a glob pattern within the workspace. Patterns "
            "containing '**' search recursively. Results are sorted and capped. Skips "
            "ignored paths (.git, build/vendor/cache dirs, .gitignore/.zakcodeignore "
            "entries); pass include_ignored=true to include them."
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
                "include_ignored": {
                    "type": "boolean",
                    "description": (
                        "Also return git-ignored and default-ignored files. Default false. "
                        "(.git is always skipped.)"
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
        # Absolute patterns are unsupported by pathlib's matcher and would let a
        # caller probe outside the base directory; reject them up front.
        if PurePath(pattern).is_absolute():
            return ToolResult.error(f"Pattern must be relative, not absolute: {pattern}")

        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return ToolResult.error("'path' must be a string.")
        soft = not bool(args.get("include_ignored"))
        base = path if path else "."

        try:
            resolved_base = resolve_path(base, ctx.workspace_root, ctx.extra_workspace_roots)
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

            try:
                raw_matches = list(matcher(effective))
            except (ValueError, NotImplementedError):
                # e.g. '../*' under rglob, or other patterns pathlib refuses;
                # treat an un-matchable pattern as simply having no matches.
                raw_matches = []

            # Defensively drop anything that escaped the workspace root(s) (e.g. via
            # a symlink the matcher followed, or a '../' pattern). We compare each
            # match by its fully resolved path and emit that canonical form, so a
            # self-referential match like 'base/../base' collapses to the base and
            # is skipped rather than leaking a confusing dotted path.
            resolved_roots = [r.resolve() for r in ctx.all_workspace_roots]
            ignore = load_ignore(Path(ctx.workspace_root))
            ignore_root = Path(ctx.workspace_root).resolve()
            kept: list[str] = []
            for p in raw_matches:
                rp = p.resolve()
                # Must live strictly under at least one root, and never be the
                # base/root itself (a glob should not return its own search dir).
                if rp == resolved_base or not any(root in rp.parents for root in resolved_roots):
                    continue
                if ignore.is_ignored_path(rp, ignore_root, is_dir=rp.is_dir(), soft=soft):
                    continue
                kept.append(str(rp))
            matches = sorted(set(kept))
            total = len(matches)
            truncated = total > _MAX_RESULTS
            if truncated:
                matches = matches[:_MAX_RESULTS]

            if not matches:
                # An empty glob for a NAMED thing (ADR-0040): the pattern's literal segment is
                # what the model was after — say where that name does occur, by path and by
                # content, so "no matches" is not read as "does not exist".
                stem = literal_stem(pattern)
                by_name, by_content = (
                    suggest(stem, ctx.workspace_root, ctx.extra_workspace_roots, soft=soft)
                    if len(stem) >= 3
                    else ([], [])
                )
                extra = render(stem, by_name, by_content)
                return ToolResult.ok(
                    "(no matches)" + (f"\n{extra}" if extra else ""),
                    data={
                        "count": 0,
                        "matches": [],
                        "suggestions": {"by_name": by_name, "by_content": by_content},
                    },
                    hint=not_found_fix(stem, bool(by_name or by_content)) if stem else None,
                )

            output = "\n".join(matches)
            if truncated:
                output += f"\n\n[... {total - _MAX_RESULTS} more matches truncated ...]"

            return ToolResult.ok(
                output,
                data={"count": total, "matches": matches, "truncated": truncated},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Glob failed for pattern {pattern!r}: {exc}")
