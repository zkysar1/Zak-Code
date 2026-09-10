"""List directory entries within the workspace."""

from __future__ import annotations

from pathlib import Path

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
from zakcode.tools.builtins._suggest import not_found_fix, render, suggest

#: Soft cap on entries rendered into the model-facing output; beyond this an explicit marker
#: points the model at glob. ``data["entries"]`` still carries the full list for clients.
_MAX_ENTRIES = 1000


class ListDirTool(Tool):
    """List the entries of a directory inside the workspace."""

    spec = ToolSpec(
        name="list_dir",
        description=(
            "List the entries of a directory within the workspace. Directories are "
            "suffixed with '/'. Defaults to the workspace root. Ignored entries (.git, "
            "build/vendor/cache dirs, .gitignore/.zakcodeignore) are hidden with a count — "
            "unless EVERY entry is ignored, in which case they are listed anyway; pass "
            "include_ignored=true to show them elsewhere."
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
                "include_ignored": {
                    "type": "boolean",
                    "description": (
                        "Also list git-ignored and default-ignored entries. Default false. "
                        "(.git is always hidden.)"
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
        soft = not bool(args.get("include_ignored"))
        target = path if path else "."

        try:
            resolved = resolve_path(target, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {target!r}: {exc}")

        try:
            if not resolved.exists():
                # Closest-path suggestions (ADR-0040) — see read_file's not-found branch.
                by_name, by_content = suggest(
                    target, ctx.workspace_root, ctx.extra_workspace_roots, soft=soft
                )
                extra = render(target, by_name, by_content)
                return ToolResult.error(
                    f"Directory not found: {target}" + (f"\n{extra}" if extra else ""),
                    fix=not_found_fix(target, bool(by_name or by_content)),
                    data={"suggestions": {"by_name": by_name, "by_content": by_content}},
                )
            if not resolved.is_dir():
                return ToolResult.error(f"Path is not a directory: {target}")

            try:
                children = sorted(resolved.iterdir(), key=lambda p: p.name)
            except PermissionError as exc:
                return ToolResult.error(f"Permission denied listing {target}: {exc}")

            ignore = load_ignore(Path(ctx.workspace_root))
            ignore_root = Path(ctx.workspace_root).resolve()
            entries: list[str] = []
            names: list[str] = []
            ignored_count = 0
            hidden_entries: list[str] = []  # soft-ignored only; .git-class stays hidden
            hidden_names: list[str] = []
            for entry in children:
                is_dir = entry.is_dir()
                # Hide ignored entries (build/vendor/.gitignore), counted so the listing never
                # SILENTLY omits something like node_modules — the agent sees there's more.
                if ignore.is_ignored_path(entry, ignore_root, is_dir=is_dir, soft=soft):
                    ignored_count += 1
                    if not ignore.is_ignored_path(entry, ignore_root, is_dir=is_dir, soft=False):
                        hidden_entries.append(f"{entry.name}/" if is_dir else entry.name)
                        hidden_names.append(entry.name)
                    continue
                entries.append(f"{entry.name}/" if is_dir else entry.name)
                names.append(entry.name)

            all_ignored = False
            if not entries and hidden_entries:
                # EVERY entry is ignored — a workspace whose data lives under a gitignored root
                # (field 2026-09-10: a Mind's world/ under .mind-data/; three listings came back
                # as one count line, the model skimmed the note four times, and the no-progress
                # rail had to fire). An empty listing of a non-empty directory is the one shape
                # that misleads, so these are shown, tagged, rather than hidden behind a count.
                all_ignored = True
                entries, names = hidden_entries, hidden_names

            # Soft cap with an explicit marker so a huge directory cannot flood the model's
            # context one-entry-per-line with no signal that it was capped. (#5 dense output)
            shown = entries
            truncated = len(entries) > _MAX_ENTRIES
            if truncated:
                hidden = len(entries) - _MAX_ENTRIES
                shown = entries[:_MAX_ENTRIES] + [
                    f"[... {hidden} more entries; use glob with a pattern to narrow ...]"
                ]
            if all_ignored:
                notes = [
                    f"[all {len(names)} entries here are ignored by .gitignore/default rules "
                    "— shown anyway because nothing else is here; elsewhere pass "
                    "include_ignored=true to see ignored entries]"
                ]
            elif ignored_count:
                notes = [
                    f"[... {ignored_count} ignored entries hidden; "
                    "include_ignored=true to show ...]"
                ]
            else:
                notes = []
            output = "\n".join(shown + notes) if (shown or notes) else "(empty directory)"
            return ToolResult.ok(
                output,
                data={
                    "path": str(resolved),
                    "count": len(names),
                    "entries": names,
                    "truncated": truncated,
                    "ignored": ignored_count,
                    "all_ignored": all_ignored,
                },
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to list {target!r}: {exc}")
