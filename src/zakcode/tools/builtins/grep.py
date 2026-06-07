"""Search file contents within the workspace using a pure-Python regex walk."""

from __future__ import annotations

import fnmatch
import os
import re
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

# Maximum number of matching lines to return.
_MAX_MATCHES = 1000
# Read at most this many bytes per file when scanning.
_MAX_FILE_BYTES = 5 * 1024 * 1024


def _is_binary(sample: bytes) -> bool:
    """Heuristically decide whether ``sample`` looks like binary data."""
    return b"\x00" in sample


class GrepTool(Tool):
    """Search for a regular expression across files in the workspace."""

    spec = ToolSpec(
        name="grep",
        description=(
            "Search file contents for a regular expression within the workspace. "
            "Skips binary files and ignored paths (.git, build/vendor/cache dirs, and "
            ".gitignore/.zakcodeignore entries); pass include_ignored=true to search them too. "
            "Returns 'file:line:match' rows, capped."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "File or directory to search (absolute or relative to the "
                        "workspace root). Defaults to the workspace root."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": "Optional filename glob filter, e.g. '*.py'.",
                },
                "include_ignored": {
                    "type": "boolean",
                    "description": (
                        "Also search git-ignored and default-ignored files (build/vendor/cache, "
                        ".gitignore entries). Default false. (.git is always skipped.)"
                    ),
                },
            },
            "required": ["pattern"],
        },
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Walk files and return matching ``file:line:text`` rows."""
        pattern = args.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return ToolResult.error("'pattern' is required and must be a string.")

        path = args.get("path")
        if path is not None and not isinstance(path, str):
            return ToolResult.error("'path' must be a string.")
        glob_filter = args.get("glob")
        if glob_filter is not None and not isinstance(glob_filter, str):
            return ToolResult.error("'glob' must be a string.")
        soft = not bool(args.get("include_ignored"))

        base = path if path else "."

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult.error(f"Invalid regex {pattern!r}: {exc}")

        try:
            resolved = resolve_path(base, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {base!r}: {exc}")

        try:
            if not resolved.exists():
                return ToolResult.error(f"Path not found: {base}")

            ignore = load_ignore(Path(ctx.workspace_root))
            ignore_root = Path(ctx.workspace_root).resolve()
            files = self._gather_files(
                resolved, glob_filter, ignore=ignore, ignore_root=ignore_root, soft=soft
            )

            rows: list[str] = []
            total = 0
            truncated = False
            file_notes: list[str] = []
            for file_path in files:
                if total >= _MAX_MATCHES:
                    truncated = True
                    break
                matches, file_truncated = self._scan_file(file_path, regex)
                if file_truncated:
                    # Explicit marker: a >5MB file was only partially scanned, so a "no match"
                    # here is not conclusive — without this the drop is silent. (#5)
                    file_notes.append(
                        f"{file_path}: [... file exceeds 5MB; only its first 5MB was scanned ...]"
                    )
                for line_no, line in matches:
                    rows.append(f"{file_path}:{line_no}:{line}")
                    total += 1
                    if total >= _MAX_MATCHES:
                        truncated = True
                        break

            if not rows and not file_notes:
                return ToolResult.ok("(no matches)", data={"count": 0, "matches": []})

            output = "\n".join(rows)
            if truncated:
                output += f"\n\n[... results truncated at {_MAX_MATCHES} matches ...]"
            if file_notes:
                output += ("\n\n" if output else "") + "\n".join(file_notes)
            return ToolResult.ok(
                output,
                data={
                    "count": total,
                    "matches": rows,
                    "truncated": truncated,
                    "files_partially_scanned": len(file_notes),
                },
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Grep failed for pattern {pattern!r}: {exc}")

    def _gather_files(
        self, root: Path, glob_filter: str | None, *, ignore, ignore_root: Path, soft: bool
    ) -> list[Path]:
        """Collect candidate files under ``root``, applying ignore + glob filters.

        ``ignore``/``ignore_root`` prune ignored directories (so os.walk never descends into
        them) and skip ignored files; ``soft=False`` keeps only the ``.git`` hard floor.
        """
        if root.is_file():
            return [root]

        collected: list[Path] = []
        # ``onerror`` is left at its default (errors swallowed). ``followlinks=False`` stops
        # os.walk recursing INTO symlinked directories, but a symlinked FILE leaf still shows
        # up in ``filenames`` and ``read_bytes()`` would follow it out of the workspace — so
        # each leaf is checked per-file below (mirroring glob's per-match re-resolve). (audit4 #1)
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            # Prune ignored directories in place so os.walk does not descend (cheap + keeps
            # huge vendor/build trees out of the scan entirely).
            dirnames[:] = [
                d
                for d in dirnames
                if not ignore.is_ignored_path(current_path / d, ignore_root, is_dir=True, soft=soft)
            ]
            for name in sorted(filenames):
                if glob_filter and not fnmatch.fnmatch(name, glob_filter):
                    continue
                leaf = current_path / name
                # A symlinked leaf is never scanned, and the real path must stay under the
                # (already workspace-confined) walked root — so grep cannot return the content
                # of a file outside the sandbox via a planted link.
                if leaf.is_symlink():
                    continue
                try:
                    resolved_leaf = leaf.resolve()
                except OSError:
                    continue
                if resolved_leaf != root and root not in resolved_leaf.parents:
                    continue
                if ignore.is_ignored_path(leaf, ignore_root, is_dir=False, soft=soft):
                    continue
                collected.append(leaf)
        return collected

    def _scan_file(
        self, file_path: Path, regex: re.Pattern[str]
    ) -> tuple[list[tuple[int, str]], bool]:
        """Return ``(matches, truncated)`` for ``file_path``.

        ``matches`` is the ``(line_no, line_text)`` pairs matching ``regex``; ``truncated`` is
        True when the file exceeded the per-file byte cap and only its first ``_MAX_FILE_BYTES``
        were scanned (so a caller can surface that the result for this file is incomplete).
        """
        try:
            raw = file_path.read_bytes()
        except OSError:
            return [], False
        if _is_binary(raw[:1024]):
            return [], False
        truncated = len(raw) > _MAX_FILE_BYTES
        if truncated:
            raw = raw[:_MAX_FILE_BYTES]
        text = raw.decode("utf-8", errors="replace")

        results: list[tuple[int, str]] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append((idx, line))
        return results, truncated
