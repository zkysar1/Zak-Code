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
from zakcode.tools.builtins._safety import PathEscapeError, resolve_path

# Maximum number of matching lines to return.
_MAX_MATCHES = 1000
# Directories we never descend into.
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache"}
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
            "Skips binary files and common vendored directories (.git, .venv, ...). "
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

            files = self._gather_files(resolved, glob_filter)

            rows: list[str] = []
            total = 0
            truncated = False
            for file_path in files:
                if total >= _MAX_MATCHES:
                    truncated = True
                    break
                for line_no, line in self._scan_file(file_path, regex):
                    rows.append(f"{file_path}:{line_no}:{line}")
                    total += 1
                    if total >= _MAX_MATCHES:
                        truncated = True
                        break

            if not rows:
                return ToolResult.ok("(no matches)", data={"count": 0, "matches": []})

            output = "\n".join(rows)
            if truncated:
                output += f"\n\n[... results truncated at {_MAX_MATCHES} matches ...]"
            return ToolResult.ok(
                output,
                data={"count": total, "matches": rows, "truncated": truncated},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Grep failed for pattern {pattern!r}: {exc}")

    def _gather_files(self, root: Path, glob_filter: str | None) -> list[Path]:
        """Collect candidate files under ``root``, applying skip/glob filters."""
        if root.is_file():
            return [root]

        collected: list[Path] = []
        # ``onerror`` is left at its default (errors swallowed). ``followlinks=False`` stops
        # os.walk recursing INTO symlinked directories, but a symlinked FILE leaf still shows
        # up in ``filenames`` and ``read_bytes()`` would follow it out of the workspace — so
        # each leaf is checked per-file below (mirroring glob's per-match re-resolve). (audit4 #1)
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            # Prune skip directories in place so os.walk does not descend.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            current_path = Path(current)
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
                collected.append(leaf)
        return collected

    def _scan_file(self, file_path: Path, regex: re.Pattern[str]) -> list[tuple[int, str]]:
        """Return ``(line_no, line_text)`` pairs in ``file_path`` matching ``regex``."""
        try:
            raw = file_path.read_bytes()
        except OSError:
            return []
        if _is_binary(raw[:1024]):
            return []
        if len(raw) > _MAX_FILE_BYTES:
            raw = raw[:_MAX_FILE_BYTES]
        text = raw.decode("utf-8", errors="replace")

        results: list[tuple[int, str]] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                results.append((idx, line))
        return results
