"""Replace an exact string within a file inside the workspace."""

from __future__ import annotations

import contextlib
import os
import tempfile

from zakcode.config import PermissionTier
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins._safety import (
    PathEscapeError,
    check_literal_content,
    check_python_syntax,
    resolve_path,
)

# Maximum number of bytes we will read before refusing to edit.
_MAX_BYTES = 100 * 1024 * 1024


class EditFileTool(Tool):
    """Replace an exact ``old_string`` with ``new_string`` in a workspace file."""

    spec = ToolSpec(
        name="edit_file",
        description=(
            "Replace an exact string in a text file within the workspace. By default "
            "the 'old_string' must match exactly once; pass 'replace_all=true' to "
            "replace every occurrence. The edit is atomic (temp file + os.replace). "
            "To create a new file, use write_file instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to the workspace root).",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace; include context for a unique match.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Text to insert in place of 'old_string'.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match.",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.PATH_SCOPED,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Perform an exact-string replacement on ``path``."""
        path = args.get("path")
        old_string = args.get("old_string")
        new_string = args.get("new_string")
        replace_all = args.get("replace_all", False)

        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")
        if not isinstance(old_string, str):
            return ToolResult.error("'old_string' is required and must be a string.")
        if not isinstance(new_string, str):
            return ToolResult.error("'new_string' is required and must be a string.")
        # ``bool`` is an ``int`` subclass; require an actual bool here.
        if not isinstance(replace_all, bool):
            return ToolResult.error("'replace_all' must be a boolean.")

        if old_string == new_string:
            return ToolResult.error("'old_string' and 'new_string' are identical.")
        if old_string == "":
            return ToolResult.error(
                "'old_string' must not be empty; use write_file to create a file."
            )

        try:
            resolved = resolve_path(path, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {path!r}: {exc}")

        try:
            if not resolved.exists():
                return ToolResult.error(
                    f"File not found: {path}",
                    fix="create it with write_file first, or check the path with list_dir/glob.",
                )
            if resolved.is_dir():
                return ToolResult.error(f"Path is a directory, not a file: {path}")

            try:
                raw = resolved.read_bytes()
            except PermissionError as exc:
                return ToolResult.error(f"Permission denied reading {path}: {exc}")
            except OSError as exc:
                return ToolResult.error(f"Could not read {path}: {exc}")

            if len(raw) > _MAX_BYTES:
                return ToolResult.error(f"File is too large to edit safely: {path}")

            # strict decode: editing relies on an exact textual match, so reject
            # files we cannot decode losslessly rather than silently mangling them.
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                return ToolResult.error(
                    f"File is not valid UTF-8 text and cannot be edited: {path}"
                )

            count = text.count(old_string)
            if count == 0:
                return ToolResult.error(
                    f"'old_string' not found in {path}",
                    fix="re-read the file (read_file) and copy old_string exactly, including "
                    "whitespace and indentation.",
                )
            if count > 1 and not replace_all:
                return ToolResult.error(
                    f"Found {count} occurrences of 'old_string' in {path}; pass "
                    "replace_all=true or add more context for a unique match."
                )

            if replace_all:
                new_text = text.replace(old_string, new_string)
                replacements = count
            else:
                new_text = text.replace(old_string, new_string, 1)
                replacements = 1

            # Write firewall: refuse a shell-command replacement, or an edit that would
            # leave a .py file unparseable (checked on the resulting file).
            guard = check_literal_content(new_string) or check_python_syntax(path, new_text)
            if guard is not None:
                return ToolResult.error(guard)

            data = new_text.encode("utf-8")
            parent = resolved.parent
            fd, tmp_name = tempfile.mkstemp(dir=str(parent), prefix=".zaktmp-")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, resolved)
            except Exception:
                # Best-effort cleanup of the temp file on failure.
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise

            suffix = "s" if replacements != 1 else ""
            return ToolResult.ok(
                f"Made {replacements} replacement{suffix} in {path}",
                data={"path": str(resolved), "replacements": replacements},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to edit {path!r}: {exc}")
