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
    check_skill_claims,
    check_skill_format,
    diagnose_python_syntax,
    resolve_path,
    skill_hosts,
)

# Maximum number of bytes we will read before refusing to edit.
_MAX_BYTES = 100 * 1024 * 1024

# Minimum length (stripped) for a single-line ``new_string`` to qualify the "edit already
# applied" idempotency no-op. Below this, a short/generic token ("pass", "30", "}") could
# coincidentally appear elsewhere and the no-op would mask a genuine wrong-target edit, so we
# fall through to the recoverable error instead. A multi-line new_string always qualifies.
_MIN_NOOP_LEN = 12


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
                # Idempotency: a retry of an ALREADY-APPLIED edit (old_string gone, new_string
                # now present) is the desired state, not a failure — return a benign no-op so a
                # small model that retries on timeout/confusion doesn't thrash. Conservative on
                # TWO axes so it never masks a genuine wrong-target edit: (1) new_string must be
                # present, and (2) DISTINCTIVE — multi-line, or >= _MIN_NOOP_LEN stripped chars.
                # A short/generic token ("pass", "30") that coincidentally appears elsewhere, or
                # an empty new_string (a pure deletion, indistinguishable from a wrong target),
                # falls through to the recoverable error below. (review: coincidental-substring)
                distinctive = "\n" in new_string or len(new_string.strip()) >= _MIN_NOOP_LEN
                if new_string and distinctive and new_string in text:
                    return ToolResult.ok(
                        f"No change needed: the edit is already applied to {path} "
                        "(old_string absent, new_string already present).",
                        data={
                            "path": str(resolved),
                            "replacements": 0,
                            "already_applied": True,
                        },
                        hint="The desired edit is already in place; continue with the next step.",
                    )
                return ToolResult.error(
                    f"'old_string' not found in {path}",
                    data={"refusal": "old_string_missing"},
                    fix="re-read the file (read_file) and copy old_string exactly, including "
                    "whitespace and indentation.",
                )
            if count > 1 and not replace_all:
                return ToolResult.error(
                    f"Found {count} occurrences of 'old_string' in {path}; pass "
                    "replace_all=true or add more context for a unique match.",
                    data={"refusal": "old_string_ambiguous"},
                )

            if replace_all:
                new_text = text.replace(old_string, new_string)
                replacements = count
            else:
                new_text = text.replace(old_string, new_string, 1)
                replacements = 1

            # Write firewall: refuse a shell-command replacement, or an edit that would
            # BREAK a .py file (checked on the resulting file). The guard protects files
            # that parse; a file that already fails to parse may still be edited — a repair
            # is made one edit at a time, and refusing every edit that does not fix the
            # whole file at once leaves a broken file unfixable by edit_file (ADR-0118).
            literal = check_literal_content(new_string)
            if literal is not None:
                return ToolResult.error(literal, data={"refusal": "literal_content"})
            after = diagnose_python_syntax(path, new_text)
            parse_note = ""
            if after is not None:
                if diagnose_python_syntax(path, text) is None:
                    return ToolResult.error(
                        after.message.replace(
                            "The file was NOT changed",
                            "This edit would break the file, so it was NOT applied",
                            1,
                        ),
                        data={
                            "refusal": "python_syntax",
                            "cause": after.cause,
                            "line": after.lineno,
                        },
                        fix=after.fix,
                    )
                parse_note = (
                    f"\nNote: {path} still does not parse — it already failed before this "
                    f"edit and still does. Fix this next:\n" + after.message.split(", at:\n", 1)[-1]
                )
            # Same shape for a skill's claims (ADR-0126): refuse only an edit that INTRODUCES
            # a host that does not exist; one the file already named is not this edit's doing.
            # …and one that makes a loadable skill UNLOADABLE (ADR-0131) — a file that never
            # parsed may still be edited: repairing it is exactly the edit a refusal asks for.
            unloadable = check_skill_format(path, new_text)
            if unloadable is not None and check_skill_format(path, text) is None:
                return ToolResult.error(
                    unloadable.replace(
                        "Refusing to write this skill", "This edit was NOT applied", 1
                    ),
                    data={"refusal": "skill_format"},
                )
            claims = check_skill_claims(path, new_text)
            if claims is not None:
                introduced = [h for h in claims[1] if h not in skill_hosts(text)]
                if introduced:
                    message = claims[0].replace(
                        "Refusing to write this skill", "This edit was NOT applied", 1
                    )
                    return ToolResult.error(
                        message, data={"refusal": "skill_claims", "hosts": introduced}
                    )

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
                f"Made {replacements} replacement{suffix} in {path}{parse_note}",
                data={"path": str(resolved), "replacements": replacements},
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to edit {path!r}: {exc}")
