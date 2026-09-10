"""Atomically write a text file within the workspace."""

from __future__ import annotations

import contextlib
import os
import tempfile

from zakcode.artifacts import ArtifactError, artifact_from_path
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
    diagnose_python_syntax,
    resolve_path,
)


class WriteFileTool(Tool):
    """Create or overwrite a file inside the workspace, writing atomically."""

    spec = ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a text file within the workspace. Parent directories "
            "are created as needed. The write is atomic (temp file + os.replace)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write (absolute or relative to the workspace root).",
                },
                "content": {
                    "type": "string",
                    "description": "Full text content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
        required_permission=PermissionTier.WORKSPACE_WRITE,
        concurrency=ConcurrencyClass.PATH_SCOPED,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        """Write ``content`` to ``path`` atomically."""
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            return ToolResult.error("'path' is required and must be a string.")
        if not isinstance(content, str):
            return ToolResult.error("'content' is required and must be a string.")

        # Deterministic write firewall (refuse-only, before any bytes land): reject a
        # shell command written as file content, or a .py file that will not compile.
        # ``data["refusal"]`` tags the error as a refusal of the model's OWN content, which
        # the loop's blocker gate distinguishes from an environmental failure (ADR-0118).
        literal = check_literal_content(content)
        if literal is not None:
            return ToolResult.error(literal, data={"refusal": "literal_content"})
        refusal = diagnose_python_syntax(path, content)
        if refusal is not None:
            return ToolResult.error(
                refusal.message,
                data={"refusal": "python_syntax", "cause": refusal.cause, "line": refusal.lineno},
                fix=refusal.fix,
            )
        # A skill names hosts a future session will call; a host that does not exist is the
        # model's own fabrication, refused the same way a .py that will not compile is
        # (ADR-0126).
        claims = check_skill_claims(path, content)
        if claims is not None:
            return ToolResult.error(claims[0], data={"refusal": "skill_claims", "hosts": claims[1]})

        try:
            resolved = resolve_path(path, ctx.workspace_root, ctx.extra_workspace_roots)
        except PathEscapeError as exc:
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to resolve path {path!r}: {exc}")

        try:
            if resolved.is_dir():
                return ToolResult.error(f"Path is a directory, not a file: {path}")

            parent = resolved.parent
            if parent.exists() and not parent.is_dir():
                return ToolResult.error(f"Parent path is not a directory: {parent}")
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return ToolResult.error(f"Could not create parent directory for {path}: {exc}")

            data = content.encode("utf-8")
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

            artifacts = []
            with contextlib.suppress(ArtifactError, OSError):
                artifacts.append(
                    artifact_from_path(
                        resolved,
                        workspace_root=ctx.workspace_root,
                        created_by_tool=self.spec.name,
                    )
                )
            result_data = {"path": str(resolved), "bytes": len(data)}
            if artifacts:
                result_data["artifact_id"] = artifacts[0].id
            return ToolResult.ok(
                f"Wrote {len(data)} bytes to {path}",
                data=result_data,
                artifacts=artifacts,
            )
        except Exception as exc:  # noqa: BLE001 - handlers must never raise
            return ToolResult.error(f"Failed to write {path!r}: {exc}")
