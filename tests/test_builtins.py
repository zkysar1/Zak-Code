"""Tests for the built-in tools and the default registry factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zakcode.config import PermissionTier
from zakcode.tools import default_registry
from zakcode.tools.base import ConcurrencyClass, ToolContext
from zakcode.tools.builtins import (
    BashTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    """A ToolContext rooted at a temporary workspace."""
    return ToolContext(workspace_root=tmp_path)


async def test_write_then_read_round_trip(ctx: ToolContext) -> None:
    write = WriteFileTool()
    read = ReadFileTool()

    res = await write.execute({"path": "sub/hello.txt", "content": "hi there"}, ctx)
    assert not res.is_error, res.output
    assert (ctx.workspace_root / "sub" / "hello.txt").read_text() == "hi there"

    res = await read.execute({"path": "sub/hello.txt"}, ctx)
    assert not res.is_error
    assert res.output == "hi there"


async def test_read_line_slice(ctx: ToolContext) -> None:
    # Write bytes explicitly so the test is independent of platform newline
    # translation (Path.write_text would convert \n to \r\n on Windows).
    (ctx.workspace_root / "lines.txt").write_bytes(b"a\nb\nc\nd\n")
    read = ReadFileTool()
    res = await read.execute({"path": "lines.txt", "offset": 2, "limit": 2}, ctx)
    assert not res.is_error
    assert res.output == "b\nc\n"


async def test_read_missing_file_is_error_not_exception(ctx: ToolContext) -> None:
    read = ReadFileTool()
    res = await read.execute({"path": "nope.txt"}, ctx)
    assert res.is_error
    assert "not found" in res.output.lower()


async def test_list_dir(ctx: ToolContext) -> None:
    (ctx.workspace_root / "adir").mkdir()
    (ctx.workspace_root / "afile.txt").write_text("x")
    res = await ListDirTool().execute({}, ctx)
    assert not res.is_error
    assert "adir/" in res.output
    assert "afile.txt" in res.output
    assert res.data is not None
    assert res.data["count"] == 2


async def test_glob(ctx: ToolContext) -> None:
    (ctx.workspace_root / "a.py").write_text("x")
    (ctx.workspace_root / "b.py").write_text("y")
    (ctx.workspace_root / "c.txt").write_text("z")
    res = await GlobTool().execute({"pattern": "*.py"}, ctx)
    assert not res.is_error
    assert res.data is not None
    assert res.data["count"] == 2
    assert all(m.endswith(".py") for m in res.data["matches"])


async def test_glob_recursive(ctx: ToolContext) -> None:
    nested = ctx.workspace_root / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "deep.py").write_text("x")
    res = await GlobTool().execute({"pattern": "**/*.py"}, ctx)
    assert not res.is_error
    assert res.data is not None
    assert any("deep.py" in m for m in res.data["matches"])


async def test_grep_file_line_match(ctx: ToolContext) -> None:
    (ctx.workspace_root / "src.txt").write_text("alpha\nbeta needle here\ngamma\n")
    res = await GrepTool().execute({"pattern": "needle"}, ctx)
    assert not res.is_error
    assert res.data is not None
    assert res.data["count"] == 1
    row = res.data["matches"][0]
    # Format is file:line:match.
    assert ":2:" in row
    assert "needle" in row


async def test_grep_skips_binary(ctx: ToolContext) -> None:
    (ctx.workspace_root / "bin.dat").write_bytes(b"needle\x00\x00more")
    (ctx.workspace_root / "text.txt").write_text("needle\n")
    res = await GrepTool().execute({"pattern": "needle"}, ctx)
    assert not res.is_error
    assert res.data is not None
    assert res.data["count"] == 1
    assert "text.txt" in res.data["matches"][0]


async def test_grep_glob_filter(ctx: ToolContext) -> None:
    (ctx.workspace_root / "a.py").write_text("found\n")
    (ctx.workspace_root / "b.txt").write_text("found\n")
    res = await GrepTool().execute({"pattern": "found", "glob": "*.py"}, ctx)
    assert not res.is_error
    assert res.data is not None
    assert res.data["count"] == 1
    assert "a.py" in res.data["matches"][0]


async def test_bash_echo_output_and_exit(ctx: ToolContext) -> None:
    res = await BashTool().execute({"command": "echo hello"}, ctx)
    assert not res.is_error
    assert "hello" in res.output
    assert "[exit code: 0]" in res.output
    assert res.data is not None
    assert res.data["exit_code"] == 0


async def test_bash_nonzero_exit_is_error(ctx: ToolContext) -> None:
    cmd = "exit 3" if sys.platform != "win32" else "cmd /c exit 3"
    res = await BashTool().execute({"command": cmd}, ctx)
    assert res.is_error
    assert res.data is not None
    assert res.data["exit_code"] == 3


async def test_path_escape_is_error(ctx: ToolContext) -> None:
    read = ReadFileTool()
    res = await read.execute({"path": "../outside.txt"}, ctx)
    assert res.is_error
    assert "outside" in res.output.lower()


async def test_write_path_escape_is_error(ctx: ToolContext) -> None:
    write = WriteFileTool()
    res = await write.execute({"path": "../escape.txt", "content": "nope"}, ctx)
    assert res.is_error
    assert not (ctx.workspace_root.parent / "escape.txt").exists()


def test_default_registry_has_all_tools_and_aliases() -> None:
    reg = default_registry()
    assert set(reg.names()) == {
        "read_file",
        "write_file",
        "list_dir",
        "glob",
        "grep",
        "bash",
    }
    # Aliases resolve to the canonical tools.
    assert reg.get("read") is reg.get("read_file")
    assert reg.get("write") is reg.get("write_file")
    assert reg.get("ls") is reg.get("list_dir")
    assert reg.get("bash") is reg.get("bash")


def test_specs_have_expected_permissions_and_concurrency() -> None:
    assert ReadFileTool.spec.required_permission == PermissionTier.READ_ONLY
    assert ReadFileTool.spec.concurrency == ConcurrencyClass.READ_ONLY_SAFE
    assert WriteFileTool.spec.required_permission == PermissionTier.WORKSPACE_WRITE
    assert WriteFileTool.spec.concurrency == ConcurrencyClass.PATH_SCOPED
    assert BashTool.spec.required_permission == PermissionTier.DANGER_FULL_ACCESS
    assert BashTool.spec.concurrency == ConcurrencyClass.NEVER_PARALLEL


async def test_registry_execute_dispatch(ctx: ToolContext) -> None:
    reg = default_registry()
    res = await reg.execute("write", {"path": "x.txt", "content": "data"}, ctx)
    assert not res.is_error
    res = await reg.execute("read", {"path": "x.txt"}, ctx)
    assert res.output == "data"
    res = await reg.execute("does_not_exist", {}, ctx)
    assert res.is_error
