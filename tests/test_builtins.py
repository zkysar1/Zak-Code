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
    assert len(res.artifacts) == 1
    assert res.artifacts[0].path == "sub/hello.txt"
    assert res.artifacts[0].filename == "hello.txt"
    assert res.data is not None
    assert res.data["artifact_id"] == res.artifacts[0].id

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
    # Slice content, then an explicit continuation marker (the slice stops before EOF; line 4
    # of 4 remains) so the partial read does not read to the model as the whole file.
    assert res.output.startswith("b\nc\n")
    assert "of 4" in res.output and "offset=4" in res.output


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


def _fake_venv(root: Path) -> Path:
    """Create a fake project venv under ``root`` and return its bin/Scripts dir."""
    import os

    bin_name = "Scripts" if os.name == "nt" else "bin"
    py = "python.exe" if os.name == "nt" else "python"
    bindir = root / ".venv" / bin_name
    bindir.mkdir(parents=True)
    (bindir / py).write_text("")  # a fake interpreter so the dir counts as a venv
    return bindir


def test_project_venv_bin_detects_workspace_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zakcode.tools.builtins._proc import _project_venv_bin

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # else the test runner's own venv wins
    bindir = _fake_venv(tmp_path)
    assert _project_venv_bin(str(tmp_path)) == str(bindir)


def test_project_venv_bin_none_without_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zakcode.tools.builtins._proc import _project_venv_bin

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    assert _project_venv_bin(str(tmp_path)) is None
    (tmp_path / ".venv").mkdir()  # an empty .venv (no interpreter) does not count
    assert _project_venv_bin(str(tmp_path)) is None


async def test_subprocess_prepends_project_venv_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End-to-end: the child's PATH starts with the workspace venv's bin dir, so `python`/`pytest`
    # resolve to the project's interpreter (the "No module named pytest" false-failure fix).
    import os

    from zakcode.tools.builtins._proc import run_capturing

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # use the workspace .venv, not the runner's
    bindir = _fake_venv(tmp_path)
    cmd = "echo %PATH%" if os.name == "nt" else 'echo "$PATH"'
    out, code = await run_capturing(shell_command=cmd, cwd=str(tmp_path), timeout=15)
    assert code == 0
    assert str(bindir) in out


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
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "read_docx",
        "read_xlsx",
        "create_docx",
        "create_xlsx",
        "inspect_image",
        "save_image",
        "create_chart_image",
        "bash",
        "powershell",
        "web_search",
        "web_fetch",
        "update_plan",
        "deep_think",
    }
    # Aliases resolve to the canonical tools (M1 added "edit" -> edit_file).
    assert reg.get("read") is reg.get("read_file")
    assert reg.get("write") is reg.get("write_file")
    assert reg.get("edit") is reg.get("edit_file")
    assert reg.get("ls") is reg.get("list_dir")
    assert reg.get("bash") is reg.get("bash")
    assert reg.get("pwsh") is reg.get("powershell")  # M10: PowerShell tool + alias
    # Guessable POSIX-muscle-memory aliases ("learn one, guess the rest").
    assert reg.get("cat") is reg.get("read_file")
    assert reg.get("dir") is reg.get("list_dir")
    assert reg.get("find") is reg.get("glob")
    assert reg.get("search") is reg.get("grep")
    assert reg.get("rg") is reg.get("grep")
    assert reg.get("read_word") is reg.get("read_docx")
    assert reg.get("read_excel") is reg.get("read_xlsx")
    assert reg.get("word") is reg.get("create_docx")
    assert reg.get("docx") is reg.get("create_docx")
    assert reg.get("excel") is reg.get("create_xlsx")
    assert reg.get("xlsx") is reg.get("create_xlsx")
    assert reg.get("image_info") is reg.get("inspect_image")
    assert reg.get("image") is reg.get("save_image")
    assert reg.get("chart") is reg.get("create_chart_image")
    assert reg.get("sh") is reg.get("bash")
    assert reg.get("shell") is reg.get("bash")
    assert reg.get("plan") is reg.get("update_plan")
    assert reg.get("todo") is reg.get("update_plan")
    assert reg.get("deliberate") is reg.get("deep_think")
    # Aliases are NOT canonical names (silent fallback; not exposed in the prompt).
    assert "cat" not in reg.names() and "search" not in reg.names()


def test_register_rejects_colliding_alias() -> None:
    from zakcode.tools.base import ToolRegistry

    # An alias that shadows another tool's canonical name is rejected.
    reg = ToolRegistry()
    reg.register(ReadFileTool())  # canonical "read_file"
    with pytest.raises(ValueError, match="collides with a registered tool name"):
        reg.register(WriteFileTool(), aliases=["read_file"])

    # An alias already mapped to a different tool is rejected.
    reg2 = ToolRegistry()
    reg2.register(ReadFileTool(), aliases=["x"])
    with pytest.raises(ValueError, match="already maps to"):
        reg2.register(WriteFileTool(), aliases=["x"])

    # A new tool whose canonical NAME equals an existing alias is rejected — otherwise it would
    # register but be permanently shadowed by the alias (every call dispatched elsewhere).
    from zakcode.tools.base import Tool, ToolResult, ToolSpec

    class _Cat(Tool):
        spec = ToolSpec(name="cat", description="a tool literally named 'cat'")

        async def execute(self, args, ctx):  # noqa: ANN001, ARG002
            return ToolResult.ok("x")

    reg3 = ToolRegistry()
    reg3.register(ReadFileTool(), aliases=["cat"])
    with pytest.raises(ValueError, match="collides with an existing alias"):
        reg3.register(_Cat())


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
