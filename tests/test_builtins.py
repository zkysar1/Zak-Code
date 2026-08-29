"""Tests for the built-in tools and the default registry factory."""

from __future__ import annotations

import os
import shutil
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


async def test_grep_glob_filter_applies_to_single_file_path(ctx: ToolContext) -> None:
    # TOOL-08: a single-file path must honor the glob filter (it used to be ignored).
    (ctx.workspace_root / "a.py").write_text("found\n")
    res = await GrepTool().execute({"pattern": "found", "glob": "*.txt", "path": "a.py"}, ctx)
    assert not res.is_error
    assert res.data is not None
    assert res.data["count"] == 0  # a.py does not match *.txt -> skipped


def test_bash_timeout_cap_raised_above_60() -> None:
    # TOOL-04: the bash timeout was a hard 60s cap (a >60s build always timed out). Default stays
    # 60, but the cap is now generous so an explicit longer timeout is honored.
    assert BashTool.spec.parameters["properties"]["timeout"]["maximum"] == 600


async def test_bash_echo_output_and_exit(ctx: ToolContext) -> None:
    res = await BashTool().execute({"command": "echo hello"}, ctx)
    assert not res.is_error
    assert "hello" in res.output
    assert "[exit code: 0]" in res.output
    assert res.data is not None
    assert res.data["exit_code"] == 0


async def test_bash_nonzero_exit_is_error(ctx: ToolContext) -> None:
    cmd = "exit 3"  # works in bash (Git Bash on Windows) and the cmd.exe fallback alike
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

    from zakcode._subprocess import find_bash
    from zakcode.tools.builtins._proc import run_capturing

    monkeypatch.delenv("VIRTUAL_ENV", raising=False)  # use the workspace .venv, not the runner's
    _fake_venv(tmp_path)
    # The bash tool now runs real Git Bash on Windows (cmd.exe only on the no-bash fallback), so
    # use bash syntax there too. Format-agnostic check: the workspace venv lives under the unique
    # tmp dir, so that dir name appears in the child's PATH whether shown as C:\... or /c/...
    use_bash = os.name != "nt" or find_bash() is not None
    cmd = 'echo "$PATH"' if use_bash else "echo %PATH%"
    out, code = await run_capturing(shell_command=cmd, cwd=str(tmp_path), timeout=15)
    assert code == 0
    assert tmp_path.name in out


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
        "read_pdf",
        "create_pdf",
        "inspect_image",
        "save_image",
        "create_chart_image",
        "bash",
        "powershell",
        "web_search",
        "web_fetch",
        "secret_names",
        "update_plan",
        "schedule_wakeup",
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
    assert reg.get("readpdf") is reg.get("read_pdf")
    assert reg.get("pdf") is reg.get("create_pdf")
    assert reg.get("makepdf") is reg.get("create_pdf")
    assert reg.get("image_info") is reg.get("inspect_image")
    assert reg.get("image") is reg.get("save_image")
    assert reg.get("chart") is reg.get("create_chart_image")
    assert reg.get("sh") is reg.get("bash")
    assert reg.get("shell") is reg.get("bash")
    assert reg.get("plan") is reg.get("update_plan")
    assert reg.get("todo") is reg.get("update_plan")
    assert reg.get("deliberate") is reg.get("deep_think")
    assert reg.get("secrets") is reg.get("secret_names")
    assert reg.get("list_secrets") is reg.get("secret_names")
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


# ── 127/126 remedy hints (2026-08-25) ─────────────────────────────────────────


async def test_bash_127_names_the_workspace_script(tmp_path) -> None:
    """A bare script name not on PATH gets a fix naming the real workspace path —
    one error instead of the model's error -> find -> retry ritual (measured on a
    mind agent 2026-08-25: dozens of identical 127s on core/scripts names)."""
    scripts = tmp_path / "core" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "pipeline-read.sh").write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "pipeline-read.sh --stage active"}, ctx)
    assert res.is_error
    assert res.data is not None and res.data["exit_code"] == 127
    assert res.fix is not None
    assert "core/scripts/pipeline-read.sh" in res.fix
    assert "bash core/scripts/pipeline-read.sh" in res.fix


async def test_bash_127_unknown_command_gets_no_fix(tmp_path) -> None:
    """A genuinely nonexistent command stays a plain 127 — no speculative hint."""
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "definitely-not-a-real-cmd-xyz"}, ctx)
    assert res.is_error
    assert res.fix is None


@pytest.mark.skipif(os.name == "nt", reason="exec-bit / exit-126 semantics are POSIX-only")
async def test_bash_126_names_the_chmod_escape(tmp_path) -> None:
    """A non-executable script run directly gets the chmod / `bash path` hint once,
    instead of the model rediscovering it per file."""
    script = tmp_path / "doit.sh"
    script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    script.chmod(0o644)
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "./doit.sh"}, ctx)
    assert res.is_error
    assert res.data is not None and res.data["exit_code"] == 126
    assert res.fix is not None and "chmod +x" in res.fix and "bash ./doit.sh" in res.fix


@pytest.mark.skipif(shutil.which("python3") is None, reason="needs a python3 on PATH")
async def test_bash_python_inline_parse_error_gets_file_hint(tmp_path) -> None:
    """A failed inline ``python -c`` program with a Syntax/IndentationError gets the
    write-a-file hint — the quoting-truncation trap (an apostrophe in a single-quoted
    program) had a model retry the identical broken command three times (2026-08-26)."""
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": 'python3 -c "x = ("'}, ctx)
    assert res.is_error
    assert res.fix is not None
    assert "write the program to a file" in res.fix


def test_python_inline_fix_predicate() -> None:
    from zakcode.tools.builtins.bash import _python_inline_fix

    err = (
        'File "<string>", line 50\n    # For simplicity, well\nIndentationError: unexpected indent'
    )
    # Fires on python -c / python3 -c / the Windows py-launcher form.
    assert _python_inline_fix("python3 -c 'import os\nprint(1)'", err) is not None
    assert _python_inline_fix('py -3 -c "print(1)"', err) is not None
    # Never fires on script-path invocations, other tools' -c flags, or non-parse errors.
    assert _python_inline_fix("python3 script.py", err) is None
    assert _python_inline_fix("grep -c foo bar.py", err) is None
    assert _python_inline_fix("python3 -c 'print(1)'", "NameError: boom") is None


def test_interpreter_mismatch_fix_predicate() -> None:
    from zakcode.tools.builtins.bash import _interpreter_mismatch_fix as fix

    # A shell script fed to Python — the reducer's verbatim shape (2026-08-29), the py
    # launcher, options before the path, a cd/env prefix.
    hint = fix("cd /w && MIND_AGENT=coach python3 core/scripts/aspirations-update-goal.sh --a b")
    assert hint is not None and "bash core/scripts/aspirations-update-goal.sh" in hint
    assert fix("py -3 core/scripts/x.sh") is not None
    assert fix("python3 -u ./x.sh; echo done") is not None
    # Python fed to a shell.
    hint = fix("bash tools/check.py --fast")
    assert hint is not None and "python3 tools/check.py" in hint
    assert fix("sh -x tools/check.py") is not None
    # Never on an inline program, a script passed later, the right interpreter, or a
    # module run.
    assert fix("python3 -c \"print(open('x.sh').read())\"") is None
    assert fix("python3 tool.py x.sh") is None
    assert fix("bash core/scripts/x.sh && python3 tool.py") is None
    assert fix("python3 -m pytest tests/x.sh") is None
    assert fix('bash -c "python3 x.py"') is None


@pytest.mark.skipif(shutil.which("python3") is None, reason="needs a python3 on PATH")
async def test_bash_python_on_a_shell_script_names_the_interpreter(tmp_path) -> None:
    """``python3 x.sh`` fails with a SyntaxError that reads like a broken script; the hint
    names the real fix — the reducer retried the identical command four times (2026-08-29)."""
    script = tmp_path / "doit.sh"
    script.write_text('case "$1" in\n  --a) echo a ;;\nesac\n', encoding="utf-8")
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "python3 doit.sh --a"}, ctx)
    assert res.is_error
    assert res.fix is not None and "bash doit.sh" in res.fix


async def test_bash_a_pipe_that_hides_the_exit_code_still_names_the_interpreter(tmp_path) -> None:
    """``python3 x.sh … | tail -40`` exits 0 — tail's status — so the hint chain never ran;
    the reducer read the SyntaxError as a broken script again, the day the hint shipped
    (2026-08-29). The mismatched command plus the interpreter's error text is the signal."""
    script = tmp_path / "doit.sh"
    script.write_text('case "$1" in\n  --a) echo a ;;\nesac\n', encoding="utf-8")
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "python3 doit.sh --a 2>&1 | tail -40"}, ctx)
    assert res.is_error
    assert res.fix is not None and "bash doit.sh" in res.fix and "pipe's" in res.fix
    assert res.data is not None and res.data["exit_code"] == 0
    # A pipe on a command that ran fine is still fine.
    ok = await BashTool().execute({"command": "bash doit.sh --a | tail -1"}, ctx)
    assert not ok.is_error and ok.output.startswith("a\n")


_ORIGINAL = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 19, in <module>\n'
    "AttributeError: 'list' object has no attribute 'get'\n"
)
_APPORT = (
    "Error in sys.excepthook:\n"
    "Traceback (most recent call last):\n"
    '  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 228, '
    "in partial_apport_excepthook\n"
    "    return apport_excepthook(binary, exc_type, exc_obj, exc_tb)\n"
    "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"
    '  File "/usr/lib/python3/dist-packages/apport_python_hook.py", line 114, '
    "in apport_excepthook\n"
    '    report["ExecutableTimestamp"] = str(int(os.stat(binary).st_mtime))\n'
    "FileNotFoundError: [Errno 2] No such file or directory: '/opt/coach-mind/-c'\n"
    "\n"
    "Original exception was:\n"
)


def test_apport_excepthook_noise_is_stripped_from_bash_output() -> None:
    """zc-03, 2026-08-29: 20 of the fleet's 61 tracebacks carried apport's own ~20-line
    crash plus a re-print of the original — read by a small model as a second error."""
    from zakcode.tools.builtins.bash import _strip_apport_noise

    # The common shape: original, apport's failure, the original again.
    assert _strip_apport_noise(_ORIGINAL + _APPORT + _ORIGINAL) == _ORIGINAL
    # Two inline programs in one command: both blocks go, both originals stay.
    twice = "one\n" + _ORIGINAL + _APPORT + _ORIGINAL + "two\n" + _ORIGINAL + _APPORT + _ORIGINAL
    assert _strip_apport_noise(twice) == "one\n" + _ORIGINAL + "two\n" + _ORIGINAL
    # An original printed only after the block is kept — it is the error.
    assert _strip_apport_noise("partial output\n" + _APPORT + _ORIGINAL) == (
        "partial output\n" + _ORIGINAL
    )
    # Some other hook's failure is real output.
    other = _ORIGINAL + _APPORT.replace("apport_python_hook", "my_hook") + _ORIGINAL
    assert _strip_apport_noise(other) == other
    # Output without the block is untouched.
    assert _strip_apport_noise(_ORIGINAL) == _ORIGINAL
    assert _strip_apport_noise("") == ""


def test_locate_basename_is_bounded_and_prunes(tmp_path) -> None:
    from zakcode.tools.builtins.bash import _locate_basename

    (tmp_path / "node_modules" / "deep").mkdir(parents=True)
    (tmp_path / "node_modules" / "deep" / "x.sh").write_text("no", encoding="utf-8")
    (tmp_path / "core" / "scripts").mkdir(parents=True)
    (tmp_path / "core" / "scripts" / "x.sh").write_text("yes", encoding="utf-8")
    assert _locate_basename(tmp_path, "x.sh") == "core/scripts/x.sh"  # pruned dir never wins
    assert _locate_basename(tmp_path, "missing.sh") is None
    assert _locate_basename(tmp_path, "core/scripts/x.sh") is None  # basenames only
