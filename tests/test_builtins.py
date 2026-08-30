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


# ── ENOENT hints (2026-08-29) ─────────────────────────────────────────────────


def _mind_workspace(tmp_path: Path) -> Path:
    """The shape of a Mind deployment: framework scripts under core/scripts, the domain's
    data under a HIDDEN .mind-data/ (which the basename locator used to prune), and a
    .git/ that must never be offered as a lead."""
    scripts = tmp_path / "core" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("reasoning-bank-add.sh", "reasoning-bank-read.sh", "wm-read.sh", "wm-set.sh"):
        (scripts / name).write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    (scripts / "history-list.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    world = tmp_path / ".mind-data" / "world"
    world.mkdir(parents=True)
    (world / "forged-skills.yaml").write_text("skills: []\n", encoding="utf-8")
    objects = tmp_path / ".git" / "objects"
    objects.mkdir(parents=True)
    (objects / "forged-skills.yaml").write_text("no", encoding="utf-8")
    return tmp_path


async def test_bash_enoent_names_where_the_file_actually_is(tmp_path) -> None:
    """`cat world/forged-skills.yaml` on a Mind deployment: the file lives at
    .mind-data/world/forged-skills.yaml. Measured 2026-08-29 (zc-03, eight Bodies): 15 of
    the day's 73 failed commands were ENOENT, every one a guessed path."""
    ctx = ToolContext(workspace_root=_mind_workspace(tmp_path))
    res = await BashTool().execute({"command": "cat world/forged-skills.yaml"}, ctx)
    assert res.is_error
    assert res.data is not None and res.data["exit_code"] != 0
    assert res.fix is not None
    assert ".mind-data/world/forged-skills.yaml" in res.fix
    assert ".git/" not in res.fix
    # A stray `world/` dir at the root does not silence it: the hit ENDS with the guess,
    # which is the wrong-prefix signature, not an optional-file check.
    (tmp_path / "world").mkdir()
    res = await BashTool().execute({"command": "cat world/forged-skills.yaml"}, ctx)
    assert res.fix is not None and ".mind-data/world/forged-skills.yaml" in res.fix


async def test_bash_wrong_prefix_hint_says_cd_will_not_help(tmp_path) -> None:
    """Measured 2026-08-30 (zc-03): refused for `bash world/scripts/yahoo/discover.sh`
    with the lead naming `.mind-data/world/scripts/yahoo/discover.sh`, the Body replied
    `cd <workspace root> && <same command>` — it read "(or `cd` there first)" as a cwd
    problem — and only after a second refusal used the path already named. When the
    guess is the real path minus its leading directory, say exactly that."""
    root = _mind_workspace(tmp_path)
    script = root / ".mind-data" / "world" / "scripts" / "yahoo" / "discover.sh"
    script.parent.mkdir(parents=True)
    script.write_text("echo discovered\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=root)
    res = await BashTool().execute({"command": "bash world/scripts/yahoo/discover.sh"}, ctx)
    assert res.is_error and res.data is not None and res.data.get("script_path_missing") is True
    assert "missing its leading '.mind-data/'" in res.output
    assert "'.mind-data/world/scripts/yahoo/discover.sh' exactly as written" in res.output
    assert "a `cd` will not help" in res.output
    assert "or `cd` there first" not in res.output
    # The generic wording survives for a same-named file in an UNRELATED directory.
    ctx2 = ToolContext(workspace_root=_mind_workspace(tmp_path / "other"))
    (tmp_path / "other" / "core" / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "other" / "core" / "scripts" / "discover.sh").write_text("", encoding="utf-8")
    res = await BashTool().execute({"command": "bash tools/discover.sh"}, ctx2)
    assert res.is_error and "or `cd` there first" in res.output
    assert "will not help" not in res.output


async def test_bash_enoent_offers_the_nearest_names_for_an_invented_script(tmp_path) -> None:
    """`python3 world/scripts/reasoning-bank.py add …` — no such script anywhere; the real
    writers are reasoning-bank-add.sh / reasoning-bank-read.sh."""
    ctx = ToolContext(workspace_root=_mind_workspace(tmp_path))
    res = await BashTool().execute(
        {"command": "python3 world/scripts/reasoning-bank.py add --entry x"}, ctx
    )
    assert res.is_error
    assert res.fix is not None
    assert "core/scripts/reasoning-bank-add.sh" in res.fix
    assert "reasoning-bank.py" in res.fix


async def test_bash_enoent_with_no_lead_gets_no_fix(tmp_path) -> None:
    """A genuinely absent file with nothing similar in the workspace stays a plain error
    — no speculative hint."""
    ctx = ToolContext(workspace_root=_mind_workspace(tmp_path))
    res = await BashTool().execute(
        {"command": "cat agents/coach/session/iteration-checkpoint.json"}, ctx
    )
    assert res.is_error
    assert res.fix is None


async def test_bash_enoent_typo_in_a_real_directory_names_its_siblings(tmp_path) -> None:
    """`bash core/scripts/wm-list.sh`: the directory is real, the name is invented. The
    family sharing its leading token (wm-read.sh, wm-set.sh) is the answer — not the
    global token match (`history-list.sh` on 'list') the live smoke on zc-03 produced."""
    ctx = ToolContext(workspace_root=_mind_workspace(tmp_path))
    res = await BashTool().execute({"command": "bash core/scripts/wm-list.sh"}, ctx)
    assert res.is_error
    assert res.fix is not None
    assert "wm-read.sh" in res.fix and "wm-set.sh" in res.fix
    assert "history-list.sh" not in res.fix


async def test_bash_enoent_same_words_in_another_order_lead_the_hint(tmp_path) -> None:
    """`bash core/scripts/blocker-create.sh` when the script is `create-blocker.sh`: the
    leading-token family (blocker-create-gate.sh, blocker-recheck.sh) is not the answer
    and used to be the whole hint — measured 2026-08-30 (zc-03), six more commands to
    find the real file. The reordered name leads, as a pasteable path."""
    root = _mind_workspace(tmp_path)
    scripts = root / "core" / "scripts"
    for name in ("create-blocker.sh", "blocker-create-gate.sh", "blocker-recheck.sh"):
        (scripts / name).write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=root)
    res = await BashTool().execute(
        {"command": "bash core/scripts/blocker-create.sh --goal g-006-22"}, ctx
    )
    assert res.is_error
    assert res.fix is not None
    assert "same words in another order" in res.fix
    assert res.fix.index("core/scripts/create-blocker.sh") < res.fix.index("blocker-create-gate.sh")
    assert "blocker-recheck.sh" in res.fix


async def test_bash_enoent_optional_file_in_a_real_directory_is_silent(tmp_path) -> None:
    """A deliberate check of an optional file (`cat <session>/iteration-checkpoint.json`)
    in a directory that exists, with nothing similar beside it, gets NO hint — the
    same-named file under another agent's dir is not a lead, it is noise."""
    root = _mind_workspace(tmp_path)
    (root / "agents" / "coach" / "session").mkdir(parents=True)
    (root / "agents" / "alpha" / "session").mkdir(parents=True)
    (root / "agents" / "alpha" / "session" / "iteration-checkpoint.json").write_text("{}")
    ctx = ToolContext(workspace_root=root)
    res = await BashTool().execute(
        {"command": "cat agents/coach/session/iteration-checkpoint.json"}, ctx
    )
    assert res.is_error
    assert res.fix is None


async def test_bash_enoent_exact_hit_lists_the_family_beside_it(tmp_path) -> None:
    """`python3 world/scripts/reasoning-bank.py add` when a `reasoning-bank.py` module
    does exist under core/scripts: the hit is named AND its siblings sharing the
    leading token, because the wrapper (`reasoning-bank-add.sh`) is what the model
    wanted — the module itself is a silent no-op when run as a script."""
    root = _mind_workspace(tmp_path)
    (root / "core" / "scripts" / "reasoning-bank.py").write_text("x = 1\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=root)
    res = await BashTool().execute(
        {"command": "python3 world/scripts/reasoning-bank.py add --entry x"}, ctx
    )
    assert res.is_error
    assert res.fix is not None
    assert "core/scripts/reasoning-bank.py" in res.fix
    assert "reasoning-bank-add.sh" in res.fix and "reasoning-bank-read.sh" in res.fix


def test_enoent_fix_predicate_reads_every_measured_shape(tmp_path) -> None:
    from zakcode.tools.builtins.bash import _enoent_fix

    root = _mind_workspace(tmp_path)
    shapes = [
        "python3: can't open file '/opt/mind/.mind-data/world/scripts/reasoning-bank.py': "
        "[Errno 2] No such file or directory",
        'Traceback (most recent call last):\n  File "<string>", line 14, in <module>\n'
        "FileNotFoundError: [Errno 2] No such file or directory: 'world/forged-skills.yaml'",
        "ls: cannot access 'world/forged-skills.yaml': No such file or directory",
        "ERROR: file or directory not found: /opt/mind/world/forged-skills.yaml\n\n",
        "bash: world/forged-skills.yaml: No such file or directory",
        "cat: world/forged-skills.yaml: No such file or directory",
        "bash: line 1: ./world/forged-skills.yaml: No such file or directory",
    ]
    for out in shapes:
        assert _enoent_fix(out, root, []) is not None, out
    nearest = _enoent_fix(shapes[0], root, [])
    assert nearest is not None and "reasoning-bank-add.sh" in nearest
    exact = _enoent_fix(shapes[2], root, [])
    assert exact is not None and ".mind-data/world/forged-skills.yaml" in exact
    # apport's own crash names '<cwd>/-c' — never a hint about a file called -c.
    assert (
        _enoent_fix(
            "FileNotFoundError: [Errno 2] No such file or directory: '/opt/mind/-c'", root, []
        )
        is None
    )
    assert _enoent_fix("cat: nothing-like-this.txt: No such file or directory", root, []) is None
    assert _enoent_fix("all good\n", root, []) is None


def test_enoent_regexes_capture_the_coreutils_and_grep_shapes() -> None:
    """The four ENOENTs the first deploy of the hint did NOT match (zc-03, 2026-08-29,
    verbatim shapes): touch, grep, and directory targets — plus the coreutils siblings
    that print the same `cannot <verb> 'x'` frame."""
    from zakcode.tools.builtins.bash import _ENOENT_RES

    def captured(out: str) -> str | None:
        for rx in _ENOENT_RES:
            m = rx.search(out)
            if m:
                return m.group(1)
        return None

    p = "/opt/mind/.mind-data/agents/coach/sessions/82a/light-prime-done"
    assert captured(f"touch: cannot touch '{p}': No such file or directory") == p
    assert captured(f"grep: {p}: No such file or directory") == p
    assert (
        captured(
            "ls: cannot access '/opt/mind/.mind-data/agents/coach/sessions/': "
            "No such file or directory"
        )
        == "/opt/mind/.mind-data/agents/coach/sessions/"
    )
    assert (
        captured(
            "mkdir: cannot create directory '/opt/mind/.mind-data/agents': "
            "No such file or directory"
        )
        == "/opt/mind/.mind-data/agents"
    )
    assert captured("stat: cannot statx 'w/x.yaml': No such file or directory") == "w/x.yaml"
    assert captured("rm: cannot remove 'w/x.yaml': No such file or directory") == "w/x.yaml"
    # Newer coreutils quote the path in the plain frame too (Git Bash on Windows CI).
    q = "C:/Users/r/AppData/Local/Temp/ws/world/forged-skills.yaml"
    assert captured(f"cat: '{q}': No such file or directory") == q
    assert captured("cat: /tmp/ws/world/forged-skills.yaml: No such file or directory") == (
        "/tmp/ws/world/forged-skills.yaml"
    )


async def test_bash_enoent_invented_prefix_names_the_real_directory(tmp_path) -> None:
    """`touch /ws/.mind-data/agents/coach/sessions/<sid>/light-prime-done` — measured
    2026-08-29 (zc-03): a Body invented `.mind-data/agents/...`; `agents/` lives at the
    workspace root. Nothing anywhere is named light-prime-done (the file was about to be
    CREATED), so the file search has no lead — the first missing path component does."""
    root = _mind_workspace(tmp_path)
    (root / "agents" / "coach" / "sessions" / "abc").mkdir(parents=True)
    ctx = ToolContext(workspace_root=root)
    bad = root / ".mind-data" / "agents" / "coach" / "sessions" / "abc" / "light-prime-done"
    # as_posix(): bash eats a WindowsPath's backslashes, and `touch C:Usersx` SUCCEEDS
    # in the cwd (measured on CI, 2026-08-29) — so the absolute form here must be posix.
    res = await BashTool().execute({"command": f"touch {bad.as_posix()}"}, ctx)
    assert res.is_error
    assert res.fix is not None, res.output
    assert "'.mind-data/agents' is the first missing part" in res.fix
    assert "a directory named 'agents' does exist: agents" in res.fix
    assert "light-prime-done" not in res.fix.split("does exist")[1]  # no phantom file lead


async def test_bash_enoent_invented_prefix_also_names_same_named_files(tmp_path) -> None:
    """`grep x /ws/.mind-data/agents/coach/sessions/<sid>/body-manifest.yaml` when another
    session's manifest exists: the same-named file is the more specific lead and keeps
    the first word (it is how the reasoning-bank.py family case has always read), and
    the invented-prefix diagnosis rides beside it."""
    root = _mind_workspace(tmp_path)
    other = root / "agents" / "coach" / "sessions" / "xyz"
    other.mkdir(parents=True)
    (other / "body-manifest.yaml").write_text("body_state: active\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=root)
    bad = root / ".mind-data" / "agents" / "coach" / "sessions" / "abc" / "body-manifest.yaml"
    res = await BashTool().execute({"command": f"grep -c body_state {bad.as_posix()}"}, ctx)
    assert res.is_error
    assert res.fix is not None, res.output
    assert "agents/coach/sessions/xyz/body-manifest.yaml" in res.fix
    assert "'.mind-data/agents' is the first missing part" in res.fix
    assert "a directory named 'agents' does exist: agents" in res.fix


async def test_bash_enoent_directory_guessed_at_the_wrong_place(tmp_path) -> None:
    """`ls world/` on a Mind deployment: no file is named `world`, its parent (the root)
    exists, and nothing at the root shares its token — the directory itself is the lead."""
    ctx = ToolContext(workspace_root=_mind_workspace(tmp_path))
    res = await BashTool().execute({"command": "ls world/"}, ctx)
    assert res.is_error
    assert res.fix is not None, res.output
    assert "a directory named 'world' does: .mind-data/world" in res.fix


async def test_bash_enoent_absolute_guess_matches_the_real_file(tmp_path) -> None:
    """An ABSOLUTE wrong-prefix guess (`cat <root>/world/forged-skills.yaml`) must match
    the hit `.mind-data/world/forged-skills.yaml` the same way the relative form does."""
    root = _mind_workspace(tmp_path)
    ctx = ToolContext(workspace_root=root)
    res = await BashTool().execute(
        {"command": f"cat {root.as_posix()}/world/forged-skills.yaml"}, ctx
    )
    assert res.is_error
    assert res.fix is not None, res.output
    assert "a file named 'forged-skills.yaml' does: .mind-data/world/forged-skills.yaml" in res.fix


def test_first_missing_component_and_guess_relative(tmp_path) -> None:
    from zakcode.tools.builtins.bash import _first_missing_component, _guess_relative

    root = _mind_workspace(tmp_path)
    (root / "agents" / "coach").mkdir(parents=True)
    roots = [root]
    # Relative: the first component is what is missing.
    assert _first_missing_component("world/x.yaml", roots) == (root, "world")
    # Absolute, deep: the anchor is the deepest EXISTING ancestor.
    p = root / ".mind-data" / "agents" / "coach" / "sessions" / "abc" / "f"
    assert _first_missing_component(str(p), roots) == (root / ".mind-data", "agents")
    # Only the leaf missing -> None (that is the sibling branch's case).
    assert _first_missing_component("agents/coach/nothing.yaml", roots) is None
    assert _first_missing_component(str(root / "agents" / "coach" / "nothing.yaml"), roots) is None
    # Guess normalisation: absolute-under-root becomes root-relative; `./` is stripped.
    assert _guess_relative(str(root / "world" / "x.yaml"), roots) == "world/x.yaml"
    assert _guess_relative("./world/x.yaml", roots) == "world/x.yaml"
    # Absolute and outside every root: returned as written, in posix form. Built from the
    # anchor because `/elsewhere` is not absolute on Windows (no drive).
    elsewhere = Path(root.anchor) / "elsewhere" / "x.yaml"
    assert _guess_relative(str(elsewhere), roots) == elsewhere.as_posix()


# ── a TOOL typed as a shell command (ADR-0098, 2026-08-29) ────────────────────


def _registry_ctx(tmp_path: Path) -> ToolContext:
    from zakcode.tools.builtins.default_registry import default_registry

    return ToolContext(workspace_root=tmp_path, tool_registry=default_registry())


async def test_bash_refuses_a_tool_written_as_a_shell_call_and_names_the_tool(tmp_path) -> None:
    """The Bodies typed the loop's deadman net — `ScheduleWakeup(prompt=…, delaySeconds=600)`
    — into the bash tool, five times in one session (zc-03, 2026-08-29): a shell syntax
    error and a lost turn each, then no real call at all. Refuse it BEFORE running, with
    the tool's real name and parameters."""
    ctx = _registry_ctx(tmp_path)
    res = await BashTool().execute(
        {"command": "ScheduleWakeup(prompt='<<autonomous-loop-dynamic>>', delaySeconds=600)"}, ctx
    )
    assert res.is_error
    assert res.data is not None and res.data.get("tool_typed_as_command") is True
    assert "[exit code" not in res.output  # nothing ran
    assert "schedule_wakeup" in res.output
    assert "prompt" in res.output and "delaySeconds" in res.output
    assert res.fix is not None and "schedule_wakeup" in res.fix


async def test_bash_still_runs_ordinary_commands_and_shell_functions(tmp_path) -> None:
    ctx = _registry_ctx(tmp_path)
    # A shell function definition has nothing between its parens: not a tool call.
    res = await BashTool().execute({"command": "greet() { echo hi; }; greet"}, ctx)
    assert not res.is_error and "hi" in res.output
    # A tool name mid-command is just text.
    res = await BashTool().execute({"command": "echo 'ScheduleWakeup(prompt=x)'"}, ctx)
    assert not res.is_error and "ScheduleWakeup" in res.output
    # An unknown name in call syntax is left to the shell (and its own error).
    res = await BashTool().execute({"command": "Frobnicate(prompt='x')"}, ctx)
    assert res.is_error and res.data is not None and "tool_typed_as_command" not in res.data


async def test_bash_without_a_registry_skips_the_check(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "ScheduleWakeup(prompt='x', delaySeconds=60)"}, ctx)
    assert res.is_error  # the shell's own syntax error, as before
    assert res.data is not None and "tool_typed_as_command" not in res.data


def test_tool_typed_as_command_predicate() -> None:
    from zakcode.tools.builtins.bash import _tool_typed_as_command
    from zakcode.tools.builtins.default_registry import default_registry

    reg = default_registry()
    assert _tool_typed_as_command("ScheduleWakeup(prompt='x')", reg) is not None
    assert _tool_typed_as_command("  wakeup(prompt='x')", reg) is not None  # alias
    assert _tool_typed_as_command("schedule_wakeup(prompt='x')", reg) is not None
    assert _tool_typed_as_command("ScheduleWakeup()", reg) is None  # no arguments: not a call
    assert _tool_typed_as_command("ls -la", reg) is None
    assert _tool_typed_as_command("ScheduleWakeup(prompt='x')", None) is None


def test_locate_basename_sees_hidden_data_dirs_but_not_vcs_or_caches(tmp_path) -> None:
    from zakcode.tools.builtins.bash import _locate_basename

    root = _mind_workspace(tmp_path)
    assert _locate_basename(root, "forged-skills.yaml") == ".mind-data/world/forged-skills.yaml"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "only-here.sh").write_text("no", encoding="utf-8")
    assert _locate_basename(root, "only-here.sh") is None


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


def test_json_first_line_fix_predicate() -> None:
    """`wrapper.sh | python3 -c` dying on `Expecting value: line 1 column 2 (char 1)` is
    json.loads on a lone `[`/`{` — the upstream printed a pretty-printed document and the
    program read it line by line (measured 2026-08-30, zc-03, two sessions)."""
    from zakcode.tools.builtins.bash import _json_first_line_fix

    err = (
        'Traceback (most recent call last):\n  File "<string>", line 3, in <module>\n'
        "json.decoder.JSONDecodeError: Expecting value: line 1 column 2 (char 1)\n"
    )
    fix = _json_first_line_fix(
        "bash core/scripts/aspirations-query.sh --full 2>&1 | python3 -c 'x'", err
    )
    assert fix is not None and "json.load(sys.stdin)" in fix and "not JSONL" in fix
    heredoc = "cat out.json | python3 - <<'PY'\nimport json\nPY"
    assert _json_first_line_fix(heredoc, err) is not None
    # The line came with its newline (`for line in sys.stdin`): same lone bracket, shifted.
    with_newline = "JSONDecodeError: Expecting value: line 2 column 1 (char 2)"
    assert _json_first_line_fix("bash x.sh | python3 -c 'x'", with_newline) is not None
    # Not a pipe into Python: the parser's input was not another command's output.
    assert (
        _json_first_line_fix("python3 -c 'import json; json.loads(open(\"x\").read())'", err)
        is None
    )
    # A different JSON error position is a different problem.
    assert (
        _json_first_line_fix(
            "bash x.sh | python3 -c 'x'",
            "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        )
        is None
    )


async def test_bash_json_first_line_hint_rides_the_error(tmp_path) -> None:
    """End to end through the tool: the remedy lands in res.fix beside the traceback."""
    ctx = ToolContext(workspace_root=tmp_path)
    cmd = (
        "printf '[\\n  {\"a\": 1}\\n]\\n' | python3 -c "
        "'import json, sys\nfor line in sys.stdin:\n    json.loads(line)'"
    )
    res = await BashTool().execute({"command": cmd}, ctx)
    assert res.is_error
    assert res.fix is not None and "json.load(sys.stdin)" in res.fix


def test_module_not_found_fix_predicate(tmp_path) -> None:
    from zakcode.tools.builtins.bash import _module_not_found_fix as fix

    pkg = tmp_path / ".mind-data" / "world" / "scripts" / "yahoo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "client.py").write_text("X = 1\n", encoding="utf-8")
    err = 'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
    # The measured shape: the package lives under a hidden data dir, cwd is the root.
    hint = fix(err + "ModuleNotFoundError: No module named 'yahoo'", tmp_path, [])
    assert hint is not None
    assert ".mind-data/world/scripts/yahoo" in hint
    assert "cd .mind-data/world/scripts && python3" in hint
    assert "PYTHONPATH=.mind-data/world/scripts" in hint
    # A dotted name whose top package exists but whose submodule does not: name what it holds.
    hint = fix(err + "ModuleNotFoundError: No module named 'yahoo.oauth'", tmp_path, [])
    assert hint is not None and "has no module 'oauth'" in hint and "client" in hint
    # A dotted name whose submodule DOES exist gets the run-from hint, not the listing.
    hint = fix(err + "ModuleNotFoundError: No module named 'yahoo.client'", tmp_path, [])
    assert hint is not None and "cd .mind-data/world/scripts" in hint
    # A single-file module counts too.
    (tmp_path / "tools" / "lib").mkdir(parents=True)
    (tmp_path / "tools" / "lib" / "helpers.py").write_text("", encoding="utf-8")
    hint = fix(err + "ModuleNotFoundError: No module named 'helpers'", tmp_path, [])
    assert hint is not None and "tools/lib/helpers.py" in hint and "cd tools/lib" in hint
    # A directory with no Python in it is not a package; a genuinely absent package is silent.
    (tmp_path / "docs" / "requests").mkdir(parents=True)
    assert fix(err + "ModuleNotFoundError: No module named 'requests'", tmp_path, []) is None
    assert fix(err + "ModuleNotFoundError: No module named 'nothing_here'", tmp_path, []) is None
    assert fix("ImportError: cannot import name 'x' from 'yahoo'", tmp_path, []) is None


@pytest.mark.skipif(shutil.which("python3") is None, reason="needs a python3 on PATH")
async def test_bash_module_not_found_names_the_package_parent(tmp_path) -> None:
    """`cd <root> && python3 -c "import yahoo"` after the package moved under
    .mind-data/world/scripts — five times in 24 h on zc-03 (2026-08-30), one identical
    retry, no hint. The hint names the parent to run from and the PYTHONPATH form."""
    pkg = tmp_path / ".mind-data" / "world" / "scripts" / "yahoo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": 'python3 -c "import yahoo.client"'}, ctx)
    assert res.is_error
    assert res.fix is not None and "cd .mind-data/world/scripts && python3" in res.fix
    # Followed as written, the remedy works.
    res = await BashTool().execute(
        {"command": 'cd .mind-data/world/scripts && python3 -c "import yahoo; print(1)"'}, ctx
    )
    assert not res.is_error


async def test_bash_module_not_found_without_a_workspace_package_is_plain(tmp_path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": 'python3 -c "import surely_not_installed_xyz"'}, ctx)
    assert res.is_error and res.fix is None


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


# ── ADR-0106: a script path that does not exist is refused BEFORE anything runs ──────


def _script_workspace(tmp_path: Path) -> Path:
    (tmp_path / "core" / "scripts").mkdir(parents=True)
    (tmp_path / "core" / "scripts" / "wm-read.sh").write_text(
        "echo wm-read ran\n", encoding="utf-8"
    )
    (tmp_path / "core" / "scripts" / "ok.sh").write_text("echo ok ran\n", encoding="utf-8")
    return tmp_path


async def test_bash_refuses_a_missing_script_path_before_running(tmp_path) -> None:
    """The piped shape that hid the ENOENT from the post-run hint: nothing runs, and the
    refusal carries the sibling lead the hint would have given."""
    ctx = ToolContext(workspace_root=_script_workspace(tmp_path))
    parse = 'python3 -c "import sys, json; json.loads(sys.stdin.read())"'
    cmd = f"bash core/scripts/wm-list.sh --json 2>&1 | {parse}"
    res = await BashTool().execute({"command": cmd}, ctx)
    assert res.is_error
    assert res.data is not None and res.data.get("script_path_missing") is True
    assert "[exit code" not in res.output  # nothing ran
    assert "was not run" in res.output and "wm-read.sh" in res.output
    assert res.fix is not None and "wm-list.sh" in res.fix


async def test_bash_runs_an_existing_script_and_follows_a_literal_cd(tmp_path) -> None:
    ctx = ToolContext(workspace_root=_script_workspace(tmp_path))
    res = await BashTool().execute({"command": "bash core/scripts/ok.sh"}, ctx)
    assert not res.is_error and "ok ran" in res.output
    res = await BashTool().execute({"command": "cd core && bash scripts/ok.sh"}, ctx)
    assert not res.is_error and "ok ran" in res.output
    # a literal cd into a directory where the script is NOT is refused with that base named
    res = await BashTool().execute({"command": "cd core && bash core/scripts/ok.sh"}, ctx)
    assert res.is_error and res.data is not None and res.data.get("script_path_missing") is True
    assert "under" in res.output


async def test_bash_preflight_fails_open_where_it_cannot_resolve(tmp_path) -> None:
    ctx = ToolContext(workspace_root=_script_workspace(tmp_path))
    # a $VAR path is not checked: the shell's own error, as before
    res = await BashTool().execute({"command": 'X=/nowhere; bash "$X/core/scripts/nope.sh"'}, ctx)
    assert res.is_error and res.data is not None and "script_path_missing" not in res.data
    # a cd to an unexpandable target: not checked
    res = await BashTool().execute({"command": 'cd "$HOME" && bash scripts/nope.sh'}, ctx)
    assert res.is_error and res.data is not None and "script_path_missing" not in res.data
    # a heredoc BODY mentioning a script is text, not an invocation
    res = await BashTool().execute(
        {"command": "python3 - <<'PY'\nprint('bash core/scripts/nope.sh')\nPY"}, ctx
    )
    assert not res.is_error and "nope.sh" in res.output


def test_script_path_missing_predicate(tmp_path) -> None:
    from zakcode.tools.builtins.bash import _script_path_missing

    root = _script_workspace(tmp_path)
    assert _script_path_missing("bash core/scripts/ok.sh", root, []) is None
    assert _script_path_missing("python3 -m pytest tests -q", root, []) is None
    assert _script_path_missing("bash -c 'echo core/scripts/nope.sh'", root, []) is None
    # written earlier in the same command: not checked
    assert (
        _script_path_missing("echo hi > core/scripts/new.sh && bash core/scripts/new.sh", root, [])
        is None
    )
    # an extra workspace root that holds the script satisfies the check
    other = tmp_path / "other"
    (other / "world" / "scripts").mkdir(parents=True)
    (other / "world" / "scripts" / "efs-ssh.sh").write_text("", encoding="utf-8")
    assert _script_path_missing("bash world/scripts/efs-ssh.sh 'echo ok'", root, [other]) is None
    missing = _script_path_missing("python3 core/scripts/aspirations-read-goal.sh g-1-1", root, [])
    assert (
        missing is not None and "aspirations-read-goal.sh" in missing and "was not run" in missing
    )
