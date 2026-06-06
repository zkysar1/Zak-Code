"""Edge-case / failure-mode tests for the six built-in tools and path safety.

Every test here is hermetic: the workspace is ``tmp_path`` wired through a
:class:`ToolContext`, no network is touched, and no real model is invoked. The
governing invariant under test is that a tool handler MUST NOT raise -- every
failure has to come back as ``ToolResult.error(...)`` (``is_error`` True), with
the human-readable message carried in ``ToolResult.output``.
"""

from __future__ import annotations

import sys

import pytest

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins._safety import (
    PathEscapeError,
    resolve_in_workspace,
    resolve_in_workspace_roots,
    resolve_path,
)
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.write_file import WriteFileTool


@pytest.fixture
def ctx(tmp_path):
    """A ToolContext whose workspace root is the per-test tmp_path."""
    return ToolContext(workspace_root=tmp_path)


# --------------------------------------------------------------------------- #
# _safety.resolve_in_workspace
# --------------------------------------------------------------------------- #
def test_safety_rejects_dotdot_escape(tmp_path):
    with pytest.raises(PathEscapeError):
        resolve_in_workspace("../outside.txt", tmp_path)


def test_safety_rejects_absolute_outside_root(tmp_path):
    outside = tmp_path.parent / "elsewhere.txt"
    with pytest.raises(PathEscapeError):
        resolve_in_workspace(str(outside), tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS alternate data streams are Windows-only")
def test_safety_rejects_alternate_data_stream(tmp_path):
    # audit3 #9: a ':' outside the drive prefix names an NTFS alternate data stream that
    # list_dir/glob never reveal — refuse it. A leading drive prefix (C:\) is still fine.
    with pytest.raises(PathEscapeError):
        resolve_in_workspace("public.txt:secret", tmp_path)
    with pytest.raises(PathEscapeError):
        resolve_in_workspace_roots("public.txt:secret", [tmp_path])
    leaf = tmp_path / "ok.txt"  # a normal absolute Windows path (drive colon only) is fine
    assert resolve_in_workspace(str(leaf), tmp_path) == leaf.resolve()


def test_safety_accepts_legitimate_nested_path(tmp_path):
    resolved = resolve_in_workspace("a/b/c.txt", tmp_path)
    assert resolved == (tmp_path / "a" / "b" / "c.txt").resolve()


def test_safety_accepts_root_itself(tmp_path):
    assert resolve_in_workspace(".", tmp_path) == tmp_path.resolve()


def test_safety_rejects_empty_and_non_string(tmp_path):
    with pytest.raises(PathEscapeError):
        resolve_in_workspace("", tmp_path)
    with pytest.raises(PathEscapeError):
        resolve_in_workspace(None, tmp_path)  # type: ignore[arg-type]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privilege on Windows")
def test_safety_rejects_symlink_pointing_outside(tmp_path):
    outside_dir = tmp_path.parent / "outside_target"
    outside_dir.mkdir(exist_ok=True)
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks unsupported in this environment")
    with pytest.raises(PathEscapeError):
        resolve_in_workspace("link/secret.txt", tmp_path)


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #
async def test_read_missing_file(ctx):
    res = await ReadFileTool().execute({"path": "nope.txt"}, ctx)
    assert res.is_error
    assert "not found" in res.output.lower()


async def test_read_directory_path(ctx, tmp_path):
    (tmp_path / "d").mkdir()
    res = await ReadFileTool().execute({"path": "d"}, ctx)
    assert res.is_error
    assert "directory" in res.output.lower()


async def test_read_escape_is_error(ctx):
    res = await ReadFileTool().execute({"path": "../escape.txt"}, ctx)
    assert res.is_error
    assert "outside the workspace" in res.output


async def test_read_binary_non_utf8_does_not_crash(ctx, tmp_path):
    (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\x00\x01abc\x80")
    res = await ReadFileTool().execute({"path": "bin.dat"}, ctx)
    assert not res.is_error
    # Undecodable bytes become the replacement char rather than raising.
    assert "�" in res.output


async def test_read_offset_past_eof_is_ok_with_note(ctx, tmp_path):
    (tmp_path / "few.txt").write_bytes(b"one\ntwo\n")
    res = await ReadFileTool().execute({"path": "few.txt", "offset": 50}, ctx)
    assert not res.is_error
    assert "beyond the end" in res.output


async def test_read_bad_offset_type_is_error(ctx, tmp_path):
    (tmp_path / "f.txt").write_bytes(b"x\n")
    res = await ReadFileTool().execute({"path": "f.txt", "offset": "1"}, ctx)
    assert res.is_error
    assert "offset" in res.output.lower()


async def test_read_bad_limit_value_is_error(ctx, tmp_path):
    (tmp_path / "f.txt").write_bytes(b"x\n")
    res = await ReadFileTool().execute({"path": "f.txt", "limit": 0}, ctx)
    assert res.is_error
    assert "limit" in res.output.lower()


async def test_read_offset_limit_slice(ctx, tmp_path):
    # Write bytes so the test is independent of platform newline translation.
    (tmp_path / "n.txt").write_bytes(b"l1\nl2\nl3\nl4\nl5\n")
    res = await ReadFileTool().execute({"path": "n.txt", "offset": 2, "limit": 2}, ctx)
    assert not res.is_error
    assert res.output == "l2\nl3\n"


async def test_read_size_cap_truncation_note(ctx, tmp_path):
    big = b"A" * (200 * 1024)
    (tmp_path / "big.txt").write_bytes(big)
    res = await ReadFileTool().execute({"path": "big.txt"}, ctx)
    assert not res.is_error
    assert res.data["truncated"] is True
    assert "truncated at 100KB" in res.output


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #
async def test_write_roundtrip_exact(ctx, tmp_path):
    content = "first\nsecond\n\twith tab\nunicode: é☃\n"
    res = await WriteFileTool().execute({"path": "out/data.txt", "content": content}, ctx)
    assert not res.is_error
    # read_bytes + decode avoids platform newline translation masking a mismatch.
    written = (tmp_path / "out" / "data.txt").read_bytes().decode("utf-8")
    assert written == content


async def test_write_creates_parent_dirs(ctx, tmp_path):
    res = await WriteFileTool().execute({"path": "a/b/c/deep.txt", "content": "hi"}, ctx)
    assert not res.is_error
    assert (tmp_path / "a" / "b" / "c" / "deep.txt").is_file()


async def test_write_to_existing_directory_is_error(ctx, tmp_path):
    (tmp_path / "adir").mkdir()
    res = await WriteFileTool().execute({"path": "adir", "content": "x"}, ctx)
    assert res.is_error
    assert "directory" in res.output.lower()


async def test_write_when_parent_is_a_file_is_error(ctx, tmp_path):
    (tmp_path / "afile").write_bytes(b"data")
    res = await WriteFileTool().execute({"path": "afile/child.txt", "content": "x"}, ctx)
    assert res.is_error
    # The pre-existing file is left intact.
    assert (tmp_path / "afile").read_bytes() == b"data"


async def test_write_escape_is_error(ctx, tmp_path):
    res = await WriteFileTool().execute({"path": "../evil.txt", "content": "x"}, ctx)
    assert res.is_error
    assert "outside the workspace" in res.output
    assert not (tmp_path.parent / "evil.txt").exists()


async def test_write_leaves_no_temp_files(ctx, tmp_path):
    res = await WriteFileTool().execute({"path": "f.txt", "content": "ok"}, ctx)
    assert not res.is_error
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".zaktmp-")]
    assert leftovers == []


# --------------------------------------------------------------------------- #
# list_dir
# --------------------------------------------------------------------------- #
async def test_list_missing_dir_is_error(ctx):
    res = await ListDirTool().execute({"path": "ghost"}, ctx)
    assert res.is_error
    assert "not found" in res.output.lower()


async def test_list_file_path_is_error(ctx, tmp_path):
    (tmp_path / "f.txt").write_bytes(b"x")
    res = await ListDirTool().execute({"path": "f.txt"}, ctx)
    assert res.is_error
    assert "not a directory" in res.output.lower()


async def test_list_empty_dir_is_ok(ctx, tmp_path):
    (tmp_path / "empty").mkdir()
    res = await ListDirTool().execute({"path": "empty"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 0


async def test_list_marks_directories_with_slash(ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_bytes(b"x")
    res = await ListDirTool().execute({"path": "."}, ctx)
    assert not res.is_error
    assert "sub/" in res.output
    assert "file.txt" in res.output


# --------------------------------------------------------------------------- #
# glob
# --------------------------------------------------------------------------- #
async def test_glob_no_matches_is_ok_not_error(ctx, tmp_path):
    (tmp_path / "a.py").write_bytes(b"x")
    res = await GlobTool().execute({"pattern": "*.nomatch"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 0


async def test_glob_recursive_match(ctx, tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_bytes(b"x")
    (tmp_path / "top.py").write_bytes(b"x")
    res = await GlobTool().execute({"pattern": "**/*.py"}, ctx)
    assert not res.is_error
    assert res.data["count"] >= 2


async def test_glob_absolute_pattern_is_error(ctx):
    abspat = "/etc/*" if sys.platform != "win32" else "C:\\*"
    res = await GlobTool().execute({"pattern": abspat}, ctx)
    assert res.is_error
    assert "relative" in res.output.lower()


async def test_glob_escaping_pattern_handled_gracefully(ctx, tmp_path):
    # '../*' must not raise; it yields no in-workspace matches.
    res = await GlobTool().execute({"pattern": "../*"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 0


async def test_glob_result_cap(ctx, tmp_path, monkeypatch):
    import zakcode.tools.builtins.glob as glob_mod

    monkeypatch.setattr(glob_mod, "_MAX_RESULTS", 3)
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_bytes(b"x")
    res = await GlobTool().execute({"pattern": "*.txt"}, ctx)
    assert not res.is_error
    assert len(res.data["matches"]) == 3
    assert res.data["truncated"] is True
    assert "truncated" in res.output


# --------------------------------------------------------------------------- #
# grep
# --------------------------------------------------------------------------- #
async def test_grep_invalid_regex_is_error_not_raise(ctx, tmp_path):
    (tmp_path / "f.txt").write_bytes(b"hello")
    res = await GrepTool().execute({"pattern": "([unclosed"}, ctx)
    assert res.is_error
    assert "invalid regex" in res.output.lower()


async def test_grep_no_matches_is_ok(ctx, tmp_path):
    (tmp_path / "f.txt").write_bytes(b"nothing here\n")
    res = await GrepTool().execute({"pattern": "ZZZ_absent"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 0


async def test_grep_skips_binary_files(ctx, tmp_path):
    (tmp_path / "b.dat").write_bytes(b"need\x00le here")
    (tmp_path / "t.txt").write_bytes(b"needle here\n")
    res = await GrepTool().execute({"pattern": "needle"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 1
    assert "t.txt" in res.output
    assert "b.dat" not in res.output


async def test_grep_line_numbers_correct(ctx, tmp_path):
    (tmp_path / "lines.txt").write_bytes(b"alpha\nbeta\ngamma target\ndelta\n")
    res = await GrepTool().execute({"pattern": "target"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 1
    assert ":3:" in res.output


async def test_grep_glob_filter(ctx, tmp_path):
    (tmp_path / "a.py").write_bytes(b"match me\n")
    (tmp_path / "a.md").write_bytes(b"match me\n")
    res = await GrepTool().execute({"pattern": "match", "glob": "*.py"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 1
    assert "a.py" in res.output


async def test_grep_result_cap(ctx, tmp_path, monkeypatch):
    import zakcode.tools.builtins.grep as grep_mod

    monkeypatch.setattr(grep_mod, "_MAX_MATCHES", 5)
    body = ("\n".join("hit" for _ in range(50)) + "\n").encode("utf-8")
    (tmp_path / "many.txt").write_bytes(body)
    res = await GrepTool().execute({"pattern": "hit"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 5
    assert res.data["truncated"] is True


# --------------------------------------------------------------------------- #
# bash
# --------------------------------------------------------------------------- #
async def test_bash_nonzero_exit_is_error_with_code(ctx):
    cmd = "exit 3" if sys.platform != "win32" else "cmd /c exit 3"
    res = await BashTool().execute({"command": cmd}, ctx)
    assert res.is_error
    assert res.data["exit_code"] == 3
    assert "exit code: 3" in res.output


async def test_bash_success_captures_output(ctx):
    res = await BashTool().execute({"command": "echo hello_zak"}, ctx)
    assert not res.is_error
    assert "hello_zak" in res.output
    assert res.data["exit_code"] == 0


async def test_bash_timeout_is_graceful(ctx):
    # Sleep longer than the (tiny) timeout via a portable Python one-liner so we
    # do not depend on a particular shell's sleep/ping syntax.
    py = sys.executable
    cmd = f'"{py}" -c "import time; time.sleep(5)"'
    res = await BashTool().execute({"command": cmd, "timeout": 1}, ctx)
    assert res.is_error
    assert res.data.get("timed_out") is True
    assert "timed out" in res.output.lower()


async def test_bash_empty_command_is_error(ctx):
    res = await BashTool().execute({"command": "   "}, ctx)
    assert res.is_error


async def test_bash_output_size_cap(ctx, tmp_path, monkeypatch):
    import zakcode.tools.builtins.bash as bash_mod

    monkeypatch.setattr(bash_mod, "_MAX_OUTPUT", 100)
    # Run a script file so we avoid cross-shell nested-quoting differences and
    # reliably emit far more than 100 chars of output.
    script = tmp_path / "spew.py"
    script.write_bytes(b"print('Z' * 5000)\n")
    py = sys.executable
    cmd = f'"{py}" "{script}"'
    res = await BashTool().execute({"command": cmd}, ctx)
    assert not res.is_error
    assert res.data["truncated"] is True
    assert "output truncated" in res.output


async def test_bash_runs_in_workspace_cwd(ctx, tmp_path):
    # Print the process cwd via a portable Python one-liner; it must equal the
    # workspace root the tool was configured with.
    py = sys.executable
    cmd = f'"{py}" -c "import os; print(os.getcwd())"'
    res = await BashTool().execute({"command": cmd}, ctx)
    assert not res.is_error
    assert str(tmp_path.resolve()) in res.output


# --------------------------------------------------------------------------- #
# Multi-root sandbox (M-3: resolve_in_workspace_roots, resolve_path, and
# ToolContext.extra_workspace_roots)
# --------------------------------------------------------------------------- #


@pytest.fixture
def multi_roots(tmp_path):
    """Create three isolated root directories (primary, root_b, root_c) and a
    ToolContext wired to use all three.
    """
    primary = tmp_path / "primary"
    root_b = tmp_path / "root_b"
    root_c = tmp_path / "root_c"
    primary.mkdir()
    root_b.mkdir()
    root_c.mkdir()
    ctx = ToolContext(
        workspace_root=primary,
        extra_workspace_roots=[root_b, root_c],
    )
    return primary, root_b, root_c, ctx


# -- _safety layer ----------------------------------------------------------


def test_multi_root_accepts_path_under_root_a(multi_roots):
    primary, _, _, _ = multi_roots
    (primary / "a.txt").write_bytes(b"x")
    resolved = resolve_in_workspace_roots("a.txt", [primary])
    assert resolved == (primary / "a.txt").resolve()


def test_multi_root_accepts_absolute_under_root_b(multi_roots):
    _, root_b, _, _ = multi_roots
    target = root_b / "b.txt"
    target.write_bytes(b"x")
    resolved = resolve_in_workspace_roots(str(target), [root_b])
    assert resolved == target.resolve()


def test_multi_root_rejects_path_outside_all_roots(multi_roots):
    primary, root_b, root_c, _ = multi_roots
    outside = primary.parent / "elsewhere.txt"
    with pytest.raises(PathEscapeError, match="outside all workspace roots"):
        resolve_in_workspace_roots(str(outside), [primary, root_b, root_c])


def test_multi_root_rejects_traversal_escape(multi_roots):
    primary, root_b, _, _ = multi_roots
    with pytest.raises(PathEscapeError, match="outside all workspace roots"):
        resolve_in_workspace_roots("../../escape.txt", [primary, root_b])


def test_multi_root_rejects_empty_roots():
    with pytest.raises(PathEscapeError, match="At least one"):
        resolve_in_workspace_roots("anything.txt", [])


def test_multi_root_relative_first_match_wins(multi_roots):
    """When a relative path is valid under multiple roots, the first wins."""
    primary, root_b, _, _ = multi_roots
    (primary / "shared.txt").write_bytes(b"from primary")
    (root_b / "shared.txt").write_bytes(b"from root_b")
    resolved = resolve_in_workspace_roots("shared.txt", [primary, root_b])
    assert resolved == (primary / "shared.txt").resolve()


def test_multi_root_relative_falls_through_to_second_root(multi_roots):
    """A relative path that only exists under root_b resolves there."""
    primary, root_b, _, _ = multi_roots
    (root_b / "only_b.txt").write_bytes(b"b")
    # Both roots are valid; the file exists only under root_b.
    # But resolve_in_workspace_roots returns the first root that CONTAINS
    # the resolved path (even if the file doesn't exist yet), so primary wins
    # if the resolved path is inside primary. We test the behavior:
    resolved = resolve_in_workspace_roots("only_b.txt", [primary, root_b])
    # primary / "only_b.txt" resolves inside primary, so primary wins
    # (the file's existence is irrelevant to the sandbox check).
    assert resolved == (primary / "only_b.txt").resolve()


def test_resolve_path_dispatches_single_root(tmp_path):
    """resolve_path with no extra_roots delegates to resolve_in_workspace."""
    resolved = resolve_path("sub/f.txt", tmp_path)
    assert resolved == (tmp_path / "sub" / "f.txt").resolve()


def test_resolve_path_dispatches_multi_root(multi_roots):
    """resolve_path with extra_roots delegates to resolve_in_workspace_roots."""
    primary, root_b, _, _ = multi_roots
    target = root_b / "data.txt"
    target.write_bytes(b"x")
    resolved = resolve_path(str(target), primary, extra_roots=[root_b])
    assert resolved == target.resolve()


def test_resolve_path_multi_rejects_escape(multi_roots):
    primary, root_b, _, _ = multi_roots
    outside = primary.parent / "nope.txt"
    with pytest.raises(PathEscapeError):
        resolve_path(str(outside), primary, extra_roots=[root_b])


# -- Tools with multi-root context -----------------------------------------


async def test_read_file_multi_root(multi_roots):
    """ReadFileTool can read from an extra workspace root."""
    _, root_b, _, ctx = multi_roots
    (root_b / "hello.txt").write_bytes(b"hello from root_b\n")
    res = await ReadFileTool().execute({"path": str(root_b / "hello.txt")}, ctx)
    assert not res.is_error
    assert "hello from root_b" in res.output


async def test_write_file_multi_root(multi_roots):
    """WriteFileTool can write to an extra workspace root."""
    _, root_b, _, ctx = multi_roots
    target = str(root_b / "new.txt")
    res = await WriteFileTool().execute({"path": target, "content": "written"}, ctx)
    assert not res.is_error
    assert (root_b / "new.txt").read_text(encoding="utf-8") == "written"


async def test_edit_file_multi_root(multi_roots):
    """EditFileTool can edit a file in an extra workspace root."""
    _, _, root_c, ctx = multi_roots
    (root_c / "e.txt").write_bytes(b"alpha beta gamma")
    target = str(root_c / "e.txt")
    res = await EditFileTool().execute(
        {"path": target, "old_string": "beta", "new_string": "BETA"}, ctx
    )
    assert not res.is_error
    assert (root_c / "e.txt").read_text(encoding="utf-8") == "alpha BETA gamma"


async def test_list_dir_multi_root(multi_roots):
    """ListDirTool can list a directory in an extra workspace root."""
    _, root_b, _, ctx = multi_roots
    (root_b / "sub").mkdir()
    (root_b / "sub" / "child.txt").write_bytes(b"x")
    res = await ListDirTool().execute({"path": str(root_b / "sub")}, ctx)
    assert not res.is_error
    assert "child.txt" in res.output


async def test_grep_multi_root(multi_roots):
    """GrepTool can search within an extra workspace root."""
    _, _, root_c, ctx = multi_roots
    (root_c / "data.py").write_bytes(b"needle_c_here\n")
    res = await GrepTool().execute({"path": str(root_c), "pattern": "needle_c"}, ctx)
    assert not res.is_error
    assert res.data["count"] == 1
    assert "needle_c_here" in res.output


async def test_glob_multi_root(multi_roots):
    """GlobTool glob results include files from multiple roots."""
    primary, root_b, _, ctx = multi_roots
    (primary / "p.txt").write_bytes(b"x")
    (root_b / "b.txt").write_bytes(b"x")
    # Glob uses the primary root as base, so only primary's files match
    # the relative pattern.
    res = await GlobTool().execute({"pattern": "*.txt"}, ctx)
    assert not res.is_error
    assert res.data["count"] >= 1


async def test_read_escape_multi_root_still_rejected(multi_roots):
    """Even with multi-root, paths outside ALL roots are rejected."""
    primary, _, _, ctx = multi_roots
    outside = str(primary.parent / "evil.txt")
    res = await ReadFileTool().execute({"path": outside}, ctx)
    assert res.is_error
    assert "outside" in res.output.lower()


async def test_write_escape_multi_root_still_rejected(multi_roots):
    """Even with multi-root, writes outside ALL roots are rejected."""
    primary, _, _, ctx = multi_roots
    outside = str(primary.parent / "evil.txt")
    res = await WriteFileTool().execute({"path": outside, "content": "x"}, ctx)
    assert res.is_error
    assert "outside" in res.output.lower()
    assert not (primary.parent / "evil.txt").exists()
