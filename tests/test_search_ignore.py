"""Tests for search-ignore hygiene (grep / glob / list_dir skip build/vendor/.gitignore).

Two layers: the matcher (_ignore.py) — where the SAFETY property is "never over-ignore"
(fail-open) — and the three search tools honoring it with an include_ignored escape hatch.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins._ignore import load_ignore
from zakcode.tools.builtins.glob import GlobTool
from zakcode.tools.builtins.grep import GrepTool
from zakcode.tools.builtins.list_dir import ListDirTool


def _spec(tmp_path: Path, *, gitignore: str | None = None, zakcodeignore: str | None = None):
    if gitignore is not None:
        (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
    if zakcodeignore is not None:
        (tmp_path / ".zakcodeignore").write_text(zakcodeignore, encoding="utf-8")
    return load_ignore(tmp_path)


# ── matcher ──────────────────────────────────────────────────────────────────────


def test_default_dirs_ignored(tmp_path: Path) -> None:
    spec = load_ignore(tmp_path)
    assert spec.match("node_modules/dep.js", is_dir=False)
    assert spec.match("a/__pycache__/x.pyc", is_dir=False)
    assert spec.match("node_modules", is_dir=True)
    assert not spec.match("src/app.py", is_dir=False)  # real source untouched


def test_git_is_hard_floor_even_with_soft_off(tmp_path: Path) -> None:
    spec = load_ignore(tmp_path)
    assert spec.match(".git/config", is_dir=False, soft=False)
    assert spec.match(".git", is_dir=True, soft=False)
    # soft=False (include_ignored) drops the default-dir + gitignore layers, keeps only .git.
    assert not spec.match("node_modules/x", is_dir=False, soft=False)


def test_gitignore_dir_pattern(tmp_path: Path) -> None:
    spec = _spec(tmp_path, gitignore="build/\n")
    assert spec.match("build", is_dir=True)
    assert spec.match("build/out.js", is_dir=False)
    assert spec.match("a/build/out.js", is_dir=False)  # any depth
    assert not spec.match("build", is_dir=False)  # a FILE named build is NOT matched by build/


def test_gitignore_basename_glob_any_depth(tmp_path: Path) -> None:
    spec = _spec(tmp_path, gitignore="*.log\n")
    assert spec.match("x.log", is_dir=False)
    assert spec.match("a/b/x.log", is_dir=False)
    assert not spec.match("x.txt", is_dir=False)


def test_gitignore_anchored_pattern(tmp_path: Path) -> None:
    spec = _spec(tmp_path, gitignore="/secret.txt\n")
    assert spec.match("secret.txt", is_dir=False)
    assert not spec.match("a/secret.txt", is_dir=False)  # anchored to root only


def test_gitignore_negation_reincludes(tmp_path: Path) -> None:
    spec = _spec(tmp_path, gitignore="*.log\n!keep.log\n")
    assert spec.match("a.log", is_dir=False)
    assert not spec.match("keep.log", is_dir=False)  # un-ignored by the later !rule


def test_fail_open_on_uncompilable_pattern(tmp_path: Path) -> None:
    # An unterminated char class can't be compiled -> the rule is dropped (fail-open: never
    # hide a file on a pattern we don't understand).
    spec = _spec(tmp_path, gitignore="[unterminated\n")
    assert not spec.match("[unterminated", is_dir=False)


def test_zakcodeignore_is_read(tmp_path: Path) -> None:
    spec = _spec(tmp_path, zakcodeignore="*.tmp\n")
    assert spec.match("x.tmp", is_dir=False)


def test_name_pattern_is_not_a_substring_match(tmp_path: Path) -> None:
    # The classic over-ignore trap: a name pattern must match a whole path component, NEVER a
    # substring. `dist` ignores dir `dist` (+ contents) but must leave `distribution.py` alone.
    spec = _spec(tmp_path, gitignore="dist\n")
    assert spec.match("dist/bundle.js", is_dir=False)
    assert spec.match("a/dist/bundle.js", is_dir=False)
    assert not spec.match("distribution.py", is_dir=False)
    assert not spec.match("redistribute.py", is_dir=False)
    assert not spec.match("a/distro.txt", is_dir=False)


def test_anchored_pattern_does_not_match_nested(tmp_path: Path) -> None:
    spec = _spec(tmp_path, gitignore="/build\n")
    assert spec.match("build/out.o", is_dir=False)
    assert not spec.match("src/build/out.o", is_dir=False)  # anchored: only root build/


def test_comments_and_blanks_skipped(tmp_path: Path) -> None:
    spec = _spec(tmp_path, gitignore="# a comment\n\n*.bak\n")
    assert spec.match("x.bak", is_dir=False)
    assert not spec.match("a comment", is_dir=False)


# ── integration: grep / glob / list_dir ────────────────────────────────────────────


def _ws(tmp_path: Path) -> ToolContext:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("NEEDLE here\n", encoding="utf-8")
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "d.py").write_text("NEEDLE in dep\n", encoding="utf-8")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.py").write_text("NEEDLE in build\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    return ToolContext(workspace_root=tmp_path)


async def test_grep_skips_default_and_gitignored(tmp_path: Path) -> None:
    res = await GrepTool().execute({"pattern": "NEEDLE", "path": "."}, _ws(tmp_path))
    assert "app.py" in res.output  # real source found
    assert "node_modules" not in res.output  # default-ignored dir
    assert "out.py" not in res.output  # .gitignore build/


async def test_grep_include_ignored_searches_them(tmp_path: Path) -> None:
    res = await GrepTool().execute(
        {"pattern": "NEEDLE", "path": ".", "include_ignored": True}, _ws(tmp_path)
    )
    assert "app.py" in res.output and "d.py" in res.output and "out.py" in res.output


async def test_grep_never_searches_git_even_with_include_ignored(tmp_path: Path) -> None:
    ctx = _ws(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("NEEDLE secret\n", encoding="utf-8")
    res = await GrepTool().execute({"pattern": "NEEDLE", "path": ".", "include_ignored": True}, ctx)
    assert ".git" not in res.output  # hard floor, always


async def test_glob_skips_then_includes_ignored(tmp_path: Path) -> None:
    ctx = _ws(tmp_path)
    res = await GlobTool().execute({"pattern": "**/*.py"}, ctx)
    assert "app.py" in res.output
    assert "node_modules" not in res.output and "out.py" not in res.output
    res2 = await GlobTool().execute({"pattern": "**/*.py", "include_ignored": True}, ctx)
    assert "d.py" in res2.output and "out.py" in res2.output


async def test_list_dir_hides_ignored_with_explicit_marker(tmp_path: Path) -> None:
    ctx = _ws(tmp_path)
    res = await ListDirTool().execute({"path": "."}, ctx)
    assert "src/" in res.output
    assert "node_modules/" not in res.output and "build/" not in res.output  # hidden
    assert "ignored entries hidden" in res.output  # but NOT silently — the count is surfaced
    assert res.data and res.data.get("ignored") >= 2
    res2 = await ListDirTool().execute({"path": ".", "include_ignored": True}, ctx)
    assert "node_modules/" in res2.output and "build/" in res2.output
