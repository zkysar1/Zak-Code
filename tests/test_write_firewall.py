"""Tests for the deterministic write firewall (Slice 0 of the Recipe Cursor design).

The firewall refuses, before any bytes land, content that is plainly a shell command
the model expected to be substituted (the exact 3B corruption seen live: ``$(cat
other.py)`` written as a file body), and ``.py`` content that does not compile.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins._safety import check_literal_content, check_python_syntax
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.write_file import WriteFileTool

# ── pure helpers ──────────────────────────────────────────────────────────────


def test_literal_content_rejects_whole_command_substitution() -> None:
    assert check_literal_content("$(cat temp_fizzbuzz.py)") is not None
    assert check_literal_content("  `ls -la`  ") is not None


def test_literal_content_allows_real_content() -> None:
    assert check_literal_content("print('cost is $(x)')") is None  # contains, not whole
    assert check_literal_content("x = 1\ny = 2\n") is None
    assert check_literal_content("") is None
    assert check_literal_content("cat sat on the mat") is None  # prose starting with 'cat'


def test_python_syntax_rejects_invalid_py() -> None:
    assert check_python_syntax("a.py", "def f(:\n    pass") is not None
    assert check_python_syntax("a.py", "$(cat x.py)") is not None  # also not valid python


def test_python_syntax_allows_valid_and_nonpy() -> None:
    assert check_python_syntax("a.py", "def f():\n    return 1\n") is None
    assert check_python_syntax("notes.txt", "def f(:") is None  # only .py is checked
    assert check_python_syntax("a.py", "") is None  # empty allowed


# ── through the tools ─────────────────────────────────────────────────────────


async def test_write_file_refuses_command_substitution(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute(
        {"path": "fizzbuzz.py", "content": "$(cat temp_fizzbuzz.py)"}, ctx
    )
    assert res.is_error
    assert "literally" in res.output.lower()
    assert not (tmp_path / "fizzbuzz.py").exists()  # no corrupt bytes landed


async def test_write_file_refuses_invalid_python(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute({"path": "broken.py", "content": "def f(:\n"}, ctx)
    assert res.is_error
    assert not (tmp_path / "broken.py").exists()


async def test_write_file_allows_valid_python(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    res = await WriteFileTool().execute({"path": "ok.py", "content": "print('ok')\n"}, ctx)
    assert not res.is_error, res.output
    assert (tmp_path / "ok.py").read_text() == "print('ok')\n"


async def test_edit_refuses_change_that_breaks_python(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)
    (tmp_path / "m.py").write_text("x = 1\n")
    res = await EditFileTool().execute(
        {"path": "m.py", "old_string": "x = 1", "new_string": "x = ("}, ctx
    )
    assert res.is_error
    assert (tmp_path / "m.py").read_text() == "x = 1\n"  # original preserved on a bad edit
