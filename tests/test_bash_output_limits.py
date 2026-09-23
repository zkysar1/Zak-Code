"""A command's output reaches the model within Claude Code's limits (ADR-0234).

A success past 30,000 characters is saved whole to the session's output directory and the
result is its start and the path. A failure past 10,000 shows its start and end, and the path.
Read may open that path: the one place outside the workspace roots it may.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from zakcode.background import BackgroundTasks
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.read_file import ReadFileTool


def _spew(tmp_path: Path, lines: int, exit_code: int = 0) -> str:
    """A command printing ``lines`` numbered lines of 40 characters, then exiting ``exit_code``.
    A script file keeps the quoting the same under bash and the cmd.exe fallback."""
    script = tmp_path / f"spew_{lines}_{exit_code}.py"
    script.write_text(
        "import sys\n"
        f"for i in range({lines}):\n"
        "    print(f'line {i:06d} ' + 'x' * 28)\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return f'"{sys.executable}" "{script}"'


@pytest.fixture
def session_ctx(tmp_path: Path) -> ToolContext:
    """A workspace, and a session whose output directory is outside it, as on a CLI box."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = SimpleNamespace(id="s1", background_tasks=[])
    tasks = BackgroundTasks(session, tasks_dir=tmp_path / "home" / "tasks" / "s1")
    return ToolContext(workspace_root=workspace, background_tasks=tasks)


async def test_a_long_success_is_saved_whole_and_its_start_shown(
    session_ctx: ToolContext, tmp_path: Path
) -> None:
    res = await BashTool().execute({"command": _spew(tmp_path, 1000)}, session_ctx)

    assert not res.is_error
    saved = Path(res.data["output_file"])
    # Bytes, not read_text: on Windows the output's line ends are "\r\n", and text mode would
    # read them back as "\n" and count a thousand characters fewer than were saved.
    whole = saved.read_bytes().decode("utf-8")
    assert whole.count("\n") == 1000 and whole.startswith("line 000000 ")
    assert res.data["output_chars"] == len(whole) and res.data["truncated"] is True
    shown = res.output
    preview = shown.split("[... the output is")[0]
    assert preview.startswith("line 000000 ")
    assert preview.rstrip("\r\n").endswith("x" * 28)  # cut back to a line end
    assert "line 000999 " not in shown  # the end is in the file, not here
    assert len(shown) < 2_500
    assert str(saved) in shown and f"{len(whole):,} characters in 1,000 lines" in shown
    assert shown.endswith("[exit code: 0]")


async def test_read_opens_the_saved_output_and_nothing_else_outside_the_roots(
    session_ctx: ToolContext, tmp_path: Path
) -> None:
    res = await BashTool().execute({"command": _spew(tmp_path, 1000)}, session_ctx)
    saved = res.data["output_file"]

    page = await ReadFileTool().execute({"path": saved, "offset": 990, "limit": 5}, session_ctx)
    assert not page.is_error
    assert "line 000989 " in page.output and "line 000993 " in page.output
    assert "line 000994 " not in page.output

    # Positive control: one directory up from the session's own output directory is still
    # outside every root, and so is the saved file for a context with no session.
    elsewhere = Path(saved).parent.parent / "elsewhere.txt"
    elsewhere.write_text("not yours\n", encoding="utf-8")
    refused = await ReadFileTool().execute({"path": str(elsewhere)}, session_ctx)
    assert refused.is_error and "outside" in refused.output
    bare = ToolContext(workspace_root=session_ctx.workspace_root)
    assert (await ReadFileTool().execute({"path": saved}, bare)).is_error


async def test_a_long_failure_shows_its_start_and_end(
    session_ctx: ToolContext, tmp_path: Path
) -> None:
    res = await BashTool().execute({"command": _spew(tmp_path, 1000, exit_code=3)}, session_ctx)

    assert res.is_error
    shown = res.output
    assert "line 000000 " in shown and "line 000999 " in shown  # both ends
    assert "line 000500 " not in shown  # the middle is in the file
    assert len(shown) < 10_600
    assert Path(res.data["output_file"]).read_text(encoding="utf-8").count("\n") == 1000
    assert shown.endswith("[exit code: 3]")


async def test_an_output_within_its_limit_is_shown_whole(
    session_ctx: ToolContext, tmp_path: Path
) -> None:
    ok = await BashTool().execute({"command": _spew(tmp_path, 500)}, session_ctx)
    failed = await BashTool().execute({"command": _spew(tmp_path, 200, exit_code=1)}, session_ctx)

    assert not ok.is_error and ok.data["truncated"] is False and "output_file" not in ok.data
    assert "line 000000 " in ok.output and "line 000499 " in ok.output
    assert failed.is_error and failed.data["truncated"] is False
    assert "line 000000 " in failed.output and "line 000199 " in failed.output
    assert session_ctx.background_tasks is not None
    assert not list(session_ctx.background_tasks.directory.glob("output-*"))


async def test_with_no_session_a_long_success_keeps_its_start_and_end(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path)

    res = await BashTool().execute({"command": _spew(tmp_path, 1000)}, ctx)

    assert not res.is_error and "output_file" not in res.data
    assert "line 000000 " in res.output and "line 000999 " in res.output
    assert len(res.output) < 30_600
    assert "Run the command again, narrower" in res.output
