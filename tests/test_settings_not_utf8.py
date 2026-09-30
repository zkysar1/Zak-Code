"""A settings file that is not UTF-8 is reported like broken JSON, never raised.

Every reader of the workspace settings files already reported broken JSON instead of raising.
A file saved in another encoding (a shell redirect or an editor writing UTF-16, a stray byte)
raised ``UnicodeDecodeError``, which none of the five readers caught, so the agent could not
start in that workspace.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.config import Settings
from zakcode.hooks.settings_loader import load_settings_hooks
from zakcode.output_styles import _read_output_style_name
from zakcode.permissions_settings import load_settings_permissions
from zakcode.status_line import load_status_line_spec
from zakcode.workspace_env import _read_blocks

#: Valid JSON with one byte that is not UTF-8 after it.
NOT_UTF8 = b'{"hooks": {}}\xff'

#: Each reader, and whether it reported the file instead of raising. The readers report in
#: three shapes, a dict of errors, a message, or None, and str() reads all three.
READERS: dict[str, Callable[[Path], bool]] = {
    "hooks": lambda ws: "parse error" in str(load_settings_hooks(ws)[1]),
    "permissions": lambda ws: "parse error" in str(load_settings_permissions(ws)[1]),
    "env": lambda ws: _read_blocks(ws)[2],
    "status-line": lambda ws: "parse error" in str(load_status_line_spec(ws)[1]),
    "output-style": lambda ws: "parse error" in str(_read_output_style_name(ws)[1]),
}


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_bytes(NOT_UTF8)
    return tmp_path


@pytest.mark.parametrize("reported", READERS.values(), ids=READERS.keys())
def test_each_reader_reports_the_file_instead_of_raising(
    tmp_path: Path, reported: Callable[[Path], bool]
) -> None:
    assert reported(_workspace(tmp_path))


def test_the_agent_starts_in_that_workspace(tmp_path: Path) -> None:
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=_workspace(tmp_path)
        )
    )
    assert agent.hook_manager.shell_hooks == []
