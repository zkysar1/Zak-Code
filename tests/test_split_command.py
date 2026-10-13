"""A settings.json hook / statusLine command is split into argv the way its author wrote it.

``split_command`` is what turns the command string of a hook or status line into the argv
that is spawned WITHOUT a shell. On POSIX it is ``shlex.split``. On Windows it used to be
``shlex.split(cmd, posix=False)``, which keeps the quote characters in every token: a quoted
path with a space (``"C:\\Program Files\\Git\\bin\\bash.exe"``, or a project directory with a
space substituted into ``"$CLAUDE_PROJECT_DIR/x.sh"``) reached the child with literal quotes
in it and never ran, and ``--dir="C:\\My Dir"`` was cut in two at the space.

``$CLAUDE_PROJECT_DIR`` is filled in after the split, argument by argument. Filling it into the
string first cut ``bash $CLAUDE_PROJECT_DIR/x.sh`` at the space of ``C:/Users/me/My Project``, so
the child was handed ``C:/Users/me/My`` and never found its script.

The Windows cases run on every platform by pointing ``sys.platform`` at ``win32``: the
splitter is pure string work, so what it returns does not depend on the host. Each case
names one kind of token, so a failure says which kind broke.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from zakcode._subprocess import split_command
from zakcode.hooks import HookEvent
from zakcode.hooks.settings_loader import load_settings_hooks
from zakcode.status_line import load_status_line_spec


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")


# ── POSIX: unchanged ─────────────────────────────────────────────────────────


def test_posix_is_plain_shlex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert split_command("bash \"a b/c.sh\" --x='1 2' d\\ e") == [
        "bash",
        "a b/c.sh",
        "--x=1 2",
        "d e",  # a backslash escapes on POSIX, as it always did
    ]


# ── Windows: the quote characters are dropped, the backslashes are kept ──────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(
            '"C:\\Program Files\\Git\\bin\\bash.exe" "C:\\My Dir\\hook.sh" --flag',
            ["C:\\Program Files\\Git\\bin\\bash.exe", "C:\\My Dir\\hook.sh", "--flag"],
            id="quoted-paths-with-spaces",
        ),
        pytest.param(
            'python hook.py --dir="C:\\My Dir"',
            ["python", "hook.py", "--dir=C:\\My Dir"],
            id="quote-inside-a-word",
        ),
        pytest.param(
            'bash "C:\\My Dir"/.claude/hooks/x.sh',
            ["bash", "C:\\My Dir/.claude/hooks/x.sh"],
            id="quoted-project-dir-then-a-suffix",
        ),
        pytest.param(
            "C:\\Users\\me\\run.cmd arg",
            ["C:\\Users\\me\\run.cmd", "arg"],
            id="unquoted-backslashes-survive",
        ),
        pytest.param(
            'tool "C:\\dir\\" next',
            ["tool", "C:\\dir\\", "next"],
            id="quoted-path-ending-in-a-backslash",
        ),
        pytest.param(
            "python -c 'import os; print(1)'",
            ["python", "-c", "import os; print(1)"],
            id="single-quotes-group-too",
        ),
        pytest.param("echo a#b", ["echo", "a#b"], id="hash-is-an-ordinary-character"),
        pytest.param("", [], id="empty"),
    ],
)
def test_windows_split(windows: None, command: str, expected: list[str]) -> None:
    assert split_command(command) == expected


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_unclosed_quote_raises_on_every_platform(
    monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    # The two callers turn this ValueError into a "command parse error" entry, never a crash.
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ValueError):
        split_command('bash "unclosed')


# ── $CLAUDE_PROJECT_DIR: one argument, however the author wrote the variable ─


_DIR = "C:/Users/me/My Project"


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param("bash $CLAUDE_PROJECT_DIR/x.sh", ["bash", f"{_DIR}/x.sh"], id="unquoted"),
        pytest.param("bash ${CLAUDE_PROJECT_DIR}/x.sh", ["bash", f"{_DIR}/x.sh"], id="braced"),
        pytest.param('bash "$CLAUDE_PROJECT_DIR/x.sh"', ["bash", f"{_DIR}/x.sh"], id="quoted"),
        pytest.param(
            "$CLAUDE_PROJECT_DIR/bin/run.cmd go",
            [f"{_DIR}/bin/run.cmd", "go"],
            id="the-program-itself",
        ),
        pytest.param(
            "tool --root=$CLAUDE_PROJECT_DIR next",
            ["tool", f"--root={_DIR}", "next"],
            id="inside-a-word",
        ),
        pytest.param(
            "py $CLAUDE_PROJECT_DIR/a.py ${CLAUDE_PROJECT_DIR}/b",
            ["py", f"{_DIR}/a.py", f"{_DIR}/b"],
            id="twice",
        ),
    ],
)
def test_project_dir_with_a_space_is_one_argument(
    monkeypatch: pytest.MonkeyPatch, platform: str, command: str, expected: list[str]
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert split_command(command, project_dir=_DIR) == expected


def test_project_dir_is_left_alone_when_none_is_given() -> None:
    assert split_command("bash $CLAUDE_PROJECT_DIR/x.sh") == ["bash", "$CLAUDE_PROJECT_DIR/x.sh"]


# ── the two callers: a project directory with a space in it ──────────────────


def _project_with_space(tmp_path: Path) -> Path:
    ws = tmp_path / "my project"
    (ws / ".claude").mkdir(parents=True)
    return ws


_VARIABLE_FORMS = [
    pytest.param('"$CLAUDE_PROJECT_DIR/{rest}"', id="quoted"),
    pytest.param("$CLAUDE_PROJECT_DIR/{rest}", id="unquoted"),
    pytest.param("${{CLAUDE_PROJECT_DIR}}/{rest}", id="braced"),
]


@pytest.mark.parametrize("form", _VARIABLE_FORMS)
def test_hook_in_a_project_dir_with_a_space(windows: None, tmp_path: Path, form: str) -> None:
    ws = _project_with_space(tmp_path)
    command = "python " + form.format(rest=".claude/hooks/x.py") + " --mode=fast"
    (ws / ".claude" / "settings.json").write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": command}]}]}}),
        encoding="utf-8",
    )
    specs, errors = load_settings_hooks(ws)
    assert errors == {}
    assert len(specs) == 1 and specs[0].event is HookEvent.TURN_END
    # The directory arrives as ONE argument, with its space and without any quote character.
    assert specs[0].command[1:] == [f"{ws}/.claude/hooks/x.py", "--mode=fast"]
    assert not any('"' in part for part in specs[0].command)


def test_hook_whose_program_is_in_a_project_dir_with_a_space(windows: None, tmp_path: Path) -> None:
    ws = _project_with_space(tmp_path)
    command = "$CLAUDE_PROJECT_DIR/.claude/hooks/x.cmd --mode=fast"
    (ws / ".claude" / "settings.json").write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": command}]}]}}),
        encoding="utf-8",
    )
    specs, errors = load_settings_hooks(ws)
    assert errors == {}
    assert [s.command for s in specs] == [[f"{ws}/.claude/hooks/x.cmd", "--mode=fast"]]


@pytest.mark.parametrize("form", _VARIABLE_FORMS)
def test_status_line_in_a_project_dir_with_a_space(
    windows: None, tmp_path: Path, form: str
) -> None:
    ws = _project_with_space(tmp_path)
    command = "python " + form.format(rest="s.py")
    (ws / ".claude" / "settings.json").write_text(
        json.dumps({"statusLine": {"type": "command", "command": command}}),
        encoding="utf-8",
    )
    spec, error = load_status_line_spec(ws)
    assert error is None
    assert spec is not None
    assert spec.command[1:] == [f"{ws}/s.py"]
