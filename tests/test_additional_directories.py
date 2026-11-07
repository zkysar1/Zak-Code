"""``permissions.additionalDirectories`` extends the file tools' reach (ADR-0375).

Claude Code names the directories a session may reach beyond its own in this array. Zak Code reads
it from the three settings files it already reads for hooks and joins it with ``--extra-root`` and
``Agent(extra_workspace_roots=...)`` into one list. These tests pin three things: what the loader
makes of each value, that the grant reaches every file tool (and every delegated child), and that it
is reach and not authority: the protected-path floor, an ingested ``deny`` rule and the ``..``
escape check bind inside an added directory as they do in the workspace.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.additional_directories import load_additional_directories
from zakcode.config import Settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply


def _declare(ws: Path, entries: object, *, file: str = ".claude/settings.json") -> None:
    """Write a settings file that names ``entries`` under ``permissions.additionalDirectories``."""
    path = ws / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"permissions": {"additionalDirectories": entries}}), encoding="utf-8"
    )


def _dir(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return _dir(tmp_path, "workspace")


# ── the loader: what each value in the array means ───────────────────────────────


def test_an_absolute_directory_is_granted_as_written(ws: Path, tmp_path: Path) -> None:
    ext = _dir(tmp_path, "external")
    _declare(ws, [str(ext)])
    assert load_additional_directories(ws) == ([ext.resolve()], {})


def test_a_relative_path_resolves_from_the_workspace_root(ws: Path, tmp_path: Path) -> None:
    sibling = _dir(tmp_path, "sibling")
    _declare(ws, ["../sibling", "docs"])
    _dir(ws, "docs")
    roots, errors = load_additional_directories(ws)
    assert roots == [sibling.resolve(), (ws / "docs").resolve()]
    assert errors == {}


def test_a_leading_tilde_expands_to_the_home_directory(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _dir(tmp_path, "home")
    shared = _dir(home, "shared")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    _declare(ws, ["~/shared"])
    assert load_additional_directories(ws)[0] == [shared.resolve()]


def test_the_three_settings_files_join_in_read_order_and_a_repeat_counts_once(
    ws: Path, tmp_path: Path
) -> None:
    a, b, c, d = (_dir(tmp_path, name) for name in "abcd")
    _declare(ws, [str(a), str(b)], file=".claude/settings.json")
    _declare(ws, [str(b), str(c)], file=".claude/settings.local.json")
    _declare(ws, [str(c), str(d)], file=".zakcode/settings.json")
    assert load_additional_directories(ws)[0] == [p.resolve() for p in (a, b, c, d)]


def test_a_directory_that_does_not_exist_grants_nothing(ws: Path, tmp_path: Path) -> None:
    (tmp_path / "a-file").write_text("not a directory", encoding="utf-8")
    _declare(ws, [str(tmp_path / "missing"), str(tmp_path / "a-file")])
    assert load_additional_directories(ws) == ([], {})


def test_a_malformed_value_is_reported_and_the_rest_still_applies(ws: Path, tmp_path: Path) -> None:
    ext = _dir(tmp_path, "external")
    _declare(ws, [7, "  ", str(ext)])
    roots, errors = load_additional_directories(ws)
    assert roots == [ext.resolve()]
    assert errors == {
        "settings.json:permissions.additionalDirectories[0]": "not a path string; skipped",
        "settings.json:permissions.additionalDirectories[1]": "not a path string; skipped",
    }

    _declare(ws, "not-a-list", file=".zakcode/settings.json")
    _, errors = load_additional_directories(ws)
    assert errors["settings.json:permissions.additionalDirectories"] == (
        "expected a list of paths, got str; skipped"
    )


def test_a_file_that_does_not_parse_is_skipped_without_an_error_of_its_own(
    ws: Path, tmp_path: Path
) -> None:
    # the hook and permission loaders report an unreadable file; a third report would be noise
    ext = _dir(tmp_path, "external")
    (ws / ".claude").mkdir()
    (ws / ".claude" / "settings.json").write_text("{not json", encoding="utf-8")
    _declare(ws, [str(ext)], file=".zakcode/settings.json")
    assert load_additional_directories(ws) == ([ext.resolve()], {})


@pytest.mark.parametrize(
    "document",
    [{}, {"permissions": {}}, {"permissions": {"allow": ["Bash"]}}, {"permissions": []}, []],
    ids=["empty", "no-keys", "other-keys", "permissions-not-an-object", "not-an-object"],
)
def test_a_file_without_the_key_declares_nothing(ws: Path, document: object) -> None:
    (ws / ".claude").mkdir()
    (ws / ".claude" / "settings.json").write_text(json.dumps(document), encoding="utf-8")
    assert load_additional_directories(ws) == ([], {})


def test_a_workspace_without_settings_declares_nothing(ws: Path) -> None:
    assert load_additional_directories(ws) == ([], {})


# ── the Agent hands the list to everything that judges a path ────────────────────


def _agent(ws: Path, script: list, *, mode: str = "allow", **kwargs: object) -> Agent:
    return Agent(
        settings=Settings(
            default_model="scripted/additional-directories",
            context_window=8192,
            workspace_root=ws,
            permission_mode=mode,
        ),
        provider=ScriptedProvider(script),
        max_iterations=8,
        **kwargs,  # type: ignore[arg-type]
    )


def _results(agent: Agent) -> list[tuple[bool, str]]:
    """``(is_error, output)`` of every tool result the turn produced, in call order."""
    return [
        (bool(block.is_error), str(block.output))
        for message in agent.session.messages
        for block in message.blocks
        if hasattr(block, "is_error")
    ]


async def test_a_declared_directory_opens_to_read_write_and_edit(ws: Path, tmp_path: Path) -> None:
    ext = _dir(tmp_path, "external")
    (ext / "note.md").write_text("one\n", encoding="utf-8")
    _declare(ws, [str(ext)])
    agent = _agent(
        ws,
        [
            call_tool("Read", {"path": str(ext / "note.md")}, id="r"),
            call_tool("Write", {"path": str(ext / "new.md"), "content": "fresh\n"}, id="w"),
            call_tool(
                "Edit",
                {"path": str(ext / "note.md"), "old_string": "one", "new_string": "two"},
                id="e",
            ),
            reply("done"),
        ],
    )
    await agent.arun_turn("go")
    results = _results(agent)
    assert [is_error for is_error, _ in results] == [False, False, False], results
    assert (ext / "new.md").read_bytes() == b"fresh\n"
    assert (ext / "note.md").read_bytes() == b"two\n"


async def test_without_the_declaration_the_same_read_is_refused(ws: Path, tmp_path: Path) -> None:
    ext = _dir(tmp_path, "external")
    (ext / "note.md").write_text("one\n", encoding="utf-8")
    agent = _agent(ws, [call_tool("Read", {"path": str(ext / "note.md")}, id="r"), reply("done")])
    await agent.arun_turn("go")
    (is_error, output), *_ = _results(agent)
    assert is_error and "outside the workspace root" in output


def test_the_list_joins_the_roots_given_in_code(ws: Path, tmp_path: Path) -> None:
    given, declared = _dir(tmp_path, "given"), _dir(tmp_path, "declared")
    _declare(ws, [str(declared)])
    agent = _agent(ws, [reply("hi")], extra_workspace_roots=[given])
    assert agent.loop.extra_workspace_roots == [given, declared.resolve()]


def test_a_workspace_that_declares_nothing_keeps_a_single_root(ws: Path) -> None:
    assert _agent(ws, [reply("hi")]).loop.extra_workspace_roots == []


def test_a_delegated_child_gets_the_same_roots(ws: Path, tmp_path: Path) -> None:
    ext = _dir(tmp_path, "external")
    _declare(ws, [str(ext)])
    agent = _agent(ws, [reply("hi")], enable_subagents=True)
    assert agent.loop.spawner is not None
    assert agent.loop.spawner._runner.extra_workspace_roots == [ext.resolve()]


def test_a_malformed_value_is_logged_once_with_its_position(
    ws: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _declare(ws, [3])
    with caplog.at_level(logging.WARNING, logger="zakcode"):
        _agent(ws, [reply("hi")])
    lines = [r.getMessage() for r in caplog.records if "additionalDirectories" in r.getMessage()]
    assert lines == [
        "settings.json:permissions.additionalDirectories[0]: not a path string; skipped"
    ]


# ── reach, not authority ─────────────────────────────────────────────────────────


async def test_the_escape_check_still_binds_inside_an_added_directory(
    ws: Path, tmp_path: Path
) -> None:
    ext = _dir(tmp_path, "external")
    (tmp_path / "outside.md").write_text("beyond every root\n", encoding="utf-8")
    _declare(ws, [str(ext)])
    agent = _agent(
        ws, [call_tool("Read", {"path": str(ext / ".." / "outside.md")}, id="r"), reply("done")]
    )
    await agent.arun_turn("go")
    (is_error, output), *_ = _results(agent)
    assert is_error and "resolves outside all workspace roots" in output


async def test_the_protected_path_floor_binds_inside_an_added_directory(
    ws: Path, tmp_path: Path
) -> None:
    ext = _dir(tmp_path, "external")
    (ext / ".env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    _declare(ws, [str(ext)])
    agent = _agent(ws, [call_tool("Read", {"path": str(ext / ".env")}, id="r"), reply("done")])
    await agent.arun_turn("go")
    (is_error, output), *_ = _results(agent)
    assert is_error and "hunter2" not in output


async def test_an_ingested_deny_rule_binds_inside_an_added_directory(
    ws: Path, tmp_path: Path
) -> None:
    ext = _dir(tmp_path, "external")
    _dir(ext, "locked")
    _dir(ext, "open")
    (ws / ".claude").mkdir()
    (ws / ".claude" / "settings.json").write_text(
        json.dumps(
            {
                "permissions": {
                    "additionalDirectories": [str(ext)],
                    "deny": ["Edit(**/locked/**)"],
                }
            }
        ),
        encoding="utf-8",
    )
    agent = _agent(
        ws,
        [
            call_tool("Write", {"path": str(ext / "locked" / "a.md"), "content": "x\n"}, id="l"),
            call_tool("Write", {"path": str(ext / "open" / "a.md"), "content": "x\n"}, id="o"),
            reply("done"),
        ],
    )
    await agent.arun_turn("go")
    (locked_error, _), (open_error, _), *_ = _results(agent)
    assert locked_error and not open_error
    assert not (ext / "locked" / "a.md").exists()
    assert (ext / "open" / "a.md").read_bytes() == b"x\n"


async def test_the_permission_mode_decides_a_write_there_as_it_does_in_the_workspace(
    ws: Path, tmp_path: Path
) -> None:
    ext = _dir(tmp_path, "external")
    _declare(ws, [str(ext)])
    write = call_tool("Write", {"path": str(ext / "a.md"), "content": "x\n"}, id="w")

    accepting = _agent(ws, [write, reply("done")], mode="acceptEdits")
    await accepting.arun_turn("go")
    assert [is_error for is_error, _ in _results(accepting)] == [False]
    assert (ext / "a.md").read_bytes() == b"x\n"

    (ext / "a.md").unlink()
    asking = _agent(ws, [write, reply("done")], mode="ask")
    await asking.arun_turn("go")
    (is_error, output), *_ = _results(asking)
    assert is_error and "Permission denied for 'Write'" in output
    assert not (ext / "a.md").exists()
