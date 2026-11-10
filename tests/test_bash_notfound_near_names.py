"""The shell's not-found answers name what the model meant (ADR-0378).

Measured 2026-11-10 on ten worker sessions over a week: about 150 commands named a script or
command that does not exist. The cases pinned here are the four shapes that got a wrong lead
or none: a name spelled almost like the real one, a real script by its bare name that the
bounded walk never reached, a skill's name typed into the shell, and a host-framework
virtual prefix that only the file tools rewrite.
"""

from __future__ import annotations

from pathlib import Path

from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.bash import (
    _FIND_MAX_DIRS,
    BashTool,
    _near_siblings,
    _posix_exit_fix,
)


def _scripts(root: Path, *names: str) -> Path:
    scripts = root / "tools" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in names:
        (scripts / name).write_text("#!/usr/bin/env bash\necho ran\n", encoding="utf-8")
    return scripts


async def test_a_name_spelled_almost_the_same_leads_the_hint(tmp_path) -> None:
    """`bash tools/scripts/retrieval.sh` for `retrieve.sh`: the first-word family used to be the
    whole answer (other scripts, "use one of those"), and the real one was never named."""
    _scripts(tmp_path, "retrieve.sh", "retrieval-gate.sh", "retrieval-pulse.sh", "retrieval_x.py")
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "bash tools/scripts/retrieval.sh --category x"}, ctx)
    assert res.is_error
    assert res.fix is not None
    assert "spelled almost the same" in res.fix
    assert res.fix.index("tools/scripts/retrieve.sh") < res.fix.index("retrieval-gate.sh")
    # the family that shares the first word (0.83 and 0.80 alike) stays an aside, not the lead
    lead = res.fix.split("spelled almost the same does:", 1)[1].split("—", 1)[0]
    assert "retrieve.sh" in lead and "retrieval-gate.sh" not in lead


async def test_the_spelling_lead_also_answers_a_command_that_ran(tmp_path) -> None:
    """Not only the preflight: a `cat` of the misspelled path fails at run time and the post-run
    hint carries the same lead."""
    _scripts(tmp_path, "retrieve.sh", "retrieval-gate.sh")
    ctx = ToolContext(workspace_root=tmp_path)
    res = await BashTool().execute({"command": "cat tools/scripts/retrieval.sh"}, ctx)
    assert res.is_error
    assert res.fix is not None and "tools/scripts/retrieve.sh" in res.fix


def test_a_different_word_is_not_a_misspelling(tmp_path) -> None:
    """`api-start.sh` is spelled within 0.88 of `api-status.sh` and does something else; the
    extension alone may differ (the script in another language is the script)."""
    scripts = _scripts(tmp_path, "api-start.sh", "net-ssh.sh", "repo-freshness.py", "state-save.sh")
    assert _near_siblings(scripts, "api-status.sh") == []
    assert _near_siblings(scripts, "state-read.sh") == []
    assert _near_siblings(scripts, "net-sh.sh") == ["net-ssh.sh"]
    assert _near_siblings(scripts, "repo-freshness.sh") == ["repo-freshness.py"]


def test_a_bare_script_name_is_found_past_a_wide_hidden_tree(tmp_path) -> None:
    """A host-framework checkout holds wide trees under dot-dirs that sort before the scripts
    directory: the depth-first walk spent its whole visit budget there and never listed it, so
    a real script run by its bare name (exit 127) got no hint 5 times out of 5."""
    _scripts(tmp_path, "query-goals.sh")
    hidden = tmp_path / ".hidden" / "skills"
    side = int(_FIND_MAX_DIRS**0.5) + 2  # side * side directories two levels below .hidden/skills
    for i in range(side):
        for j in range(side):
            (hidden / f"s{i:03d}" / f"p{j:03d}").mkdir(parents=True)
    fix = _posix_exit_fix(
        "query-goals.sh --id g-1",
        "bash: line 1: query-goals.sh: command not found",
        127,
        tmp_path,
    )
    assert fix is not None and "tools/scripts/query-goals.sh" in fix


class _Skills:
    def names(self) -> list[str]:
        return ["verify-unit", "work-loop"]

    def body(self, name: str) -> str | None:
        return None

    async def load(self, name: str, *, query: str = "", args: str = "") -> object:
        raise NotImplementedError  # pragma: no cover — not exercised by this path


async def test_a_skill_name_typed_into_the_shell_is_named_as_a_skill(tmp_path) -> None:
    """A skill's name as a command: one of them 10 times on six sessions in a week, each a bare
    "command not found". The tool registry already answers a skill called as a tool."""
    ctx = ToolContext(workspace_root=tmp_path, skill_resolver=_Skills())
    res = await BashTool().execute({"command": "verify-unit g-1"}, ctx)
    assert res.is_error
    assert res.data is not None and res.data["exit_code"] == 127
    assert res.fix is not None and 'Skill(skill="verify-unit")' in res.fix
    plain = await BashTool().execute({"command": "definitely-not-a-command-zz"}, ctx)
    assert plain.is_error
    assert "Skill(" not in (plain.fix or "")


async def test_a_virtual_prefix_in_a_command_names_its_real_directory(tmp_path) -> None:
    """`bash world/scripts/net-ssh.sh` from the workspace: the file tools rewrite `world/`, the
    shell does not, and the world directory can live outside the workspace. Measured: one
    session ran a script that way 80 times, 5 of them with a lead."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    world = tmp_path / "elsewhere" / "world"
    (world / "scripts").mkdir(parents=True)
    (world / "scripts" / "net-ssh.sh").write_text("echo ran\n", encoding="utf-8")
    real = (world / "scripts" / "net-ssh.sh").as_posix()
    ctx = ToolContext(workspace_root=workspace, virtual_roots={"world": world})
    res = await BashTool().execute({"command": "bash world/scripts/net-ssh.sh 'echo ok'"}, ctx)
    assert res.is_error
    assert res.data is not None and res.data.get("script_path_missing") is True
    assert res.fix is not None and "virtual prefix" in res.fix and real in res.fix
    typo = await BashTool().execute({"command": "bash world/scripts/net-sh.sh x"}, ctx)
    assert typo.fix is not None and "spelled almost the same" in typo.fix and real in typo.fix
    # without a declared directory for the prefix there is nothing to say about it
    bare = ToolContext(workspace_root=workspace)
    res = await BashTool().execute({"command": "bash world/scripts/net-ssh.sh 'echo ok'"}, bare)
    assert res.is_error and "virtual prefix" not in (res.fix or "")


async def test_a_virtual_prefix_in_a_command_that_ran_names_its_real_directory(tmp_path) -> None:
    """Not only a script run: `cat world/notes.txt` runs, fails with "No such file", and the
    post-run hint names the directory the prefix stands for."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    world = tmp_path / "elsewhere" / "world"
    world.mkdir(parents=True)
    (world / "notes.txt").write_text("hello\n", encoding="utf-8")
    ctx = ToolContext(workspace_root=workspace, virtual_roots={"world": world})
    res = await BashTool().execute({"command": "cat world/notes.txt"}, ctx)
    assert res.is_error
    assert res.fix is not None and "virtual prefix" in res.fix
    assert (world / "notes.txt").as_posix() in res.fix
