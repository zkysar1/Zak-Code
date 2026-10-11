"""Read, Edit and Write resolve the host framework's ``world/`` and ``meta/`` prefixes (ADR-0282).

Field transcript 2026-10-10 (a served Mind, five minutes into its first turn): the framework's
``retrieve.sh`` answered with ``world/knowledge/tree/…``, the spelling its own hook resolves, and
the first batch of four parallel Reads all came back ``File not found``. The served Mind's
``WORLD_PATH`` is ``<workspace>/knowledge``, so the file was at
``<workspace>/knowledge/knowledge/tree/…``. The layout below is that one, written by the same
provisioner step (``agents/<id>/local-paths.conf``, unquoted absolute values).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from zakcode import Agent, _mind_virtual_roots
from zakcode.agent.budget import IterationBudget
from zakcode.agent.loop import TurnResult
from zakcode.config import PermissionTier, Settings
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    compile_protected_paths,
)
from zakcode.tools.base import ConcurrencyClass, ToolContext, ToolSpec
from zakcode.tools.builtins._safety import PathEscapeError, resolve_path
from zakcode.tools.builtins.default_registry import default_registry
from zakcode.tools.builtins.edit import EditFileTool
from zakcode.tools.builtins.read_file import ReadFileTool
from zakcode.tools.builtins.write_file import WriteFileTool
from zakcode.virtual_paths import expand_virtual_prefix

_NODE = "knowledge/tree/intelligence/installed-packs/pack-a/n03-scores-to-probabilities.md"
_NODE_BODY = "# Scores to probabilities\nsoftmax turns scores into probabilities\n"


@pytest.fixture
def served(tmp_path: Path) -> Path:
    """A served Mind's workspace, laid out as the provisioner leaves it."""
    ws = tmp_path / "mind-workspace"
    (ws / "agents" / "char-1").mkdir(parents=True)
    node = ws / "knowledge" / _NODE
    node.parent.mkdir(parents=True)
    node.write_text(_NODE_BODY, encoding="utf-8")
    (ws / "knowledge" / "program.md").write_text("# The Program\n", encoding="utf-8")
    (ws / "meta").mkdir()
    (ws / "meta" / "reflection-strategy.yaml").write_text("depth: 1\n", encoding="utf-8")
    (ws / "notes.md").write_text("workspace note\n", encoding="utf-8")
    (ws / "agents" / "char-1" / "local-paths.conf").write_text(
        f"WORLD_PATH={ws}/knowledge\nMETA_PATH={ws}/meta\n", encoding="utf-8"
    )
    return ws


@pytest.fixture
def hosted(tmp_path: Path) -> Path:
    """A framework repo whose world and meta live OUTSIDE it (the ADR-0229 / ADR-0230 shape).

    The served layout cannot tell a ``meta/`` rewrite from no rewrite: its ``META_PATH`` is
    ``<workspace>/meta``, the very directory a plain relative ``meta/…`` already reaches. Here
    neither prefix names a directory that also exists in the workspace.
    """
    repo = tmp_path / "Workspace"
    (repo / "agents" / "assistant").mkdir(parents=True)
    world, meta = tmp_path / "Agents-World", tmp_path / "Agents-Meta"
    world.mkdir()
    meta.mkdir()
    (meta / "reflection-strategy.yaml").write_text("depth: 1\n", encoding="utf-8")
    (repo / "agents" / "assistant" / "local-paths.conf").write_text(
        f"WORLD_PATH={world}\nMETA_PATH={meta}\n", encoding="utf-8"
    )
    return repo


def _ctx(ws: Path, *, declared: bool = True) -> ToolContext:
    """The tool context the loop builds for ``ws``: roots as the Agent derives them."""
    roots = _mind_virtual_roots(ws) if declared else {}
    return ToolContext(
        workspace_root=ws, extra_workspace_roots=list(roots.values()), virtual_roots=roots
    )


# ── the conf: what each prefix stands for ───────────────────────────────────────


def test_each_prefix_stands_for_the_directory_the_conf_declares(served: Path) -> None:
    assert _mind_virtual_roots(served) == {
        "world": (served / "knowledge").resolve(),
        "meta": (served / "meta").resolve(),
    }


def test_quoted_conf_values_declare_the_prefixes_too(served: Path) -> None:
    conf = served / "agents" / "char-1" / "local-paths.conf"
    conf.write_text(f"WORLD_PATH=\"{served}/knowledge\"\nMETA_PATH='{served}/meta'\n")
    assert set(_mind_virtual_roots(served)) == {"world", "meta"}


def test_a_prefix_whose_directory_is_missing_is_not_declared(served: Path) -> None:
    conf = served / "agents" / "char-1" / "local-paths.conf"
    conf.write_text(f"WORLD_PATH={served}/knowledge\nMETA_PATH={served}/no-such-dir\n")
    assert set(_mind_virtual_roots(served)) == {"world"}


def test_a_workspace_without_a_conf_declares_nothing(tmp_path: Path) -> None:
    assert _mind_virtual_roots(tmp_path) == {}


def test_agents_that_agree_declare_the_prefix_and_agents_that_differ_do_not(
    served: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other_meta = tmp_path / "other-meta"
    other_meta.mkdir()
    second = served / "agents" / "char-2"
    second.mkdir()
    (second / "local-paths.conf").write_text(
        f"WORLD_PATH={served}/knowledge\nMETA_PATH={other_meta}\n"
    )
    with caplog.at_level(logging.WARNING, logger="zakcode"):
        roots = _mind_virtual_roots(served)
    # Two agents, one world: it is unambiguous. Two metas: a write to meta/x would land in
    # whichever one was picked, so neither is.
    assert roots == {"world": (served / "knowledge").resolve()}
    assert any(
        "meta/ is declared as 2 different directories" in r.getMessage() for r in caplog.records
    )


# ── the rewrite: exactly the two prefixes, nothing else ─────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("world/knowledge/tree/x.md", "/W/knowledge/tree/x.md"),
        ("meta/reflection-strategy.yaml", "/M/reflection-strategy.yaml"),
        ("world//x.md", "/W/x.md"),  # a doubled slash must not become an absolute /x.md
        ("world/", "/W/"),
        ("world", "world"),  # the framework's contract needs the slash
        ("worldly/x.md", "worldly/x.md"),
        ("./world/x.md", "./world/x.md"),
        ("src/world/x.md", "src/world/x.md"),
        ("/srv/world/x.md", "/srv/world/x.md"),
        ("World/x.md", "World/x.md"),
        ("notes.md", "notes.md"),
        ("", ""),
    ],
)
def test_expand_virtual_prefix_rewrites_only_the_two_prefixes(path: str, expected: str) -> None:
    roots = {"world": Path("/W"), "meta": Path("/M")}
    assert expand_virtual_prefix(path, roots).replace("\\", "/") == expected


def test_a_prefix_nobody_declared_is_left_alone() -> None:
    assert expand_virtual_prefix("meta/x.md", {"world": Path("/W")}) == "meta/x.md"
    assert expand_virtual_prefix("world/x.md", {}) == "world/x.md"


# ── the resolver: rewrite first, then confine like any path ─────────────────────


def test_resolve_path_rewrites_and_still_confines(served: Path) -> None:
    ctx = _ctx(served)
    got = resolve_path("world/program.md", served, ctx.extra_workspace_roots, ctx.virtual_roots)
    assert got == (served / "knowledge" / "program.md").resolve()
    # Leaving the declared directory is judged exactly as an absolute path would be: the
    # parent of the workspace belongs to no root.
    with pytest.raises(PathEscapeError):
        resolve_path("world/../../outside.md", served, ctx.extra_workspace_roots, ctx.virtual_roots)


def test_resolve_path_without_virtual_roots_is_the_old_behaviour(served: Path) -> None:
    ctx = _ctx(served)
    assert (
        resolve_path("world/program.md", served, ctx.extra_workspace_roots)
        == (served / "world" / "program.md").resolve()
    )


# ── the three tools ─────────────────────────────────────────────────────────────


async def test_read_opens_the_node_retrieve_pointed_at_on_the_first_try(served: Path) -> None:
    result = await ReadFileTool().execute({"path": f"world/{_NODE}"}, _ctx(served))
    assert not result.is_error, result.output
    assert result.output == _NODE_BODY
    assert result.data is not None
    assert Path(result.data["path"]) == (served / "knowledge" / _NODE).resolve()


async def test_the_same_read_without_the_fix_is_the_field_failure(served: Path) -> None:
    """Positive control: on this layout, with no virtual roots, the call is the u4 one."""
    result = await ReadFileTool().execute({"path": f"world/{_NODE}"}, _ctx(served, declared=False))
    assert result.is_error
    assert result.output.startswith(f"File not found: world/{_NODE}")


async def test_read_passes_a_workspace_path_through_untouched(served: Path) -> None:
    ctx = _ctx(served)
    relative = await ReadFileTool().execute({"path": "notes.md"}, ctx)
    absolute = await ReadFileTool().execute({"path": str(served / "notes.md")}, ctx)
    assert (relative.output, absolute.output) == ("workspace note\n", "workspace note\n")


async def test_read_resolves_meta_where_the_conf_puts_it(hosted: Path) -> None:
    result = await ReadFileTool().execute({"path": "meta/reflection-strategy.yaml"}, _ctx(hosted))
    assert not result.is_error, result.output
    assert result.output == "depth: 1\n"


async def test_edit_rewrites_the_file_under_the_declared_meta(hosted: Path) -> None:
    meta = _mind_virtual_roots(hosted)["meta"]
    result = await EditFileTool().execute(
        {
            "path": "meta/reflection-strategy.yaml",
            "old_string": "depth: 1",
            "new_string": "depth: 2",
        },
        _ctx(hosted),
    )
    assert not result.is_error, result.output
    assert (meta / "reflection-strategy.yaml").read_text() == "depth: 2\n"
    assert not (hosted / "meta").exists()


async def test_write_creates_the_file_under_the_declared_world_not_a_stray_world_dir(
    served: Path,
) -> None:
    result = await WriteFileTool().execute(
        {"path": "world/knowledge/tree/new-node.md", "content": "# New\n"}, _ctx(served)
    )
    assert not result.is_error, result.output
    assert (served / "knowledge" / "knowledge" / "tree" / "new-node.md").read_text() == "# New\n"
    assert not (served / "world").exists()


async def test_write_of_any_other_path_still_lands_in_the_workspace(served: Path) -> None:
    result = await WriteFileTool().execute({"path": "docs/a.md", "content": "a\n"}, _ctx(served))
    assert not result.is_error, result.output
    assert (served / "docs" / "a.md").read_text() == "a\n"


async def test_a_workspace_that_declares_no_world_keeps_its_own_world_directory(
    tmp_path: Path,
) -> None:
    """``world/`` is an ordinary directory name until a conf says it is the framework's."""
    (tmp_path / "world").mkdir()
    (tmp_path / "world" / "map.md").write_text("a game map\n")
    result = await ReadFileTool().execute({"path": "world/map.md"}, _ctx(tmp_path))
    assert not result.is_error, result.output
    assert result.output == "a game map\n"


# ── the Agent derives the roots once and hands them to everything that needs them ──


def _agent(ws: Path, script: list, **kwargs) -> Agent:
    return Agent(
        provider=ScriptedProvider(script),
        default_model="scripted/virtual-prefix",
        workspace_root=str(ws),
        max_iterations=6,
        **kwargs,
    )


def test_the_agent_derives_the_roots_from_the_workspace_conf(served: Path, tmp_path: Path) -> None:
    agent = _agent(served, [reply("hi")])
    assert agent.loop.virtual_roots == _mind_virtual_roots(served)
    plain = tmp_path / "plain"
    plain.mkdir()
    assert _agent(plain, [reply("hi")]).loop.virtual_roots == {}


def test_the_agent_hands_its_roots_to_the_sub_agents_it_spawns(served: Path) -> None:
    agent = _agent(served, [reply("hi")], enable_subagents=True)
    assert agent.loop.spawner is not None
    assert agent.loop.spawner._runner.virtual_roots == _mind_virtual_roots(served)


async def test_a_turn_reads_the_node_through_the_whole_stack(served: Path) -> None:
    agent = _agent(
        served,
        [call_tool("Read", {"path": f"world/{_NODE}"}, id="r1"), reply("read it")],
        permission_policy=PermissionPolicy("allow"),
    )
    result = await agent.arun_turn("what does the pack say about probabilities?")
    assert result.stop_reason == "completed"
    block = agent.session.messages[2].blocks[0]
    assert (block.is_error, block.output) == (False, _NODE_BODY)


async def test_a_delegated_child_gets_the_same_roots(
    served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The audit4 #4 shape (test_subagent.py): a child must not be handed a narrower sandbox,
    # and a child that cannot open ``world/…`` would fail the very Read its parent just made.
    import zakcode.agent.subagent as sub

    captured: dict = {}

    class _FakeLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)

        async def aclose(self) -> None:
            pass

        async def arun_turn(self, prompt: str) -> TurnResult:
            return TurnResult(stop_reason="completed")

    monkeypatch.setattr(sub, "AgentLoop", _FakeLoop)
    roots = _mind_virtual_roots(served)
    runner = sub.SubAgentRunner(
        provider=ScriptedProvider([reply("x")]),
        registry=default_registry(),
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=served
        ),
        budget=IterationBudget(10),
        workspace_root=served,
        virtual_roots=roots,
    )
    await runner.run(sub.GENERAL_PURPOSE, "do it")
    assert captured["virtual_roots"] == roots


# ── the permission floor sees the rewritten spelling (ADR-0031) ─────────────────

_WRITE = ToolSpec(
    name="write_file",
    description="write_file test tool",
    parameters={"type": "object", "properties": {}},
    required_permission=PermissionTier.WORKSPACE_WRITE,
    concurrency=ConcurrencyClass.NEVER_PARALLEL,
)


def _guarding(prod: str, *, virtual: dict[str, str] | None) -> PermissionPolicy:
    # what ``deny Edit(/srv/prod/**)`` ingests to: an operator rule on the REAL location
    return PermissionPolicy(
        PermissionMode.AUTONOMOUS,
        extra_protected_paths=compile_protected_paths([rf"{prod}/"], write_only=True),
        workspace_root="/ws",
        virtual_roots=virtual,
    )


def test_a_rule_on_the_real_world_directory_binds_the_world_spelling_too() -> None:
    policy = _guarding("/srv/prod/world", virtual={"world": "/srv/prod/world"})
    for path in ("/srv/prod/world/guardrails.md", "world/guardrails.md"):
        assert policy.decide(_WRITE, {"path": path})[0] is PermissionDecision.DENY, path
    assert policy.decide(_WRITE, {"path": "docs/guardrails.md"})[0] is PermissionDecision.ALLOW
    # a delegated child is judged the same way
    child = policy.child_view()
    assert child.decide(_WRITE, {"path": "world/guardrails.md"})[0] is PermissionDecision.DENY


def test_without_the_roots_the_world_spelling_walks_past_a_rule_on_the_real_location() -> None:
    """Control for the pin above: the rule only bites the alias because the floor knows it."""
    policy = _guarding("/srv/prod/world", virtual=None)
    assert policy.decide(_WRITE, {"path": "/srv/prod/world/guardrails.md"})[0] is (
        PermissionDecision.DENY
    )
    assert policy.decide(_WRITE, {"path": "world/guardrails.md"})[0] is PermissionDecision.ALLOW


def test_the_agent_hands_its_roots_to_the_policy_it_builds(served: Path) -> None:
    agent = _agent(served, [reply("hi")])
    expected = {k: str(v) for k, v in _mind_virtual_roots(served).items()}
    assert agent.permission_policy._virtual_roots == expected
