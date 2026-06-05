"""M-3 integration test: cross-runtime skill portability via multi-root sandbox.

Proves that when ``--skill-dir`` points at claude-mind's ``.claude/skills``
directory, the auto-detection logic correctly identifies:

1. The mind repo root (the ``.git``-owning ancestor of the skill dir).
2. External ``WORLD_PATH`` and ``META_PATH`` parsed from
   ``agents/*/local-paths.conf`` under that repo root.

Then proves the sandbox accepts file reads from all detected roots (and rejects
paths outside them).

These tests use hermetic temp directories; they do NOT hit the real claude-mind
repo. The _infer_roots_from_skill_dir helper is exercised directly so the test
does not require an LLM provider.
"""

from __future__ import annotations

import pytest

from zakcode import _find_repo_root, _infer_roots_from_skill_dir, _parse_local_paths_conf
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins._safety import PathEscapeError, resolve_in_workspace_roots
from zakcode.tools.builtins.read_file import ReadFileTool

# --------------------------------------------------------------------------- #
# Helper unit tests
# --------------------------------------------------------------------------- #


def test_find_repo_root_detects_git(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    deep = repo / "a" / "b" / "c"
    deep.mkdir(parents=True)
    assert _find_repo_root(deep) == repo.resolve()


def test_find_repo_root_returns_none_when_absent(tmp_path):
    assert _find_repo_root(tmp_path) is None


def test_parse_local_paths_conf_extracts_world_and_meta(tmp_path):
    world = tmp_path / "world"
    meta = tmp_path / "meta"
    world.mkdir()
    meta.mkdir()
    conf = tmp_path / "local-paths.conf"
    conf.write_text(
        f"# comment\nWORLD_PATH={world}\nMETA_PATH={meta}\nOTHER=ignored\n",
        encoding="utf-8",
    )
    result = _parse_local_paths_conf(conf)
    assert len(result) == 2
    assert world.resolve() in [p.resolve() for p in result]
    assert meta.resolve() in [p.resolve() for p in result]


def test_parse_local_paths_conf_skips_nonexistent(tmp_path):
    conf = tmp_path / "local-paths.conf"
    conf.write_text("WORLD_PATH=C:/nonexistent/dir/surely\n", encoding="utf-8")
    assert _parse_local_paths_conf(conf) == []


def test_parse_local_paths_conf_missing_file(tmp_path):
    assert _parse_local_paths_conf(tmp_path / "nope.conf") == []


# --------------------------------------------------------------------------- #
# _infer_roots_from_skill_dir — full integration
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_mind_repo(tmp_path):
    """Build a minimal repo structure mirroring claude-mind's layout."""
    repo = tmp_path / "Mind"
    repo.mkdir()
    (repo / ".git").mkdir()

    # Skills dir
    skills = repo / ".claude" / "skills" / "prime"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("---\nname: prime\n---\nPseudocode.", encoding="utf-8")

    # External world and meta dirs (outside the repo, like a real OneDrive location)
    world = tmp_path / "Agents-World"
    meta = tmp_path / "Agents-Meta"
    world.mkdir()
    meta.mkdir()
    (world / "program.md").write_text("# The Program", encoding="utf-8")
    (meta / "config-overrides.yaml").write_text("# overrides", encoding="utf-8")

    # Agent config pointing at the external dirs
    agent_dir = repo / "agents" / "omni"
    agent_dir.mkdir(parents=True)
    conf = agent_dir / "local-paths.conf"
    conf.write_text(
        f"WORLD_PATH={world}\nMETA_PATH={meta}\n",
        encoding="utf-8",
    )

    return repo, world, meta, skills.parent  # skills.parent = .claude/skills


def test_infer_roots_discovers_repo_and_externals(mock_mind_repo):
    repo, world, meta, skill_dir = mock_mind_repo
    roots = _infer_roots_from_skill_dir(skill_dir)
    resolved = [r.resolve() for r in roots]

    # Must include the repo root.
    assert repo.resolve() in resolved
    # Must include the external world and meta dirs.
    assert world.resolve() in resolved
    assert meta.resolve() in resolved


def test_infer_roots_deduplicates(mock_mind_repo):
    repo, _, _, skill_dir = mock_mind_repo
    roots = _infer_roots_from_skill_dir(skill_dir)
    resolved = [r.resolve() for r in roots]
    # Each root appears exactly once.
    assert len(resolved) == len(set(resolved))


# --------------------------------------------------------------------------- #
# End-to-end: sandbox accepts reads across all inferred roots
# --------------------------------------------------------------------------- #


async def test_m3_sandbox_accepts_reads_across_all_roots(mock_mind_repo):
    """The multi-root sandbox accepts file reads from every auto-detected root."""
    repo, world, meta, skill_dir = mock_mind_repo

    # Simulate the auto-detection that Agent.__init__ performs.
    inferred = _infer_roots_from_skill_dir(skill_dir)
    # Primary workspace is the zak-code repo (would be CWD in a real invocation).
    # For this test, use a dummy primary.
    primary = repo.parent / "ZakCode"
    primary.mkdir()
    (primary / "pyproject.toml").write_text("[project]\nname='zakcode'\n", encoding="utf-8")

    ctx = ToolContext(
        workspace_root=primary,
        extra_workspace_roots=inferred,
    )

    # Read from primary workspace — should work.
    res = await ReadFileTool().execute({"path": str(primary / "pyproject.toml")}, ctx)
    assert not res.is_error, f"Primary read failed: {res.output}"
    assert "zakcode" in res.output

    # Read from mind repo root — should work.
    res = await ReadFileTool().execute(
        {"path": str(repo / ".claude" / "skills" / "prime" / "SKILL.md")}, ctx
    )
    assert not res.is_error, f"Repo read failed: {res.output}"
    assert "Pseudocode" in res.output

    # Read from external world dir — should work.
    res = await ReadFileTool().execute({"path": str(world / "program.md")}, ctx)
    assert not res.is_error, f"World read failed: {res.output}"
    assert "The Program" in res.output

    # Read from external meta dir — should work.
    res = await ReadFileTool().execute({"path": str(meta / "config-overrides.yaml")}, ctx)
    assert not res.is_error, f"Meta read failed: {res.output}"
    assert "overrides" in res.output


async def test_m3_sandbox_rejects_outside_all_roots(mock_mind_repo):
    """Even with multi-root, a path outside ALL roots is rejected."""
    repo, _, _, skill_dir = mock_mind_repo
    inferred = _infer_roots_from_skill_dir(skill_dir)
    primary = repo.parent / "ZakCode"
    primary.mkdir(exist_ok=True)

    ctx = ToolContext(
        workspace_root=primary,
        extra_workspace_roots=inferred,
    )

    # A path outside all roots.
    outside = repo.parent.parent / "totally_outside.txt"
    res = await ReadFileTool().execute({"path": str(outside)}, ctx)
    assert res.is_error
    assert "outside" in res.output.lower()


def test_m3_safety_layer_rejects_traversal_from_multi_root(mock_mind_repo):
    """Traversal escape is rejected even with multi-root."""
    repo, world, meta, _ = mock_mind_repo
    with pytest.raises(PathEscapeError):
        resolve_in_workspace_roots(
            "../../../../../../etc/passwd",
            [repo, world, meta],
        )
