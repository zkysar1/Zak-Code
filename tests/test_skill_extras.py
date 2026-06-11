"""Skill-frontmatter extras + targeted logging instrumentation (audit P1-2 / P1-5).

Acceptance name from the audit: ``test_skill_extras`` — unknown frontmatter keys are
preserved in an ``extras`` dict (Mind skills carry ``minimum_mode``,
``companion_scripts``, ``user_invocable``, ``triggers``, …) and round-trip through
``save_skill``. The logging tests pin the new operator-facing instrumentation.
"""

from __future__ import annotations

import asyncio
import logging

from zakcode.config import PermissionTier
from zakcode.permissions import PermissionMode, PermissionPolicy
from zakcode.skills import parse_frontmatter, save_skill
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)

MIND_SKILL = """\
---
name: deep-focus
description: Enter deep focus mode.
allowed-tools: [read_file, bash]
version: 1.2.0
minimum_mode: contemplative
companion_scripts: [prime.sh, encode.sh]
user_invocable: true
triggers: [focus, deep-work]
---
Body of the skill.
"""


# ── acceptance 9: test_skill_extras ───────────────────────────────────────────


def test_skill_extras() -> None:
    """Unknown frontmatter keys are preserved in ``extras``, typed fields untouched."""
    fm, body = parse_frontmatter(MIND_SKILL)
    assert fm.name == "deep-focus"
    assert fm.allowed_tools == ["read_file", "bash"]
    assert fm.version == "1.2.0"
    assert fm.extras == {
        "minimum_mode": "contemplative",
        "companion_scripts": ["prime.sh", "encode.sh"],
        "user_invocable": "true",
        "triggers": ["focus", "deep-work"],
    }
    assert body == "Body of the skill."


def test_skill_extras_round_trip_through_save_skill(tmp_path) -> None:
    extras = {"minimum_mode": "swift", "companion_scripts": ["a.sh", "b.sh"]}
    path = save_skill(
        "round-trip",
        "desc",
        "the body",
        skills_dir=tmp_path,
        allowed_tools=["bash"],
        extras=extras,
    )
    fm, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert fm.extras == extras
    assert fm.name == "round-trip" and body == "the body"


def test_skill_without_extras_unchanged() -> None:
    fm, _ = parse_frontmatter("---\nname: plain\n---\nbody")
    assert fm.extras == {}


# ── logging instrumentation (audit P1-5) ──────────────────────────────────────


class _BoomTool(Tool):
    spec = ToolSpec(
        name="boom",
        description="always raises",
        parameters={"type": "object", "properties": {}},
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )

    async def execute(self, args: dict, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("kaboom")


def test_registry_logs_wrapped_tool_exception(tmp_path, caplog) -> None:
    """The biggest former silent-swallow now leaves an operator traceback."""
    reg = ToolRegistry()
    reg.register(_BoomTool())
    ctx = ToolContext(workspace_root=tmp_path)
    with caplog.at_level(logging.ERROR, logger="zakcode.tools"):
        result = asyncio.run(reg.execute("boom", {}, ctx))
    assert result.is_error and "kaboom" in result.output  # model-facing wrap unchanged
    assert any("boom" in r.message for r in caplog.records)  # operator-facing log added


def test_permission_denial_is_logged(caplog) -> None:
    shell = ToolSpec(
        name="bash",
        description="shell",
        parameters={"type": "object", "properties": {}},
        required_permission=PermissionTier.DANGER_FULL_ACCESS,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    with caplog.at_level(logging.WARNING, logger="zakcode.permissions"):
        allowed, _ = asyncio.run(policy.authorize(shell, {"command": "sudo rm -rf /"}))
    assert not allowed
    assert any("denied" in r.message and "bash" in str(r.args) for r in caplog.records)
