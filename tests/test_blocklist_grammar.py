"""Tests for the operator deny-rule grammar (compile_deny_patterns + extra patterns)."""

from __future__ import annotations

from pathlib import Path

from zakcode.config import PermissionTier
from zakcode.permissions import (
    PermissionDecision,
    PermissionPolicy,
    compile_deny_patterns,
)
from zakcode.tools.base import ToolSpec

_SHELL = ToolSpec(
    name="bash", description="run", required_permission=PermissionTier.DANGER_FULL_ACCESS
)


def test_compile_skips_invalid_regex() -> None:
    compiled = compile_deny_patterns(["valid-rule", "[unterminated(", "  ", "another"])
    # The two valid regexes compile; the bad one and the blank are skipped.
    assert len(compiled) == 2


def test_extra_pattern_escalates_otherwise_allowed_command() -> None:
    extra = compile_deny_patterns(["secret-tool"])
    policy = PermissionPolicy("allow", extra_dangerous_patterns=extra)
    decision, reason = policy.decide(_SHELL, {"command": "run secret-tool now"})
    assert decision is PermissionDecision.ASK  # tightened from allow
    assert "secret-tool" in reason


def test_extra_pattern_does_not_remove_baseline() -> None:
    policy = PermissionPolicy("allow", extra_dangerous_patterns=compile_deny_patterns(["foo"]))
    # A built-in footgun still escalates even though we supplied an unrelated extra.
    decision, _reason = policy.decide(_SHELL, {"command": "sudo rm something"})
    assert decision is PermissionDecision.ASK


def test_without_extra_the_command_is_allowed() -> None:
    policy = PermissionPolicy("allow")
    decision, _reason = policy.decide(_SHELL, {"command": "run secret-tool now"})
    assert decision is PermissionDecision.ALLOW  # not a baseline footgun


def test_deny_mode_blocks_extra_pattern() -> None:
    policy = PermissionPolicy("allow", extra_dangerous_patterns=compile_deny_patterns(["nope-cmd"]))
    # In allow mode it escalates to ASK; in deny mode a match is a hard DENY.
    deny_policy = PermissionPolicy(
        "deny", extra_dangerous_patterns=compile_deny_patterns(["nope-cmd"])
    )
    assert policy.decide(_SHELL, {"command": "nope-cmd"})[0] is PermissionDecision.ASK
    assert deny_policy.decide(_SHELL, {"command": "nope-cmd"})[0] is PermissionDecision.DENY


def test_facade_denied_commands_are_applied(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.evals.harness import ScriptedProvider, reply

    agent = Agent(
        provider=ScriptedProvider(script=[reply("ok")]),
        permission_mode="allow",
        denied_commands=["forbidden-binary"],
        default_model="scripted/test",
        workspace_root=str(tmp_path),
    )
    decision, _reason = agent.permission_policy.decide(_SHELL, {"command": "forbidden-binary --go"})
    assert decision is PermissionDecision.ASK
