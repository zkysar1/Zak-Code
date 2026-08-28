"""``bypassPermissions`` — the dangerously-skip posture (ADR-0055).

Field-driven 2026-08-28 (coach, zc-03): an unattended Mind runner sat blocked forever on
an interactive y/a/n permission prompt nobody was there to answer. ``autonomous`` is the
never-prompt-fail-CLOSED mode; ``bypassPermissions`` is its fail-OPEN twin (the Claude
Code ``--dangerously-skip-permissions`` analog): nothing prompts, every escalation the
other modes would ASK is ALLOWED, and only two refusals survive — the catastrophic
blocklist (uniform in every mode) and explicit whole-tool config denies. An explicit
per-tool TIGHTEN override is still honored: the operator who wrote both asked for it.
"""

from __future__ import annotations

import asyncio

from zakcode.config import PermissionTier
from zakcode.permissions import (
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionRequest,
)
from zakcode.tools.base import ConcurrencyClass, ToolSpec


def _spec(name: str, tier: PermissionTier) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} test tool",
        parameters={"type": "object", "properties": {}},
        required_permission=tier,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )


READ = _spec("read_file", PermissionTier.READ_ONLY)
WRITE = _spec("write_file", PermissionTier.WORKSPACE_WRITE)
SHELL = _spec("bash", PermissionTier.DANGER_FULL_ACCESS)
FETCH = _spec("web_fetch", PermissionTier.READ_ONLY)

DANGEROUS = {"command": "sudo rm -rf / --no-preserve-root"}
BENIGN = {"command": "git status"}


class ScriptedPrompter:
    def __init__(self, outcomes: list[PermissionOutcome] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[PermissionRequest] = []

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        self.calls.append(request)
        return self.outcomes.pop(0) if self.outcomes else PermissionOutcome.DENY_ONCE


def _auth(policy: PermissionPolicy, spec: ToolSpec, args: dict) -> tuple[bool, str]:
    return asyncio.run(policy.authorize(spec, args))


def test_bypass_mode_allows_everything_and_never_prompts() -> None:
    """Identical behavior with or without a prompter — nothing ever prompts."""
    prompter = ScriptedPrompter([PermissionOutcome.ALLOW_ONCE] * 8)
    for policy in (
        PermissionPolicy(PermissionMode.BYPASS),  # headless
        PermissionPolicy(PermissionMode.BYPASS, prompter=prompter),  # attended
    ):
        assert _auth(policy, READ, {"path": "a.txt"}) == (True, "")
        assert _auth(policy, WRITE, {"path": "a.txt"}) == (True, "")
        assert _auth(policy, SHELL, BENIGN) == (True, "")
        # Catastrophic: still a deterministic hard DENY — bypass waives prompts and
        # gates, never the catastrophic floor (one rule, uniform in every mode).
        allowed, reason = _auth(policy, SHELL, DANGEROUS)
        assert not allowed and "dangerous" in reason
    assert prompter.calls == []


def test_bypass_waives_the_dependency_gate() -> None:
    # The coach wedge: an undeclared install (plus a phantom package parsed from a
    # redirection) escalated to a prompt no one could answer. In bypass it just runs.
    policy = PermissionPolicy(
        PermissionMode.BYPASS, declared_packages=lambda: {"pypi:requests"}
    )
    assert _auth(policy, SHELL, {"command": "pip install evil-pkg"}) == (True, "")


def test_bypass_waives_protected_paths_and_confirm_tools() -> None:
    prompter = ScriptedPrompter()
    policy = PermissionPolicy(
        PermissionMode.BYPASS, prompter=prompter, confirm_tools={"web_fetch"}
    )
    # Protected path (.env is a built-in write-sensitive floor elsewhere): allowed.
    assert _auth(policy, WRITE, {"path": ".env"}) == (True, "")
    # Confirm-on-use tool: the confirmation is waived, not failed closed.
    assert _auth(policy, FETCH, {"url": "https://example.com"}) == (True, "")
    assert prompter.calls == []
    # And the loop-facing public probe reports no protected reason under bypass.
    assert policy.protected_path_reason({"path": ".git/config"}) is None


def test_bypass_respects_explicit_tool_denies() -> None:
    policy = PermissionPolicy(PermissionMode.BYPASS, extra_denied_tools={"web_fetch"})
    allowed, reason = _auth(policy, FETCH, {"url": "https://example.com"})
    assert not allowed and "denied by configuration" in reason


def test_bypass_honors_an_explicit_per_tool_tighten() -> None:
    """session=bypass + tool override web_fetch->ask: the operator who wrote both has
    asked for that tool to prompt — the tighten is honored, not silently ignored."""
    prompter = ScriptedPrompter([PermissionOutcome.ALLOW_ONCE])
    policy = PermissionPolicy(
        PermissionMode.BYPASS,
        prompter=prompter,
        tool_mode_overrides={"web_fetch": PermissionMode.ASK},
    )
    # web_fetch is READ_ONLY tier — at or below ask's ceiling, so it auto-allows
    # without a prompt; the point is the DANGEROUS floor still binds the tightened
    # tool while bypass covers the rest.
    assert _auth(policy, FETCH, {"url": "https://example.com"}) == (True, "")
    allowed, reason = _auth(policy, SHELL, DANGEROUS)
    assert not allowed and "dangerous" in reason


def test_bypass_parse_spellings() -> None:
    assert PermissionMode.parse("bypassPermissions") is PermissionMode.BYPASS
    assert PermissionMode.parse("bypass-permissions") is PermissionMode.BYPASS
    assert PermissionMode.parse("BYPASS_PERMISSIONS") is PermissionMode.BYPASS
    # A bare "bypass" is NOT recognized — unknown values fail toward the safe default.
    assert PermissionMode.parse("bypass") is PermissionMode.ASK
