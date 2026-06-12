"""The protected-path floor (self-remediation Step 2).

A write to a sensitive location (``.git/``, ``.env``, the virtualenv, the agent's own config)
is never auto-allowed: it escalates to a confirmation prompt — and a hard DENY in
``autonomous`` — even under ``allow``/``acceptEdits`` mode or a session grant. This closes the
"a blanket grant (or a loose mode) silently waives a sensitive write" hole, and the
self-permission-escalation / secret-rewrite / dependency-tamper / repo-corruption vectors an
unattended agent must not reach. Tighten-only, like the dangerous-command floor.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from zakcode.config import PermissionTier
from zakcode.permissions import (
    PROTECTED_PATH_PATTERNS,
    PermissionDecision,
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionRequest,
    compile_protected_paths,
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


WRITE = _spec("write_file", PermissionTier.WORKSPACE_WRITE)
SHELL = _spec("bash", PermissionTier.DANGER_FULL_ACCESS)


class _ScriptedPrompter:
    def __init__(self, outcomes: list[PermissionOutcome] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[PermissionRequest] = []

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        self.calls.append(request)
        return self.outcomes.pop(0) if self.outcomes else PermissionOutcome.DENY_ONCE


def _auth(policy: PermissionPolicy, spec: ToolSpec, args: dict) -> tuple[bool, str]:
    return asyncio.run(policy.authorize(spec, args))


def _matches(path: str) -> bool:
    return any(p.search(path) for p, _ in PROTECTED_PATH_PATTERNS)


# ── 1. pattern precision: sensitive paths match, look-alikes do NOT ───────────────


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        # layered (Next.js/Vite) + backup secrets — any chain of dotted segments (review finding)
        ".env.development.local",
        ".env.production.local",
        ".env.test.local",
        ".env.local.bak",
        ".env~",
        "config/.env.staging.local",
        "config/.env",
        "/abs/path/.env",
        ".git/config",
        ".git/hooks/pre-commit",
        "repo/.git/HEAD",
        ".venv/bin/activate",
        "venv/lib/python3.11/x",
        "site-packages/requests/__init__.py",
        ".venv/lib/site-packages/foo",
        ".claude/settings.json",
        "x/.claude/agents/self.md",
    ],
)
def test_protected_paths_match(path: str) -> None:
    assert _matches(path), path


@pytest.mark.parametrize(
    "path",
    [
        # look-alikes that must NOT be protected
        ".env.example",
        ".env.sample",
        ".env.template",
        ".gitignore",
        ".gitattributes",
        ".github/workflows/ci.yml",
        "src/main.py",
        "environment.py",
        "my_environment.txt",
        "README.md",
        "docs/.venvnotes.md",  # 'venv' not a path segment here
        "prevent/x.py",
    ],
)
def test_lookalike_paths_do_not_match(path: str) -> None:
    assert not _matches(path), path


# ── 2. decide(): the floor never auto-allows a protected write ────────────────────


def test_protected_write_escalates_in_allow_mode() -> None:
    policy = PermissionPolicy(PermissionMode.ALLOW)
    decision, reason = policy.decide(WRITE, {"path": ".env"})
    assert decision is PermissionDecision.ASK
    assert ".env" in reason


def test_protected_write_escalates_in_accept_edits_mode() -> None:
    # acceptEdits auto-applies a NORMAL workspace edit, but a protected path still escalates.
    policy = PermissionPolicy(PermissionMode.ACCEPT_EDITS)
    assert policy.decide(WRITE, {"path": "src/main.py"})[0] is PermissionDecision.ALLOW
    assert policy.decide(WRITE, {"path": ".env"})[0] is PermissionDecision.ASK


def test_protected_write_hard_denies_in_autonomous() -> None:
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    for path in (".env", ".git/config", ".venv/bin/pip", ".claude/settings.json"):
        decision, _ = policy.decide(WRITE, {"path": path})
        assert decision is PermissionDecision.DENY, path


def test_normal_write_passes_through() -> None:
    policy = PermissionPolicy(PermissionMode.ALLOW)
    assert policy.decide(WRITE, {"path": "src/app.py"})[0] is PermissionDecision.ALLOW
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    assert policy.decide(WRITE, {"path": "docs/readme.md"})[0] is PermissionDecision.ALLOW


def test_shell_args_are_not_scanned_for_protected_paths() -> None:
    # The floor scans WRITE-tool path args, NOT shell commands: a path in a command is usually a
    # read/execute (running the venv python, reading .git/config), so scanning over-blocks. A
    # shell-driven write to a protected path is best-effort residual (the Step 3 sandbox's job).
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    # the key regression: executing the venv python must NOT be blocked
    assert policy.decide(SHELL, {"command": ".venv/bin/python -m pytest"})[0] is (
        PermissionDecision.ALLOW
    )
    assert policy.decide(SHELL, {"command": "cat .git/config"})[0] is PermissionDecision.ALLOW
    # a shell redirect into .env is NOT caught by this floor (documented residual)
    assert policy.decide(SHELL, {"command": "echo x >> .env"})[0] is PermissionDecision.ALLOW


def test_dangerous_floor_still_applies_to_shell() -> None:
    # The dangerous-command floor is unaffected by the protected-path scoping.
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    decision, reason = policy.decide(SHELL, {"command": "sudo tee /etc/hosts"})
    assert decision is PermissionDecision.DENY
    assert "dangerous" in reason.lower()


# ── 3. a session grant cannot waive the protected-path floor ──────────────────────


def test_session_grant_does_not_waive_protected_path_interactive() -> None:
    prompter = _ScriptedPrompter([PermissionOutcome.DENY_ONCE])
    policy = PermissionPolicy(PermissionMode.ALLOW, prompter=prompter)
    policy._session_allow.add("write_file")  # blanket grant

    # a normal write rides the grant unprompted
    allowed, _ = _auth(policy, WRITE, {"path": "src/x.py"})
    assert allowed is True
    assert prompter.calls == []

    # a protected write re-prompts despite the grant (operator denies here)
    allowed, _ = _auth(policy, WRITE, {"path": ".env"})
    assert allowed is False
    assert len(prompter.calls) == 1


def test_session_grant_does_not_waive_protected_path_autonomous() -> None:
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    policy._session_allow.add("write_file")
    allowed, reason = _auth(policy, WRITE, {"path": ".git/config"})
    assert allowed is False
    assert "protected" in reason.lower()


# ── 4. public accessor + child_view + operator extras ─────────────────────────────


def test_protected_path_reason_accessor() -> None:
    policy = PermissionPolicy(PermissionMode.ALLOW)
    assert policy.protected_path_reason({"path": ".env"}) is not None
    assert policy.protected_path_reason({"path": "src/x.py"}) is None
    # shell args are not scanned (file-path args only)
    assert policy.protected_path_reason({"command": "echo > .git/x"}) is None


def test_child_view_keeps_the_floor() -> None:
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS)
    child = policy.child_view()
    assert child.decide(WRITE, {"path": ".env"})[0] is PermissionDecision.DENY


def test_operator_extra_protected_paths_tighten() -> None:
    extra = compile_protected_paths([r"secrets/.*\.key"])
    policy = PermissionPolicy(PermissionMode.AUTONOMOUS, extra_protected_paths=extra)
    assert policy.decide(WRITE, {"path": "secrets/prod.key"})[0] is PermissionDecision.DENY
    # built-ins still present alongside the extra
    assert policy.decide(WRITE, {"path": ".env"})[0] is PermissionDecision.DENY
    # and the extra rides through child_view (combined list is propagated)
    assert (
        policy.child_view().decide(WRITE, {"path": "secrets/x.key"})[0] is PermissionDecision.DENY
    )


def test_compile_protected_paths_skips_invalid_regex() -> None:
    compiled = compile_protected_paths(["valid.*", "(unclosed", "", "  "])
    assert len(compiled) == 1
    assert compiled[0][0].search("validxyz")
    assert all(isinstance(p, re.Pattern) for p, _ in compiled)
