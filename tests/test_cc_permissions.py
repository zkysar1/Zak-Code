"""Tests for Claude Code ``permissions`` ingestion (Phase 3).

Covers the translation of CC ``permissions.{allow,deny,ask}`` ``Tool(pattern)`` gestures into Zak
Code's deny-first :class:`~zakcode.permissions.PermissionPolicy` seams, and — the whole point — the
TIGHTEN-ONLY security invariant: an ingested ``allow`` can never let the never-waivable
catastrophic-command floor or protected-path floor through.

Generic by construction: no plug-in is named. The gestures used here (``Bash(rm -rf*)``,
``Write(*/secrets/*)``, a bare-tool deny/allow) are the grammar, exercised on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode import Agent
from zakcode.config import Settings
from zakcode.permissions import PermissionDecision, PermissionMode, PermissionPolicy
from zakcode.permissions_settings import IngestedPermissions, load_settings_permissions

# ── fixtures ────────────────────────────────────────────────────────────────────


def _write_permissions(ws: Path, permissions: dict, *, local: bool = False) -> None:
    """Write a ``{"permissions": ...}`` settings file into ``<ws>/.claude``."""
    d = ws / ".claude"
    d.mkdir(parents=True, exist_ok=True)
    name = "settings.local.json" if local else "settings.json"
    (d / name).write_text(json.dumps({"permissions": permissions}), encoding="utf-8")


def _agent(ws: Path, *, mode: str = "ask") -> Agent:
    """A real offline Agent (scripted provider, no prompter); ingestion is always-on (ADR-0029).

    No prompter is supplied, so an ``ask`` decision FAILS CLOSED (denied) — the safe headless
    default, and exactly the configuration in which the floor-holds proof is strongest.
    """
    return Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=ws,
            permission_mode=mode,
        ),
    )


def _spec(agent: Agent, tool_name: str) -> object:
    """The real ToolSpec for ``tool_name`` from the agent's registry (faithful tier/parameters)."""
    tool = agent.registry.get(tool_name)
    assert tool is not None, tool_name
    return tool.spec


# ── deny: a Bash command glob denies that command ───────────────────────────────


def test_deny_bash_command_glob_denies_that_command(tmp_path: Path) -> None:
    # An ingested ``deny: ["Bash(rm -rf*)"]`` must deny ``rm -rf <anything>``. Run under 'allow'
    # mode so bash would otherwise AUTO-ALLOW — isolating the ingested rule's effect: it escalates
    # the matched command, and with no prompter that ASK fails closed to DENY.
    _write_permissions(tmp_path, {"deny": ["Bash(rm -rf*)"]})
    agent = _agent(tmp_path, mode="allow")
    bash = _spec(agent, "bash")

    decision, reason = agent.permission_policy.decide(bash, {"command": "rm -rf ./build"})
    assert decision is PermissionDecision.ASK, reason  # escalated by the ingested deny pattern
    assert "operator deny rule" in reason or "rm -rf" in reason

    # A non-matching command still auto-allows under 'allow' mode (the rule is narrow, not blanket).
    ok, _ = agent.permission_policy.decide(bash, {"command": "ls -la"})
    assert ok is PermissionDecision.ALLOW


async def test_deny_bash_command_glob_fails_closed_without_prompter(tmp_path: Path) -> None:
    # The same ingested deny, all the way through authorize(): no prompter ⇒ the ASK escalation
    # fails closed to a hard DENY.
    _write_permissions(tmp_path, {"deny": ["Bash(rm -rf*)"]})
    agent = _agent(tmp_path, mode="allow")
    bash = _spec(agent, "bash")
    allowed, reason = await agent.permission_policy.authorize(bash, {"command": "rm -rf ./build"})
    assert allowed is False
    assert "confirmation" in reason or "deny" in reason


# ── deny: a path glob protects that path ────────────────────────────────────────


def test_deny_path_glob_protects_that_path(tmp_path: Path) -> None:
    # An ingested ``deny: ["Write(*/secrets/*)"]`` must protect that path: a write there never
    # auto-allows, even under 'allow' mode — it re-decides to ASK (the protected-path floor).
    _write_permissions(tmp_path, {"deny": ["Write(*/secrets/*)"]})
    agent = _agent(tmp_path, mode="allow")
    write = _spec(agent, "write_file")

    decision, reason = agent.permission_policy.decide(
        write, {"path": "config/secrets/api.txt", "content": "x"}
    )
    assert decision is PermissionDecision.ASK, reason
    assert "protected path" in reason

    # A path outside the rule still auto-allows.
    ok, _ = agent.permission_policy.decide(write, {"path": "config/app.txt", "content": "x"})
    assert ok is PermissionDecision.ALLOW


def test_deny_path_glob_matches_windows_separators(tmp_path: Path) -> None:
    # The translated path regex normalizes ``/`` to match a Windows ``\`` too, so a POSIX-written CC
    # rule still bites on a Windows path argument.
    _write_permissions(tmp_path, {"deny": ["Edit(*/world/*.jsonl)"]})
    agent = _agent(tmp_path, mode="allow")
    edit = _spec(agent, "edit_file")
    decision, reason = agent.permission_policy.decide(edit, {"path": r"agent\world\events.jsonl"})
    assert decision is PermissionDecision.ASK, reason
    assert "protected path" in reason


# ── allow: a bare-tool allow pre-approves a normally-prompting tool ──────────────


def test_allow_bare_tool_pre_approves_a_prompting_tool(tmp_path: Path) -> None:
    # Under 'ask' mode, write_file (WORKSPACE_WRITE) normally PROMPTS. An ingested
    # ``allow: ["Write"]`` pins write_file to 'allow' mode, so a benign workspace write auto-allows.
    _write_permissions(tmp_path, {"allow": ["Write"]})
    agent = _agent(tmp_path, mode="ask")
    write = _spec(agent, "write_file")

    decision, _ = agent.permission_policy.decide(write, {"path": "notes.txt", "content": "hi"})
    assert decision is PermissionDecision.ALLOW

    # Control: in a workspace WITHOUT the allow gesture (ingestion is always-on, so the control
    # is a clean workspace, not a flag) the same write would PROMPT (ASK), proving the
    # pre-approval is what changed it.
    clean = tmp_path / "clean-ws"
    clean.mkdir()
    plain = _agent(clean, mode="ask")
    pdec, _ = plain.permission_policy.decide(
        _spec(plain, "write_file"), {"path": "notes.txt", "content": "hi"}
    )
    assert pdec is PermissionDecision.ASK


# ── THE SECURITY TEST: an ingested allow can NEVER bypass the floor ──────────────


async def test_allow_bash_star_still_denies_catastrophic_and_protected(tmp_path: Path) -> None:
    # The crux. Ingest the most permissive gesture possible — ``allow: ["Bash(*)", "Write(*)"]`` —
    # and prove the tighten-only invariant: a catastrophic ``rm -rf /`` is STILL denied, and a write
    # to the secrets file ``.env`` is STILL protected. The floor runs BEFORE any allow.
    _write_permissions(tmp_path, {"allow": ["Bash(*)", "Write(*)"]})
    agent = _agent(tmp_path, mode="ask")  # no prompter ⇒ any escalation fails closed
    policy = agent.permission_policy
    bash = _spec(agent, "bash")
    write = _spec(agent, "write_file")

    # 1) Catastrophic command: rm -rf / is never auto-allowed; with no prompter it is DENIED.
    allowed, reason = await policy.authorize(bash, {"command": "rm -rf /"})
    assert allowed is False, reason
    assert policy.dangerous_reason({"command": "rm -rf /"}) is not None

    # 2) Protected path: a write to .env is never auto-allowed; with no prompter it is DENIED.
    allowed, reason = await policy.authorize(write, {"path": ".env", "content": "SECRET=1"})
    assert allowed is False, reason
    assert policy.protected_path_reason({"path": ".env"}) is not None


def test_allow_bash_star_in_autonomous_is_hard_deny(tmp_path: Path) -> None:
    # Strengthen the invariant for the unattended posture: even with ``allow: ["Bash(*)"]``
    # ingested, an AUTONOMOUS session hard-DENIES rm -rf / deterministically (not a prompt). The
    # ingested allow cannot loosen the autonomous floor.
    _write_permissions(tmp_path, {"allow": ["Bash(*)"]})
    agent = _agent(tmp_path, mode="autonomous")
    decision, reason = agent.permission_policy.decide(_spec(agent, "bash"), {"command": "rm -rf /"})
    assert decision is PermissionDecision.DENY
    assert "dangerous" in reason


# ── deny: a bare-tool deny pins that tool off ───────────────────────────────────


def test_deny_bare_tool_denies_the_whole_tool(tmp_path: Path) -> None:
    # A bare ``deny: ["Bash"]`` denies bash UNCONDITIONALLY (via extra_denied_tools, not a mode
    # override): every bash call is blocked outright, even under a permissive 'allow' session.
    # (See test_extra_denied_tools_binds_a_read_only_tool — not a mode override.)
    _write_permissions(tmp_path, {"deny": ["Bash"]})
    ingested, _ = load_settings_permissions(tmp_path)
    assert "bash" in ingested.denied_tools
    agent = _agent(tmp_path, mode="allow")
    decision, reason = agent.permission_policy.decide(_spec(agent, "bash"), {"command": "ls"})
    assert decision is PermissionDecision.DENY
    assert "denied" in reason.lower()


# ── unmappable gestures are recorded, never crashed or mis-mapped ───────────────


def test_unmappable_gesture_is_recorded_not_crashed(tmp_path: Path) -> None:
    # A CC-only tool with no Zak counterpart (AskUserQuestion) and an exotic per-pattern allow must
    # be RECORDED in the errors dict with a reason — never silently dropped or mis-mapped, never a
    # crash.
    _write_permissions(
        tmp_path,
        {
            "deny": ["AskUserQuestion(*)"],
            "allow": ["Write(**/.claude/skills/**)"],  # per-pattern allow: no tighten-only mapping
            "ask": ["Read(*/.env)"],  # ask → default posture, recorded as a benign no-op
        },
    )
    ingested, errors = load_settings_permissions(tmp_path)
    # Nothing mappable was produced from these three gestures.
    assert isinstance(ingested, IngestedPermissions)
    assert not ingested.denied_command_regexes
    # Each gesture is accounted for in errors, with a human reason.
    joined = " ".join(errors.values()).lower()
    assert any("askuserquestion" in k.lower() for k in errors)
    assert "no zak code tool" in joined
    assert any("per-pattern allow" in v.lower() for v in errors.values())
    assert any("ask gesture" in v.lower() for v in errors.values())


def test_malformed_gesture_does_not_crash(tmp_path: Path) -> None:
    # A malformed gesture (missing closing paren) and a non-string entry must not crash ingestion.
    _write_permissions(tmp_path, {"deny": ["Bash(unterminated", 42, "", "  "]})
    ingested, errors = load_settings_permissions(tmp_path)
    # 'Bash(unterminated' parses as a bare (unknown) tool name 'Bash(unterminated' → recorded.
    assert isinstance(ingested, IngestedPermissions)
    assert any("bash(unterminated" in k.lower() for k in errors)


# ── always on: declared permissions are live, no flag exists (ADR-0029) ─────────


def test_ingestion_is_always_on(tmp_path: Path) -> None:
    # A workspace's ``permissions`` block binds with NO flag and NO Agent parameter — the same
    # "declared config is live" posture as settings hooks (ADR-0025). A bare ``deny: ["Bash"]``
    # in a plain Agent construction denies bash even under 'allow' mode. This matters doubly
    # since the built-in agent-config protected class was removed: these ingested rules are the
    # only authority over ``.claude/``, so they must always load.
    _write_permissions(tmp_path, {"deny": ["Bash"]})
    agent = Agent(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            permission_mode="allow",
        ),
    )
    decision, reason = agent.permission_policy.decide(_spec(agent, "bash"), {"command": "ls"})
    assert decision is PermissionDecision.DENY
    assert "denied by configuration" in reason


def test_no_settings_file_is_a_silent_noop(tmp_path: Path) -> None:
    # Ingestion ON but no settings file at all: empty result, no errors, no crash.
    ingested, errors = load_settings_permissions(tmp_path)
    assert ingested.is_empty()
    assert errors == {}


# ── settings.local.json is also read, layered after settings.json ───────────────


def test_settings_local_json_permissions_are_also_ingested(tmp_path: Path) -> None:
    # CC splits config across settings.json (shared) and settings.local.json (per-machine); both
    # contribute their deny rules (a tighten-only union).
    _write_permissions(tmp_path, {"deny": ["Bash(rm -rf*)"]})
    _write_permissions(tmp_path, {"deny": ["Write(*/secrets/*)"]}, local=True)
    ingested, _ = load_settings_permissions(tmp_path)
    assert ingested.denied_command_regexes  # from settings.json
    assert ingested.protected_path_regexes_write_only  # from settings.local.json (a Write deny)


# ── the verb is retained: an Edit/Write deny binds writes, not reads (ADR-0030) ─


def test_write_deny_does_not_block_reading_the_path(tmp_path: Path) -> None:
    # The Mind-shaped case measured in the fresh-eyes dry-run: deny Edit+Write on a path the
    # framework REQUIRES agents to read (36 of 44 real gestures were Edit/Write-only). CC leaves
    # the path readable; so must we. Autonomous mode makes the outcome binary (deny vs allow).
    _write_permissions(
        tmp_path,
        {"deny": ["Edit(*/knowledge/tree/_tree.yaml)", "Write(*/knowledge/tree/_tree.yaml)"]},
    )
    agent = _agent(tmp_path, mode="autonomous")
    read = _spec(agent, "read_file")
    write = _spec(agent, "write_file")
    path = "world/knowledge/tree/_tree.yaml"

    rdec, rreason = agent.permission_policy.decide(read, {"path": path})
    assert rdec is PermissionDecision.ALLOW, rreason  # readable, exactly as CC leaves it
    wdec, wreason = agent.permission_policy.decide(write, {"path": path, "content": "x"})
    assert wdec is PermissionDecision.DENY, wreason  # the write deny still binds
    assert "write to a protected path" in wreason


def test_read_deny_blocks_reads_and_writes(tmp_path: Path) -> None:
    # A deny Read(glob) is the operator saying the CONTENT is sensitive — it binds read-only
    # tools AND writes (strict pool, no write-only mark).
    _write_permissions(tmp_path, {"deny": ["Read(*/vault/*)"]})
    agent = _agent(tmp_path, mode="autonomous")
    read = _spec(agent, "read_file")
    write = _spec(agent, "write_file")
    assert agent.permission_policy.decide(read, {"path": "x/vault/key.txt"})[0] is (
        PermissionDecision.DENY
    )
    assert (
        agent.permission_policy.decide(write, {"path": "x/vault/key.txt", "content": "x"})[0]
        is PermissionDecision.DENY
    )


# ── a relative spelling cannot bypass a parent-prefixed glob (ADR-0031) ─────────


def test_relative_path_argument_binds_parent_prefixed_deny(tmp_path: Path) -> None:
    # The Mind-shaped rule ``Edit(*/.claude/skills/start/*)`` needs a parent segment (CC matches
    # absolute paths). The Agent hands its workspace root to the policy, so the RELATIVE
    # spelling a model actually emits resolves to the absolute form and binds too.
    _write_permissions(
        tmp_path, {"deny": ["Edit(*/.claude/skills/start/*)", "Write(*/.claude/skills/start/*)"]}
    )
    agent = _agent(tmp_path, mode="autonomous")
    write = _spec(agent, "write_file")
    read = _spec(agent, "read_file")
    relative = ".claude/skills/start/SKILL.md"
    absolute = str(tmp_path / relative)
    for path in (relative, absolute):
        wdec, wreason = agent.permission_policy.decide(write, {"path": path, "content": "x"})
        assert wdec is PermissionDecision.DENY, (path, wreason)
        # verb retained (ADR-0030): the Edit/Write deny leaves the file readable either way
        assert agent.permission_policy.decide(read, {"path": path})[0] is (
            PermissionDecision.ALLOW
        ), path
    # an unrelated skill stays editable — the deny is exactly as wide as written
    assert (
        agent.permission_policy.decide(
            write, {"path": ".claude/skills/other/SKILL.md", "content": "x"}
        )[0]
        is PermissionDecision.ALLOW
    )


# ── deny beats allow within an ingestion (tighten-only at the gesture level) ────


def test_deny_beats_allow_for_same_tool(tmp_path: Path) -> None:
    # If the same file both denies and allows a tool as a whole, deny wins (the stricter intent):
    # bash lands in denied_tools (unconditional) and the allow is dropped, not applied as a mode.
    _write_permissions(tmp_path, {"deny": ["Bash"], "allow": ["Bash(*)"]})
    ingested, errors = load_settings_permissions(tmp_path)
    assert "bash" in ingested.denied_tools
    assert "bash" not in ingested.tool_mode_overrides
    assert any("deny wins" in v.lower() for v in errors.values())


# ── a directly-constructed policy with the ingested seams holds the floor ───────


async def test_ingested_seams_on_a_bare_policy_hold_the_floor() -> None:
    # Belt-and-suspenders, independent of the Agent: feed the translated seams straight into a
    # PermissionPolicy the way the Agent does, and prove the floor still holds for the worst case.
    from zakcode.permissions import compile_deny_patterns, compile_protected_paths

    ingested = IngestedPermissions(tool_mode_overrides={"bash": "allow", "write_file": "allow"})
    policy = PermissionPolicy(
        PermissionMode.ASK,  # no prompter ⇒ ask fails closed
        extra_dangerous_patterns=compile_deny_patterns(ingested.denied_command_regexes),
        tool_mode_overrides=ingested.tool_mode_overrides,
        extra_protected_paths=compile_protected_paths(ingested.protected_path_regexes),
    )
    bash = _BashSpecStub()
    allowed, _ = await policy.authorize(bash, {"command": "rm -rf /"})
    assert allowed is False


class _BashSpecStub:
    """Minimal ToolSpec stand-in: decide() only reads .name and .required_permission."""

    name = "bash"

    @property
    def required_permission(self):  # noqa: ANN202 - tiny stub
        from zakcode.config import PermissionTier

        return PermissionTier.DANGER_FULL_ACCESS


class _WebFetchSpecStub:
    """A READ_ONLY tool stub — the tier a deny MODE override silently fails to bind."""

    name = "web_fetch"

    @property
    def required_permission(self):  # noqa: ANN202 - tiny stub
        from zakcode.config import PermissionTier

        return PermissionTier.READ_ONLY


def test_extra_denied_tools_binds_a_read_only_tool() -> None:
    # THE BLOCKER REGRESSION (policy level): a whole-tool deny must bind even a READ_ONLY tool. A
    # deny MODE override cannot (deny's ceiling is READ_ONLY, which a read-only tool satisfies); the
    # tier-independent extra_denied_tools does.
    policy = PermissionPolicy(PermissionMode.ALLOW, extra_denied_tools={"web_fetch"})
    decision, reason = policy.decide(_WebFetchSpecStub(), {"url": "http://x"})
    assert decision is PermissionDecision.DENY and "denied" in reason.lower()


def test_ingested_bare_deny_binds_a_read_only_tool(tmp_path: Path) -> None:
    # End-to-end: an ingested ``deny: ["WebFetch"]`` (a READ_ONLY CC tool) denies it through
    # the wired Agent policy — the read-only path the original mode-override silently missed.
    _write_permissions(tmp_path, {"deny": ["WebFetch"]})
    agent = _agent(tmp_path, mode="allow")
    decision, reason = agent.permission_policy.decide(_WebFetchSpecStub(), {"url": "http://x"})
    assert decision is PermissionDecision.DENY and "denied" in reason.lower()
