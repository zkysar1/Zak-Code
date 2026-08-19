"""Claude Code compatibility — the conformance suite (the guardian).

Zak Code's strategic promise is to be a *faithful, GENERIC Claude-Code host*: anything built for
Claude Code (skills, hooks, settings, commands) runs on it unmodified, while the *behavior* comes
from whatever plugs in — claude-mind, another skill system, a third-party tool. This module is the
guardian of that promise.

THE RULE: every assertion here proves a piece of the Claude Code extension contract using only
GENERIC fixtures — never claude-mind (or any single framework) by name. If a conformance test can
only be written by referencing a specific plug-in, the host has leaked plug-in behavior into the
core and the design is wrong. See docs/CLAUDE-CODE-HOST-ROADMAP.md and docs/CLAUDE-MIND-COMPAT.md.

Contract areas (each roadmap row lands its test here):
  - Skills      (done)    — .claude/skills/<name>/SKILL.md discovery + tolerant frontmatter
  - Hooks       (done)    — settings.json ingestion, Stop->TurnEnd mapping, $CLAUDE_PROJECT_DIR
  - Commands    (Phase 1) — triggers-based dispatch, args, user-invocable
  - Settings    (Phase 1) — settings.json + settings.local.json layering
  - Permissions (Phase 3) — permissions.{allow,deny} Tool(glob) gestures → the deny-first policy,
                            tighten-only (the safety floor outranks any ingested allow)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.config import Settings
from zakcode.hooks import HookEvent, LifecyclePayload
from zakcode.hooks.settings_loader import load_settings_hooks
from zakcode.messages import Message
from zakcode.skills import (
    SkillRegistry,
    default_skill_dirs,
    discover_skill_dir,
    parse_frontmatter,
)
from zakcode.tools.base import ToolContext

pytestmark = pytest.mark.cc_conformance


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ===========================================================================
# Skills — the Claude Code `.claude/skills/<name>/SKILL.md` contract
# ===========================================================================


def test_claude_skills_dir_is_a_discovery_root(tmp_path: Path) -> None:
    # A faithful host discovers skills from <workspace>/.claude/skills (Claude Code's location).
    assert (tmp_path / ".claude" / "skills") in default_skill_dirs(tmp_path)


def test_skill_in_claude_dir_is_discovered(tmp_path: Path) -> None:
    _write(
        tmp_path / ".claude" / "skills" / "greet" / "SKILL.md",
        "---\nname: greet\ndescription: say hi\n---\nbody",
    )
    skills, errors = discover_skill_dir(tmp_path / ".claude" / "skills")
    assert [s.name for s in skills] == ["greet"]
    assert not errors


def test_skill_frontmatter_preserves_cognitive_keys() -> None:
    # A CC skill carries metadata the host does not type natively (triggers, user-invocable, and
    # arbitrary cognitive keys). The host must PRESERVE them, not drop or choke on them — that is
    # exactly what lets a framework layer its own semantics on top of a generic host.
    fm, _body = parse_frontmatter(
        "---\n"
        "name: looper\n"
        "description: a self-driving skill\n"
        "user-invocable: false\n"
        "triggers: [/start, /go]\n"
        "minimum_mode: autonomous\n"
        "---\n"
        "body"
    )
    assert fm.name == "looper"
    assert fm.extras["user_invocable"] == "false"  # key normalized -/_; value preserved verbatim
    assert fm.extras["triggers"] == ["/start", "/go"]  # bracketed value -> list
    assert fm.extras["minimum_mode"] == "autonomous"


def test_skill_is_resolvable_by_trigger_not_just_name(tmp_path: Path) -> None:
    # Claude Code maps a skill to a slash via its `triggers:` frontmatter: a skill named `looper`
    # with `triggers: [/start]` must be reachable as `/start`, not only as `/looper`. Name wins.
    _write(
        tmp_path / ".claude" / "skills" / "looper" / "SKILL.md",
        "---\nname: looper\ndescription: a looping skill\ntriggers: [/start, /go]\n---\nbody",
    )
    skills, _errors = discover_skill_dir(tmp_path / ".claude" / "skills")
    registry = SkillRegistry()
    for skill in skills:
        registry.add(skill)
    for token in ("looper", "start", "/go"):  # by name, by trigger, by trigger-with-slash
        resolved = registry.resolve(token)
        assert resolved is not None and resolved.name == "looper", token
    assert registry.resolve("missing") is None


def test_skill_resolution_is_case_insensitive(tmp_path: Path) -> None:
    # Claude Code matches names/triggers case-insensitively, and the CLI lower-cases a typed
    # `/Command` — so `/start` must reach a `triggers: [/Start]` skill, `/BOOTER` a `name: Booter`.
    _write(
        tmp_path / ".claude" / "skills" / "booter" / "SKILL.md",
        "---\nname: Booter\ndescription: d\ntriggers: [/Start]\n---\nbody",
    )
    skills, _errors = discover_skill_dir(tmp_path / ".claude" / "skills")
    registry = SkillRegistry()
    for skill in skills:
        registry.add(skill)
    for token in ("Booter", "booter", "BOOTER", "start", "/Start"):
        resolved = registry.resolve(token)
        assert resolved is not None and resolved.name == "Booter", token


# ===========================================================================
# Hooks — the Claude Code `settings.json` hook contract
# ===========================================================================


def _settings_json(workspace: Path, hooks: dict) -> Path:
    return _write(workspace / ".claude" / "settings.json", json.dumps({"hooks": hooks}))


def test_settings_json_stop_event_maps_to_turn_end(tmp_path: Path) -> None:
    # Claude Code's "Stop" hook (the perpetual-loop driver) IS Zak Code's generic TURN_END seam.
    _settings_json(
        tmp_path, {"Stop": [{"hooks": [{"type": "command", "command": "bash loop.sh"}]}]}
    )
    specs, errors = load_settings_hooks(tmp_path)
    assert [s.event for s in specs] == [HookEvent.TURN_END]
    assert not errors


def test_settings_json_substitutes_claude_project_dir(tmp_path: Path) -> None:
    # CC hooks reference scripts via $CLAUDE_PROJECT_DIR; a faithful host substitutes the workspace.
    _settings_json(
        tmp_path,
        {
            "PreToolUse": [
                {"hooks": [{"type": "command", "command": "bash $CLAUDE_PROJECT_DIR/g.sh"}]}
            ]
        },
    )
    specs, _errors = load_settings_hooks(tmp_path)
    assert any(tmp_path.as_posix() in part for part in specs[0].command)
    assert not any("$CLAUDE_PROJECT_DIR" in part for part in specs[0].command)


def test_settings_json_core_event_names_are_mapped(tmp_path: Path) -> None:
    # The Claude Code hook-event vocabulary the host speaks today.
    expected = {
        "PreToolUse": HookEvent.PRE_TOOL_USE,
        "PostToolUse": HookEvent.POST_TOOL_USE,
        "SessionStart": HookEvent.SESSION_START,
        "PreCompact": HookEvent.PRE_COMPACT,
        "Stop": HookEvent.TURN_END,
    }
    for name, event in expected.items():
        workspace = tmp_path / name  # isolate each event in its own workspace
        _settings_json(
            workspace, {name: [{"hooks": [{"type": "command", "command": "bash x.sh"}]}]}
        )
        specs, _errors = load_settings_hooks(workspace)
        assert [s.event for s in specs] == [event], name


def test_settings_json_unimplemented_event_is_skipped_not_crashed(tmp_path: Path) -> None:
    # Events the host does not yet implement (StopFailure, UserPromptExpansion) must be skipped with
    # a recorded reason — never a crash. When a later phase implements one, THIS assertion flips,
    # which is the point: it documents the current edge of the contract.
    _settings_json(
        tmp_path, {"StopFailure": [{"hooks": [{"type": "command", "command": "bash sf.sh"}]}]}
    )
    specs, errors = load_settings_hooks(tmp_path)
    assert specs == []
    assert "StopFailure" in errors


def test_settings_local_json_hooks_are_also_loaded(tmp_path: Path) -> None:
    # Claude Code splits config across settings.json (shared) and settings.local.json (per-machine);
    # a faithful host reads both, and hooks from each contribute.
    _settings_json(
        tmp_path, {"PreToolUse": [{"hooks": [{"type": "command", "command": "bash a.sh"}]}]}
    )
    _write(
        tmp_path / ".claude" / "settings.local.json",
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "bash b.sh"}]}]}}),
    )
    specs, _errors = load_settings_hooks(tmp_path)
    events = {s.event for s in specs}
    assert HookEvent.PRE_TOOL_USE in events  # from settings.json
    assert HookEvent.TURN_END in events  # from settings.local.json


def test_posttooluse_additional_context_is_parsed() -> None:
    # Claude Code's PostToolUse hook returns hookSpecificOutput.additionalContext to inject context
    # for the model after a tool runs; a faithful host parses it off the hook's stdout JSON.
    from zakcode.hooks import HookManager

    stdout = json.dumps(
        {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "extra info"}}
    ).encode()
    _message, _mutated, _deny, additional = HookManager._parse_stdout(stdout)
    assert additional == "extra info"


async def test_posttooluse_additional_context_reaches_aggregated_result(tmp_path: Path) -> None:
    # The REAL path: a PostToolUse shell hook returning additionalContext must survive
    # HookManager.run aggregation (not just the static _parse_stdout), so the loop can inject it.
    import sys

    from zakcode.hooks import HookManager, HookPayload, HookSpec

    script = tmp_path / "ctx_hook.py"
    script.write_text(
        'import json; print(json.dumps({"hookSpecificOutput": {"additionalContext": "INJECTED"}}))'
    )
    mgr = HookManager(
        [HookSpec(event=HookEvent.POST_TOOL_USE, command=[sys.executable, str(script)])]
    )
    result = await mgr.run(HookPayload(event=HookEvent.POST_TOOL_USE, tool_name="bash"))
    assert result.additional_context == "INJECTED"


# ===========================================================================
# Commands — Claude Code slash arguments (`/skill args`, use_skill args=…)
# ===========================================================================


def _scripted_agent(workspace: Path) -> Agent:
    # A real Agent on the offline scripted provider: exercises the actual skill-loading path with
    # no network and no model call. enable_skills wires the resolver + the use_skill tool.
    return Agent(
        settings=Settings(default_model="scripted/test", workspace_root=workspace),
        enable_skills=True,
    )


def _write_claude_skill(workspace: Path, name: str, body: str, *, frontmatter: str = "") -> None:
    extra = f"{frontmatter}\n" if frontmatter else ""
    _write(
        workspace / ".claude" / "skills" / name / "SKILL.md",
        f"---\nname: {name}\ndescription: {name} skill.\n{extra}---\n{body}\n",
    )


async def test_slash_command_invocation_surfaces_arguments(tmp_path: Path) -> None:
    # `/skill the args` (Claude Code): the trailing text reaches the skill as arguments, surfaced
    # ahead of the body so a skill can branch on a sub-command. The human CLI invoke_skill path.
    _write_claude_skill(tmp_path, "looper", "Loop body.")
    agent = _scripted_agent(tmp_path)
    result = await agent.invoke_skill("looper", "loop")
    assert result.invoked
    injected = agent.session.messages[-1].text
    assert "[arguments: loop]" in injected and "loop body" in injected.lower()


async def test_slash_command_composes_an_immediate_turn(tmp_path: Path) -> None:
    # Claude Code parity: typing `/looper loop` RUNS the skill — compose_skill_turn returns
    # the text the CLI executes as THIS turn (no second "describe your task" message), and
    # composing must not itself touch the session, or the body would double-inject when the
    # turn runs. This is the fix for the live 2026-08-19 report: `/start sera` loaded the
    # skill and then sat at the prompt waiting for another message.
    _write_claude_skill(tmp_path, "looper", "Loop body.")
    agent = _scripted_agent(tmp_path)
    before = len(agent.session.messages)
    result = await agent.compose_skill_turn("looper", "loop")
    assert result.invoked and result.turn_text is not None
    assert result.turn_text.startswith("[skill: looper]")
    assert "[arguments: loop]" in result.turn_text and "loop body" in result.turn_text.lower()
    assert len(agent.session.messages) == before


async def test_use_skill_tool_passes_arguments_to_the_body(tmp_path: Path) -> None:
    # The model-facing counterpart: use_skill accepts `args` and forwards them, so chaining like
    # use_skill(name, args='loop') carries the sub-command through to the body it returns.
    _write_claude_skill(tmp_path, "looper", "Loop body.")
    agent = _scripted_agent(tmp_path)
    tool = agent.registry.get("use_skill")
    assert tool is not None
    ctx = ToolContext(workspace_root=tmp_path, skill_resolver=agent.loop._skill_resolver)
    res = await tool.execute({"name": "looper", "args": "loop"}, ctx)
    assert res.is_error is False
    assert "[arguments: loop]" in res.output


async def test_skill_without_arguments_has_no_frame(tmp_path: Path) -> None:
    # No args (and whitespace-only args) must not emit a stray empty `[arguments: …]` frame.
    _write_claude_skill(tmp_path, "plain", "Plain body.")
    agent = _scripted_agent(tmp_path)
    result = await agent.invoke_skill("plain", "   ")  # whitespace-only → treated as no args
    assert result.invoked
    injected = agent.session.messages[-1].text
    assert "[arguments:" not in injected and "plain body" in injected.lower()


async def test_user_invocable_false_blocks_human_path_not_model_chaining(tmp_path: Path) -> None:
    # Claude Code's `user-invocable: false` marks an internal skill: reachable by another skill
    # chaining to it (the model's use_skill), but NOT by a human typing /<name>. Honor both sides.
    _write_claude_skill(tmp_path, "boot", "Boot body.", frontmatter="user-invocable: false")
    agent = _scripted_agent(tmp_path)
    before = len(agent.session.messages)
    # Human `/boot` is refused, and the session is left untouched (nothing injected).
    refused = await agent.invoke_skill("boot")
    assert refused.denied_reason is not None and "user-invocable" in refused.denied_reason
    assert len(agent.session.messages) == before
    # The model may still reach it via use_skill (internal chaining is allowed).
    tool = agent.registry.get("use_skill")
    assert tool is not None
    ctx = ToolContext(workspace_root=tmp_path, skill_resolver=agent.loop._skill_resolver)
    res = await tool.execute({"name": "boot"}, ctx)
    assert res.is_error is False and "boot body" in res.output.lower()


# ===========================================================================
# Lifecycle — Claude Code SessionStart `source` + PreCompact `trigger`
# ===========================================================================


def test_lifecycle_payload_surfaces_source_and_trigger_at_top_level() -> None:
    # Claude Code puts SessionStart `source` and PreCompact `trigger` at the stdin TOP LEVEL (not
    # nested) — a faithful host's lifecycle payload must serialize them there for shell hooks.
    start = LifecyclePayload(event=HookEvent.SESSION_START, source="resume").model_dump()
    assert start["source"] == "resume"
    compact = LifecyclePayload(event=HookEvent.PRE_COMPACT, trigger="auto").model_dump()
    assert compact["trigger"] == "auto"


async def test_sessionstart_source_distinguishes_fresh_from_resumed(tmp_path: Path) -> None:
    # SessionStart fires `source="startup"` for a fresh session and `source="resume"` for one that
    # already carries history — the signal a framework branches on (prime fresh vs reconcile).
    fresh = _scripted_agent(tmp_path)
    seen: list[str] = []
    fresh.hook_manager.register_lifecycle(HookEvent.SESSION_START, lambda p: seen.append(p.source))
    await fresh.loop._fire_session_start_once()
    assert seen == ["startup"]

    resumed = _scripted_agent(tmp_path)
    resumed.session.add_message(Message.user("an earlier turn"))
    seen2: list[str] = []
    resumed.hook_manager.register_lifecycle(
        HookEvent.SESSION_START, lambda p: seen2.append(p.source)
    )
    await resumed.loop._fire_session_start_once()
    assert seen2 == ["resume"]


# ===========================================================================
# Transcript — Claude Code `.jsonl` view at `transcript_path`
# ===========================================================================


async def test_turn_end_materializes_a_readable_cc_transcript(tmp_path: Path) -> None:
    # A faithful host exposes a Claude-Code-shaped transcript at `transcript_path` for hooks that
    # read the full history (e.g. a Stop hook). The materialized file parses as CC `.jsonl`, with
    # assistant text findable the way a CC Stop hook reads it (type=="assistant", message.content).
    agent = _scripted_agent(tmp_path)
    agent.session.add_message(Message.user("hello there"))
    agent.session.add_message(Message.assistant_text("hi back"))
    path = agent.loop._cc_transcript_path()
    assert path and Path(path).exists()
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assistant = [e for e in events if e.get("type") == "assistant"]
    assert assistant and any(
        block.get("text") == "hi back"
        for e in assistant
        for block in e["message"]["content"]
        if isinstance(block, dict)
    )


# ===========================================================================
# Status line — the Claude Code `statusLine` command contract
# ===========================================================================


def test_status_line_command_is_read_from_settings_json(tmp_path: Path) -> None:
    # A faithful host reads CC's `statusLine` object from .claude/settings.json and runs the
    # command after a turn with a status JSON on stdin, rendering its first stdout line. Generic
    # (no plug-in named): a configured `{type: command, command: ...}` yields a runnable spec.
    from zakcode.status_line import load_status_line_spec

    _write(
        tmp_path / ".claude" / "settings.json",
        json.dumps({"statusLine": {"type": "command", "command": "bash status.sh"}}),
    )
    spec, errors = load_status_line_spec(tmp_path)
    assert spec is not None and "status.sh" in " ".join(spec.command)
    assert not errors  # a clean type:command gesture maps without warning


# ===========================================================================
# Output styles — the Claude Code `outputStyle` + `.claude/output-styles/<name>.md` contract
# ===========================================================================


def test_active_output_style_body_is_loaded_from_settings(tmp_path: Path) -> None:
    # A faithful host reads CC's `outputStyle` name from .claude/settings.json and loads the
    # body from .claude/output-styles/<name>.md as a block to fold into the system prompt.
    # Generic (no plug-in named): a configured style yields a framed, injectable block.
    from zakcode.output_styles import load_active_output_style

    _write(tmp_path / ".claude" / "settings.json", json.dumps({"outputStyle": "terse"}))
    _write(tmp_path / ".claude" / "output-styles" / "terse.md", "Answer tersely.")
    block, reason = load_active_output_style(tmp_path)
    assert block is not None and "Answer tersely." in block
    assert reason is None  # a clean selection + body maps without a reason


def test_output_style_is_off_by_default(tmp_path: Path) -> None:
    # A workspace may carry ANOTHER runtime's output-style config; without the opt-in flag a
    # faithful host must NOT reshape its prompt from it. The configured style is inert when off.
    from zakcode.agent import DYNAMIC_BOUNDARY

    _write(tmp_path / ".claude" / "settings.json", json.dumps({"outputStyle": "terse"}))
    _write(tmp_path / ".claude" / "output-styles" / "terse.md", "OFF_DEFAULT_MARKER")
    agent = Agent(
        settings=Settings(default_model="scripted/test", workspace_root=tmp_path),
    )
    prompt = agent.loop.prompt_builder.build(agent.settings)
    assert "OFF_DEFAULT_MARKER" not in prompt
    assert DYNAMIC_BOUNDARY in prompt  # a normal prompt, just without the style


# ===========================================================================
# Permissions — the Claude Code `permissions.{allow,deny}` Tool(glob) contract
# ===========================================================================


def test_settings_permissions_deny_path_glob_is_translated_to_a_protected_path(
    tmp_path: Path,
) -> None:
    # A faithful host reads CC's permissions.deny path gestures and protects those paths. Here a
    # generic ``Write(*/state/*)`` deny must yield a protected-path rule (a write there can never
    # silently auto-allow). Translation only — enforcement is the policy's existing floor.
    from zakcode.permissions_settings import load_settings_permissions

    _write(
        tmp_path / ".claude" / "settings.json",
        json.dumps({"permissions": {"deny": ["Write(*/state/*)", "Bash(git push --force*)"]}}),
    )
    ingested, errors = load_settings_permissions(tmp_path)
    assert ingested.protected_path_regexes  # the Write path-glob
    assert ingested.denied_command_regexes  # the Bash command-glob
    assert not errors  # both gestures mapped cleanly


async def test_settings_permissions_ingestion_is_tighten_only(tmp_path: Path) -> None:
    # THE host promise for permissions: ingesting a framework's config can only ever TIGHTEN. Even
    # with the most permissive gesture (``allow: ["Bash(*)"]``) ingested, the never-waivable
    # catastrophic floor still denies ``rm -rf /``. (Generic — no plug-in named.)
    _write(
        tmp_path / ".claude" / "settings.json",
        json.dumps({"permissions": {"allow": ["Bash(*)"]}}),
    )
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", workspace_root=tmp_path, permission_mode="ask"
        ),
        enable_settings_permissions=True,
    )
    bash = agent.registry.get("bash")
    assert bash is not None
    allowed, _reason = await agent.permission_policy.authorize(bash.spec, {"command": "rm -rf /"})
    assert allowed is False  # the floor outranks the ingested allow


def test_settings_permissions_off_by_default(tmp_path: Path) -> None:
    # A workspace may carry ANOTHER runtime's permission config; without the opt-in flag a faithful
    # host must NOT reshape its posture from it. A bare ``deny: ["Bash"]`` is inert when off.
    _write(
        tmp_path / ".claude" / "settings.json",
        json.dumps({"permissions": {"deny": ["Bash"]}}),
    )
    agent = Agent(
        settings=Settings(
            default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
        ),
    )
    assert "bash" not in agent.permission_policy.tool_mode_overrides
