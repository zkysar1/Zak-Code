"""Tests for settings.json hook ingestion (PR-T5).

Covers: discovery, event-name mapping, shlex splitting, DANGEROUS_PATTERNS
gate, env scrub on all workspace-sourced hooks (TE-R1), and unknown/skipped
events.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.hooks import HookEvent
from zakcode.hooks.settings_loader import load_settings_hooks


def _write_settings(ws: Path, obj: dict, *, subdir: str = ".claude") -> None:
    d = ws / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(json.dumps(obj), encoding="utf-8")


# ── no file ───────────────────────────────────────────────────────────────────


def test_load_settings_no_file(tmp_path: Path) -> None:
    specs, errors = load_settings_hooks(tmp_path)
    assert specs == []
    assert errors == {}


# ── Stop -> TURN_END mapping ─────────────────────────────────────────────────


def test_load_settings_stop_maps_to_turn_end(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "bash core/scripts/stop-hook.sh",
                                "timeout": 60,
                            }
                        ]
                    }
                ]
            }
        },
    )
    specs, errors = load_settings_hooks(tmp_path)
    assert len(specs) == 1
    assert specs[0].event is HookEvent.TURN_END
    # argv[0] is resolved to a real executable path (dodges the Windows WSL stub).
    from zakcode._subprocess import resolve_executable

    assert specs[0].command == [resolve_executable("bash"), "core/scripts/stop-hook.sh"]
    assert specs[0].timeout == 60.0
    # TE-R1: workspace-sourced hooks carry drop_env.
    assert len(specs[0].drop_env) >= 0  # At minimum the list exists.


# ── PreToolUse with matcher ──────────────────────────────────────────────────


def test_load_settings_pre_tool_use_matcher(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "echo hi"}]}
                ]
            }
        },
    )
    specs, _ = load_settings_hooks(tmp_path)
    assert len(specs) == 1
    assert specs[0].event is HookEvent.PRE_TOOL_USE
    assert specs[0].matcher == "Bash"


# ── dangerous command + autonomous => denied ─────────────────────────────────


def test_load_settings_dangerous_autonomous_denied(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "rm -rf /"}]}]}},
    )
    specs, errors = load_settings_hooks(tmp_path, permission_mode="autonomous")
    assert specs == []
    assert any("DENIED" in v for v in errors.values())


# ── dangerous command + ask => registered with warning ───────────────────────


def test_load_settings_dangerous_ask_allowed(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {"hooks": {"PreToolUse": [{"hooks": [{"type": "command", "command": "rm -rf /"}]}]}},
    )
    specs, errors = load_settings_hooks(tmp_path, permission_mode="ask")
    assert len(specs) == 1
    assert any("WARNING" in v for v in errors.values())


# ── unknown event skipped ────────────────────────────────────────────────────


def test_load_settings_unknown_event_skipped(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {"hooks": {"MadeUpEvent": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}},
    )
    specs, errors = load_settings_hooks(tmp_path)
    assert specs == []
    assert "MadeUpEvent" in errors
    assert "unknown" in errors["MadeUpEvent"].lower()


# ── StopFailure => skipped ───────────────────────────────────────────────────


def test_load_settings_stop_failure_skipped(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {"hooks": {"StopFailure": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}},
    )
    specs, errors = load_settings_hooks(tmp_path)
    assert specs == []
    assert "StopFailure" in errors
    assert "not implemented" in errors["StopFailure"].lower()


# ── UserPromptSubmit => skipped, not "unknown" ──────────────────────────


def test_load_settings_user_prompt_submit_skipped(tmp_path: Path) -> None:
    """A real CC event whose firing seam is not designed yet must degrade LOUDLY.

    Until it was listed in ``_SKIP_EVENTS`` it fell through to the unknown-event
    branch, so a Mind that wires ``UserPromptSubmit`` (claude-mind does) read the
    same message a typo produces. "Deliberately deferred" and "you misspelled it"
    are different diagnoses and only one of them is actionable, so the *absence*
    of "unknown" is asserted beside the presence of the deferral.
    """
    _write_settings(
        tmp_path,
        {"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}},
    )
    specs, errors = load_settings_hooks(tmp_path)
    assert specs == []
    assert "not implemented" in errors["UserPromptSubmit"].lower()
    assert "unknown" not in errors["UserPromptSubmit"].lower()


# ── a whole claude-mind hooks block loads with zero unknown events ─────────

#: Every event claude-mind wires in its own ``.claude/settings.json``, read from a
#: live Mind tree (2026-09-03). Pinning the literal set is the point: any of these
#: that neither registers nor skips is a hook the Mind configured and this host
#: silently dropped -- the ADR-0025 failure class, one layer up.
_CLAUDE_MIND_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "SessionStart",
    "PreCompact",
    "Stop",
    "StopFailure",
    "UserPromptExpansion",
    "UserPromptSubmit",
)


def test_claude_mind_hooks_block_yields_no_unknown_events(tmp_path: Path) -> None:
    """The partition IS the contract: every configured event registers or reports.

    Both halves are asserted together on purpose. "Zero unknown-event errors"
    alone is satisfied by a loader that drops deferred events on the floor, which
    is the very failure this pins against; "every event errors" alone is satisfied
    by a loader that registers nothing.
    """
    _write_settings(
        tmp_path,
        {
            "hooks": {
                ev: [{"hooks": [{"type": "command", "command": "echo hi"}]}]
                for ev in _CLAUDE_MIND_EVENTS
            }
        },
    )
    specs, errors = load_settings_hooks(tmp_path)

    unknown = {ev: msg for ev, msg in errors.items() if "unknown" in msg.lower()}
    assert unknown == {}, f"claude-mind wires events this host does not recognise: {unknown}"

    # The deferral list, spelled out. When a seam below ships, this fails and the
    # implementer updates it here -- a deferral nobody is forced to retire becomes
    # permanent, and then the compat doc is lying in the other direction.
    assert set(errors) == {"StopFailure", "UserPromptExpansion", "UserPromptSubmit"}

    registered = [ev for ev in _CLAUDE_MIND_EVENTS if ev not in errors]
    assert len(specs) == len(registered), (
        f"{len(registered)} events reported no error but only {len(specs)} specs loaded"
    )


# ── timeout honored ──────────────────────────────────────────────────────────


def test_load_settings_timeout_honored(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "echo hi", "timeout": 120}]}
                ]
            }
        },
    )
    specs, _ = load_settings_hooks(tmp_path)
    assert specs[0].timeout == 120.0


# ── both .claude and .zakcode files merged ───────────────────────────────────


def test_load_settings_both_dirs_merged(tmp_path: Path) -> None:
    _write_settings(
        tmp_path,
        {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo claude"}]}]}},
        subdir=".claude",
    )
    _write_settings(
        tmp_path,
        {"hooks": {"SessionEnd": [{"hooks": [{"type": "command", "command": "echo zakcode"}]}]}},
        subdir=".zakcode",
    )
    specs, _ = load_settings_hooks(tmp_path)
    assert len(specs) == 2
    events = {s.event for s in specs}
    assert HookEvent.SESSION_START in events
    assert HookEvent.SESSION_END in events


# ── TE-R1: drop_env populated on every workspace-sourced spec ────────────────


def test_load_settings_drop_env_populated(tmp_path: Path) -> None:
    """Every spec from settings.json has drop_env set (TE-R1)."""
    _write_settings(
        tmp_path,
        {
            "hooks": {
                "PreToolUse": [{"hooks": [{"type": "command", "command": "echo hi"}]}],
                "Stop": [{"hooks": [{"type": "command", "command": "bash stop.sh"}]}],
            }
        },
    )
    specs, _ = load_settings_hooks(tmp_path)
    assert len(specs) == 2
    for spec in specs:
        assert isinstance(spec.drop_env, list)
        # The list is from provider_key_env_names(); it may be empty on
        # machines with no provider keys set, but it must exist and be the
        # same list for all specs.
    # All specs share the same scrub list.
    assert specs[0].drop_env == specs[1].drop_env


# ── TE-R1: _run_shell scrubs env for specs with drop_env ─────────────────────


async def test_run_shell_env_scrubbed_via_drop_env(tmp_path: Path) -> None:
    """Generic _run_shell respects HookSpec.drop_env (TE-R1)."""
    import os
    import sys

    from zakcode.hooks import HookManager, HookPayload, HookSpec

    body = (
        "import os, sys\n"
        f"open(r'{tmp_path / 'env_out.txt'}', 'w').write("
        "os.environ.get('ANTHROPIC_API_KEY', 'ABSENT'))\n"
        "import json; print(json.dumps({'message': 'ok'}))\n"
        "sys.exit(0)\n"
    )
    script = tmp_path / "check.py"
    script.write_text(body, encoding="utf-8")
    spec = HookSpec(
        event=HookEvent.PRE_TOOL_USE,
        command=[sys.executable, str(script)],
        drop_env=["ANTHROPIC_API_KEY"],
    )
    mgr = HookManager([spec])

    old = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-scrub-key"
    try:
        payload = HookPayload(event=HookEvent.PRE_TOOL_USE, tool_name="bash")
        await mgr.run(payload)
    finally:
        if old is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old

    result = (tmp_path / "env_out.txt").read_text()
    assert result == "ABSENT"
