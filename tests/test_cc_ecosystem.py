"""Ecosystem proof: a COMPLETE, GENERIC Claude-Code plug-in runs on Zak Code unmodified.

The conformance suite (test_cc_conformance.py) proves each contract piece in isolation. THIS proves
the bonus the whole effort is for — assemble a self-contained Claude-Code plug-in (a skill with a
slash trigger, a settings.json with a Stop hook + permission denies, an output style, an always-on
rule), none of it Zak-Code- or claude-mind-specific, and show every piece works together on ONE
Agent. If this passes, anything built for Claude Code's extension surface rides in the same way.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode import Agent
from zakcode.config import PermissionTier, Settings
from zakcode.permissions import PermissionDecision


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _lay_down_plugin(ws: Path) -> None:
    """A complete, generic Claude-Code plug-in under <ws>/.claude (names no framework)."""
    # A skill reachable by a slash trigger (/hi -> greeter).
    _w(
        ws / ".claude" / "skills" / "greeter" / "SKILL.md",
        "---\nname: greeter\ndescription: greet the user\ntriggers: [/hi]\n---\nSay hello warmly.",
    )
    # An always-on rule and a named output style.
    _w(ws / ".claude" / "rules" / "house.md", "Always be concise.")
    _w(ws / ".claude" / "output-styles" / "terse.md", "Respond in as few words as possible.")
    # settings.json: a Stop hook + permission denies + the active output style.
    _w(
        ws / ".claude" / "settings.json",
        json.dumps(
            {
                "permissions": {"deny": ["Bash(rm -rf*)", "WebFetch"]},
                "outputStyle": "terse",
                "hooks": {
                    "Stop": [{"hooks": [{"type": "command", "command": "bash keepgoing.sh"}]}]
                },
            }
        ),
    )


def _full_agent(ws: Path) -> Agent:
    # Every CC-compat seam on — the way an operator hosting a CC plug-in would run it.
    return Agent(
        settings=Settings(
            default_model="scripted/test", workspace_root=ws, permission_mode="allow"
        ),
        enable_skills=True,
        enable_rules=True,
        enable_settings_hooks=True,
        enable_settings_permissions=True,
        enable_output_style=True,
    )


class _Spec:
    """Minimal ToolSpec stand-in (decide() reads only .name + .required_permission)."""

    def __init__(self, name: str, tier: PermissionTier) -> None:
        self.name = name
        self.required_permission = tier


def test_a_complete_cc_plugin_works_end_to_end(tmp_path: Path) -> None:
    _lay_down_plugin(tmp_path)
    agent = _full_agent(tmp_path)
    policy = agent.permission_policy
    bash, web = PermissionTier.DANGER_FULL_ACCESS, PermissionTier.READ_ONLY

    # 1) SKILL — discoverable AND routable by its slash trigger (/hi, not the name).
    assert agent.skill_registry is not None
    greeter = agent.skill_registry.resolve("hi")
    assert greeter is not None and greeter.name == "greeter"

    # 2) HOOK — the settings.json Stop hook loaded as a TURN_END hook.
    assert agent.hook_manager.has_turn_end_hooks()

    # 3) PERMISSIONS — the denies bind: a read-only tool (WebFetch) is denied UNCONDITIONALLY, the
    #    ingested command rule escalates `rm -rf`, while a benign command still auto-allows.
    assert policy.decide(_Spec("web_fetch", web), {"url": "http://x"})[0] is PermissionDecision.DENY
    assert policy.decide(_Spec("bash", bash), {"command": "ls -la"})[0] is PermissionDecision.ALLOW
    assert policy.decide(_Spec("bash", bash), {"command": "rm -rf /tmp/x"})[0] is not (
        PermissionDecision.ALLOW
    )

    # 4) PROMPT — the always-on rule AND the active output style are both in the system prompt.
    prompt = agent.loop.prompt_builder.build(agent.settings)
    assert "be concise" in prompt.lower()  # the rule
    assert "few words" in prompt.lower()  # the output style
