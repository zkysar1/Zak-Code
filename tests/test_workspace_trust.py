"""Folder-trust adoption for workspace settings.json hooks (``zakcode.workspace_trust``).

The policy under test: the compatibility surfaces stay OFF by default at the library layer,
but an interactive host resolves an UNSET ``settings_hooks`` by asking the operator once per
workspace and remembering the answer — Claude Code folder-trust semantics. The decision
logic and persistence live in core; the CLI only renders (see ``_ask_hooks_adoption``).
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.cli import _parse_adoption_answer
from zakcode.hooks.settings_loader import summarize_settings_hooks
from zakcode.workspace_trust import (
    ADOPT_ALWAYS,
    ADOPT_NEVER,
    ADOPT_SESSION,
    hooks_decision,
    remember_hooks_decision,
    resolve_hooks_adoption,
)

# ── detection: summarize_settings_hooks ─────────────────────────────────────────


def _write_settings(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hook_entry(*commands: str) -> list[dict]:
    return [{"matcher": "*", "hooks": [{"type": "command", "command": c} for c in commands]}]


def test_summarize_counts_loadable_hooks_across_files(tmp_path: Path) -> None:
    _write_settings(
        tmp_path / ".claude" / "settings.json",
        {"hooks": {"Stop": _hook_entry("echo stop"), "PreToolUse": _hook_entry("a", "b")}},
    )
    _write_settings(
        tmp_path / ".claude" / "settings.local.json",
        {"hooks": {"SessionStart": _hook_entry("echo boot")}},
    )
    summary = summarize_settings_hooks(tmp_path)
    assert summary == {"PreToolUse": 2, "SessionStart": 1, "Stop": 1}
    # Order is the event-map declaration order (stable for rendering), not file order.
    assert list(summary) == ["PreToolUse", "SessionStart", "Stop"]


def test_summarize_skips_unloadable_events_and_malformed_entries(tmp_path: Path) -> None:
    _write_settings(
        tmp_path / ".claude" / "settings.json",
        {
            "hooks": {
                "StopFailure": _hook_entry("never loads"),  # recognised but unimplemented
                "NoSuchEvent": _hook_entry("unknown"),
                "Stop": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": "   "}]},  # blank
                    {"matcher": "*", "hooks": [{"type": "prompt", "command": "x"}]},  # not command
                ],
            }
        },
    )
    # A workspace whose hooks would all load to ZERO specs must not trigger the question.
    assert summarize_settings_hooks(tmp_path) == {}


def test_summarize_empty_and_corrupt_fail_open(tmp_path: Path) -> None:
    assert summarize_settings_hooks(tmp_path) == {}  # nothing declared
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text("{not json", encoding="utf-8")
    assert summarize_settings_hooks(tmp_path) == {}  # parse error → nothing to adopt


# ── persistence: the per-workspace decision store ────────────────────────────────


def test_store_roundtrip_always_and_never(tmp_path: Path) -> None:
    store = tmp_path / "trust.json"
    ws = tmp_path / "ws"
    ws.mkdir()
    assert hooks_decision(ws, store_path=store) is None
    remember_hooks_decision(ws, ADOPT_ALWAYS, store_path=store)
    assert hooks_decision(ws, store_path=store) == ADOPT_ALWAYS
    remember_hooks_decision(ws, ADOPT_NEVER, store_path=store)
    assert hooks_decision(ws, store_path=store) == ADOPT_NEVER


def test_store_session_answer_is_never_persisted(tmp_path: Path) -> None:
    store = tmp_path / "trust.json"
    remember_hooks_decision(tmp_path, ADOPT_SESSION, store_path=store)
    assert not store.exists()
    assert hooks_decision(tmp_path, store_path=store) is None


def test_store_corrupt_fails_open_to_ask_again(tmp_path: Path) -> None:
    store = tmp_path / "trust.json"
    store.write_text("[1, 2, 3]", encoding="utf-8")  # wrong shape
    assert hooks_decision(tmp_path, store_path=store) is None
    store.write_text("{broken", encoding="utf-8")  # not json
    assert hooks_decision(tmp_path, store_path=store) is None
    # And a write over the corrupt file recovers it rather than crashing.
    remember_hooks_decision(tmp_path, ADOPT_ALWAYS, store_path=store)
    assert hooks_decision(tmp_path, store_path=store) == ADOPT_ALWAYS


# ── policy: resolve_hooks_adoption ───────────────────────────────────────────────


def _no_ask(_summary: dict[str, int]) -> str | None:
    raise AssertionError("ask must not be called on this path")


SUMMARY = {"Stop": 1, "PreToolUse": 2}


def test_explicit_setting_short_circuits_everything() -> None:
    # An operator's ZAKCODE_SETTINGS_HOOKS answer is global and silent — no ask, no notice.
    on = resolve_hooks_adoption(
        configured=True, summary=SUMMARY, decision="never", interactive=True, ask=_no_ask
    )
    assert on.enable is True and on.remember is None and on.notice is None
    off = resolve_hooks_adoption(
        configured=False, summary=SUMMARY, decision="always", interactive=True, ask=_no_ask
    )
    assert off.enable is False and off.remember is None and off.notice is None


def test_nothing_declared_stays_silent() -> None:
    r = resolve_hooks_adoption(
        configured=None, summary={}, decision=None, interactive=True, ask=_no_ask
    )
    assert r.enable is False and r.remember is None and r.notice is None


def test_remembered_always_loads_without_asking() -> None:
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=ADOPT_ALWAYS, interactive=True, ask=_no_ask
    )
    assert r.enable is True and r.remember is None
    assert r.notice is not None and "trusted workspace" in r.notice


def test_remembered_never_stays_off_but_says_so() -> None:
    # The original defect was SILENCE — even a remembered "never" must leave one line.
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=ADOPT_NEVER, interactive=True, ask=_no_ask
    )
    assert r.enable is False and r.remember is None
    assert r.notice is not None and "off for this workspace" in r.notice


def test_non_interactive_never_prompts_and_points_at_the_lever() -> None:
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=None, interactive=False, ask=_no_ask
    )
    assert r.enable is False and r.remember is None
    assert r.notice is not None and "ZAKCODE_SETTINGS_HOOKS=1" in r.notice


def test_interactive_answers_map_to_outcomes() -> None:
    def ask_returning(value: str | None):
        calls: list[dict[str, int]] = []

        def _ask(summary: dict[str, int]) -> str | None:
            calls.append(summary)
            return value

        return _ask, calls

    ask, calls = ask_returning(ADOPT_ALWAYS)
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=None, interactive=True, ask=ask
    )
    assert calls == [SUMMARY]  # asked exactly once, with the summary
    assert r.enable is True and r.remember == ADOPT_ALWAYS

    ask, _ = ask_returning(ADOPT_SESSION)
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=None, interactive=True, ask=ask
    )
    assert r.enable is True and r.remember is None  # loads now, nothing persisted

    ask, _ = ask_returning(ADOPT_NEVER)
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=None, interactive=True, ask=ask
    )
    assert r.enable is False and r.remember == ADOPT_NEVER

    ask, _ = ask_returning(None)  # dismissed (EOF / no clear answer)
    r = resolve_hooks_adoption(
        configured=None, summary=SUMMARY, decision=None, interactive=True, ask=ask
    )
    assert r.enable is False and r.remember is None  # off, and asked again next session


# ── the CLI's answer parser (the only UI logic worth pinning) ────────────────────


def test_parse_adoption_answers() -> None:
    for raw, expected in [
        ("1", ADOPT_ALWAYS),
        ("y", ADOPT_ALWAYS),
        (" YES ", ADOPT_ALWAYS),
        ("2", ADOPT_SESSION),
        ("o", ADOPT_SESSION),
        ("session", ADOPT_SESSION),
        ("3", ADOPT_NEVER),
        ("N", ADOPT_NEVER),
        ("never", ADOPT_NEVER),
        ("", None),
        ("maybe", None),
    ]:
        assert _parse_adoption_answer(raw) == expected, raw
