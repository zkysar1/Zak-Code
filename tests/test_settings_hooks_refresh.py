"""ADR-0079: settings.json hooks are re-read at the next turn when the file changes.

A Mind pulls framework updates by git while its sessions run for hours; a gate that
lands in ``.claude/settings.json`` mid-session must fire from the next turn on, not
after the next restart (measured 2026-08-29: a store-write guard promoted onto a live
deployment was invisible to all four running sessions).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from zakcode.hooks import HookEvent, HookManager, HookSpec
from zakcode.hooks.settings_loader import SettingsHooks, settings_hooks_signature


def _settings(*commands: str) -> dict:
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": c, "timeout": 5} for c in commands],
                }
            ]
        }
    }


def _write(ws: Path, obj: dict) -> Path:
    d = ws / ".claude"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "settings.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _bump_mtime(p: Path) -> None:
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def _programmatic() -> HookSpec:
    return HookSpec(event=HookEvent.PRE_TOOL_USE, command=["true"], matcher="*")


def test_unchanged_files_are_not_re_read(tmp_path: Path) -> None:
    _write(tmp_path, _settings("bash a.sh"))
    hooks = SettingsHooks(tmp_path, permission_mode="ask")
    specs, _ = hooks.load()
    manager = HookManager(shell_hooks=[_programmatic(), *specs])
    assert hooks.refresh(manager) == (False, {})
    assert len(manager.shell_hooks) == 2


def test_a_hook_added_to_settings_fires_after_refresh(tmp_path: Path) -> None:
    p = _write(tmp_path, _settings("bash a.sh"))
    hooks = SettingsHooks(tmp_path, permission_mode="ask")
    specs, _ = hooks.load()
    prog = _programmatic()
    manager = HookManager(shell_hooks=[prog, *specs])

    _write(tmp_path, _settings("bash a.sh", "bash b.sh"))
    _bump_mtime(p)
    changed, errs = hooks.refresh(manager)

    assert changed and errs == {}
    assert manager.shell_hooks[0] is prog  # programmatic hooks untouched, order kept
    assert [h.command[-1] for h in manager.shell_hooks[1:]] == ["a.sh", "b.sh"]
    assert len(manager.shell_hooks) == 3  # the old settings slice was REPLACED, not appended


def test_a_removed_settings_file_drops_only_its_hooks(tmp_path: Path) -> None:
    p = _write(tmp_path, _settings("bash a.sh"))
    hooks = SettingsHooks(tmp_path, permission_mode="ask")
    specs, _ = hooks.load()
    prog = _programmatic()
    manager = HookManager(shell_hooks=[prog, *specs])

    p.unlink()
    changed, _ = hooks.refresh(manager)

    assert changed
    assert manager.shell_hooks == [prog]


def test_a_broken_edit_keeps_the_previous_hooks(tmp_path: Path) -> None:
    p = _write(tmp_path, _settings("bash a.sh"))
    hooks = SettingsHooks(tmp_path, permission_mode="ask")
    specs, _ = hooks.load()
    manager = HookManager(shell_hooks=[*specs])

    p.write_text("{not json", encoding="utf-8")
    _bump_mtime(p)
    changed, errs = hooks.refresh(manager)

    assert not changed
    assert any("parse error" in e for e in errs.values())
    assert [h.command[-1] for h in manager.shell_hooks] == ["a.sh"]  # gates never stripped
    # The broken file is not re-parsed every turn; a later good edit is picked up.
    assert hooks.refresh(manager) == (False, {})
    _write(tmp_path, _settings("bash c.sh"))
    _bump_mtime(p)
    changed, errs = hooks.refresh(manager)
    assert changed and errs == {}
    assert [h.command[-1] for h in manager.shell_hooks] == ["c.sh"]


def test_signature_covers_every_candidate_file(tmp_path: Path) -> None:
    before = settings_hooks_signature(tmp_path)
    assert before == ()
    _write(tmp_path, _settings("bash a.sh"))
    (tmp_path / ".zakcode").mkdir()
    (tmp_path / ".zakcode" / "settings.json").write_text("{}", encoding="utf-8")
    after = settings_hooks_signature(tmp_path)
    assert [Path(p).name for p, *_ in after] == ["settings.json", "settings.json"]
    assert after != before
