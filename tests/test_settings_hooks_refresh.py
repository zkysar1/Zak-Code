"""ADR-0079: settings.json hooks are re-read before every model call when the file changes.

A host framework pulls updates by git while its sessions run for hours; a gate that
lands in ``.claude/settings.json`` mid-session must fire from the next model call on, not
after the next restart (measured 2026-08-29: a store-write guard promoted onto a live
deployment was invisible to all four running sessions).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from zakcode import Agent
from zakcode.evals.harness import ScriptedProvider, call_tool, reply
from zakcode.hooks import HookEvent, HookManager, HookPayload, HookSpec, settings_loader
from zakcode.hooks.settings_loader import SettingsHooks, settings_hooks_signature
from zakcode.permissions import PermissionPolicy


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


#: Two ways an edit leaves a settings file unreadable: broken JSON, and bytes that are not
#: UTF-8 (an editor saving in another encoding).
BROKEN = pytest.mark.parametrize(
    "broken", [b"{not json", b'{"hooks": {}}\xff'], ids=["not-json", "not-utf8"]
)


@BROKEN
def test_a_broken_edit_keeps_the_previous_hooks(tmp_path: Path, broken: bytes) -> None:
    p = _write(tmp_path, _settings("bash a.sh"))
    hooks = SettingsHooks(tmp_path, permission_mode="ask")
    specs, _ = hooks.load()
    manager = HookManager(shell_hooks=[*specs])

    p.write_bytes(broken)
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


def test_a_file_that_can_no_longer_be_looked_at_keeps_the_previous_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Its directory can no longer be entered: every stat of the file is refused, so the
    # signature drops it and the refresh re-reads, and there Path.is_file() raises
    # PermissionError. That must be reported like a broken edit, never raised, because the
    # refresh runs before every model call of a turn that may have run for hours.
    p = _write(tmp_path, _settings("bash a.sh"))
    hooks = SettingsHooks(tmp_path, permission_mode="ask")
    specs, _ = hooks.load()
    manager = HookManager(shell_hooks=[*specs])
    real_stat = Path.stat

    def stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        if self == p:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", stat)
    changed, errs = hooks.refresh(manager)

    assert not changed
    assert list(errs) == [str(p)] and errs[str(p)].startswith("parse error")
    assert [h.command[-1] for h in manager.shell_hooks] == ["a.sh"]  # gates never stripped


def test_signature_covers_every_candidate_file(tmp_path: Path) -> None:
    before = settings_hooks_signature(tmp_path)
    assert before == ()
    _write(tmp_path, _settings("bash a.sh"))
    (tmp_path / ".zakcode").mkdir()
    (tmp_path / ".zakcode" / "settings.json").write_text("{}", encoding="utf-8")
    after = settings_hooks_signature(tmp_path)
    assert [Path(p).name for p, *_ in after] == ["settings.json", "settings.json"]
    assert after != before


# ── mid-turn: the main loop re-reads before every model call ─────────────────────
#
# A served session runs ONE turn for many hours, so a hook a pull lands mid-turn must gate
# the next tool call of that same turn. Claude Code does this (measured 2026-09-30 on
# versions 2.1.283 and 2.1.285): a PreToolUse hook added while a turn runs, by the model's
# own tool call or by another process, fires on the turn's next tool call.


def _gate_every_tool(*scripts: str) -> dict:
    """Settings whose PreToolUse hooks match every tool, so a cheap read-only call trips them."""
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": f"bash {s}"} for s in scripts],
                }
            ]
        }
    }


class _Dispatches:
    """Records the script of every settings hook dispatched for a tool call, in order.

    It stands in for the hook subprocess (the runner has its own tests): what these tests pin is
    WHICH hooks the manager holds when a tool call is gated. ``during_first_call`` runs once,
    while the turn's first tool call is being gated, so the edit lands after that iteration's
    re-read and before the next model call.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, during_first_call: Callable[[], None]
    ) -> None:
        self.scripts: list[str] = []
        pending: list[Callable[[], None]] = [during_first_call]

        async def run_shell(manager: HookManager, spec: HookSpec, payload: HookPayload) -> None:
            if payload.event is HookEvent.PRE_TOOL_USE:
                self.scripts.append(spec.command[-1])
                if pending:
                    pending.pop()()
            return None

        monkeypatch.setattr(HookManager, "_run_shell", run_shell)


def _agent(workspace: Path, tool_calls: int) -> tuple[Agent, ScriptedProvider]:
    """A real Agent whose scripted model makes ``tool_calls`` read-only calls, then answers.

    Each call lists its own directory: the loop's stuck detection treats a run of identical
    calls as a loop and steps in, which would cut the script short.
    """
    dirs = [workspace / f"d{i}" for i in range(tool_calls)]
    for d in dirs:
        d.mkdir()
    provider = ScriptedProvider(
        script=[*(call_tool("LS", {"path": str(d)}) for d in dirs), reply("done")]
    )
    agent = Agent(
        provider=provider,
        permission_policy=PermissionPolicy("allow"),
        default_model="scripted/test",
        workspace_root=str(workspace),
        max_iterations=tool_calls + 2,
    )
    return agent, provider


def _count_refreshes_and_parses(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count re-checks (``SettingsHooks.refresh``) and full re-parses from here on."""
    counts = {"refresh": 0, "parse": 0}
    real_refresh = SettingsHooks.refresh
    real_load = settings_loader.load_settings_hooks

    def refresh(self: SettingsHooks, manager: HookManager) -> tuple[bool, dict[str, str]]:
        counts["refresh"] += 1
        return real_refresh(self, manager)

    def load(*args: object, **kwargs: object) -> tuple[list[HookSpec], dict[str, str]]:
        counts["parse"] += 1
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SettingsHooks, "refresh", refresh)
    monkeypatch.setattr(settings_loader, "load_settings_hooks", load)
    return counts


async def _one_turn(agent: Agent, path: str) -> None:
    if path == "buffered":
        await agent.arun_turn("go")
    else:
        async for _event in agent.astream_turn("go"):
            pass
    await agent.aclose()


#: The loop's two turn paths each re-read at the top of every iteration; pin both.
PATHS = pytest.mark.parametrize("path", ["buffered", "streaming"])


@PATHS
async def test_a_hook_added_mid_turn_fires_on_the_next_tool_call_of_the_same_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    settings = _write(tmp_path, _gate_every_tool("base.sh"))

    def add_a_hook() -> None:
        _write(tmp_path, _gate_every_tool("base.sh", "new.sh"))
        _bump_mtime(settings)

    hooks = _Dispatches(monkeypatch, during_first_call=add_a_hook)
    agent, _ = _agent(tmp_path, tool_calls=2)
    counts = _count_refreshes_and_parses(monkeypatch)
    await _one_turn(agent, path)
    # One turn, two tool calls: the hook written while the first call was gated fires on the
    # second. Re-read only at a turn's start, it would have waited for the next turn.
    assert hooks.scripts == ["base.sh", "base.sh", "new.sh"]
    assert counts["parse"] == 1  # one edit, one re-parse


@PATHS
async def test_an_unchanged_settings_file_is_only_stat_checked_before_each_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    _write(tmp_path, _gate_every_tool("base.sh"))
    hooks = _Dispatches(monkeypatch, during_first_call=lambda: None)
    agent, provider = _agent(tmp_path, tool_calls=3)
    counts = _count_refreshes_and_parses(monkeypatch)
    await _one_turn(agent, path)
    assert hooks.scripts == ["base.sh"] * 3
    assert provider.calls == 4  # three tool calls, then the answer
    # A re-check at the turn's start and one before each model call, each only the signature
    # stats: nothing changed, so nothing was parsed again.
    assert counts == {"refresh": 5, "parse": 0}


@BROKEN
async def test_a_broken_edit_mid_turn_keeps_the_previous_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    broken: bytes,
) -> None:
    settings = _write(tmp_path, _gate_every_tool("base.sh"))

    def break_the_file() -> None:
        settings.write_bytes(broken)
        _bump_mtime(settings)

    hooks = _Dispatches(monkeypatch, during_first_call=break_the_file)
    agent, _ = _agent(tmp_path, tool_calls=3)
    counts = _count_refreshes_and_parses(monkeypatch)
    with caplog.at_level(logging.WARNING):
        await _one_turn(agent, "buffered")
    assert hooks.scripts == ["base.sh"] * 3  # the gate is never stripped mid-turn
    # The broken file is read and reported once, not before every later call.
    assert counts["parse"] == 1
    reported = [r for r in caplog.records if "parse error" in r.getMessage()]
    assert len(reported) == 1
