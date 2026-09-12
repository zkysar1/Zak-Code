"""The conductor's endings route through the FRAMEWORK's graceful stop (ADR-0047).

`test_framework_stop.py` pins the CONTRACT (what a raise writes, in what order).
These pin the WIRING — that the run's two endings actually reach it — and, as the
control, that a workspace which is not a framework seed is left exactly as it was.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.framework_stop import SIGNAL_SET_SCRIPT, framework_session_dir
from zakcode.session.store import SessionStore

AGENT = "alpha"


def _plant_framework_seed(root: Path) -> None:
    """A minimal stand-in for the seed's own signal writer, at the real path."""
    script = root / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'dir="agents/${AYOAI_AGENT}/session"\n'
        '[ -f "$dir/stop-target-mode" ] || exit 9\n'
        'touch "$dir/$1"\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _app(tmp_path: Path, *, run_stop_agent: str | None) -> Any:
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_stop_agent=run_stop_agent,
    )
    return create_app(settings=settings, store=SessionStore(base_dir=tmp_path / "sessions"))


def test_run_stop_raises_the_minds_own_graceful_stop(tmp_path: Path) -> None:
    _plant_framework_seed(tmp_path)
    client = TestClient(_app(tmp_path, run_stop_agent=AGENT))

    response = client.post("/run/stop", json={"reason": "stopped"})

    assert response.status_code == 200, response.text
    session = framework_session_dir(tmp_path, AGENT)
    assert (session / "stop-requested").exists()
    assert (session / "stop-target-mode").read_text(encoding="utf-8") == "assistant"


def test_run_stop_without_a_stop_agent_changes_nothing(tmp_path: Path) -> None:
    """The control. Without it, a raise that fired unconditionally would look correct."""
    _plant_framework_seed(tmp_path)
    client = TestClient(_app(tmp_path, run_stop_agent=None))

    response = client.post("/run/stop", json={"reason": "stopped"})

    assert response.status_code == 200, response.text
    assert not (tmp_path / "agents").exists()


def test_run_stop_stays_idempotent_across_repeats(tmp_path: Path) -> None:
    _plant_framework_seed(tmp_path)
    client = TestClient(_app(tmp_path, run_stop_agent=AGENT))

    first = client.post("/run/stop", json={"reason": "stopped"})
    second = client.post("/run/stop", json={"reason": "other_reason"})

    assert first.status_code == 200 and second.status_code == 200
    # The first reason wins, and the signal is still exactly one stop.
    assert second.json()["reason"] == "stopped"
    assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()


def test_the_reserve_is_held_for_the_minds_own_ending(tmp_path: Path) -> None:
    """`run_stop_agent` earns the graceful-stop budget the digest message used to.

    Without this, the stop would be armed and the loop cancelled before the mind could
    consolidate — the severed ending, reintroduced one layer down.
    """
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_consolidation_reserve=30.0,
        run_stop_agent=AGENT,
    )
    app = create_app(settings=settings, store=SessionStore(base_dir=tmp_path / "sessions"))
    assert app.state.graceful_stop_budget() >= 30.0

    bare = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_consolidation_reserve=30.0,
    )
    bare_app = create_app(settings=bare, store=SessionStore(base_dir=tmp_path / "sessions"))
    assert bare_app.state.graceful_stop_budget() == 0.0
