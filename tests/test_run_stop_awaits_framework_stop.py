"""The run must keep BEATING while the mind runs its own graceful stop (g-373-16).

Raising a framework stop writes signals the mind reads at the TOP of its next beat.
Setting ``run_stopping`` in the same breath ended the loop before that beat could
happen, so the mind never read the signal it had just been sent. Measured on two live
dev vessels: ``stop-requested`` landed, ``stop-loop`` and ``handoff.yaml`` never
appeared, and the vessel was killed at grace expiry with no consolidation. The served
process IS the mind's only reader, so stopping the beat removes the reader.

Each test below fails against the pre-fix loop, and the first one fails LOUDLY: the
run ends while the mind is still mid-stop.
"""

from __future__ import annotations

import asyncio
import stat
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import create_app
from zakcode.session.framework_stop import (
    SIGNAL_SET_SCRIPT,
    STOP_LOOP_SIGNAL,
    framework_session_dir,
    framework_stop_complete,
)
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

AGENT = "probe"


def _plant_setter(root: Path) -> None:
    """Plant a faithful stand-in for the framework's ``session-signal-set.sh``."""
    script = root / SIGNAL_SET_SCRIPT
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'dir="agents/${AYOAI_AGENT}/session"\n'
        'mkdir -p "$dir"\n'
        'touch "$dir/$1"\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)


def _sign_off(root: Path, agent: str = AGENT) -> Path:
    """What the framework's Phase -1.4 does when its obligations are complete."""
    marker = framework_session_dir(root, agent) / STOP_LOOP_SIGNAL
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    return marker


class _QuietAgent:
    """A turn that does nothing interesting — these tests are about the LOOP."""

    def __init__(self, session: Session) -> None:
        self.session = session

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _build(
    tmp_path: Path,
    *,
    agent: str | None = AGENT,
    reserve: float = 5.0,
    max_duration: float | None = None,
) -> tuple[Any, list[str]]:
    """An app whose run can be stopped; returns (app, endings)."""
    if agent is not None:
        _plant_setter(tmp_path)
    endings: list[str] = []
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_max_duration=max_duration,
        run_consolidation_reserve=reserve,
        run_stop_agent=agent,
    )

    async def _on_run_end(reason: str) -> None:
        endings.append(reason)

    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _QuietAgent(session),
        on_run_end=_on_run_end,
    )
    return app, endings


async def _post_stop(app: Any) -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/run/stop", json={"reason": "stopped"})
        assert response.status_code == 200


# ── the defect itself ────────────────────────────────────────────────────────────


def test_run_stop_keeps_beating_until_the_mind_signs_off(tmp_path: Path) -> None:
    """THE regression. The run must still be alive after the stop is raised.

    Pre-fix, `run_stopping.set()` ended the loop on the very next guard evaluation, so
    the assertion below that the loop is STILL RUNNING is what fails — which is exactly
    the live-vessel finding, reproduced without a vessel.
    """
    app, endings = _build(tmp_path, reserve=5.0)

    async def scenario() -> None:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        await asyncio.sleep(0.3)

        # The mind has been ASKED but has not signed off: the door must still be open.
        assert not loop_task.done(), "the run ended before the mind could run its stop"
        assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists()

        _sign_off(tmp_path)  # Phase -1.4 completes
        await asyncio.wait_for(loop_task, timeout=5)

    asyncio.run(scenario())
    assert endings == ["stopped"]


def test_the_sign_off_ends_the_run_promptly_not_at_grace_expiry(tmp_path: Path) -> None:
    """The wait ends EARLY on the mind's own permit — the grace is a bound, not a sleep."""
    app, endings = _build(tmp_path, reserve=30.0)  # a grace far longer than the test

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        await _post_stop(app)
        _sign_off(tmp_path)
        started = time.monotonic()
        await asyncio.wait_for(loop_task, timeout=10)
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed < 5.0, f"waited {elapsed:.1f}s on a 30s grace — the permit was not read"
    assert endings == ["stopped"]


def test_a_mind_that_never_signs_off_cannot_hold_the_vessel_open(tmp_path: Path) -> None:
    """The other bound. A wedged mind costs the grace and not one second more."""
    app, endings = _build(tmp_path, reserve=0.6)

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=10)  # never signed off
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.4, f"ended in {elapsed:.2f}s — the door was never held"
    assert elapsed < 6.0, f"took {elapsed:.2f}s on a 0.6s grace — the bound did not hold"
    assert endings == ["stopped"]


def test_a_non_seed_workspace_ends_exactly_as_before(tmp_path: Path) -> None:
    """No `run_stop_agent` => no raise, no window, and the old ending is untouched."""
    app, endings = _build(tmp_path, agent=None)

    async def scenario() -> float:
        loop_task = asyncio.create_task(app.state.consume_say_loop())
        await asyncio.sleep(0.2)
        started = time.monotonic()
        await _post_stop(app)
        await asyncio.wait_for(loop_task, timeout=5)
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed < 1.0, f"a non-seed run waited {elapsed:.2f}s for a stop it never raised"
    assert endings == ["stopped"]


def test_the_cap_landing_between_beats_still_raises_the_mind_s_own_stop(tmp_path: Path) -> None:
    """The cap path, where the mid-turn watcher never fires (nothing is `inflight`).

    The watcher skips whenever no turn is running, so an idle run that reaches its cap
    was ending with no graceful stop raised at all — a second starvation path, on a
    different trigger from the `/run/stop` one above.
    """
    app, endings = _build(tmp_path, reserve=0.6, max_duration=0.9)

    asyncio.run(asyncio.wait_for(app.state.consume_say_loop(), timeout=10))

    assert (framework_session_dir(tmp_path, AGENT) / "stop-requested").exists(), (
        "the cap ended the run without ever asking the mind to stop"
    )
    assert endings == ["duration_cap"]


# ── the permit reader ────────────────────────────────────────────────────────────


def test_framework_stop_complete_reads_the_permit(tmp_path: Path) -> None:
    assert framework_stop_complete(tmp_path, AGENT) is False  # absent = "not yet"
    _sign_off(tmp_path)
    assert framework_stop_complete(tmp_path, AGENT) is True


def test_framework_stop_complete_without_an_agent_is_false(tmp_path: Path) -> None:
    """No agent = no seed = nothing to wait for; never a wait on an empty path."""
    assert framework_stop_complete(tmp_path, "") is False
