"""Tests for :class:`zakcode.server.driver.ServeDriver`.

The driver is *mechanism* — it keeps a perpetual turn running against a local serve
daemon and records the driven session in ``.current-session`` (the watch ``current``
alias source). These tests pin its supervision behavior against a scriptable fake
client that stands in for :class:`~zakcode.server.client.ServerClient`: the driver
calls only ``health`` / ``create_session`` / ``astream_turn``, so a duck-typed double
exercises every path without a real HTTP transport or a running daemon.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.server.driver import CURRENT_SESSION_FILE, ServeDriver
from zakcode.usage import Usage


def _done(reason: str = "completed") -> AgentDone:
    return AgentDone(stop_reason=reason, iterations=1, usage=Usage(total_tokens=1))


class FakeServerClient:
    """A scriptable stand-in for ServerClient.

    ``turn_script`` is consumed one entry per ``astream_turn`` call: a ``list`` of
    events to yield, or an ``Exception`` instance to raise (on first iteration, like a
    real transport fault). When the script runs out, every further turn is a clean
    ``completed`` turn. ``health`` fails its first ``health_fail_times`` calls.
    """

    def __init__(
        self,
        *,
        health_fail_times: int = 0,
        turn_script: list[list[AgentEvent] | Exception] | None = None,
    ) -> None:
        self.health_calls = 0
        self._health_fail_times = health_fail_times
        self.create_calls = 0
        self.created_sids: list[str] = []
        self.turn_calls: list[dict[str, object]] = []
        self.marker_calls: list[dict[str, object]] = []
        self._turn_script = list(turn_script or [])
        self._turn_index = 0

    async def health(self) -> dict[str, object]:
        self.health_calls += 1
        if self.health_calls <= self._health_fail_times:
            raise httpx.ConnectError("daemon not up yet")
        return {"status": "ok"}

    async def create_session(self) -> str:
        self.create_calls += 1
        sid = f"sess-{self.create_calls}"
        self.created_sids.append(sid)
        return sid

    async def publish_watch_marker(self, session_id: str, *, reason: str = "") -> None:
        self.marker_calls.append({"session_id": session_id, "reason": reason})

    def _next_action(self) -> list[AgentEvent] | Exception:
        if self._turn_index < len(self._turn_script):
            action = self._turn_script[self._turn_index]
            self._turn_index += 1
            return action
        return [_done("completed")]

    async def astream_turn(
        self, message: str, session_id: str | None = None, model: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        self.turn_calls.append({"message": message, "session_id": session_id, "model": model})
        action = self._next_action()
        if isinstance(action, Exception):
            raise action  # first-iteration fault, exactly like a broken transport
        for event in action:
            yield event


def _driver(client: FakeServerClient, tmp_path: Path, **kw: object) -> ServeDriver:
    kw.setdefault("backoff_initial", 0.001)
    kw.setdefault("backoff_max", 0.01)
    return ServeDriver(client, tmp_path, **kw)  # type: ignore[arg-type]


async def test_writes_current_session_and_drives_turns(tmp_path: Path) -> None:
    seen: list[AgentEvent] = []
    client = FakeServerClient(turn_script=[[AgentTextDelta(text="hi"), _done()], [_done()]])
    driver = _driver(client, tmp_path, max_turns=2, on_event=seen.append)
    await driver.run()

    assert client.create_calls == 1
    assert (tmp_path / CURRENT_SESSION_FILE).read_text(encoding="utf-8").strip() == "sess-1"
    assert len(client.turn_calls) == 2  # boot + one continue
    assert all(c["session_id"] == "sess-1" for c in client.turn_calls)
    assert any(isinstance(e, AgentTextDelta) for e in seen)  # events reach the on_event seam


async def test_boot_then_continue_messages(tmp_path: Path) -> None:
    client = FakeServerClient()
    driver = _driver(client, tmp_path, boot_message="BOOT", continue_message="MORE", max_turns=3)
    await driver.run()

    assert [c["message"] for c in client.turn_calls] == ["BOOT", "MORE", "MORE"]


async def test_model_passthrough(tmp_path: Path) -> None:
    client = FakeServerClient()
    driver = _driver(client, tmp_path, model="groq/x", max_turns=1)
    await driver.run()

    assert client.turn_calls[0]["model"] == "groq/x"


async def test_backoff_on_turn_error_then_recovers(tmp_path: Path) -> None:
    # First turn faults; the second (same session) succeeds. One turn should complete.
    client = FakeServerClient(turn_script=[httpx.ConnectError("blip"), [_done()]])
    driver = _driver(client, tmp_path, max_turns=1)
    await driver.run()

    assert len(client.turn_calls) == 2  # failed attempt + successful retry
    assert client.create_calls == 1  # a transient fault does NOT recreate the session
    assert client.turn_calls[1]["session_id"] == "sess-1"  # retried on the same session


async def test_recreates_session_after_repeated_failures(tmp_path: Path) -> None:
    # Three consecutive faults (the daemon dropped the session); the driver must
    # re-health, mint a fresh session, and rewrite .current-session before continuing.
    client = FakeServerClient(
        turn_script=[
            httpx.ConnectError("gone"),
            httpx.ConnectError("gone"),
            httpx.ConnectError("gone"),
            [_done()],
        ]
    )
    driver = _driver(client, tmp_path, max_turns=1, recreate_after_failures=3)
    await driver.run()

    assert client.create_calls == 2  # original + one recreate
    assert client.created_sids == ["sess-1", "sess-2"]
    assert (tmp_path / CURRENT_SESSION_FILE).read_text(encoding="utf-8").strip() == "sess-2"
    # The recreate re-boots (boot_message) on the fresh session, not a bare continue.
    assert client.turn_calls[-1]["session_id"] == "sess-2"
    assert client.turn_calls[-1]["message"] == driver.boot_message


async def test_rotation_publishes_session_rotated_marker_to_old_session(tmp_path: Path) -> None:
    # On rotation the driver notifies watch observers on the OLD session BEFORE minting a new
    # one, so they reconnect to `current` cleanly instead of guessing at the stream close.
    client = FakeServerClient(
        turn_script=[
            httpx.ConnectError("gone"),
            httpx.ConnectError("gone"),
            httpx.ConnectError("gone"),
            [_done()],
        ]
    )
    driver = _driver(client, tmp_path, max_turns=1, recreate_after_failures=3)
    await driver.run()

    assert len(client.marker_calls) == 1
    assert client.marker_calls[0]["session_id"] == "sess-1"  # the OLD session, before the re-mint
    assert "re-minted" in str(client.marker_calls[0]["reason"])


async def test_rotation_marker_failure_does_not_break_the_serve_loop(tmp_path: Path) -> None:
    # The marker publish is best-effort: a failure must never break the serve loop (the driver
    # wraps it in contextlib.suppress), so rotation still completes.
    client = FakeServerClient(
        turn_script=[httpx.ConnectError("gone")] * 3 + [[_done()]],
    )

    async def _boom(session_id: str, *, reason: str = "") -> None:
        raise httpx.ConnectError("marker publish failed")

    client.publish_watch_marker = _boom  # type: ignore[method-assign]
    driver = _driver(client, tmp_path, max_turns=1, recreate_after_failures=3)
    await driver.run()  # must NOT raise

    assert client.create_calls == 2  # rotation still happened despite the marker publish failing


async def test_waits_for_health_before_first_turn(tmp_path: Path) -> None:
    client = FakeServerClient(health_fail_times=2)  # up on the 3rd probe
    driver = _driver(client, tmp_path, max_turns=1)
    await driver.run()

    assert client.health_calls == 3
    assert client.create_calls == 1  # session only minted after health passed
    assert len(client.turn_calls) == 1


async def test_stop_before_start_creates_no_session(tmp_path: Path) -> None:
    client = FakeServerClient()
    stop = asyncio.Event()
    stop.set()
    driver = _driver(client, tmp_path, stop=stop)
    await driver.run()

    assert client.create_calls == 0  # never got past the health/stop gate
    assert client.turn_calls == []
    assert not (tmp_path / CURRENT_SESSION_FILE).exists()


async def test_stop_mid_run_exits_after_current_turn(tmp_path: Path) -> None:
    client = FakeServerClient()  # perpetual clean turns
    driver = _driver(client, tmp_path, max_turns=None)

    # Stop the driver from inside the stream, after the first turn's AgentDone.
    def stopper(event: AgentEvent) -> None:
        if isinstance(event, AgentDone):
            driver.request_stop()

    driver.on_event = stopper
    await asyncio.wait_for(driver.run(), timeout=5)

    assert len(client.turn_calls) == 1  # exited after the in-flight turn, no re-kick


async def test_error_stop_reason_grows_backoff_but_clean_resets(tmp_path: Path) -> None:
    # A provider_error turn should leave the backoff grown (slow-retry, not hot-loop);
    # a clean completed turn resets it.
    err_client = FakeServerClient(turn_script=[[_done("provider_error")]])
    err_driver = _driver(err_client, tmp_path, backoff_initial=0.001, max_turns=1)
    await err_driver.run()
    assert err_driver._backoff_current > 0.001  # grew after the error stop reason

    ok_client = FakeServerClient(turn_script=[[_done("completed")]])
    ok_driver = _driver(ok_client, tmp_path, backoff_initial=0.001, max_turns=1)
    await ok_driver.run()
    assert ok_driver._backoff_current == 0.001  # reset after a clean turn


async def test_consecutive_provider_errors_compound_backoff(tmp_path: Path) -> None:
    # Regression pin for the 2026-07-19 decommissioned-model incident: a
    # provider_error turn completes its HTTP stream successfully, and the old loop
    # reset the backoff on every HTTP-successful turn BEFORE the stop-reason sleep —
    # so consecutive provider errors retried at backoff_initial forever (observed
    # live: 1,941 retries at ~1.6s). Consecutive error turns must COMPOUND:
    # 0.001 → 0.002 → 0.004 → 0.008 with factor 2, never re-pinning to initial.
    err = [_done("provider_error")]
    client = FakeServerClient(turn_script=[err, err, err])
    driver = _driver(
        client,
        tmp_path,
        backoff_initial=0.001,
        backoff_max=1.0,  # roomy cap so growth (not clamping) is what's measured
        provider_error_escalate_after=10,  # keep escalation out of this test
        max_turns=3,
    )
    await driver.run()
    assert driver._backoff_current == 0.008  # compounded across all three error turns


async def test_provider_error_streak_escalates_to_cap_then_clean_turn_heals(
    tmp_path: Path,
) -> None:
    # A provider fault that survives the escalation threshold is non-transient
    # (removed model, revoked key): the backoff jumps straight to backoff_max —
    # still perpetual, but paced — and the resume cue keeps flowing.
    err = [_done("provider_error")]
    client = FakeServerClient(turn_script=[err, err])
    driver = _driver(client, tmp_path, provider_error_escalate_after=2, max_turns=2)
    await driver.run()
    assert driver._provider_error_streak == 2
    assert driver._backoff_current == driver.backoff_max  # jumped to the cap
    assert "interrupted partway" in str(client.turn_calls[1]["message"])  # resume cue intact

    # A later clean turn heals fully: streak zeroed, backoff reset to initial.
    heal_client = FakeServerClient(turn_script=[err, err, [_done("completed")], [_done()]])
    heal_driver = _driver(heal_client, tmp_path, provider_error_escalate_after=2, max_turns=4)
    await heal_driver.run()
    assert heal_driver._provider_error_streak == 0
    assert heal_driver._backoff_current == 0.001


async def test_nudge_is_framed_into_the_turn_then_consumed(tmp_path: Path) -> None:
    (tmp_path / ".nudge").write_text("look at coral reefs\n", encoding="utf-8")
    client = FakeServerClient()
    driver = _driver(client, tmp_path, boot_message="RESEARCH", nudge_file=".nudge", max_turns=1)
    await driver.run()

    msg = str(client.turn_calls[0]["message"])
    assert "coral reefs" in msg  # the suggestion is present
    assert "suggestion, not an instruction" in msg  # framed, not an order
    assert msg.endswith("RESEARCH")  # prepended to the base message
    assert not (tmp_path / ".nudge").exists()  # consumed exactly once


async def test_no_nudge_file_leaves_the_message_unchanged(tmp_path: Path) -> None:
    client = FakeServerClient()
    driver = _driver(client, tmp_path, boot_message="RESEARCH", nudge_file=".nudge", max_turns=1)
    await driver.run()
    assert client.turn_calls[0]["message"] == "RESEARCH"


async def test_nudge_fires_once_not_on_the_following_turn(tmp_path: Path) -> None:
    (tmp_path / ".nudge").write_text("check tide pools", encoding="utf-8")
    client = FakeServerClient()
    driver = _driver(
        client,
        tmp_path,
        boot_message="BOOT",
        continue_message="MORE",
        nudge_file=".nudge",
        max_turns=2,
    )
    await driver.run()
    assert "tide pools" in str(client.turn_calls[0]["message"])  # turn 1 carries it
    assert client.turn_calls[1]["message"] == "MORE"  # turn 2 is clean again


async def test_provider_error_cues_resume_then_clean_turn_reverts(tmp_path: Path) -> None:
    # Turn 1 dies mid-work (provider_error) -> turn 2 must carry the RESUME cue, not the
    # generic continue; turn 2 completes cleanly -> turn 3 reverts to the plain continue.
    client = FakeServerClient(turn_script=[[_done("provider_error")], [_done()], [_done()]])
    driver = _driver(client, tmp_path, boot_message="BOOT", continue_message="MORE", max_turns=3)
    await driver.run()

    msgs = [str(c["message"]) for c in client.turn_calls]
    assert msgs[0] == "BOOT"
    assert "interrupted" in msgs[1] and "resume" in msgs[1]  # the resume cue fired
    assert "did NOT finish" in msgs[1]  # says the in-flight step is unfinished
    assert msgs[2] == "MORE"  # recovery is one-shot; a clean turn goes back to normal


async def test_provider_error_detail_is_folded_into_the_resume_cue(tmp_path: Path) -> None:
    err = AgentDone(
        stop_reason="provider_error",
        iterations=1,
        usage=Usage(total_tokens=1),
        error="tool_use_failed: malformed arguments",
    )
    client = FakeServerClient(turn_script=[[err]])
    driver = _driver(client, tmp_path, max_turns=2)
    await driver.run()

    # The (redacted) provider detail reaches the mind, so it can see WHY (e.g. its own
    # malformed tool call) and avoid repeating the same failure.
    assert "tool_use_failed" in str(client.turn_calls[1]["message"])


async def test_budget_exhausted_keeps_the_plain_continue(tmp_path: Path) -> None:
    # budget_exhausted still backs off (BACKOFF_STOP_REASONS) but is a whole-turn wall,
    # not a half-done step — the resume cue is scoped to provider_error only.
    client = FakeServerClient(turn_script=[[_done("budget_exhausted")]])
    driver = _driver(client, tmp_path, continue_message="MORE", max_turns=2)
    await driver.run()

    assert client.turn_calls[1]["message"] == "MORE"


async def test_custom_resume_message_passthrough(tmp_path: Path) -> None:
    client = FakeServerClient(turn_script=[[_done("provider_error")]])
    driver = _driver(client, tmp_path, resume_message="RESUME NOW{error}", max_turns=2)
    await driver.run()

    # No error detail on the done frame -> the {error} seam collapses to nothing.
    assert client.turn_calls[1]["message"] == "RESUME NOW"
