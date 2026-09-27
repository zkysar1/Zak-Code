"""Tests for the sidecar read-only surface: GET /workspace/summary (P0-4) + /sidecar/health (P0-5).

Both endpoints read artifacts a SEPARATE component writes into the loop workspace
(research/journal.md, research/findings*, .current-session) and MUST degrade gracefully when
they are absent — the surface is safe to poll before the agent loop has produced anything.
These are plain JSON GETs (no streaming), so Starlette's TestClient drives them directly (unlike
the /watch SSE tail, which needs a real server). Auth: _AUTH_EXEMPT_PATHS is the EXACT string
"/health", so BOTH endpoints — including /sidecar/health — require the bearer token when one is
configured; only the unauth liveness /health is exempt.
"""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.framework_stop import (
    SIGNAL_SET_SCRIPT,
    STOP_REQUESTED_SIGNAL,
    framework_session_dir,
)
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on these read-only routes)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(workspace: Path, *, auth_token: str | None = None) -> FastAPI:
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=workspace,
        auth_token=auth_token,
    )
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _client(workspace: Path, *, auth_token: str | None = None) -> TestClient:
    return TestClient(_make_app(workspace, auth_token=auth_token))


def _seed_research(workspace: Path, journal: str, findings: list[str]) -> None:
    research = workspace / "research"
    research.mkdir(parents=True, exist_ok=True)
    (research / "journal.md").write_text(journal, encoding="utf-8")
    (research / "findings.md").write_text("\n".join(f"- {f}" for f in findings), encoding="utf-8")


def test_workspace_summary_returns_journal_findings_and_session(tmp_path: Path) -> None:
    _seed_research(tmp_path, "## Journal\nline one\n", ["alpha finding", "beta finding"])
    (tmp_path / ".current-session").write_text("sess-123\n", encoding="utf-8")
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body["journal"].startswith("## Journal")
    assert body["finding_count"] == 2
    assert body["session_id"] == "sess-123"


def test_workspace_summary_degrades_when_nothing_written(tmp_path: Path) -> None:
    # No research/ dir, no .current-session — safe to poll before the loop's first turn.
    body = _client(tmp_path).get("/workspace/summary").json()
    # "findings" joined this payload in g-373-28 (the client renders the findings
    # themselves, not just the count). Degrades to [] exactly where finding_count
    # degrades to 0, so polling before the loop's first turn is still safe.
    assert body == {"journal": "", "finding_count": 0, "findings": [], "session_id": None}


def test_workspace_summary_caps_journal_at_5000_chars(tmp_path: Path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    (research / "journal.md").write_text("x" * 12000, encoding="utf-8")
    body = _client(tmp_path).get("/workspace/summary").json()
    assert len(body["journal"]) == 5000


def test_finding_count_counts_directory_files_excluding_dotfiles(tmp_path: Path) -> None:
    findings = tmp_path / "research" / "findings"
    findings.mkdir(parents=True)
    (findings / "f1.md").write_text("one", encoding="utf-8")
    (findings / "f2.md").write_text("two", encoding="utf-8")
    (findings / ".gitkeep").write_text("", encoding="utf-8")  # dotfile — not a finding
    body = _client(tmp_path).get("/workspace/summary").json()
    assert body["finding_count"] == 2


# ── g-373-28: the findings THEMSELVES, newest first ──────────────────────────
# The count alone left the client showing "2 findings logged" and nothing to read.
# These pin the two accepted shapes, and pin that the ORDER's provenance is reported
# rather than smoothed over: a directory has real mtimes, a flat list has none.


def test_findings_directory_returns_bodies_newest_first_with_real_dates(
    tmp_path: Path,
) -> None:
    import os

    findings = tmp_path / "research" / "findings"
    findings.mkdir(parents=True)
    (findings / "old.md").write_text("the older finding", encoding="utf-8")
    (findings / "new.md").write_text("the newer finding", encoding="utf-8")
    (findings / ".gitkeep").write_text("", encoding="utf-8")  # dotfile — not a finding
    os.utime(findings / "old.md", (1_000_000, 1_000_000))
    os.utime(findings / "new.md", (2_000_000, 2_000_000))

    body = _client(tmp_path).get("/workspace/summary").json()

    assert [f["name"] for f in body["findings"]] == ["new.md", "old.md"], (
        "newest first by mtime, and the dotfile is not a finding"
    )
    assert body["findings"][0]["text"] == "the newer finding"
    newer, older = body["findings"]
    assert newer["modified_at"] > older["modified_at"], (
        "a directory carries a real per-file mtime, so the date is MEASURED and orders"
    )
    assert newer["modified_at"].startswith("1970-01-24"), (
        "mtime 2_000_000 is a real epoch instant, not a placeholder — pin it so a future "
        "change from mtime to 'now' cannot pass this test"
    )
    assert all(f["truncated"] is False for f in body["findings"])
    assert body["finding_count"] == 2, "the count field is unchanged (additive)"


def test_flat_findings_file_reverses_and_reports_no_date(tmp_path: Path) -> None:
    # A flat list records no per-entry time. Newest-first is the append-order ASSUMPTION,
    # so modified_at must be null rather than a fabricated date a reader would trust.
    _seed_research(tmp_path, "j", ["first written", "second written"])

    body = _client(tmp_path).get("/workspace/summary").json()

    assert [f["text"] for f in body["findings"]] == ["- second written", "- first written"]
    assert all(f["modified_at"] is None for f in body["findings"])
    assert all(f["name"] is None for f in body["findings"])


def test_finding_body_is_truncated_and_says_so(tmp_path: Path) -> None:
    findings = tmp_path / "research" / "findings"
    findings.mkdir(parents=True)
    (findings / "big.md").write_text("y" * 9000, encoding="utf-8")

    entry = _client(tmp_path).get("/workspace/summary").json()["findings"][0]

    assert len(entry["text"]) == 4000
    assert entry["truncated"] is True, (
        "a clipped finding must SAY it was clipped — this rides a polled endpoint"
    )


def test_findings_list_is_capped_while_the_count_is_not(tmp_path: Path) -> None:
    findings = tmp_path / "research" / "findings"
    findings.mkdir(parents=True)
    for i in range(60):
        (findings / f"f{i:02d}.md").write_text(str(i), encoding="utf-8")

    body = _client(tmp_path).get("/workspace/summary").json()

    assert len(body["findings"]) == 50, "the payload is bounded"
    assert body["finding_count"] == 60, (
        "the COUNT still reports the true total, so a capped list never understates how "
        "many findings exist"
    )


def test_sidecar_health_reports_active_session(tmp_path: Path) -> None:
    (tmp_path / ".current-session").write_text("sess-777\n", encoding="utf-8")
    body = _client(tmp_path).get("/sidecar/health").json()
    # last_run_stop_reason joined this payload in g-369-28 so the env-server can end the
    # ENVIRONMENT when a bounded run finishes, instead of idling until the generic
    # idle-monitor reaps it and marks the run failed. None until this session's run ends.
    #
    # observation_intake joined in g-373-03: the perception bridge was unobservable from
    # either end, so a 4xx read exactly like a delivered frame. It rides THIS payload
    # rather than a new endpoint because the env-server already polls this one.
    #
    # Kept as whole-payload equality on purpose. This is a contract another service
    # polls, so a subset assertion would let a field be dropped or renamed without a
    # single test going red — and every counter is asserted PRESENT-and-zero because a
    # counter that only appears after the first failure cannot tell "healthy" from
    # "nobody called it" (guard-3169).
    assert body == {
        "status": "ok",
        "active_session_id": "sess-777",
        "last_run_stop_reason": None,
        "observation_intake": {
            "accepted": 0,
            "superseded": 0,
            "refused_bad_version": 0,
            "refused_missing_ref": 0,
            "refused_too_large": 0,
            "wake_delivered": 0,
            "wake_dropped": 0,
            "last_frame_age_seconds": None,
            "last_observed_at": None,
        },
    }


def test_sidecar_health_active_session_none_before_first_turn(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/sidecar/health").json()
    assert body == {
        "status": "ok",
        "active_session_id": None,
        "last_run_stop_reason": None,
        # g-373-03 — see the note on the sibling test above.
        "observation_intake": {
            "accepted": 0,
            "superseded": 0,
            "refused_bad_version": 0,
            "refused_missing_ref": 0,
            "refused_too_large": 0,
            "wake_delivered": 0,
            "wake_dropped": 0,
            "last_frame_age_seconds": None,
            "last_observed_at": None,
        },
    }


def test_sidecar_health_reports_this_sessions_run_ending(tmp_path: Path) -> None:
    """A marker written for the CURRENT session is reported."""
    (tmp_path / ".current-session").write_text("sess-777\n", encoding="utf-8")
    (tmp_path / ".run-stop-reason").write_text("sess-777\nduration_cap", encoding="utf-8")
    body = _client(tmp_path).get("/sidecar/health").json()
    assert body["last_run_stop_reason"] == "duration_cap"


def test_sidecar_health_suppresses_a_prior_sessions_ending(tmp_path: Path) -> None:
    """A marker from an EARLIER session must NOT be reported (rb-5759).

    This is the whole reason the marker carries a session id. A bare reason marker is
    write-once and never false — until the next run, when a stale ``duration_cap`` still
    sitting on disk would be read as *this* run's ending and terminate a fresh run at
    birth.

    The fence is necessary and NOT sufficient, and this docstring used to claim
    otherwise ("so no clear-on-start hook is needed and a missed clear cannot strand the
    next run"). It self-invalidates across session rotation inside a LIVING server only.
    Across a reboot of the same persistent workspace both halves come back matching —
    ``.current-session`` is persistent and only advances at the driver's next iteration —
    so the comparison below is stale-to-stale and passes. That is g-369-86, and it ended
    a production world 5.1 minutes after boot. The reboot case is covered by
    ``test_sidecar_health_drops_a_prior_runs_ending_across_a_reboot`` and by the
    clear-at-run-start it pins; this test still owns the rotation case.
    """
    (tmp_path / ".current-session").write_text("sess-NEW\n", encoding="utf-8")
    (tmp_path / ".run-stop-reason").write_text("sess-OLD\nduration_cap", encoding="utf-8")
    body = _client(tmp_path).get("/sidecar/health").json()
    assert body["last_run_stop_reason"] is None, (
        "a prior session's ending leaked into the current run — the env-server would "
        "terminate this run at birth"
    )
    # Positive control: the same fence must PASS the matching case, or the assertion
    # above would hold vacuously for any always-None implementation.
    (tmp_path / ".run-stop-reason").write_text("sess-NEW\nstopped", encoding="utf-8")
    assert _client(tmp_path).get("/sidecar/health").json()["last_run_stop_reason"] == "stopped"


@pytest.mark.parametrize(
    "marker",
    ["sess-777\nduration_cap\nboot-BEFORE", "sess-777\nduration_cap"],
    ids=["recorded-under-an-earlier-boot", "no-boot-line"],
)
def test_sidecar_health_drops_a_prior_runs_ending_across_a_reboot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    """A marker that survived a REBOOT must not end the run that just started (g-369-86).

    The session fence cannot catch this on its own, and the gap took down a production
    world. ``.current-session`` is persistent and only advances at the driver's NEXT
    iteration, so at boot it still names the session the PREVIOUS run ended on: marker
    session and current session agree, the fence passes on a stale-to-stale comparison,
    and the first ``/sidecar/health`` poll of a brand-new run reports ``duration_cap``.
    Env debc47de died 5.1 minutes after boot on a day whose cap was 7080s.

    Both files are therefore left in the MATCHING state a reboot produces — the exact
    state the fence is blind to — not the mismatched state the rotation test uses. This
    boot's id is pinned, so each marker is cleared for the reason its id names. One was
    recorded under another boot. The other carries no boot line, the shape written
    before ADR-0256 and by the provisioner.
    """
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: "boot-AFTER")
    (tmp_path / ".current-session").write_text("sess-777\n", encoding="utf-8")
    (tmp_path / ".run-stop-reason").write_text(marker, encoding="utf-8")

    # Positive control FIRST, and it is what makes this test meaningful: with no run
    # started, this is precisely the state the session fence green-lights. Without it
    # the assertion below would hold vacuously for any always-None implementation, and
    # it is also the literal pre-fix production behaviour.
    before = _client(tmp_path).get("/sidecar/health").json()
    assert before["last_run_stop_reason"] == "duration_cap"

    # The reboot: a NEW server comes up on the SAME persistent workspace. Entering the
    # TestClient context runs the real lifespan, which is what starts the run — so this
    # exercises the production wiring, not a hand-called helper.
    with TestClient(_make_app(tmp_path)) as client:
        body = client.get("/sidecar/health").json()
        assert body["last_run_stop_reason"] is None, (
            "a prior run's ending survived a reboot and would terminate this run at "
            "birth — the g-369-86 production incident"
        )

    # Assert the CLEANUP itself, not only its visible effect (guard-3218): a reader that
    # merely suppressed the value would leave the landmine for the next consumer.
    assert not (tmp_path / ".run-stop-reason").exists()


def test_sidecar_health_still_reports_an_ending_from_the_RUNNING_process(
    tmp_path: Path,
) -> None:
    """Clearing at run start must not disarm the signal the env-server depends on.

    The clear above is a change to a SHARED contract — the Java BudgetMeterVerticle
    polls this field to end the environment when a bounded run finishes, instead of
    idling ~9.5min until the generic idle-monitor reaps it and marks the run FAILED. So
    the fix is only correct if an ending written AFTER the run started is still
    reported; a clear that swallowed those would trade a 5-minute death for a 9.5-minute
    one and a false failure.
    """
    (tmp_path / ".current-session").write_text("sess-777\n", encoding="utf-8")
    with TestClient(_make_app(tmp_path)) as client:
        # Re-read rather than assuming: the consumer heals a marker that names a session
        # the store does not have, and `_current_run_stop_reason` compares "" to None as
        # equal, so this stays correct whether or not the id survived startup.
        try:
            sid = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
        except OSError:
            sid = ""
        (tmp_path / ".run-stop-reason").write_text(f"{sid}\nduration_cap", encoding="utf-8")
        body = client.get("/sidecar/health").json()
        assert body["last_run_stop_reason"] == "duration_cap"


@pytest.mark.parametrize(
    ("restart_boot", "reported"),
    [("boot-A", "duration_cap"), ("boot-B", None)],
    ids=["same-boot-restart-keeps-it", "reboot-clears-it"],
)
def test_a_bounded_runs_ending_is_kept_across_a_restart_on_the_same_boot_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restart_boot: str, reported: str | None
) -> None:
    """The run ends, the process exits, and it starts again: the ending must still be on
    ``/sidecar/health`` for the env-server's next poll when the start is a RESTART on the
    same boot, and cleared when it is a reboot (g-373-159, ADR-0256).

    mind-serve@ runs with Restart=always, so every bounded run ends in exactly this restart,
    seconds after ``on_run_end`` brings the process down, while the env-server polls every
    60s. Clearing at every process start deleted the ending before any poll could read it,
    and the environment idled on to its own session cap (measured on DEV: a 12m37s tail
    after a ``duration_cap`` ending). The boot is the discriminator: the SAME ending, met by
    a start under another boot, is the g-369-86 landmine, so both cases run the same code.
    Both halves are production code. The real writer records the ending when the capped
    run ends, and entering the TestClient runs the real lifespan that the restart runs.
    """
    boot = {"id": "boot-A"}
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: boot["id"])
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=tmp_path,
        run_max_duration=0.3,
    )
    ended = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=_factory,
    )
    asyncio.run(ended.state.consume_say_loop())  # returns because the cap ended the run
    # Read by position (guard-6956): session (none was ever current), reason, boot.
    recorded = (tmp_path / ".run-stop-reason").read_text(encoding="utf-8").splitlines()
    assert recorded == ["", "duration_cap", "boot-A"], recorded

    boot["id"] = restart_boot
    with TestClient(_make_app(tmp_path)) as client:
        assert client.get("/sidecar/health").json()["last_run_stop_reason"] == reported
    assert (tmp_path / ".run-stop-reason").exists() is (reported is not None)


def test_a_start_with_no_readable_boot_id_clears_even_a_boot_stamped_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no boot id to read (not Linux, or /proc unreadable), no ending can be proved to
    be this boot's, so the start clears it. A stale ending ends a live run (g-369-86), while
    a lost one only delays the environment's end, so the doubtful case takes the clear."""
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: None)
    (tmp_path / ".run-stop-reason").write_text("\nduration_cap\nboot-A", encoding="utf-8")
    # Positive control: before any start, the fence passes this marker (no session on
    # either side), so the None below can only come from the clear.
    assert _client(tmp_path).get("/sidecar/health").json()["last_run_stop_reason"] == "duration_cap"
    with TestClient(_make_app(tmp_path)) as client:
        assert client.get("/sidecar/health").json()["last_run_stop_reason"] is None
    assert not (tmp_path / ".run-stop-reason").exists()


@pytest.mark.parametrize("reason", ["stopped", "idle", "budget_exhausted"])
def test_only_a_duration_cap_ending_survives_a_restart_on_the_same_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """Every ending but ``duration_cap`` is cleared at start, even one THIS boot recorded.

    After any other ending the world can stay up, as a stopped "Keep it running" run does, so
    this start opens a run of its own. The env-server's step-out wait reads ANY ending on
    ``/sidecar/health`` as that run's receipt (SidecarProxyVerticle ``awaitSidecarExit``).
    A kept ending would end the wait before the new run's digest was written.
    """
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: "boot-A")
    (tmp_path / ".run-stop-reason").write_text(f"\n{reason}\nboot-A", encoding="utf-8")
    # Positive control: before any start, the fence passes this marker (no session on either
    # side), so the None below can only come from the clear.
    assert _client(tmp_path).get("/sidecar/health").json()["last_run_stop_reason"] == reason
    with TestClient(_make_app(tmp_path)) as client:
        assert client.get("/sidecar/health").json()["last_run_stop_reason"] is None
    assert not (tmp_path / ".run-stop-reason").exists()


def _plant_signal_setter(root: Path) -> None:
    """A stand-in for the framework's own signal writer, at the path the sidecar calls."""
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


@pytest.mark.parametrize(
    ("ending", "opens_a_run"),
    [("duration_cap", False), ("stopped", True)],
    ids=["kept-cap-starts-ended", "stopped-ending-opens-its-own-run"],
)
def test_a_restart_that_keeps_this_boots_cap_starts_ended_and_raises_no_second_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: str, opens_a_run: bool
) -> None:
    """A start that keeps this boot's cap is that run's restart, not a new run (g-373-161).

    Keeping the ending (ADR-0256) let the env-server read it, and it then ended the
    environment with POST /run/stop. But the restarted process had armed a fresh run clock
    and started its say consumer, so that stop reached an OPEN run and raised a second
    framework stop on a mind that had already finished its own. Measured on DEV run
    1790481966000_g373159: an 8-iteration graceful-stop turn of 653,630 tokens, run while
    the vessel was being torn down. The kept start must come up ENDED: the stop is answered
    ``ended``, nothing is raised, no run clock is armed, and ``on_run_end`` is not called
    again (it would only bring the process down into another restart).

    The control is a restart after a ``stopped`` ending. That ending is cleared, so the
    restart opens its own run and the same stop DOES raise the mind's graceful stop, which
    is also what keeps the ``not raised`` assertion from holding vacuously.
    """
    monkeypatch.setattr("zakcode.server.app._boot_id", lambda: "boot-A")
    agent = "probe"
    # The first run ends through the real writer: on its cap, or on a person's stop.
    first = create_app(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            run_max_duration=0.3 if ending == "duration_cap" else None,
        ),
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=_factory,
    )

    async def end_the_first_run() -> None:
        loop = asyncio.create_task(first.state.consume_say_loop())
        if ending == "stopped":
            transport = httpx.ASGITransport(app=first)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                assert (await c.post("/run/stop", json={"reason": "stopped"})).status_code == 200
        await asyncio.wait_for(loop, timeout=10)

    asyncio.run(end_the_first_run())
    recorded = (tmp_path / ".run-stop-reason").read_text(encoding="utf-8").splitlines()
    assert recorded == ["", ending, "boot-A"], recorded

    # The supervisor's restart on the same boot, configured as a vessel is: capped, and a
    # framework seed, so a stop that reaches an open run raises the mind's own.
    _plant_signal_setter(tmp_path)
    endings: list[str] = []

    async def _on_run_end(reason: str) -> None:
        endings.append(reason)

    restarted = create_app(
        settings=Settings(
            default_model="scripted/test",
            context_window=8192,
            workspace_root=tmp_path,
            run_max_duration=60.0,
            run_consolidation_reserve=1.0,
            run_stop_agent=agent,
        ),
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=_factory,
        on_run_end=_on_run_end,
    )
    stop_requested = framework_session_dir(tmp_path, agent) / STOP_REQUESTED_SIGNAL
    # Entering the TestClient runs the real lifespan, which is what the restart runs. The
    # stop is the env-server's own call: BudgetMeterVerticle sends the reason it acts on.
    with TestClient(restarted) as client:
        answer = client.post("/run/stop", json={"reason": "duration_cap"}).json()
        raised = stop_requested.exists()
        reported = client.get("/sidecar/health").json()["last_run_stop_reason"]

    if opens_a_run:
        assert answer == {"stopping": True, "reason": "duration_cap"}, answer
        assert raised, "a run the restart opened must still get the mind's own stop"
        assert reported is None
        return
    assert answer == {"stopping": False, "ended": True, "reason": "duration_cap"}, answer
    assert not raised, "a second framework stop was raised on a run that had already ended"
    assert not (tmp_path / "agents").exists(), "the raise wrote the framework's stop files"
    # No say consumer, so no run clock: `_consume_say_loop` is what arms the deadlines.
    assert not hasattr(restarted.state, "deadline_watcher")
    assert endings == []
    # The env-server still reads the ending it came for, before and after its stop.
    assert reported == "duration_cap"


def test_sidecar_endpoints_require_bearer_when_auth_configured(tmp_path: Path) -> None:
    token = "sidecar-secret"
    (tmp_path / ".current-session").write_text("sess-9\n", encoding="utf-8")
    client = _client(tmp_path, auth_token=token)
    # /sidecar/health is NOT the exempt /health — no token → 401 (proves the distinction).
    assert client.get("/sidecar/health").status_code == 401
    assert client.get("/workspace/summary").status_code == 401
    auth = {"Authorization": f"Bearer {token}"}
    assert client.get("/sidecar/health", headers=auth).status_code == 200
    assert client.get("/workspace/summary", headers=auth).status_code == 200
    assert client.get("/health").status_code == 200  # exempt liveness — still no token needed
