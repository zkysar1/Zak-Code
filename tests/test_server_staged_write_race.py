"""Concurrent POSTs to the routes that stage a file must not collide on its temp name.

``/observe`` and ``/nudge`` publish with a temp-write followed by ``os.replace``. The temp
name used to carry only the PID (``.observation.<pid>.tmp``), but both routes are sync
``def`` handlers, which Starlette runs on worker threads that all share one PID. Two
overlapping POSTs therefore wrote the SAME temp path: the first ``os.replace`` consumed it
and the second raised ``FileNotFoundError`` and answered 500 (ADR-0318).

The window is microseconds wide, so firing threads at the route proves nothing: it passes
on the broken code most runs. These tests close the window on purpose. Every ``os.replace``
onto the staged file waits at a barrier until ALL the writers have written their temp, which
is the interleaving production reaches by chance, and then they rename together. On a temp
name shared between threads that is a certain failure rather than a likely one.

Since ADR-0328 ``/observe`` stages under a lock, so the writers can no longer be held in the
rename together and only the nudge test can force that interleaving. The observe test pins the
lock instead: its writers are answered, one at a time, and one frame stands whole.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from zakcode.config import Settings
from zakcode.server.app import create_app
from zakcode.session.store import Session, SessionStore

WRITERS = 8
# A writer that never reaches the barrier (refused earlier, say) must fail the test in
# seconds instead of hanging it.
BARRIER_TIMEOUT_S = 10.0


class _FakeAgent:
    """Minimal AgentLike — never invoked here (no turn runs on these routes)."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _make_app(workspace: Path) -> FastAPI:
    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=workspace
    )
    store = SessionStore(base_dir=workspace / "sessions")
    return create_app(settings=settings, store=store, agent_factory=_factory)


def _hold_replaces_at_barrier(monkeypatch: pytest.MonkeyPatch, staged_name: str) -> list[str]:
    """Make each writer's first ``os.replace`` onto ``staged_name`` wait until all WRITERS arrive.

    Arriving means the writer has already written its temp file, so once the barrier opens
    every writer sits between "temp written" and "temp renamed" at the same moment. Returns
    the name of the temp each writer arrived with, in arrival order.

    Only a writer's FIRST rename waits. Windows refuses most of eight renames that hit one
    target together, and the helper then retries from the same thread; a retry must not wait
    for a barrier that has already opened.
    """
    real_replace = os.replace
    barrier = threading.Barrier(WRITERS, timeout=BARRIER_TIMEOUT_S)
    arrivals: list[str] = []
    arrived_threads: set[int] = set()

    def gated(src: str | os.PathLike[str], dst: str | os.PathLike[str], **kwargs: Any) -> None:
        if Path(dst).name == staged_name and threading.get_ident() not in arrived_threads:
            arrived_threads.add(threading.get_ident())
            arrivals.append(Path(src).name)
            barrier.wait()
        real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", gated)
    return arrivals


def _assert_each_writer_had_its_own_temp(arrivals: list[str]) -> None:
    """The gate saw every writer, and no two of them published from the same temp file.

    The first half stops the test passing without having forced anything: a publish that no
    longer goes through ``os.replace`` never reaches the barrier. The second pins the cause
    and not only the symptom, since a fix that swallowed the ``FileNotFoundError`` would
    answer 200 while two writers still shared one temp.
    """
    assert len(arrivals) == WRITERS, "the barrier was not reached, so nothing was forced"
    assert len(set(arrivals)) == WRITERS, arrivals


def _post_concurrently(
    app: FastAPI, path: str, body: Callable[[int], dict[str, object]]
) -> list[int | BaseException]:
    """POST ``body(i)`` to ``path`` from WRITERS threads at once; one outcome per writer.

    The outcome is the status code, or the exception the route raised: TestClient re-raises
    a route's unhandled exception into the calling thread, so an exception here IS a 500.
    """
    outcomes: dict[int, int | BaseException] = {}

    def post(i: int) -> None:
        try:
            outcomes[i] = TestClient(app).post(path, json=body(i)).status_code
        except BaseException as exc:  # noqa: BLE001 — recorded, and asserted on by the caller
            outcomes[i] = exc

    threads = [threading.Thread(target=post, args=(i,)) for i in range(WRITERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=BARRIER_TIMEOUT_S * 3)
    assert not any(thread.is_alive() for thread in threads), "a writer never returned"
    return [outcomes[i] for i in range(WRITERS)]


def _labels(outcomes: list[int | BaseException]) -> list[int | str]:
    """The outcomes as a short list a failure message can show whole."""
    return [o if isinstance(o, int) else type(o).__name__ for o in outcomes]


def _strays(workspace: Path, staged_name: str) -> list[str]:
    """Anything left beside the staged file that carries its name: an orphaned temp."""
    return sorted(
        entry.name
        for entry in workspace.iterdir()
        if entry.name.startswith(staged_name) and entry.name != staged_name
    )


def test_concurrent_observes_are_staged_one_at_a_time_and_one_stands_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eight overlapping frames are all answered, never inside the rename together, one stands.

    The barrier the nudge test uses cannot open here. ``/observe`` stages under one lock
    (ADR-0328), because looking at the staged frame, deciding what the new one carries forward
    and writing it are one read-modify-write, so no two writers are ever between "temp
    written" and "temp renamed" at once, and the temp name is no longer what keeps them apart.
    This pins the lock instead: each rename is slowed so that a second writer would be inside
    it if it could be, and the most writers ever inside is counted. The nudge test, which takes
    no lock, still forces the interleaving and still pins the temp name.
    """
    app = _make_app(tmp_path)
    real_replace = os.replace
    counting = threading.Lock()
    inside = 0
    peak = 0

    def slowed(src: str | os.PathLike[str], dst: str | os.PathLike[str], **kwargs: Any) -> None:
        nonlocal inside, peak
        if Path(dst).name != ".observation":
            real_replace(src, dst, **kwargs)
            return
        with counting:
            inside += 1
            peak = max(peak, inside)
        try:
            time.sleep(0.02)
            real_replace(src, dst, **kwargs)
        finally:
            with counting:
                inside -= 1

    monkeypatch.setattr(os, "replace", slowed)

    outcomes = _post_concurrently(
        app,
        "/observe",
        lambda i: {
            "envelopeVersion": 1,
            "externalClientRef": "char-42",
            "observedAt": f"2026-09-06T21:00:{i:02d}Z",
            "observation": {"writer": {"n": i}},
            "droppedSlices": [],
        },
    )

    # /observe is latest-wins and never refuses a frame, so every writer is answered 200.
    assert outcomes == [200] * WRITERS, _labels(outcomes)
    assert peak == 1, f"{peak} frames were being staged at once"
    # One writer's WHOLE frame stands: the file parses, and its parts agree with each other.
    staged = json.loads((tmp_path / ".observation").read_text(encoding="utf-8"))
    n = staged["observation"]["writer"]["n"]
    assert staged["observedAt"] == f"2026-09-06T21:00:{n:02d}Z"
    assert _strays(tmp_path, ".observation") == []


def test_concurrent_nudges_each_stage_without_colliding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _make_app(tmp_path)
    arrivals = _hold_replaces_at_barrier(monkeypatch, ".nudge")

    outcomes = _post_concurrently(app, "/nudge", lambda i: {"text": f"suggestion {i}"})

    # The barrier holds every writer past the single-slot ``exists()`` check, so none is
    # refused with 429 here and all are accepted. What this pins is that none of them errors.
    assert outcomes == [200] * WRITERS, _labels(outcomes)
    _assert_each_writer_had_its_own_temp(arrivals)
    text = (tmp_path / ".nudge").read_text(encoding="utf-8").strip()
    assert text in {f"suggestion {i}" for i in range(WRITERS)}
    assert _strays(tmp_path, ".nudge") == []
