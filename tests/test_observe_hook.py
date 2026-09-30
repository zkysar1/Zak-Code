"""A workspace's ``ObservationReceived`` hook hears each accepted CHANGE frame (ADR-0270).

Before this event the perception intake woke a host by running two scripts it knew by path,
for an agent it knew by name, so the wake worked for one host framework and for no other.
The hook hands both decisions to the host: whether it runs a loop worth waking, and how to
wake it. The route keeps what belongs to the envelope: only a change is handed over, and
only after it is staged.

What these pin:

* A change frame runs the hook once, after the frame is staged, at the workspace root, with
  the frame's kind and staged path on stdin beside Claude Code's ``hook_event_name``.
* The exit code is the disposition. 0 is ``delivered``; a non-zero exit, a timeout and a hook
  that cannot start are ``dropped``; the frame stays staged either way.
* Only a change runs it. A heartbeat, an unstamped envelope and an unknown kind run nothing.
* It runs whichever ending the run uses, the host-neutral ``run_stop_message`` included.
* A workspace that declares it never also gets the signal-file wake. The control in the same
  test shows that wake firing before the hook is declared.
* The settings files are re-read when they change, both ways.

Hook scripts are small Python files run by this interpreter, so the tests hold on every
platform CI runs.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_observe_perception_wake import AGENT, _marker, _plant_seed
from zakcode.config import Settings
from zakcode.hooks import HookEvent, HookManager, LifecyclePayload, settings_loader
from zakcode.hooks.settings_loader import load_settings_hooks
from zakcode.server.app import create_app
from zakcode.session.observation_inbox import KIND_CHANGE, KIND_HEARTBEAT
from zakcode.session.store import Session, SessionStore


class _FakeAgent:
    """Minimal AgentLike. Never invoked here: no turn runs on this route."""

    def __init__(self, session: Session) -> None:
        self.session = session


def _factory(session: Session, model: str | None, prompter: object = None) -> _FakeAgent:  # noqa: ARG001
    return _FakeAgent(session)


def _client(workspace: Path, *, agent: str | None = None, message: str | None = None) -> TestClient:
    settings = Settings(
        default_model="scripted/test",
        context_window=8192,
        workspace_root=workspace,
        run_stop_agent=agent,
        run_stop_message=message,
    )
    store = SessionStore(base_dir=workspace / "sessions")
    return TestClient(create_app(settings=settings, store=store, agent_factory=_factory))


def _envelope(**over: object) -> dict[str, object]:
    body: dict[str, object] = {
        "envelopeVersion": 1,
        "externalClientRef": "char-42",
        "observedAt": "2026-09-30T04:00:00Z",
        "observation": {"nearbyPerception": {"units": ["a", "b"]}},
        "droppedSlices": [],
    }
    body.update(over)
    return body


def _hook(tmp_path: Path, name: str, *, exit_code: int = 0, sleep_s: float = 0.0) -> Path:
    """A hook that appends what it was handed to ``<name>.jsonl``, then exits ``exit_code``.

    It also records whether the frame was already staged when it ran and the directory it
    ran in. The script and its record live outside the workspace, so neither can be taken
    for something the route wrote.
    """
    record = tmp_path / f"{name}.jsonl"
    script = tmp_path / f"{name}.py"
    script.write_text(
        "import json, os, sys, time\n"
        "doc = json.load(sys.stdin)\n"
        "doc['_staged_before'] = os.path.exists('.observation')\n"
        "doc['_ran_in'] = os.getcwd()\n"
        f"time.sleep({sleep_s})\n"
        f"with open({str(record)!r}, 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(doc) + '\\n')\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    return script


def _records(tmp_path: Path, name: str) -> list[dict[str, object]]:
    record = tmp_path / f"{name}.jsonl"
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()]


def _command(script: Path) -> str:
    """The settings-file command that runs ``script`` under this interpreter.

    Quoted POSIX-style where the loader splits POSIX-style. On Windows it splits on
    whitespace and keeps quotes, so the parts go in bare there (CI paths carry no spaces).
    """
    parts = [Path(sys.executable).as_posix(), script.as_posix()]
    if sys.platform != "win32":
        parts = [shlex.quote(p) for p in parts]
    return " ".join(parts)


def _declare(workspace: Path, *commands: str, timeout: float | None = None) -> None:
    """Declare ``commands`` as ObservationReceived hooks in ``.zakcode/settings.json``."""
    hooks: list[dict[str, object]] = []
    for command in commands:
        hook: dict[str, object] = {"type": "command", "command": command}
        if timeout is not None:
            hook["timeout"] = timeout
        hooks.append(hook)
    path = workspace / ".zakcode" / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"hooks": {"ObservationReceived": [{"hooks": hooks}]}}), encoding="utf-8"
    )


def _intake(client: TestClient) -> dict[str, object]:
    resp = client.get("/sidecar/health")
    assert resp.status_code == 200, resp.text
    return resp.json()["observation_intake"]


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


# ── what the hook is handed, and when ────────────────────────────────────────────────


def test_a_change_frame_runs_the_hook_once_after_it_is_staged(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "seen")))
    (ws / ".current-session").write_text("sess-1\n", encoding="utf-8")

    resp = _client(ws).post("/observe", json=_envelope(kind=KIND_CHANGE))

    assert resp.status_code == 200
    assert resp.json()["wake"] == "delivered"
    [doc] = _records(tmp_path, "seen")
    assert doc["hook_event_name"] == "ObservationReceived"
    assert doc["_staged_before"] is True, "the hook ran before its own frame was on disk"
    assert os.path.samefile(str(doc["_ran_in"]), ws), "a hook runs at the workspace root"
    assert os.path.samefile(str(doc["cwd"]), ws)
    assert doc["session_id"] == "sess-1", "the run's current session rides on the payload"
    data = doc["data"]
    assert isinstance(data, dict)
    assert data["kind"] == KIND_CHANGE
    assert os.path.samefile(str(data["observation_path"]), ws / ".observation")


@pytest.mark.parametrize(
    "over",
    [{"kind": KIND_HEARTBEAT}, {}, {"kind": "snapshot"}],
    ids=["heartbeat", "unstamped", "unknown-kind"],
)
def test_only_a_change_runs_the_hook(tmp_path: Path, over: dict[str, object]) -> None:
    """The gate is equality with "change": a timer frame or a kind this receiver does not
    know must not spawn a process on every round."""
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "seen")))

    body = _client(ws).post("/observe", json=_envelope(**over)).json()

    assert body["wake"] == "not-attempted"
    assert _records(tmp_path, "seen") == []
    assert (ws / ".observation").exists(), "not handing a frame over is not dropping it"


# ── the exit code is the disposition ────────────────────────────────────────────────


def test_a_hook_that_exits_zero_is_a_delivered_wake(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "ok")))
    client = _client(ws)

    assert client.post("/observe", json=_envelope(kind=KIND_CHANGE)).json()["wake"] == "delivered"
    intake = _intake(client)
    assert intake["wake_delivered"] == 1
    assert intake["wake_dropped"] == 0


def test_a_failing_hook_is_a_dropped_wake_and_the_frame_stays(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "fails", exit_code=3)))
    client = _client(ws)

    resp = client.post("/observe", json=_envelope(kind=KIND_CHANGE))

    assert resp.status_code == 200
    body = resp.json()
    assert body["wake"] == "dropped"
    assert body["accepted"] is True, "the frame was staged; only the hand-over failed"
    assert len(_records(tmp_path, "fails")) == 1, "precondition: the hook did run"
    assert json.loads((ws / ".observation").read_text(encoding="utf-8"))["kind"] == KIND_CHANGE
    intake = _intake(client)
    assert intake["wake_dropped"] == 1
    assert intake["wake_delivered"] == 0


def test_a_hook_that_times_out_is_a_dropped_wake(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "slow", sleep_s=30)), timeout=0.5)

    started = time.monotonic()
    body = _client(ws).post("/observe", json=_envelope(kind=KIND_CHANGE)).json()

    assert body["wake"] == "dropped"
    assert time.monotonic() - started < 20, "the hook's timeout, not its sleep, ended the call"
    assert _records(tmp_path, "slow") == [], "the hook was stopped before it finished"


def test_a_hook_that_cannot_start_is_a_dropped_wake(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(ws, f"{(tmp_path / 'no-such-interpreter').as_posix()} hook.py")

    body = _client(ws).post("/observe", json=_envelope(kind=KIND_CHANGE)).json()

    assert body["wake"] == "dropped"


@pytest.mark.parametrize("stage", ["read", "run"])
def test_hooks_that_cannot_be_read_or_run_are_a_dropped_wake_never_a_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Fail-open: the frame is on disk before the hooks are read, so the vessel sees it
    accepted whether reading the settings files or running the hooks fails."""
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "seen")))

    def _cannot_read(self: settings_loader.SettingsHooks, manager: HookManager) -> Any:  # noqa: ARG001
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "a settings file is not UTF-8")

    async def _cannot_run(self: HookManager, payload: LifecyclePayload) -> bool | None:  # noqa: ARG001
        raise RuntimeError("this event loop cannot start a subprocess")

    if stage == "read":
        monkeypatch.setattr(settings_loader.SettingsHooks, "refresh", _cannot_read)
    else:
        monkeypatch.setattr(HookManager, "deliver_observation", _cannot_run)
    resp = _client(ws).post("/observe", json=_envelope(kind=KIND_CHANGE))

    assert resp.status_code == 200
    assert resp.json()["wake"] == "dropped"
    assert (ws / ".observation").exists()


def test_every_hook_runs_and_any_failure_drops_the_wake(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(
        ws,
        _command(_hook(tmp_path, "first", exit_code=1)),
        _command(_hook(tmp_path, "second")),
    )

    body = _client(ws).post("/observe", json=_envelope(kind=KIND_CHANGE)).json()

    assert body["wake"] == "dropped", "one failed hand-over must not read as a clean one"
    assert len(_records(tmp_path, "first")) == 1
    assert len(_records(tmp_path, "second")) == 1, "a failing hook does not stop the next"


# ── which ending, which wake ─────────────────────────────────────────────────────────


def test_the_hook_runs_when_the_run_ends_by_message(tmp_path: Path) -> None:
    """The host-neutral ending turns the signal-file wake off. It must not turn this off."""
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "seen")))

    body = _client(ws, message="/stop host").post("/observe", json=_envelope(kind=KIND_CHANGE))

    assert body.json()["wake"] == "delivered"
    assert len(_records(tmp_path, "seen")) == 1


def test_a_declared_hook_replaces_the_signal_file_wake(tmp_path: Path) -> None:
    """One wake per change: the host's hook, never the hook plus the signal file."""
    ws = _workspace(tmp_path)
    _plant_seed(ws)  # an autonomous agent with a working signal writer
    client = _client(ws, agent=AGENT)

    # CONTROL, same workspace and same writer: with no hook declared the signal file lands.
    control = client.post("/observe", json=_envelope(kind=KIND_CHANGE)).json()
    assert control["wake"] == "delivered"
    assert _marker(ws).exists(), "control: the planted signal-file wake fires"
    _marker(ws).unlink()

    _declare(ws, _command(_hook(tmp_path, "seen")))
    body = client.post("/observe", json=_envelope(kind=KIND_CHANGE)).json()

    assert body["wake"] == "delivered"
    assert len(_records(tmp_path, "seen")) == 1, "the hook declared after start was used"
    assert not _marker(ws).exists(), "the signal file was written beside the hook"


def test_the_settings_files_are_re_read_both_ways(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    client = _client(ws)
    change = _envelope(kind=KIND_CHANGE)

    assert client.post("/observe", json=change).json()["wake"] == "not-attempted"
    _declare(ws, _command(_hook(tmp_path, "seen")))
    assert client.post("/observe", json=change).json()["wake"] == "delivered"
    (ws / ".zakcode" / "settings.json").unlink()
    assert client.post("/observe", json=change).json()["wake"] == "not-attempted"
    assert len(_records(tmp_path, "seen")) == 1


def test_a_frame_arriving_mid_re_read_runs_the_new_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-read advances the file signature before it parses. Without the route's lock a
    second frame arriving mid-parse finds nothing changed and runs the old, empty list."""
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "seen")))
    first_client = _client(ws)
    second_client = TestClient(first_client.app)  # the same app: one hook list, two callers
    real_load = settings_loader.load_settings_hooks
    parsing, release = threading.Event(), threading.Event()

    def slow_load(*args: Any, **kwargs: Any) -> Any:
        parsing.set()
        release.wait(10)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(settings_loader, "load_settings_hooks", slow_load)
    wakes: dict[str, str] = {}

    def post(name: str, client: TestClient) -> None:
        wakes[name] = client.post("/observe", json=_envelope(kind=KIND_CHANGE)).json()["wake"]

    first = threading.Thread(target=post, args=("first", first_client))
    first.start()
    assert parsing.wait(10), "precondition: the first frame is re-reading the settings file"
    second = threading.Thread(target=post, args=("second", second_client))
    second.start()
    second.join(0.5)  # unlocked, the second frame is done by now, on the old list
    release.set()
    first.join(10)
    second.join(10)

    # Only the wakes are compared: the two hooks may run at once, and two processes appending
    # to one record file is not a race this test should depend on (Windows has no atomic append).
    assert wakes == {"first": "delivered", "second": "delivered"}


# ── the pieces ───────────────────────────────────────────────────────────────────────


def test_the_loader_reads_the_event(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    _declare(ws, _command(_hook(tmp_path, "seen")))

    specs, errors = load_settings_hooks(ws)

    assert errors == {}
    assert [spec.event for spec in specs] == [HookEvent.OBSERVATION_RECEIVED]


async def test_no_hook_means_nothing_was_attempted() -> None:
    """``None``, never ``False``: a workspace with no hook has not failed a delivery."""
    payload = LifecyclePayload(event=HookEvent.OBSERVATION_RECEIVED)
    assert await HookManager().deliver_observation(payload) is None
