"""ADR-0191: ``Bash(run_in_background=true)`` — Claude Code's background command for a Zak
Code session, with the exit reported as a ``<task-notification>`` at the session's idle door.

A Mind's playbooks say "background the suite, END the turn; the harness notifies", and the
framework's rules forbid polling a background job with ``ScheduleWakeup`` because the
harness reports on it. Zak Code had no such thing, so the framework carried a table of
which harness could notify — the last harness-capability branch. These tests pin the
contract: the tool returns at once with an id and an output file; the record is persisted;
an exited task is reported ONCE, as Claude Code's block, at the REPL's idle door; the exit
survives a process that did not spawn the task; ``TaskOutput`` reads, ``TaskStop`` kills the
whole group; a session's end kills what it started. Hermetic: real (short) shell commands
in a tmp workspace, no network.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import zakcode.background as background_module
from zakcode.background import (
    MAX_OUTPUT_CHARS,
    NOTIFICATION_HEAD,
    BackgroundTask,
    BackgroundTasks,
    notification_block,
    output_tail,
    pid_alive,
    task_status,
    tasks_dir_for,
)
from zakcode.cli import _InputMux
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.hooks import HookEvent, HookPayload, wire_payload
from zakcode.session.store import Session, SessionStore
from zakcode.tools import default_registry
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.bash import BashTool
from zakcode.tools.builtins.task_output import TaskOutputTool
from zakcode.tools.builtins.task_stop import TaskStopTool

# `sleep` and `exit` behave the same in bash everywhere (Git Bash on Windows).
LONG = "sleep 30"


def _tasks(tmp_path: Path, changes: list[int] | None = None) -> tuple[Session, BackgroundTasks]:
    session = Session(cwd=str(tmp_path), model="test")
    on_change = None if changes is None else (lambda: changes.append(1))
    return session, BackgroundTasks(session, on_change=on_change, tasks_dir=tmp_path / "tasks")


def _ctx(tmp_path: Path, tasks: BackgroundTasks | None) -> ToolContext:
    return ToolContext(workspace_root=tmp_path, background_tasks=tasks)


async def _settle(tasks: BackgroundTasks, task: BackgroundTask, timeout: float = 15.0) -> None:
    status, _ = await tasks.wait(task, timeout)
    assert status != "running", f"task still running after {timeout}s"


def _forget_in_process(task_id: str) -> None:
    """Drop THIS process's handles on a task — the shape of a process that never spawned
    it (the ADR-0034 restart in between)."""
    watcher = background_module._WATCHERS.pop(task_id, None)
    if watcher is not None:
        watcher.cancel()
    background_module._PROCS.pop(task_id, None)


async def _until_dead(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while pid_alive(pid):
        assert time.monotonic() < deadline, f"pid {pid} still alive after {timeout}s"
        await asyncio.sleep(0.05)


# ── the tool returns at once; the record is persisted ─────────────────────────────


async def test_run_in_background_returns_at_once_with_an_id_and_an_output_file(
    tmp_path: Path,
) -> None:
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)
    started = time.monotonic()
    res = await BashTool().execute(
        {"command": "sleep 1; echo finished", "run_in_background": True, "description": "probe"},
        _ctx(tmp_path, tasks),
    )
    assert not res.is_error, res.output
    assert time.monotonic() - started < 0.9, "the tool waited for the command"
    assert res.data is not None and res.data["background"] is True
    task_id = res.data["task_id"]
    assert f"Command running in background with ID: {task_id}" in res.output
    assert res.data["output_file"] in res.output
    assert "notified when it completes" in res.output
    # The record is on the session — and was persisted before the tool returned.
    (record,) = session.background_tasks
    assert record.id == task_id and record.command == "sleep 1; echo finished"
    assert record.description == "probe" and record.pid > 0
    assert changes, "the table was not persisted when the task started"
    assert tasks.status(record) == ("running", None)
    await _settle(tasks, record)
    assert tasks.status(record) == ("completed", 0)
    assert Path(record.output_file).read_text(encoding="utf-8").strip() == "finished"


async def test_the_record_round_trips_through_the_session_store(tmp_path: Path) -> None:
    session, tasks = _tasks(tmp_path)
    task = await tasks.start("echo persisted", cwd=str(tmp_path))
    await _settle(tasks, task)
    store = SessionStore(tmp_path / "store")
    store.save(session)
    loaded = store.load(session.id)
    (record,) = loaded.background_tasks
    assert record.model_dump() == task.model_dump()
    # A process that only READ the record still knows how the task ended.
    assert BackgroundTasks(loaded).status(record) == ("completed", 0)


def test_task_output_lives_beside_the_session_store() -> None:
    sessions = Path("/srv/mind/.zakcode/sessions")
    assert tasks_dir_for(sessions, "s1") == Path("/srv/mind/.zakcode/tasks/s1")


async def test_without_a_session_table_background_is_refused(tmp_path: Path) -> None:
    res = await BashTool().execute(
        {"command": "echo x", "run_in_background": True}, _ctx(tmp_path, None)
    )
    assert res.is_error and "foreground" in res.output
    assert res.data is not None and res.data["background"] is False


# ── the exit is reported ONCE, as Claude Code's block ─────────────────────────────


async def test_an_exited_task_is_reported_once_as_a_task_notification(tmp_path: Path) -> None:
    changes: list[int] = []
    session, tasks = _tasks(tmp_path, changes)
    task = await tasks.start("echo hi", cwd=str(tmp_path), description="say hi")
    await _settle(tasks, task)
    note = tasks.take_notifications()
    assert note is not None
    assert note.startswith(NOTIFICATION_HEAD)
    assert "<task-notification>" in note and "</task-notification>" in note
    assert f"<task-id>{task.id}</task-id>" in note
    assert f"<output-file>{task.output_file}</output-file>" in note
    assert "<status>completed</status>" in note
    assert '<summary>Background command "say hi" completed (exit code 0)</summary>' in note
    # Reported once: the flag is on the record and persisted.
    assert task.notified is True and session.background_tasks[0].notified is True
    assert tasks.take_notifications() is None
    assert len(changes) >= 2  # the start, then the report


async def test_a_running_task_is_not_reported(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start(LONG, cwd=str(tmp_path))
    try:
        assert tasks.take_notifications() is None
        assert task.notified is False
    finally:
        await tasks.stop(task.id)


async def test_a_nonzero_exit_is_reported_with_its_code(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start("echo bad; exit 3", cwd=str(tmp_path))
    await _settle(tasks, task)
    note = tasks.take_notifications()
    assert note is not None and "completed (exit code 3)" in note


async def test_several_exited_tasks_ride_one_line(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    a = await tasks.start("echo a", cwd=str(tmp_path))
    b = await tasks.start("echo b", cwd=str(tmp_path))
    await _settle(tasks, a)
    await _settle(tasks, b)
    note = tasks.take_notifications()
    assert note is not None
    assert note.count("<task-notification>") == 2
    assert f"<task-id>{a.id}</task-id>" in note and f"<task-id>{b.id}</task-id>" in note
    assert note.count(NOTIFICATION_HEAD) == 1


# ── TaskOutput / TaskStop ─────────────────────────────────────────────────────────


async def test_task_output_blocks_until_exit_and_returns_the_tail(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start("sleep 0.3; echo tail-line", cwd=str(tmp_path))
    res = await TaskOutputTool().execute({"task_id": task.id}, _ctx(tmp_path, tasks))
    assert not res.is_error, res.output
    assert res.data is not None
    assert res.data["status"] == "completed" and res.data["exit_code"] == 0
    assert "tail-line" in res.output
    assert f"[task {task.id}] status: completed, exit code: 0" in res.output


async def test_task_output_without_blocking_reports_a_running_task(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start(LONG, cwd=str(tmp_path))
    try:
        started = time.monotonic()
        res = await TaskOutputTool().execute(
            {"task_id": task.id, "block": False}, _ctx(tmp_path, tasks)
        )
        assert time.monotonic() - started < 1.0
        assert not res.is_error
        assert res.data is not None and res.data["status"] == "running"
        assert res.data["exit_code"] is None
        assert "(no output yet)" in res.output
    finally:
        await tasks.stop(task.id)


async def test_task_output_bounds_the_wait(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start(LONG, cwd=str(tmp_path))
    try:
        started = time.monotonic()
        res = await TaskOutputTool().execute(
            {"task_id": task.id, "timeout": 300}, _ctx(tmp_path, tasks)
        )
        assert 0.25 <= time.monotonic() - started < 3.0
        assert res.data is not None and res.data["status"] == "running"
    finally:
        await tasks.stop(task.id)


async def test_task_output_flags_a_failed_command_as_an_error(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start("echo boom; exit 2", cwd=str(tmp_path))
    res = await TaskOutputTool().execute({"task_id": task.id}, _ctx(tmp_path, tasks))
    assert res.is_error
    assert res.data is not None and res.data["exit_code"] == 2
    assert "boom" in res.output


async def test_task_output_unknown_id_is_an_error(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    res = await TaskOutputTool().execute({"task_id": "nope"}, _ctx(tmp_path, tasks))
    assert res.is_error and "no background task with id 'nope'" in res.output
    res = await TaskOutputTool().execute({"task_id": "nope"}, _ctx(tmp_path, None))
    assert res.is_error and "not available" in res.output


async def test_task_stop_kills_the_whole_process_group(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    # The command's own child (a grandchild of the wrapper) must die with it.
    task = await tasks.start("sleep 30 & sleep 30; wait", cwd=str(tmp_path))
    res = await TaskStopTool().execute({"task_id": task.id}, _ctx(tmp_path, tasks))
    assert not res.is_error, res.output
    assert res.data is not None and res.data["stopped"] is True
    await _until_dead(task.pid)
    status, _code = tasks.status(task)
    assert status == "killed"
    note = tasks.take_notifications()
    assert note is not None and "<status>killed</status>" in note and "was stopped" in note


async def test_task_stop_refuses_a_finished_or_unknown_task(tmp_path: Path) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start("echo done", cwd=str(tmp_path))
    await _settle(tasks, task)
    res = await TaskStopTool().execute({"task_id": task.id}, _ctx(tmp_path, tasks))
    assert res.is_error and "not running" in res.output
    res = await TaskStopTool().execute({"task_id": "nope"}, _ctx(tmp_path, tasks))
    assert res.is_error and "no background task" in res.output
    res = await TaskStopTool().execute({}, _ctx(tmp_path, tasks))
    assert res.is_error and "'task_id' is required" in res.output


# ── the exit survives the process that spawned the task ───────────────────────────


async def test_a_process_that_did_not_spawn_the_task_still_learns_its_exit(
    tmp_path: Path,
) -> None:
    """The bash wrapper writes the exit file itself: with THIS process's handles gone (the
    restart shape), a fresh handle over the reloaded record reads ``completed``."""
    session, tasks = _tasks(tmp_path)
    task = await tasks.start("sleep 0.3; echo after-restart; exit 4", cwd=str(tmp_path))
    _forget_in_process(task.id)
    fresh = BackgroundTasks(Session.model_validate(session.model_dump(mode="json")))
    record = fresh.get(task.id)
    assert record is not None
    status, code = await fresh.wait(record, 15.0)
    assert (status, code) == ("completed", 4)
    assert Path(record.exit_file).read_text(encoding="utf-8").strip() == "4"
    assert "after-restart" in fresh.output(record)
    note = fresh.take_notifications()
    assert note is not None and "completed (exit code 4)" in note


async def test_stop_after_a_restart_kills_by_pid(tmp_path: Path) -> None:
    session, tasks = _tasks(tmp_path)
    task = await tasks.start(LONG, cwd=str(tmp_path))
    _forget_in_process(task.id)
    fresh = BackgroundTasks(Session.model_validate(session.model_dump(mode="json")))
    stopped, reason = await fresh.stop(task.id)
    assert stopped, reason
    await _until_dead(task.pid)
    record = fresh.get(task.id)
    assert record is not None and record.stopped is True
    status, code = fresh.status(record)
    assert status == "killed"
    note = fresh.take_notifications()
    assert note is not None and "<status>killed</status>" in note


def test_a_task_gone_without_a_recorded_exit_is_lost_not_alive(tmp_path: Path) -> None:
    # A pid that has exited and been reaped, and no exit file: nothing recorded the exit.
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()  # exited AND reaped: the pid no longer names a live process
    record = BackgroundTask(
        id="t1",
        command="ghost",
        cwd=str(tmp_path),
        output_file=str(tmp_path / "t1.out"),
        exit_file=str(tmp_path / "t1.exit"),
        pid=done.pid if not pid_alive(done.pid) else -1,
        started_at="2026-09-18T00:00:00+00:00",
    )
    assert task_status(record) == ("lost", None)
    block = notification_block(record, "lost", None)
    assert "<status>lost</status>" in block and "without reporting an exit code" in block
    assert task_status(record.model_copy(update={"stopped": True})) == ("killed", None)


def test_output_tail_is_bounded(tmp_path: Path) -> None:
    big = tmp_path / "big.out"
    big.write_text("x" * (MAX_OUTPUT_CHARS + 5_000) + "END", encoding="utf-8")
    tail = output_tail(str(big))
    assert tail.startswith("[... 5003 earlier chars omitted")
    assert tail.endswith("END")
    assert output_tail(str(tmp_path / "missing.out")) == ""


# ── the doors and the registry ────────────────────────────────────────────────────


async def test_the_repl_idle_door_hands_the_notification_over_as_a_harness_line(
    tmp_path: Path,
) -> None:
    _session, tasks = _tasks(tmp_path)
    task = await tasks.start("echo door", cwd=str(tmp_path))
    await _settle(tasks, task)
    mux = _InputMux(
        tmp_path / "say", tmp_path / "stop", keyboard=False, wakeup_probe=tasks.take_notifications
    )
    got = mux.try_input()
    assert got is not None and got[0] == "harness"
    assert got[1] is not None and "<task-notification>" in got[1]
    assert mux.try_input() is None  # reported once


def test_the_registry_advertises_claude_codes_names() -> None:
    registry = default_registry()
    assert {"TaskOutput", "TaskStop"} <= set(registry.names())
    assert registry.canonical("task_output") == "TaskOutput"
    assert registry.canonical("task_stop") == "TaskStop"
    bash = registry.get("Bash")
    assert bash is not None
    props = bash.spec.parameters["properties"]
    assert "run_in_background" in props and "description" in props


def test_the_hook_wire_spells_the_new_tools_as_claude_code_does(tmp_path: Path) -> None:
    import json

    payload = HookPayload(
        event=HookEvent.PRE_TOOL_USE,
        tool_name="TaskStop",
        arguments={"task_id": "t1"},
        cwd=str(tmp_path),
        session_id="sid-1",
    )
    doc = json.loads(wire_payload(payload))
    assert doc["tool_name"] == "TaskStop" and doc["tool_input"] == {"task_id": "t1"}


def test_the_served_consumer_beat_delivers_the_notification_verbatim(tmp_path: Path) -> None:
    """A served mind has no idle prompt: its consumer beat is where an exited background
    command is reported — verbatim (no nudge folded in, no slash dispatch), once."""
    from fastapi.testclient import TestClient

    from zakcode.agent.loop import TurnResult
    from zakcode.config import Settings
    from zakcode.events import AgentDone, AgentTextDelta
    from zakcode.messages import Message
    from zakcode.server.app import create_app
    from zakcode.usage import Usage

    class _Mind:
        def __init__(self, session: Session) -> None:
            self.session = session
            self.turns: list[str] = []

        async def arun_turn(self, user_text: str) -> TurnResult:
            self.turns.append(user_text)
            self.session.add_message(Message.user(user_text))
            assistant = Message.assistant_text("ok")
            self.session.add_message(assistant)
            return TurnResult(
                assistant_messages=[assistant], tool_results=[], iterations=1, usage=Usage()
            )

        async def astream_turn(self, user_text: str):  # type: ignore[no-untyped-def]
            self.turns.append(user_text)
            self.session.add_message(Message.user(user_text))
            self.session.add_message(Message.assistant_text("ok"))
            yield AgentTextDelta(text="ok")
            yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())

    made: list[_Mind] = []

    def factory(session: Session, model: str | None, prompter: object = None) -> _Mind:  # noqa: ARG001
        agent = _Mind(session)
        made.append(agent)
        return agent

    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    # The session's record says a task exited (the files say so); nothing reported it yet.
    session = Session(cwd=str(tmp_path), model="scripted/test")
    out = tmp_path / "t9.out"
    out.write_text("suite: 12 passed\n", encoding="utf-8")
    (tmp_path / "t9.exit").write_text("0\n", encoding="utf-8")
    session.background_tasks.append(
        BackgroundTask(
            id="t9",
            command="pytest -q",
            cwd=str(tmp_path),
            output_file=str(out),
            exit_file=str(tmp_path / "t9.exit"),
            pid=-1,
            started_at="2026-09-18T00:00:00+00:00",
        )
    )
    store.save(session)
    (tmp_path / ".current-session").write_text(session.id + "\n", encoding="utf-8")
    app = create_app(settings=settings, store=store, agent_factory=factory)
    # No lifespan context: the app's own consumer loop must not race this beat.
    TestClient(app, raise_server_exceptions=False)
    assert asyncio.run(app.state.consume_one_say()) is True
    assert asyncio.run(app.state.consume_one_say()) is False  # reported once
    (mind,) = made
    (turn,) = mind.turns
    assert turn.startswith(NOTIFICATION_HEAD)
    assert "<task-id>t9</task-id>" in turn and "completed (exit code 0)" in turn
    assert "<command-name>" not in turn  # verbatim: no slash dispatch, no nudge
    assert store.load(session.id).background_tasks[0].notified is True


async def test_a_session_end_kills_what_it_started(tmp_path: Path) -> None:
    from zakcode import Agent
    from zakcode.config import Settings

    agent = Agent(
        settings=Settings(
            default_model="scripted/test", context_window=8192, workspace_root=tmp_path
        ),
        provider=ScriptedProvider([reply("x")]),
    )
    task = await agent.loop.background_tasks.start(LONG, cwd=str(tmp_path))
    assert pid_alive(task.pid)
    await agent.aclose()
    await _until_dead(task.pid)
    assert agent.loop.background_tasks.status(task)[0] == "killed"
