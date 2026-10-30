"""A served web app follows ``zakcode update``, and the page says what a failed turn raised
(ADR-0333, ADR-0334).

ADR-0333: ``zakcode update`` replaces the install a running web app loaded. The terminal REPL
restarts itself into the new build at its next idle prompt (ADR-0034); the web app kept serving
the build it started on until someone restarted it by hand, so an update reached the CLI and not
the page. These drive the build follower through the ``start_consumer`` / ``stop_consumer`` seams
with ``install_changed`` patched, so "a newer build is installed" is a value the test sets, and
``on_new_build`` records what the CLI would exec on.

ADR-0334: a turn that raised past the agent loop's own failure handling showed the page one fixed
line, "turn failed before completing — see server log", whatever was raised. A fallback model with
no known context window and a revoked key read the same.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import zakcode.server.app as app_module
from tests.test_server_consumer import _LiveServer
from zakcode.agent.loop import TurnResult
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.providers.base import UnknownContextWindow
from zakcode.server.app import NewBuild, create_app
from zakcode.server.safe_projection import SafeEventProjection
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage

SKILL_CALL = 'Call Skill(skill="work-loop") now. That call was not executed.'


class _Agent:
    """A scripted turn. ``gate`` holds it open; ``carry`` is what the loop sets aside for a
    restarted process, as the real loop does at a skill boundary (ADR-0101)."""

    def __init__(
        self,
        session: Session,
        turns: list[str],
        gate: asyncio.Event | None = None,
        carry: str | None = None,
    ) -> None:
        self.session = session
        self.loop = SimpleNamespace(restart_continuation=None, restart_boundary=None)
        self._turns = turns
        self._gate = gate
        self._carry = carry

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self._turns.append(user_text)
        self.session.add_message(Message.user(user_text))
        if self._gate is not None:
            await self._gate.wait()
        self.session.add_message(Message.assistant_text("ok"))
        if self._carry is not None:
            self.loop.restart_continuation = self._carry
            self.loop.restart_boundary = "skill"
        yield AgentTextDelta(text="ok")
        yield AgentDone(
            stop_reason="restart" if self._carry else "completed", iterations=1, usage=Usage()
        )

    async def arun_turn(self, user_text: str) -> TurnResult:
        self._turns.append(user_text)
        self.session.add_message(Message.user(user_text))
        if self._gate is not None:
            await self._gate.wait()
        return TurnResult(stop_reason="completed")


class _Install:
    """Stands in for ``install_changed``: nothing new until a test sets ``changed``."""

    def __init__(self) -> None:
        self.changed: tuple[str, str] | None = None

    def __call__(self) -> tuple[str, str] | None:
        return self.changed


def _patch_install(monkeypatch: pytest.MonkeyPatch) -> _Install:
    install = _Install()
    monkeypatch.setattr(app_module, "install_changed", install)
    monkeypatch.setattr(app_module, "_NEW_BUILD_POLL_SECONDS", 0.05)
    monkeypatch.setattr(app_module, "_RESTART_HOLD_SECONDS", 0.02)
    # The consumer acknowledges a pending restart at its next beat, 0.5 s apart when idle: as
    # long as these tests' waits, so a restart that should never come could land just after
    # the check. Measured: two of these tests passed against code that restarted anyway.
    monkeypatch.setattr(app_module, "_IDLE_BEAT_SECONDS", 0.05)
    return install


def _build(
    tmp_path: Path,
    *,
    turns: list[str],
    gate: asyncio.Event | None = None,
    carry: str | None = None,
    resume_turn: str | None = None,
    **run: Any,
) -> tuple[Any, list[NewBuild]]:
    """An app that follows new builds; returns (app, the builds it asked to restart into)."""
    builds: list[NewBuild] = []

    async def _on_new_build(build: NewBuild) -> None:
        builds.append(build)

    settings = Settings(
        default_model="scripted/test", context_window=8192, workspace_root=tmp_path, **run
    )
    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=lambda session, model, prompter: _Agent(session, turns, gate, carry),
        on_new_build=_on_new_build,
        resume_turn=resume_turn,
    )
    return app, builds


async def _until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.02)


# ── ADR-0333: the web app follows an update ──────────────────────────────────


def test_an_idle_web_app_restarts_into_a_newer_build_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install = _patch_install(monkeypatch)
    app, builds = _build(tmp_path, turns=[])

    async def scenario() -> None:
        await app.state.start_consumer()
        await asyncio.sleep(0.3)
        assert builds == [], "it restarted with no newer build installed"
        install.changed = ("1111111", "2222222")
        await _until(lambda: bool(builds))
        await asyncio.sleep(0.3)  # long enough for a second restart, were there one
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert builds == [NewBuild("1111111", "2222222")]


def test_a_chat_turn_in_flight_holds_the_restart_until_it_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A POST /chat turn runs beside the say consumer, so a held consumer is not enough: the
    restart also waits for every turn in flight, or it would cut this one short."""
    install = _patch_install(monkeypatch)
    turns: list[str] = []
    gate = asyncio.Event()
    app, builds = _build(tmp_path, turns=turns, gate=gate)

    async def scenario() -> None:
        await app.state.start_consumer()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            chat = asyncio.create_task(client.post("/chat", json={"message": "a long one"}))
            await _until(lambda: "a long one" in turns)
            install.changed = ("1111111", "2222222")
            await asyncio.sleep(0.5)
            assert builds == [], "the restart fired with a chat turn in flight"
            gate.set()
            response = await chat
            assert response.status_code == 200
        await _until(lambda: bool(builds))
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert builds == [NewBuild("1111111", "2222222")]


def test_no_say_starts_a_turn_on_the_build_being_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once a restart is pending the consumer is held: a say that arrives then waits in its file
    for the new process, rather than running on the old build and being cut by the exec. (The
    recorder here never stops the server, so the hold is all that keeps the say waiting.)"""
    install = _patch_install(monkeypatch)
    turns: list[str] = []
    app, builds = _build(tmp_path, turns=turns)

    async def scenario() -> None:
        await app.state.start_consumer()
        install.changed = ("1111111", "2222222")
        await _until(lambda: bool(builds))
        assert write_say(say_path(tmp_path), "arrived during the restart")
        await asyncio.sleep(0.5)
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert turns == []
    assert say_path(tmp_path).exists(), "the say was taken on the build being left"


@pytest.mark.parametrize(
    "run",
    [{"run_max_duration": 3600.0}, {"run_idle_timeout": 3600.0}],
    ids=["run-cap", "idle-bound"],
)
def test_a_run_a_host_manages_keeps_its_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: dict[str, Any]
) -> None:
    """A bounded run's build and clocks are its host's: a restart would re-arm the run's clock
    and its ending, so such a run never restarts itself, whatever is installed under it."""
    install = _patch_install(monkeypatch)
    install.changed = ("1111111", "2222222")
    app, builds = _build(tmp_path, turns=[], **run)

    async def scenario() -> None:
        await app.state.start_consumer()
        await asyncio.sleep(0.5)
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert builds == []


def test_a_turn_ended_for_the_restart_hands_its_line_on_and_holds_the_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop ends a turn at a skill boundary when a newer build is installed (ADR-0101) and
    sets the call aside for the restarted process. The server keeps it for the restart, and
    holds the consumer from that moment: the follower's next look is a second away here, and a
    say taken before it would run on the build being left."""
    install = _patch_install(monkeypatch)
    monkeypatch.setattr(app_module, "_NEW_BUILD_POLL_SECONDS", 1.0)
    install.changed = ("1111111", "2222222")
    turns: list[str] = []
    app, builds = _build(tmp_path, turns=turns, carry=SKILL_CALL)
    assert write_say(say_path(tmp_path), "keep going")

    async def scenario() -> None:
        await app.state.start_consumer()
        await _until(lambda: turns == ["keep going"] and not say_path(tmp_path).exists())
        await asyncio.sleep(0.1)  # the turn's own clean-up
        assert write_say(say_path(tmp_path), "a second say")
        await asyncio.sleep(0.4)
        assert say_path(tmp_path).exists(), "a say ran after the turn that ended for the restart"
        await _until(lambda: bool(builds))
        await app.state.stop_consumer()

    asyncio.run(scenario())

    assert turns == ["keep going"]
    assert builds == [NewBuild("1111111", "2222222", SKILL_CALL, "skill")]


def test_a_restarted_web_app_runs_the_carried_line_first_as_it_is(tmp_path: Path) -> None:
    """The line the old process handed on runs before a say that was already waiting, and as the
    harness's own line: never read as a slash command, never given a nudge in front of it."""
    turns: list[str] = []
    resume = "[harness] this session was restarted into build 2222222.\n/not-a-slash-command"
    app, _builds = _build(tmp_path, turns=turns, resume_turn=resume)
    assert write_say(say_path(tmp_path), "a say that waited")
    (tmp_path / ".nudge").write_text("look at the tower\n", encoding="utf-8")

    async def scenario() -> None:
        assert await app.state.consume_one_say() is True
        assert await app.state.consume_one_say() is True

    asyncio.run(scenario())

    assert turns[0] == resume
    assert turns[1].endswith("a say that waited")


# ── ADR-0334: the page says what a failed turn raised ────────────────────────


def test_the_failure_line_names_what_was_raised_on_one_line() -> None:
    exc = UnknownContextWindow(
        "these configured models have no known context window:\n"
        "  - fallback_model: model 'openai/no-such-model' has no known context window"
    )

    line = app_module._turn_failure_error(exc, lambda text: text)

    assert line == (
        "turn failed before completing: UnknownContextWindow: these configured models have no "
        "known context window: - fallback_model: model 'openai/no-such-model' has no "
        "known context window (full trace in the server log)"
    )


def test_the_failure_line_is_redacted() -> None:
    secret = "sk-ant-api03-" + "Q7x" * 20
    exc = RuntimeError(f"the provider refused key {secret}")

    line = app_module._turn_failure_error(exc, SafeEventProjection(env={}).redact)

    secret_shown = secret in line
    assert secret_shown is False
    assert line.startswith("turn failed before completing: RuntimeError: the provider refused key")


def test_the_failure_line_is_capped() -> None:
    exc = RuntimeError("x" * 5000)

    line = app_module._turn_failure_error(exc, lambda text: text)

    detail = line.removeprefix("turn failed before completing: ").removesuffix(
        " (full trace in the server log)"
    )
    assert len(detail) == app_module._TURN_FAILURE_DETAIL_CHARS
    assert detail.endswith("…")


def test_a_redactor_that_fails_still_gives_the_done_frame_its_words() -> None:
    def _broken(text: str) -> str:
        raise ValueError("cannot redact")

    line = app_module._turn_failure_error(RuntimeError("boom"), _broken)

    assert line == "turn failed before completing — see server log"


async def test_a_say_turn_that_raises_tells_the_page_what_failed(tmp_path: Path) -> None:
    """The path the page reads: a say whose agent cannot be built (the refusal a fallback model
    with no known context window raises at construction) ends with a done frame naming it."""
    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)

    def refusing_factory(session: Session, model: object, prompter: object) -> _Agent:
        raise UnknownContextWindow(
            "these configured models have no known context window:\n"
            "  - fallback_model: model 'openai/no-such-model' has no known context window"
        )

    app = create_app(
        settings=settings,
        store=SessionStore(base_dir=tmp_path / "sessions"),
        agent_factory=refusing_factory,
    )
    with _LiveServer(app) as base_url:
        async with httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(10.0)) as ac:
            assert (await ac.post("/say", json={"text": "hello?"})).status_code in (200, 202)
            deadline = time.monotonic() + 10.0
            while (current := await ac.get("/sessions/current")).status_code != 200:
                assert time.monotonic() < deadline, "the say never made a session"
                await asyncio.sleep(0.05)
            sid = current.json()["id"]
            frames: list[dict[str, Any]] = []
            async with ac.stream("GET", f"/watch/{sid}?full=1") as watch:
                async for line in watch.aiter_lines():
                    if line.startswith("data:"):
                        frames.append(json.loads(line[len("data:") :].strip()))
                        if frames[-1].get("event") == "done":
                            break

    done = frames[-1]
    assert done["stop_reason"] == "provider_error"
    assert done["error"].startswith("turn failed before completing: UnknownContextWindow: ")
    assert "fallback_model: model 'openai/no-such-model'" in done["error"]
