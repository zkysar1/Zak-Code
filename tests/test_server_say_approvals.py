"""The say-inbox permission prompter: approvals ride the ONE contract on the server.

Server-run turns (REST /chat, /chat/stream — every turn the autonomous driver runs)
used to get no prompter at all, so `ask` mode denied with nobody asked (2026-08-25
operator ruling: the say-inbox file is THE input contract, and approvals must be
answerable through it on every surface)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from zakcode.permissions import PermissionOutcome, PermissionRequest, PermissionTier
from zakcode.server.app import SayInboxPrompter
from zakcode.session.say_inbox import (
    interrupt_path,
    read_say,
    request_interrupt,
    say_path,
    say_pending,
    write_say,
)


def _request() -> PermissionRequest:
    return PermissionRequest(
        tool_name="bash",
        tier=PermissionTier.WORKSPACE_WRITE,
        arguments={"command": "ls"},
        reason="test",
    )


def _prompter(tmp_path: Path, **kw) -> SayInboxPrompter:
    kw.setdefault("poll", 0.02)
    kw.setdefault("timeout", 5.0)
    return SayInboxPrompter(tmp_path, **kw)


def test_inbox_answer_resolves_the_prompt(tmp_path: Path) -> None:
    assert write_say(say_path(tmp_path), "a")
    outcome = asyncio.run(_prompter(tmp_path).confirm(_request()))
    assert outcome is PermissionOutcome.ALLOW_SESSION


def test_non_answer_say_is_held_and_requeued_after_resolution(tmp_path: Path) -> None:
    async def run() -> PermissionOutcome:
        prompter = _prompter(tmp_path)
        assert write_say(say_path(tmp_path), "also please add tests")

        async def answer_later() -> None:
            # Wait until the non-answer was consumed (held), then answer.
            while say_pending(say_path(tmp_path)):
                await asyncio.sleep(0.01)
            write_say(say_path(tmp_path), "y")

        answering = asyncio.create_task(answer_later())
        outcome = await prompter.confirm(_request())
        await answering
        # The held message returns to the inbox for the between-turn consumer.
        for _ in range(200):
            if say_pending(say_path(tmp_path)):
                break
            await asyncio.sleep(0.01)
        assert read_say(say_path(tmp_path)) == "also please add tests"
        return outcome

    assert asyncio.run(run()) is PermissionOutcome.ALLOW_ONCE


def test_interrupt_file_denies_and_is_consumed(tmp_path: Path) -> None:
    request_interrupt(interrupt_path(tmp_path))
    outcome = asyncio.run(_prompter(tmp_path).confirm(_request()))
    assert outcome is PermissionOutcome.DENY_ONCE
    assert not interrupt_path(tmp_path).exists()


def test_timeout_denies(tmp_path: Path) -> None:
    outcome = asyncio.run(_prompter(tmp_path, timeout=0.1).confirm(_request()))
    assert outcome is PermissionOutcome.DENY_ONCE


def test_request_is_announced_to_the_watch_bus(tmp_path: Path) -> None:
    frames: list[dict] = []
    assert write_say(say_path(tmp_path), "y")
    asyncio.run(_prompter(tmp_path, publish=frames.append).confirm(_request()))
    assert frames and frames[0]["type"] == "action_required"
    assert frames[0]["tool_name"] == "bash"


def test_rest_turns_receive_the_inbox_prompter(tmp_path: Path, monkeypatch) -> None:
    """Both REST routes hand the factory a SayInboxPrompter — never None."""
    import httpx

    from zakcode.config import Settings
    from zakcode.server.app import create_app
    from zakcode.session.store import Session, SessionStore

    received: list[object] = []

    class _FakeAgent:
        def __init__(self, session: Session) -> None:
            self.session = session

        async def arun_turn(self, message: str):  # noqa: ANN202
            from zakcode.agent import TurnResult

            return TurnResult(text="ok", stop_reason="completed", iterations=1)

    def factory(session: Session, model: str | None, prompter: object) -> _FakeAgent:
        received.append(prompter)
        return _FakeAgent(session)

    settings = Settings(workspace_root=tmp_path, session_dir=tmp_path / "sessions")
    app = create_app(
        settings=settings, store=SessionStore(tmp_path / "sessions"), agent_factory=factory
    )
    transport = httpx.ASGITransport(app=app)

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": "hi"})
            assert resp.status_code == 200

    asyncio.run(run())
    assert received and isinstance(received[0], SayInboxPrompter)
