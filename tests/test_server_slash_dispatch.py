"""Server-side slash dispatch (ADR-0037) — every server door runs a leading-slash message
through ``Agent.compose_skill_turn`` exactly like the CLI: the say consumer, ``POST /chat``
and ``POST /chat/stream``.

The field finding: a served Mind could never be STARTED. Its framework's boot command
(``/start <agent> --mode assistant``) is a user-invocable-only skill and the server passed
raw text to the turn, so the model refused its own boot command as self-invocation. A
headless deployment (systemd ``mind-serve@``, a recipe writing ``.say``) has only these doors.

Pinned here: an invoked skill's provenance-framed body IS the turn (nudge left queued); a
denied / unreadable skill runs NO turn (``/chat`` 403 / 500; stream + bus get ``status`` +
``done(skill_refused)``); an unknown ``/token`` and a thin agent without a skills surface stay
prose; the parse mirrors the CLI one-shot (first token lower-cased + slash-stripped, rest args).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient

from zakcode import SkillInvocation
from zakcode.agent.loop import TurnResult
from zakcode.config import Settings
from zakcode.events import AgentDone, AgentEvent, AgentTextDelta
from zakcode.messages import Message
from zakcode.server.app import SlashDispatch, create_app, dispatch_slash
from zakcode.session.say_inbox import say_path, write_say
from zakcode.session.store import Session, SessionStore
from zakcode.usage import Usage


class _SkillAgent:
    """AgentLike with a scripted skill registry: ``start`` invokes, ``hidden`` is denied
    (user-invocable: false), ``broken`` is unreadable, anything else is unknown."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.compose_calls: list[tuple[str, str]] = []
        self.turns: list[str] = []

    async def compose_skill_turn(self, name: str, args: str = "") -> SkillInvocation:
        self.compose_calls.append((name, args))
        if name == "start":
            frame = (
                "<command-message>start is running</command-message>\n"
                "<command-name>/start</command-name>"
            )
            if args:
                frame += f"\n<command-args>{args}</command-args>"
            return SkillInvocation(invoked=True, name="start", turn_text=f"{frame}\n\nSTART BODY")
        if name == "hidden":
            return SkillInvocation(
                invoked=True, name="hidden", denied_reason="hidden is user-invocable: false"
            )
        if name == "broken":
            return SkillInvocation(invoked=True, name="broken", error="ENOENT")
        return SkillInvocation(invoked=False)

    async def arun_turn(self, user_text: str) -> TurnResult:
        self.turns.append(user_text)
        self.session.add_message(Message.user(user_text))
        assistant = Message.assistant_text("ok")
        self.session.add_message(assistant)
        return TurnResult(
            assistant_messages=[assistant], tool_results=[], iterations=1, usage=Usage()
        )

    async def astream_turn(self, user_text: str) -> AsyncIterator[AgentEvent]:
        self.turns.append(user_text)
        self.session.add_message(Message.user(user_text))
        self.session.add_message(Message.assistant_text("ok"))
        yield AgentTextDelta(text="ok")
        yield AgentDone(stop_reason="completed", iterations=1, usage=Usage())


def _build(tmp_path: Path) -> tuple[TestClient, SessionStore, list[_SkillAgent]]:
    made: list[_SkillAgent] = []

    def factory(session: Session, model: str | None, prompter: object = None) -> _SkillAgent:  # noqa: ARG001
        agent = _SkillAgent(session)
        made.append(agent)
        return agent

    settings = Settings(default_model="scripted/test", context_window=8192, workspace_root=tmp_path)
    store = SessionStore(base_dir=tmp_path / "sessions")
    app = create_app(settings=settings, store=store, agent_factory=factory)
    return TestClient(app, raise_server_exceptions=False), store, made


def _sse_events(text: str) -> list[dict]:
    return [
        json.loads(line[len("data:") :].strip())
        for line in text.splitlines()
        if line.startswith("data:")
    ]


# ── the say inbox (the headless door a recipe writes) ───────────────────────────


def test_say_slash_dispatches_the_skill_as_a_framed_turn(tmp_path: Path) -> None:
    client, store, made = _build(tmp_path)
    assert write_say(say_path(tmp_path), "/start tricks --mode assistant")

    assert asyncio.run(client.app.state.consume_one_say()) is True
    assert not say_path(tmp_path).exists()  # consumed exactly once

    (agent,) = made
    assert agent.compose_calls == [("start", "tricks --mode assistant")]
    (turn,) = agent.turns
    assert turn.startswith("<command-message>start is running</command-message>")
    assert "<command-args>tricks --mode assistant</command-args>" in turn
    assert turn.endswith("START BODY")
    # the framed turn is what the session persisted — the boot survives the vessel
    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    assert store.load(marker).messages[0].text == turn


def test_say_slash_leaves_a_queued_nudge_for_the_next_prose_turn(tmp_path: Path) -> None:
    """The provenance frame must OPEN the message; a nudge preamble in front of it would
    demote the skill to prose. So a skill turn does not consume the nudge slot."""
    client, _store, made = _build(tmp_path)
    assert client.post("/nudge", json={"text": "look at the moon"}).status_code == 200

    assert write_say(say_path(tmp_path), "/start tricks")
    assert asyncio.run(client.app.state.consume_one_say()) is True
    assert made[0].turns[0].startswith("<command-message>")
    assert "look at the moon" not in made[0].turns[0]

    assert write_say(say_path(tmp_path), "hello")
    assert asyncio.run(client.app.state.consume_one_say()) is True
    assert "look at the moon" in made[1].turns[0]  # folded into the prose turn, as before
    assert made[1].turns[0].endswith("hello")


def test_say_denied_skill_runs_no_turn_but_consumes_the_slot(tmp_path: Path) -> None:
    client, store, made = _build(tmp_path)
    assert write_say(say_path(tmp_path), "/hidden")

    assert asyncio.run(client.app.state.consume_one_say()) is True
    assert not say_path(tmp_path).exists()
    (agent,) = made
    assert agent.compose_calls == [("hidden", "")]
    assert agent.turns == []  # never handed to the model as prose
    marker = (tmp_path / ".current-session").read_text(encoding="utf-8").strip()
    assert store.load(marker).messages == []


def test_say_unknown_slash_token_stays_prose(tmp_path: Path) -> None:
    client, _store, made = _build(tmp_path)
    assert write_say(say_path(tmp_path), "/usr/bin/env is where?")
    assert asyncio.run(client.app.state.consume_one_say()) is True
    assert made[0].compose_calls == [("usr/bin/env", "is where?")]
    assert made[0].turns == ["/usr/bin/env is where?"]


def test_say_parse_mirrors_the_cli_one_shot(tmp_path: Path) -> None:
    client, _store, made = _build(tmp_path)
    assert write_say(say_path(tmp_path), "  /START   tricks   --mode autonomous  ")
    assert asyncio.run(client.app.state.consume_one_say()) is True
    assert made[0].compose_calls == [("start", "tricks   --mode autonomous")]


# ── POST /chat ─────────────────────────────────────────────────────────────────


def test_chat_slash_dispatches_the_skill(tmp_path: Path) -> None:
    client, _store, made = _build(tmp_path)
    resp = client.post("/chat", json={"message": "/start tricks --mode assistant"})
    assert resp.status_code == 200
    assert made[0].turns[0].startswith("<command-message>start is running")
    assert resp.json()["text"] == "ok"


def test_chat_denied_is_403_and_unreadable_is_500_and_neither_strands_the_session(
    tmp_path: Path,
) -> None:
    client, _store, made = _build(tmp_path)
    sid = client.post("/sessions").json()["id"]

    denied = client.post("/chat", json={"message": "/hidden", "session_id": sid})
    assert denied.status_code == 403
    assert "user-invocable" in denied.json()["detail"]

    broken = client.post("/chat", json={"message": "/broken now", "session_id": sid})
    assert broken.status_code == 500
    assert "could not load skill broken" in broken.json()["detail"]

    assert all(agent.turns == [] for agent in made)
    # the per-session reservation was released both times: a later turn is not 409'd
    assert client.post("/chat", json={"message": "hi", "session_id": sid}).status_code == 200


def test_chat_plain_text_is_untouched(tmp_path: Path) -> None:
    client, _store, made = _build(tmp_path)
    assert client.post("/chat", json={"message": "no slash here"}).status_code == 200
    assert made[0].compose_calls == []
    assert made[0].turns == ["no slash here"]


# ── POST /chat/stream ──────────────────────────────────────────────────────────


def test_chat_stream_slash_dispatches_the_skill(tmp_path: Path) -> None:
    client, _store, made = _build(tmp_path)
    resp = client.post("/chat/stream", json={"message": "/start tricks"})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert [e["event"] for e in events] == ["text", "done"]
    assert made[0].turns[0].startswith("<command-message>start is running")


def test_chat_stream_refusal_is_terminal_frames_not_a_turn(tmp_path: Path) -> None:
    client, _store, made = _build(tmp_path)
    sid = client.post("/sessions").json()["id"]
    resp = client.post("/chat/stream", json={"message": "/hidden", "session_id": sid})
    assert resp.status_code == 200
    events = _sse_events(resp.text)
    assert [e["event"] for e in events] == ["status", "done"]
    assert events[0]["message"] == "hidden is user-invocable: false"
    assert events[1]["stop_reason"] == "skill_refused"
    assert events[1]["degraded"] is True
    assert events[1]["error"] == "hidden is user-invocable: false"
    assert made[0].turns == []
    # reservation released: the session takes a later turn
    assert client.post("/chat", json={"message": "hi", "session_id": sid}).status_code == 200


# ── the helper itself ──────────────────────────────────────────────────────────


def test_dispatch_slash_thin_agent_without_skills_surface_is_prose() -> None:
    outcome = asyncio.run(dispatch_slash(object(), "/start tricks"))
    assert outcome == SlashDispatch(handled=False)


def test_dispatch_slash_bare_slash_is_prose(tmp_path: Path) -> None:
    agent = _SkillAgent(Session(cwd=str(tmp_path), model="scripted/test"))
    outcome = asyncio.run(dispatch_slash(agent, "/"))
    assert outcome.handled is False
    assert agent.compose_calls == [("", "")]
