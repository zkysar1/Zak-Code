"""ADR-0187: a turn-end veto that names a skill re-entry is DELIVERED — the harness composes
the skill the way the say inbox runs a typed slash — and the veto-stall fence bounds a loop
whose model never runs a skill; a stall leaves a wake-up behind.

Hermetic: scripted providers, an in-memory tool registry, a fake composer, ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import (
    _MAX_PLAN_NUDGES,
    _VETO_STALL_THRESHOLD,
    AgentLoop,
    harness_skill_turn_text,
    skill_reentry_in,
)
from zakcode.config import load_settings
from zakcode.events import AgentStatus
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import RESUME_COMPACT_STOP_REASONS, Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec
from zakcode.wakeup import LOOP_SENTINEL

# ── the framework's own words (stop-hook.sh, verbatim shapes) ────────────────

REDUCER_REASON = (
    "Turn ended without a Skill(aspirations) re-entry (autocompact OR a text summary "
    "terminated the turn). Your FIRST action MUST be: Skill('aspirations') with args='loop'. "
    "Do NOT manually select goals. Do NOT run Bash commands first. Call the Skill tool "
    "IMMEDIATELY. Agent: sera. Prefix all Bash with AYOAI_AGENT=sera."
)
WORKER_REASON = (
    "Worker Body turn ended without a Skill(worker-loop) re-entry (a text summary or "
    "autocompact terminated the turn). Your FIRST action MUST be: Skill('worker-loop') — NOT "
    "Skill('aspirations'), which is the REDUCER-only re-entry (guard-517/guard-463). Do NOT "
    "emit a text summary first."
)
PLAIN_REASON = "Not done: verify your work."

ASPIRATIONS_TURN = (
    "<command-message>aspirations is running</command-message>\n"
    "<command-name>/aspirations</command-name>\n"
    "<command-args>loop</command-args>\n\n"
    "# Aspirations\n\n## Phase -1.5: Enter\n\nread the state\n\n## Phase 0: Select\n\npick a goal\n"
)
#: The same skill without sections: nothing is seeded into the plan, so a text-only
#: completion reaches the Stop-hook seam directly instead of the open-plan gate first —
#: the fence tests count vetoes, not plan nudges.
FLAT_TURN = ASPIRATIONS_TURN.split("# Aspirations", 1)[0] + "# Aspirations\n\nEnter the loop.\n"


# ── fixtures ─────────────────────────────────────────────────────────────────


class ScriptedProvider(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(self, messages: list[Message], *, system=None, tools=None, **kw):  # type: ignore[override]
        self.calls += 1
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(context_window=8192)


class EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


class UseSkillStub(Tool):
    spec = ToolSpec(name="Skill", description="Load a skill by name.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=f"[skill body of {args.get('name')}]")


class RecordingHook:
    def __init__(self, results: list[TurnEndResult | None]) -> None:
        self._results = list(results)
        self.payloads: list[TurnEndPayload] = []

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        self.payloads.append(payload)
        if not self._results:
            return None
        return self._results.pop(0)


class _Composed:
    def __init__(self, **kw: Any) -> None:
        self.invoked = kw.get("invoked", True)
        self.name = kw.get("name", "")
        self.turn_text = kw.get("turn_text")
        self.denied_reason = kw.get("denied_reason")
        self.error = kw.get("error")


def _composer(
    calls: list[tuple[str, str, str]],
    *,
    outcome: str = "deliver",
    turn_text: str = ASPIRATIONS_TURN,
):
    """A fake ``compose_skill_turn`` with the ``source`` seam, scripted per outcome."""

    async def compose(name: str, args: str = "", *, fuzzy: bool = True, source: str = "command"):
        calls.append((name, args, source))
        if outcome == "unknown":
            return _Composed(invoked=False)
        if outcome == "denied":
            return _Composed(name=name, denied_reason=f"/{name} is user-only")
        return _Composed(name=name, turn_text=turn_text)

    return compose


async def _legacy_composer(name: str, args: str = "", *, fuzzy: bool = True) -> _Composed:
    """A composer WITHOUT the ``source`` seam (a stand-in predating ADR-0187)."""
    return _Composed(name=name, turn_text=ASPIRATIONS_TURN)


def _veto(reason: str) -> TurnEndResult:
    return TurnEndResult(vetoed=True, continuation_prompt=reason)


def _texts(n: int) -> list[LLMResult]:
    """``n`` distinct text-only completions: the turn keeps ending in words (the spiral's
    shape) without tripping the broken-record guard, which is a different rail (ADR-0026)."""
    return [LLMResult(text=f"Verdict: loop started ({i})") for i in range(n)]


_TEXT = _texts(1)[0]


def _loop(provider: Provider, tmp_path: Path, compose: Any) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(UseSkillStub())
    return AgentLoop(
        provider,
        registry,
        Session(cwd=str(tmp_path), model="test/model"),
        settings=load_settings(workspace_root=tmp_path),
        max_iterations=20,
        turn_end_vetoable=True,
        compose_skill=compose,
    )


def _delivered(loop: AgentLoop) -> list[Message]:
    return [
        m
        for m in loop.session.messages
        if m.role == "user" and m.text.startswith("<command-message>aspirations is running")
    ]


# ── the parser ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (REDUCER_REASON, ("aspirations", "loop")),
        (WORKER_REASON, ("worker-loop", "")),
        ('Call Skill(skill="aspirations", args="loop") now.', ("aspirations", "loop")),
        ("Run Skill(skill='worker-loop') as your first action.", ("worker-loop", "")),
        (
            "Skill(aspirations-spark) first, then Skill(aspirations) with args='loop'.",
            ("aspirations", "loop"),
        ),
        (
            "Call Skill(aspirations) with args='loop' as your VERY NEXT tool call.",
            ("aspirations", "loop"),
        ),
        (PLAIN_REASON, None),
        ("Continue.", None),
        ("NOT Skill('aspirations') — this Body is closed.", None),
        ("", None),
    ],
)
def test_skill_reentry_in_reads_both_harnesses_vocabulary(reason: str, expected) -> None:
    assert skill_reentry_in(reason) == expected


def test_harness_skill_turn_text_folds_the_note_into_the_frame() -> None:
    text = harness_skill_turn_text(
        ASPIRATIONS_TURN, "a turn-end hook asked for it:\n  " + PLAIN_REASON
    )
    first, rest = text.split("\n", 1)
    # One line, [harness]-tagged, the note flattened; the rest of the turn byte-identical.
    assert first == (
        "<command-message>aspirations is running — [harness] a turn-end hook asked for it: "
        "Not done: verify your work.</command-message>"
    )
    assert rest == ASPIRATIONS_TURN.split("\n", 1)[1]
    # Not a composed turn, or nothing to say: unchanged.
    assert harness_skill_turn_text("plain text", "note") == "plain text"
    assert harness_skill_turn_text(ASPIRATIONS_TURN, "   ") == ASPIRATIONS_TURN


# ── delivery ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_veto_naming_a_skill_delivers_that_skill(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []
    hook = RecordingHook([_veto(REDUCER_REASON), None])
    # The delivered skill seeds plan steps, so the model's next text-only finish meets the
    # open-plan gate before the Stop hook again: script the nudges it takes to fall through.
    provider = ScriptedProvider(_texts(2 + _MAX_PLAN_NUDGES))
    loop = _loop(provider, tmp_path, _composer(calls))
    loop.hook_manager.register_turn_end(hook)

    result = await loop.arun_turn("start the loop")

    assert result.stop_reason == "completed"
    assert len(hook.payloads) == 2
    # The composer ran the skill the hook named, with its args, as the HARNESS (not a human).
    assert calls == [("aspirations", "loop", "harness")]
    # The re-entry message IS the composed turn: frame first (provenance, elision, the
    # transcript all key on it), the hook's reason folded into the frame's message line,
    # then the body — never a rail asking the model to fetch it.
    delivered = _delivered(loop)
    assert len(delivered) == 1
    assert not any(  # the hook's reason never rides a rail of its own
        m.role == "user"
        and m.text.startswith("[harness] Hint:")
        and "Skill('aspirations')" in m.text
        for m in loop.session.messages
    )
    # …and the provider saw exactly that message on the re-entry call.
    assert loop.session.loop_skill == "aspirations loop"  # the sentinel wake-up resolves to it
    # The skill's sections were seeded into the plan, like a turn-opening skill.
    titles = [t.title for t in loop.session.task_network.tasks]
    assert any("Enter" in t for t in titles) and any("Select" in t for t in titles)
    # At turn end the body is elided (ADR-0045) — the frame stays, with the hook's words.
    head = delivered[0].text.split("\n", 1)[0]
    assert head.startswith(
        "<command-message>aspirations is running — [harness] a turn-end hook asked for it: "
        "Turn ended without"
    )
    assert "<command-body elided" in delivered[0].text


@pytest.mark.asyncio
async def test_the_provider_sees_the_delivered_skill_on_the_re_entry_call(tmp_path: Path) -> None:
    class Seeing(ScriptedProvider):
        def __init__(self, results):
            super().__init__(results)
            self.seen: list[list[Message]] = []

        async def acomplete(self, messages, *, system=None, tools=None, **kw):  # type: ignore[override]
            self.seen.append(list(messages))
            return await super().acomplete(messages, system=system, tools=tools, **kw)

    hook = RecordingHook([_veto(REDUCER_REASON), None])
    provider = Seeing(_texts(2 + _MAX_PLAN_NUDGES))
    loop = _loop(provider, tmp_path, _composer([]))
    loop.hook_manager.register_turn_end(hook)
    await loop.arun_turn("start the loop")
    composed = [
        m
        for m in provider.seen[1]
        if m.role == "user"
        and m.text.startswith("<command-message>aspirations is running — [harness]")
    ]
    assert len(composed) == 1
    assert "<command-args>loop</command-args>" in composed[0].text
    assert "## Phase -1.5: Enter" in composed[0].text
    assert "Skill('aspirations') with args='loop'" in composed[0].text  # the hook's own words
    # The skeleton's rail ("I added the sections … to your plan") FOLLOWS the body it
    # points into — the order a typed slash gets — never precedes it.
    users = [m.text for m in provider.seen[1] if m.role == "user"]
    body_at = next(i for i, t in enumerate(users) if t.startswith("<command-message>"))
    rail_at = next(i for i, t in enumerate(users) if "to your plan as steps" in t)
    assert body_at < rail_at


@pytest.mark.asyncio
async def test_the_streaming_twin_names_the_delivered_skill(tmp_path: Path) -> None:
    hook = RecordingHook([_veto(WORKER_REASON), None])
    provider = ScriptedProvider(_texts(2 + _MAX_PLAN_NUDGES))
    calls: list[tuple[str, str, str]] = []
    loop = _loop(provider, tmp_path, _composer(calls))
    loop.hook_manager.register_turn_end(hook)
    events = [e async for e in loop.astream_turn("go")]
    assert calls == [("worker-loop", "", "harness")]
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert "turn_end hook vetoed stop; /worker-loop delivered" in statuses


@pytest.mark.asyncio
async def test_a_veto_naming_no_skill_is_the_plain_rail_as_before(tmp_path: Path) -> None:
    calls: list[tuple[str, str, str]] = []
    hook = RecordingHook([_veto(PLAIN_REASON), None])
    provider = ScriptedProvider([_TEXT, LLMResult(text="done")])
    loop = _loop(provider, tmp_path, _composer(calls))
    loop.hook_manager.register_turn_end(hook)
    events = [e async for e in loop.astream_turn("go")]
    assert calls == []  # nothing to compose
    rails = [m for m in loop.session.messages if m.role == "user" and "verify your work" in m.text]
    assert len(rails) == 1 and rails[0].text.startswith("[harness] Hint:")
    assert loop.session.loop_skill == ""
    statuses = [e.message for e in events if isinstance(e, AgentStatus)]
    assert "turn_end hook vetoed stop; continuing" in statuses


@pytest.mark.parametrize("outcome", ["unknown", "denied", "legacy"])
@pytest.mark.asyncio
async def test_a_skill_that_cannot_be_delivered_falls_back_to_the_rail(
    tmp_path: Path, outcome: str
) -> None:
    calls: list[tuple[str, str, str]] = []
    compose = _legacy_composer if outcome == "legacy" else _composer(calls, outcome=outcome)
    hook = RecordingHook([_veto(REDUCER_REASON), None])
    provider = ScriptedProvider([_TEXT, LLMResult(text="done")])
    loop = _loop(provider, tmp_path, compose)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "completed"
    assert _delivered(loop) == []
    rails = [
        m
        for m in loop.session.messages
        if m.role == "user" and m.text.startswith("[harness] Hint:")
    ]
    assert len(rails) == 1 and "Skill('aspirations') with args='loop'" in rails[0].text
    assert loop.session.loop_skill == ""


@pytest.mark.asyncio
async def test_no_composer_means_the_rail(tmp_path: Path) -> None:
    hook = RecordingHook([_veto(REDUCER_REASON), None])
    provider = ScriptedProvider([_TEXT, LLMResult(text="done")])
    loop = _loop(provider, tmp_path, None)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "completed"
    assert _delivered(loop) == []


# ── the fence ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_fence_ends_a_turn_whose_model_never_runs_the_delivered_skill(
    tmp_path: Path,
) -> None:
    """The measured spiral (2026-09-17, serene): text → BLOCK naming Skill('aspirations') →
    text → BLOCK …, for hours. Three deliveries are honoured; the fourth such veto ends the
    turn as ``veto_stall`` with a wake-up behind it."""
    vetoes = _VETO_STALL_THRESHOLD + 3
    hook = RecordingHook([_veto(REDUCER_REASON)] * vetoes)
    provider = ScriptedProvider(_texts(vetoes + 1))
    calls: list[tuple[str, str, str]] = []
    loop = _loop(provider, tmp_path, _composer(calls, turn_text=FLAT_TURN))
    loop.hook_manager.register_turn_end(hook)

    result = await loop.arun_turn("start the loop")

    assert result.stop_reason == "veto_stall"
    assert result.degraded is True
    assert result.iterations == _VETO_STALL_THRESHOLD + 1  # three re-entries, then the stop
    assert len(hook.payloads) == _VETO_STALL_THRESHOLD + 1  # consulted; its 4th veto refused
    assert len(calls) == _VETO_STALL_THRESHOLD  # delivered three times, never a fourth
    assert len(_delivered(loop)) == _VETO_STALL_THRESHOLD
    assert loop.session.last_stop_reason == "veto_stall"
    assert "veto_stall" in RESUME_COMPACT_STOP_REASONS  # a resume drops the spiral
    # The net: no wake-up was held, so the harness armed the sentinel — which the REPL
    # resolves to the very skill the hook asked for (Session.loop_skill).
    held = loop.wakeup_slot.pending()
    assert held is not None and held.prompt == LOOP_SENTINEL
    assert loop.session.loop_skill == "aspirations loop"


@pytest.mark.asyncio
async def test_the_fence_keeps_a_wakeup_the_framework_already_held(tmp_path: Path) -> None:
    vetoes = _VETO_STALL_THRESHOLD + 1
    hook = RecordingHook([_veto(REDUCER_REASON)] * vetoes)
    provider = ScriptedProvider(_texts(vetoes + 1))
    loop = _loop(provider, tmp_path, _composer([], turn_text=FLAT_TURN))
    loop.hook_manager.register_turn_end(hook)
    loop.wakeup_slot.arm("poll the reducer", 900)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "veto_stall"
    held = loop.wakeup_slot.pending()
    assert held is not None and held.prompt == "poll the reducer"  # the framework's net wins


@pytest.mark.asyncio
async def test_a_model_skill_call_starts_the_fence_over(tmp_path: Path) -> None:
    """A use_skill call between vetoes is the model following the loop: the count restarts."""
    vetoes = 6
    hook = RecordingHook([_veto(REDUCER_REASON)] * (vetoes + 1))
    texts = _texts(6)
    provider = ScriptedProvider(
        [
            texts[0],  # veto 1 (count 1)
            texts[1],  # veto 2 (count 2)
            LLMResult(
                tool_calls=[
                    ToolCall(id="s1", name="Skill", arguments={"name": "aspirations-execute"})
                ]
            ),
            texts[2],  # veto 3 → the skill ran: count 1
            texts[3],  # veto 4 (count 2)
            texts[4],  # veto 5 (count 3)
            texts[5],  # veto 6: refused — veto_stall
        ]
    )
    loop = _loop(provider, tmp_path, _composer([], turn_text=FLAT_TURN))
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "veto_stall"
    assert len(hook.payloads) == vetoes  # without the reset it would have stopped at 4
    assert result.iterations == vetoes + 1


@pytest.mark.asyncio
async def test_generic_vetoes_stay_unbounded(tmp_path: Path) -> None:
    """A Stop hook naming no skill keeps Claude Code's contract: it stands down, not the loop."""
    vetoes = _VETO_STALL_THRESHOLD + 4
    hook = RecordingHook([_veto(PLAIN_REASON)] * vetoes + [None])
    provider = ScriptedProvider(_texts(vetoes + 1))
    loop = _loop(provider, tmp_path, _composer([]))
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "completed"
    assert len(hook.payloads) == vetoes + 1
    assert loop.wakeup_slot.pending() is None


@pytest.mark.asyncio
async def test_the_fence_applies_to_the_streaming_twin(tmp_path: Path) -> None:
    vetoes = _VETO_STALL_THRESHOLD + 1
    hook = RecordingHook([_veto(REDUCER_REASON)] * vetoes)
    provider = ScriptedProvider(_texts(vetoes + 1))
    loop = _loop(provider, tmp_path, _composer([], turn_text=FLAT_TURN))
    loop.hook_manager.register_turn_end(hook)
    events = [e async for e in loop.astream_turn("go")]
    done = events[-1]
    assert getattr(done, "stop_reason", None) == "veto_stall"
    assert getattr(done, "degraded", None) is True


@pytest.mark.asyncio
async def test_the_fence_is_per_turn(tmp_path: Path) -> None:
    """A new turn starts with a clean count: honoured vetoes in one turn do not tax the next."""
    hook = RecordingHook(
        [_veto(REDUCER_REASON)] * _VETO_STALL_THRESHOLD + [None, _veto(REDUCER_REASON), None]
    )
    provider = ScriptedProvider(_texts(_VETO_STALL_THRESHOLD + 1) + [_TEXT, LLMResult(text="done")])
    loop = _loop(provider, tmp_path, _composer([], turn_text=FLAT_TURN))
    loop.hook_manager.register_turn_end(hook)
    first = await loop.arun_turn("go")
    assert first.stop_reason == "completed"  # three honoured, then the hook allowed the stop
    second = await loop.arun_turn("again")
    assert second.stop_reason == "completed"  # its one veto honoured: the count was fresh
    assert second.iterations == 2
