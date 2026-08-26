"""TURN_END loop integration (T2/T3/T4): the Stop-hook veto seam in the agent loop.

Covers the design acceptance matrix: budget-zero = byte-identical default (no hook
fires), allow vs veto on "completed", per-turn budget exhaustion, the non-vetoable
stop reasons (max_iterations / provider_error / recipe_stalled), doom-loop veto with
the synthetic tool_result pairing fix + stall-state reset, payload contents, the
streaming twin's AgentStatus announcement, and the Settings/Agent plumbing.

Everything is hermetic: scripted in-memory providers (no network, no model), a tiny
in-memory tool registry, and a ``tmp_path`` workspace.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import zakcode
from zakcode.agent.loop import DOOM_LOOP_THRESHOLD, AgentLoop
from zakcode.config import load_settings
from zakcode.events import AgentDone, AgentStatus
from zakcode.hooks import TurnEndPayload, TurnEndResult
from zakcode.messages import Message
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    RequestFailed,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

# ── scripted providers / tools (test_loop_edge.py patterns) ───────────────────


class ScriptedProvider(Provider):
    """Replays a fixed list of :class:`LLMResult`s, one per ``acomplete`` call."""

    def __init__(self, results: list[LLMResult]) -> None:
        self._results = list(results)
        self.calls = 0

    async def acomplete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        self.calls += 1
        if not self._results:
            raise AssertionError("provider ran out of scripted results")
        return self._results.pop(0)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()


class FailingProvider(ScriptedProvider):
    """Raises ``RequestFailed`` on every call (drives stop_reason=provider_error)."""

    def __init__(self) -> None:
        super().__init__([])

    async def acomplete(self, messages, *, system=None, tools=None, **kwargs):  # type: ignore[override]
        raise RequestFailed("scripted failure")


class EchoTool(Tool):
    spec = ToolSpec(name="echo", description="Echo back the provided text.")

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(output=str(args.get("text", "")))


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


def _make_loop(
    provider: Provider,
    tmp_path: Path,
    *,
    max_iterations: int = 10,
    turn_end_vetoable: bool = True,
    **settings_over: Any,
) -> AgentLoop:
    settings = load_settings(workspace_root=tmp_path, **settings_over)
    return AgentLoop(
        provider,
        _registry(),
        Session(cwd=str(tmp_path), model="test/model"),
        settings=settings,
        max_iterations=max_iterations,
        turn_end_vetoable=turn_end_vetoable,
    )


class RecordingHook:
    """In-process TURN_END hook scripted with per-call results (None == allow)."""

    def __init__(self, results: list[TurnEndResult | None]) -> None:
        self._results = list(results)
        self.payloads: list[TurnEndPayload] = []

    def __call__(self, payload: TurnEndPayload) -> TurnEndResult | None:
        self.payloads.append(payload)
        if not self._results:
            return None
        return self._results.pop(0)


def _veto(reason: str = "Not done: verify your work.") -> TurnEndResult:
    return TurnEndResult(vetoed=True, continuation_prompt=reason)


_TEXT_DONE = LLMResult(text="final answer")


def _same_call(i: int) -> LLMResult:
    """An identical (name, arguments) echo batch — doom-loop fodder; ids vary."""
    return LLMResult(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": "same"})])


# ── non-vetoable loop (the sub-agent shape): seam structurally off ────────────


@pytest.mark.asyncio
async def test_turn_end_subagent_loop_never_vetoes(tmp_path: Path) -> None:
    """A loop built without ``turn_end_vetoable`` (the sub-agent construction shape)
    never consults veto hooks — a Stop hook must not resurrect a sub-agent whose
    completion returns to its parent."""
    hook = RecordingHook([_veto()])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path, turn_end_vetoable=False)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert hook.payloads == []  # gate short-circuits before any payload is built


# ── allow / veto on "completed" ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_hook_allows_completed(tmp_path: Path) -> None:
    hook = RecordingHook([None])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.iterations == 1
    assert len(hook.payloads) == 1


@pytest.mark.asyncio
async def test_turn_end_hook_vetoes_completed_once(tmp_path: Path) -> None:
    hook = RecordingHook([_veto("Run the tests before declaring done."), None])
    provider = ScriptedProvider([_TEXT_DONE, LLMResult(text="ran them; done")])
    loop = _make_loop(provider, tmp_path)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.iterations == 2  # re-entered once
    assert len(hook.payloads) == 2
    # The continuation prompt rides a control-rail user message in the session, carrying
    # the [harness] provenance tag (ADR-0021) so it can never read as the user speaking.
    injected = [m for m in loop.session.messages if m.role == "user" and "Run the tests" in m.text]
    assert len(injected) == 1
    assert injected[0].text.startswith("[harness] Hint:")


@pytest.mark.asyncio
async def test_turn_end_vetoes_are_unbounded(tmp_path: Path) -> None:
    """No per-turn veto cap (no-knobs ruling): the hook is consulted at EVERY vetoable
    stop and is itself in charge of standing down (here: after three vetoes)."""
    hook = RecordingHook([_veto(), _veto(), _veto(), None])
    provider = ScriptedProvider([_TEXT_DONE, _TEXT_DONE, _TEXT_DONE, _TEXT_DONE])
    loop = _make_loop(provider, tmp_path)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.iterations == 4  # 3 vetoes consumed, the hook then allowed the stop
    assert len(hook.payloads) == 4  # consulted every time — no budget short-circuit


# ── non-vetoable stop reasons ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_max_iterations_not_vetoable(tmp_path: Path) -> None:
    hook = RecordingHook([_veto()])
    # Distinct calls each iteration: only the cap can stop this turn.
    provider = ScriptedProvider(
        [
            LLMResult(tool_calls=[ToolCall(id=f"c{i}", name="echo", arguments={"text": str(i)})])
            for i in range(4)
        ]
    )
    loop = _make_loop(provider, tmp_path, max_iterations=2)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "max_iterations"
    assert hook.payloads == []


@pytest.mark.asyncio
async def test_turn_end_provider_error_not_vetoable(tmp_path: Path) -> None:
    hook = RecordingHook([_veto()])
    loop = _make_loop(FailingProvider(), tmp_path)  # RequestFailed is never retried
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "provider_error"
    assert hook.payloads == []


@pytest.mark.asyncio
async def test_turn_end_fire_refuses_non_vetoable_reasons(tmp_path: Path) -> None:
    # Unit check on the gate itself: even with budget + hooks, the non-vetoable
    # reasons (incl. recipe_stalled, whose full ladder is heavy to script) get None.
    hook = RecordingHook([_veto(), _veto(), _veto()])
    loop = _make_loop(ScriptedProvider([]), tmp_path)
    loop.hook_manager.register_turn_end(hook)
    for reason in ("recipe_stalled", "max_iterations", "provider_error"):
        prompt = await loop._fire_turn_end(
            reason, iterations=1, veto_count=0, turn_assistant=[], stuck_took_action=False
        )
        assert prompt is None
    assert hook.payloads == []


# ── doom-loop veto: pairing fix + stall-state reset ───────────────────────────


@pytest.mark.asyncio
async def test_turn_end_doom_loop_veto_pairs_and_resets(tmp_path: Path) -> None:
    hook = RecordingHook([_veto("Different approach, please."), None])
    # Identical batch forever. Three doom cycles now: the built-in RECOVERY fires first (one nudge
    # + re-enter, the hook is NOT consulted), THEN the turn_end veto re-enters, THEN the final
    # non-veto break. The RESET means each needs a full fresh streak -> 3 * THRESHOLD iterations.
    provider = ScriptedProvider([_same_call(i) for i in range(3 * DOOM_LOOP_THRESHOLD + 1)])
    loop = _make_loop(provider, tmp_path, max_iterations=12)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("repeat")
    assert result.stop_reason == "doom_loop"
    assert result.iterations == 3 * DOOM_LOOP_THRESHOLD
    # The turn_end hook is consulted only after the recovery is spent: twice (veto, then None).
    assert [p.stop_reason for p in hook.payloads] == ["doom_loop", "doom_loop"]
    # Pairing fix: the vetoed batch's tool_use got a synthetic error tool_result, and the FINAL
    # (non-veto) doom_loop break pairs its unexecuted batch too — two `doom_loop_intervention`
    # markers (the recovery's own pairing uses the distinct `doom_recovery` marker, not counted).
    synthetic = [
        b
        for m in loop.session.messages
        if m.role == "tool"
        for b in m.blocks
        if getattr(b, "data", None) and b.data.get("doom_loop_intervention")
    ]
    assert len(synthetic) == 2
    assert all(b.is_error for b in synthetic)


# ── payload contents ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_payload_contents(tmp_path: Path) -> None:
    hook = RecordingHook([None])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path)
    loop.hook_manager.register_turn_end(hook)
    await loop.arun_turn("hi")
    payload = hook.payloads[0]
    assert payload.stop_reason == "completed"
    assert payload.session_id == loop.session.id
    assert payload.cwd == str(tmp_path)
    assert payload.iterations == 1
    assert payload.max_iterations == 10
    assert payload.degraded is False
    assert payload.last_assistant_message == "final answer"


# ── streaming twin ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_streaming_veto_yields_status(tmp_path: Path) -> None:
    hook = RecordingHook([_veto("Keep going."), None])
    provider = ScriptedProvider([_TEXT_DONE, LLMResult(text="done now")])
    loop = _make_loop(provider, tmp_path)
    loop.hook_manager.register_turn_end(hook)
    events = [ev async for ev in loop.astream_turn("hi")]
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any("turn_end hook vetoed stop" in s for s in statuses)
    done = [ev for ev in events if isinstance(ev, AgentDone)][-1]
    assert done.stop_reason == "completed"
    assert len(hook.payloads) == 2


@pytest.mark.asyncio
async def test_turn_end_streaming_subagent_loop_unchanged(tmp_path: Path) -> None:
    hook = RecordingHook([_veto()])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path, turn_end_vetoable=False)
    loop.hook_manager.register_turn_end(hook)
    events = [ev async for ev in loop.astream_turn("hi")]
    assert [ev for ev in events if isinstance(ev, AgentDone)][-1].stop_reason == "completed"
    assert hook.payloads == []


# ── fail-open ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_crashing_hook_fails_open(tmp_path: Path) -> None:
    def explode(payload: TurnEndPayload) -> TurnEndResult | None:
        raise RuntimeError("hook bug")

    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path)
    loop.hook_manager.register_turn_end(explode)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"  # a broken hook never blocks the stop


# ── settings / Agent plumbing (T4, no-knobs) ──────────────────────────────────


def test_no_turn_end_env_knob(monkeypatch, tmp_path: Path) -> None:
    """The veto-budget knob is GONE (no-knobs ruling): the retired env var is inert
    and Settings carries no field for it."""
    monkeypatch.setenv("ZAKCODE_TURN_END_VETO_BUDGET", "50")
    settings = load_settings(workspace_root=tmp_path)
    assert not hasattr(settings, "turn_end_veto_budget")


def test_agent_main_loop_is_always_vetoable(tmp_path: Path) -> None:
    agent = zakcode.Agent(
        default_model="ollama_chat/qwen2.5",
        workspace_root=tmp_path,
    )
    assert agent.loop.turn_end_vetoable is True


def test_no_max_iterations_setting(monkeypatch, tmp_path: Path) -> None:
    """ZAKCODE_MAX_ITERATIONS is inert: no Settings field — a hard cap is an SDK
    constructor arg only, and the product runs unlimited."""
    monkeypatch.setenv("ZAKCODE_MAX_ITERATIONS", "200")
    settings = load_settings(workspace_root=tmp_path)
    assert not hasattr(settings, "max_iterations")
