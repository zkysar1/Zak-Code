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


def _make_loop(provider: Provider, tmp_path: Path, **settings_over: Any) -> AgentLoop:
    settings_over.setdefault("max_iterations", 10)
    settings = load_settings(workspace_root=tmp_path, **settings_over)
    return AgentLoop(
        provider,
        _registry(),
        Session(cwd=str(tmp_path), model="test/model"),
        settings=settings,
        turn_end_veto_budget=settings.turn_end_veto_budget,
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


# ── budget zero: byte-identical default ───────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_budget_zero_no_hook_fires(tmp_path: Path) -> None:
    hook = RecordingHook([_veto()])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path)  # budget defaults to 0
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert hook.payloads == []  # gate short-circuits before any payload is built


# ── allow / veto / budget on "completed" ──────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_hook_allows_completed(tmp_path: Path) -> None:
    hook = RecordingHook([None])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.iterations == 1
    assert len(hook.payloads) == 1


@pytest.mark.asyncio
async def test_turn_end_hook_vetoes_completed_once(tmp_path: Path) -> None:
    hook = RecordingHook([_veto("Run the tests before declaring done."), None])
    provider = ScriptedProvider([_TEXT_DONE, LLMResult(text="ran them; done")])
    loop = _make_loop(provider, tmp_path, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.iterations == 2  # re-entered once
    assert len(hook.payloads) == 2
    # The continuation prompt rides a control-rail user message in the session.
    injected = [m for m in loop.session.messages if m.role == "user" and "Run the tests" in m.text]
    assert len(injected) == 1
    assert injected[0].text.startswith("Hint:")


@pytest.mark.asyncio
async def test_turn_end_budget_exhausted(tmp_path: Path) -> None:
    hook = RecordingHook([_veto(), _veto(), _veto()])
    provider = ScriptedProvider([_TEXT_DONE, _TEXT_DONE, _TEXT_DONE])
    loop = _make_loop(provider, tmp_path, turn_end_veto_budget=2)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"
    assert result.iterations == 3  # 2 vetoes consumed, third stop is final
    assert len(hook.payloads) == 2  # the third gate short-circuits on the spent budget


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
    loop = _make_loop(provider, tmp_path, max_iterations=2, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("go")
    assert result.stop_reason == "max_iterations"
    assert hook.payloads == []


@pytest.mark.asyncio
async def test_turn_end_provider_error_not_vetoable(tmp_path: Path) -> None:
    hook = RecordingHook([_veto()])
    loop = _make_loop(FailingProvider(), tmp_path, provider_max_retries=0, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "provider_error"
    assert hook.payloads == []


@pytest.mark.asyncio
async def test_turn_end_fire_refuses_non_vetoable_reasons(tmp_path: Path) -> None:
    # Unit check on the gate itself: even with budget + hooks, the non-vetoable
    # reasons (incl. recipe_stalled, whose full ladder is heavy to script) get None.
    hook = RecordingHook([_veto(), _veto(), _veto()])
    loop = _make_loop(ScriptedProvider([]), tmp_path, turn_end_veto_budget=5)
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
    # Identical batch forever; doom fires at the threshold, the veto re-enters,
    # and the RESET means the guard needs a full fresh streak to fire again.
    provider = ScriptedProvider([_same_call(i) for i in range(2 * DOOM_LOOP_THRESHOLD + 1)])
    loop = _make_loop(provider, tmp_path, max_iterations=10, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(hook)
    result = await loop.arun_turn("repeat")
    assert result.stop_reason == "doom_loop"
    # threshold iterations -> veto -> threshold MORE iterations (proves the reset:
    # without it the very next identical batch would re-trip immediately).
    assert result.iterations == 2 * DOOM_LOOP_THRESHOLD
    assert [p.stop_reason for p in hook.payloads] == ["doom_loop", "doom_loop"]
    # Pairing fix: the vetoed batch's tool_use got a synthetic error tool_result,
    # and the FINAL (non-veto) doom_loop break pairs its unexecuted batch too —
    # two interventions total, so the resumed session has no dangling tool_use.
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
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path, turn_end_veto_budget=1)
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
    loop = _make_loop(provider, tmp_path, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(hook)
    events = [ev async for ev in loop.astream_turn("hi")]
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any("turn_end hook vetoed stop" in s for s in statuses)
    done = [ev for ev in events if isinstance(ev, AgentDone)][-1]
    assert done.stop_reason == "completed"
    assert len(hook.payloads) == 2


@pytest.mark.asyncio
async def test_turn_end_streaming_budget_zero_unchanged(tmp_path: Path) -> None:
    hook = RecordingHook([_veto()])
    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path)  # budget 0
    loop.hook_manager.register_turn_end(hook)
    events = [ev async for ev in loop.astream_turn("hi")]
    assert [ev for ev in events if isinstance(ev, AgentDone)][-1].stop_reason == "completed"
    assert hook.payloads == []


# ── fail-open ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_end_crashing_hook_fails_open(tmp_path: Path) -> None:
    def explode(payload: TurnEndPayload) -> TurnEndResult | None:
        raise RuntimeError("hook bug")

    loop = _make_loop(ScriptedProvider([_TEXT_DONE]), tmp_path, turn_end_veto_budget=5)
    loop.hook_manager.register_turn_end(explode)
    result = await loop.arun_turn("hi")
    assert result.stop_reason == "completed"  # a broken hook never blocks the stop


# ── settings / Agent plumbing (T4) ────────────────────────────────────────────


def test_turn_end_budget_parses_from_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ZAKCODE_TURN_END_VETO_BUDGET", "50")
    settings = load_settings(workspace_root=tmp_path)
    assert settings.turn_end_veto_budget == 50


def test_agent_threads_budget_to_loop(tmp_path: Path) -> None:
    agent = zakcode.Agent(
        default_model="ollama_chat/qwen2.5",
        workspace_root=tmp_path,
        turn_end_veto_budget=7,
    )
    assert agent.loop.turn_end_veto_budget == 7


def test_default_budget_is_zero(tmp_path: Path) -> None:
    assert load_settings(workspace_root=tmp_path).turn_end_veto_budget == 0
