"""Small-model containment (ADR-0024): degenerate tool arguments + the false "done".

Field incident 2026-08-26 (small local model): one turn carried a python -c payload whose
arguments had collapsed into repetition ("import json; " ×28, "YOUR_" ×38) — executed
unjudged, because the completion-text guard deliberately skips tool-call batches — and a
later turn ENDED on "Now I will use the `create_file` command … I will then use `mv` …"
with none of it done. These tests pin the argument veto, the false-done nudge on both
turn paths, and the struggle latch.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from zakcode.agent.degeneration import burst_repetition
from zakcode.agent.loop import _INTENT_NUDGE, AgentLoop, _announces_future_work
from zakcode.events import AgentStatus
from zakcode.messages import Message, ToolResultBlock
from zakcode.providers.base import (
    Capabilities,
    LLMResult,
    Provider,
    ProviderStreamEvent,
    StreamDone,
    StreamTextDelta,
    ToolCall,
)
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry
from zakcode.tools.builtins.write_file import WriteFileTool

# ── burst_repetition (pure detector) ─────────────────────────────────────────


def test_convicts_the_import_flood() -> None:
    blob = ("import json; " * 28) + "\ndef list_files(folder=None):\n    return []\n"
    verdict = burst_repetition(blob)
    assert verdict is not None
    unit, repeats = verdict
    assert "import json" in unit * 2  # phase-shifted unit still spells the fragment
    assert repeats >= 12


def test_convicts_the_token_stutter_mid_text() -> None:
    blob = 'access_token = "' + "YOUR_" * 38 + 'TOKEN_HERE"\n# more code follows\n' + "x = 1\n" * 20
    verdict = burst_repetition(blob)
    assert verdict is not None
    _unit, repeats = verdict
    assert repeats >= 12


def test_healthy_code_is_not_convicted() -> None:
    code = "\n".join(f"def handler_{i}(x):\n    return x + {i}" for i in range(60))
    assert burst_repetition(code) is None


def test_divider_lines_and_padding_are_never_convicted() -> None:
    assert burst_repetition("-" * 400) is None  # single-char unit: formatting
    assert burst_repetition("\n" * 400) is None
    assert burst_repetition("print('=' * 60)\n" + "=" * 300) is None


def test_short_runs_stay_legal() -> None:
    assert burst_repetition("0, " * 8) is None  # a small literal array is not a loop


# ── the argument veto at the execution seam ──────────────────────────────────


class _ScriptProvider(Provider):
    def __init__(self, results: list[LLMResult]) -> None:
        self._results = results
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.calls += 1
        return self._results[self.calls - 1]

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def _loop(results: list[LLMResult], tmp_path: Path) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(WriteFileTool())
    return AgentLoop(
        _ScriptProvider(results),
        registry,
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=8,
    )


def _tool_blocks(loop: AgentLoop) -> list[ToolResultBlock]:
    return [b for m in loop.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def test_degenerate_arguments_are_vetoed_not_executed(tmp_path: Path) -> None:
    degenerate = ("import json; " * 30) + "print('hi')\n"
    loop = _loop(
        [
            LLMResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        arguments={"path": "out.py", "content": degenerate},
                    )
                ]
            ),
            LLMResult(text="understood"),
        ],
        tmp_path,
    )
    result = asyncio.run(loop.arun_turn("write the script"))
    assert result.stop_reason == "completed"
    blocks = _tool_blocks(loop)
    assert len(blocks) == 1
    assert blocks[0].is_error is True
    assert "degenerated into repetition" in blocks[0].output
    assert "was not executed" in blocks[0].output
    assert not (tmp_path / "out.py").exists()
    assert loop._turn_struggle is True  # zakpick sees a struggle signal


def test_clean_arguments_still_execute(tmp_path: Path) -> None:
    loop = _loop(
        [
            LLMResult(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="write_file",
                        # non-runnable target: keeps the quality gate (which would spend a
                        # third provider call scoring a runnable write) out of this control
                        arguments={"path": "notes.txt", "content": "hello\n"},
                    )
                ]
            ),
            LLMResult(text="done"),
        ],
        tmp_path,
    )
    result = asyncio.run(loop.arun_turn("write it"))
    assert result.stop_reason == "completed"
    assert (tmp_path / "notes.txt").exists()
    assert loop._turn_struggle is False


# ── the false-done guard ─────────────────────────────────────────────────────


def test_future_intent_matcher_is_surgical() -> None:
    assert _announces_future_work("Now I will use the `create_file` command to create it.")
    assert _announces_future_work("I will then use `mv` to move the file and make it executable.")
    assert _announces_future_work("Let me now edit the registry entry.")
    assert not _announces_future_work("I created the file and registered the skill; done.")
    assert not _announces_future_work("I'll let you know if anything changes.")
    assert not _announces_future_work("I will need you to provide the API key first.")


def test_completion_announcing_work_is_nudged_once(tmp_path: Path) -> None:
    loop = _loop(
        [
            LLMResult(text="Now I will use the create_file command to create the file."),
            LLMResult(text="Nothing more was actually needed; finishing."),
        ],
        tmp_path,
    )
    result = asyncio.run(loop.arun_turn("forge the skill"))
    assert result.stop_reason == "completed"
    assert loop.provider.calls == 2  # type: ignore[attr-defined]
    nudges = [m for m in loop.session.messages if m.role == "user" and _INTENT_NUDGE[:40] in m.text]
    assert len(nudges) == 1
    assert nudges[0].text.startswith("[harness] Hint:")


def test_reporting_completion_is_not_nudged(tmp_path: Path) -> None:
    loop = _loop([LLMResult(text="I created the file and verified it; done.")], tmp_path)
    result = asyncio.run(loop.arun_turn("forge the skill"))
    assert result.stop_reason == "completed"
    assert loop.provider.calls == 1  # type: ignore[attr-defined]


def test_intent_nudge_fires_at_most_once_per_turn(tmp_path: Path) -> None:
    loop = _loop(
        [
            LLMResult(text="Now I will create the file."),
            LLMResult(text="Next, I will register the skill."),  # still announcing — no 3rd ask
        ],
        tmp_path,
    )
    result = asyncio.run(loop.arun_turn("forge the skill"))
    assert result.stop_reason == "completed"
    assert loop.provider.calls == 2  # type: ignore[attr-defined]


class _StreamScript(Provider):
    """Streams each scripted text as one delta + done."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:  # pragma: no cover — streaming path only
        raise AssertionError("buffered path must not run")

    async def astream(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> AsyncIterator[ProviderStreamEvent]:
        self.calls += 1
        yield StreamTextDelta(text=self._texts[self.calls - 1])
        yield StreamDone(finish_reason="stop")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=200_000)


def test_streaming_completion_announcing_work_is_nudged(tmp_path: Path) -> None:
    provider = _StreamScript(
        [
            "Now I will use the create_file command to create the file.",
            "Nothing more was actually needed; finishing.",
        ]
    )
    loop = AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=8,
    )

    async def _collect() -> list[Any]:
        return [ev async for ev in loop.astream_turn("forge the skill")]

    events = asyncio.run(_collect())
    assert provider.calls == 2
    statuses = [ev.message for ev in events if isinstance(ev, AgentStatus)]
    assert any("asking for it" in s for s in statuses)


# ── the broken-record guard (ADR-0026) ───────────────────────────────────────


def _veto_once_hook() -> Any:
    """A Stop-hook stand-in: veto the first turn end, allow after (the Mind loop shape)."""
    from zakcode.hooks import TurnEndResult

    vetoes = [0]

    def hook(payload: Any) -> TurnEndResult | None:
        if vetoes[0] < 1:
            vetoes[0] += 1
            return TurnEndResult(vetoed=True, continuation_prompt="continue with the work")
        return None

    return hook


def test_parroted_completion_gets_the_broken_record_rail(tmp_path: Path) -> None:
    stale = (
        "The plan shows all steps are complete and the loop is active. "
        "No further action is needed - the perpetual learning loop is running."
    )
    loop = _loop(
        [
            LLMResult(text=stale),
            LLMResult(text=stale),  # verbatim re-send after the veto re-prompt
            LLMResult(text="New information: claimed goal g-1 and started execution."),
        ],
        tmp_path,
    )
    loop.turn_end_vetoable = True  # the Mind loop runs with a vetoable Stop seam
    loop.hook_manager.register_turn_end(_veto_once_hook())
    result = asyncio.run(loop.arun_turn("keep the loop going"))
    assert result.stop_reason == "completed"
    assert loop.provider.calls == 3  # type: ignore[attr-defined]
    rails = [
        m
        for m in loop.session.messages
        if m.role == "user" and "already said exactly this" in m.text
    ]
    assert len(rails) == 1
    assert rails[0].text.startswith("[harness] Hint:")
    assert loop._turn_struggle is True  # parroting is a struggle signal


def test_distinct_completions_are_not_parroting(tmp_path: Path) -> None:
    loop = _loop(
        [
            LLMResult(text="Precheck finished: fourteen goals scored and ranked for selection."),
            LLMResult(text="Selection finished: claimed goal g-1; execution starts next cycle."),
        ],
        tmp_path,
    )
    loop.turn_end_vetoable = True  # the Mind loop runs with a vetoable Stop seam
    loop.hook_manager.register_turn_end(_veto_once_hook())
    result = asyncio.run(loop.arun_turn("keep the loop going"))
    assert result.stop_reason == "completed"
    assert not any("already said exactly this" in m.text for m in loop.session.messages)


def test_short_repeats_stay_below_the_floor(tmp_path: Path) -> None:
    loop = _loop([LLMResult(text="Done."), LLMResult(text="Done.")], tmp_path)
    loop.turn_end_vetoable = True  # the Mind loop runs with a vetoable Stop seam
    loop.hook_manager.register_turn_end(_veto_once_hook())
    result = asyncio.run(loop.arun_turn("quick check"))
    assert result.stop_reason == "completed"
    assert not any("already said exactly this" in m.text for m in loop.session.messages)
