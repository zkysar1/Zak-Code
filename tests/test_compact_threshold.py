"""ADR-0240: the compaction threshold leaves the call room for one answer.

Field 2026-09-23 (three worker Bodies on one 131k pod): compaction was ~37% of a Body's
model time. The old threshold, a fixed 0.8 of the window, left ~26k tokens unused on every
call it guarded, while the largest completion across 945 measured calls was 4,468 tokens.
The threshold is now 0.9 of the window where the window can spare the answer's room, pulled
earlier where it cannot, and never below the old 0.8.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.compact import CompactionConfig, Compactor
from zakcode.agent.loop import AgentLoop
from zakcode.messages import Message, ToolResultBlock, ToolUseBlock
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
from zakcode.tools.base import ToolRegistry

BENCH_RUNNER = Path(__file__).resolve().parent.parent / "bench" / "run_task.py"


class _Provider(Provider):
    """A fixed token count, a declared window and output cap, and a canned summary."""

    def __init__(self, *, tokens: int, window: int, max_output: int | None = None) -> None:
        self.tokens = tokens
        self.window = window
        self.max_output = max_output
        self.summaries = 0

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        self.summaries += 1
        return LLMResult(text="summary")

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return self.tokens

    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_tools=True, context_window=self.window, max_output=self.max_output
        )


def _loop(provider: Provider, tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        provider,
        ToolRegistry(),
        Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        compactor=Compactor(),
    )


def _history(n: int) -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out.append(Message.user(f"question {i}"))
        out.append(Message.assistant_text(f"answer {i}"))
    return out


# --- the threshold itself -------------------------------------------------------------------


def test_a_131k_pod_compacts_at_nine_tenths() -> None:
    # The pod declares no output cap, so the reserve is the 4,096 floor. The window spares it
    # (131,072 - 6,553 margin - 4,096 = 120,423), so the 0.9 cap is what binds.
    assert Compactor().threshold(131_072, answer_room=4_096) == 117_964


def test_a_large_answer_reserve_pulls_the_threshold_earlier() -> None:
    # A 200k model with a 64k output cap reserves the 16,384 ceiling: 200,000 - 10,000 - 16,384.
    assert Compactor().threshold(200_000, answer_room=16_384) == 173_616
    # A 32k window with the 4,096 floor: 32,768 - 1,638 - 4,096, under 0.9 of it (29,491).
    assert Compactor().threshold(32_768, answer_room=4_096) == 27_034


def test_the_reserve_never_pulls_below_the_old_threshold() -> None:
    # Ollama's 16,384 cap: the reserve alone would say 11,469, into the fixed floor.
    assert Compactor().threshold(16_384, answer_room=4_096) == 13_107
    # The eval probe's toy window keeps its documented 160.
    assert Compactor().threshold(200, answer_room=4_096) == 160


def test_an_explicit_fraction_under_the_floor_still_wins() -> None:
    # The bench forces its compaction arm with threshold_fraction=0.15 on the full window.
    forced = Compactor(CompactionConfig(threshold_fraction=0.15))
    assert forced.threshold(131_072, answer_room=4_096) == 19_660


def test_should_compact_counts_against_the_answer_room() -> None:
    c = Compactor()
    msgs = [Message.user("x")]

    def over(tokens: int, **kw: int) -> bool:
        return c.should_compact(msgs, context_window=32_768, count_tokens=lambda _m: tokens, **kw)

    assert over(27_035, answer_room=4_096)
    assert not over(27_034, answer_room=4_096)
    # No reserve given: the 0.9 cap alone, since the margin by itself leaves 0.95.
    assert over(29_492)
    assert not over(29_491)


def test_an_unknown_window_still_refuses() -> None:
    with pytest.raises(ValueError):
        Compactor().threshold(None, answer_room=4_096)
    with pytest.raises(ValueError):
        Compactor().should_compact([], context_window=0, count_tokens=lambda _m: 1)


# --- the loop's check -------------------------------------------------------------------------


def test_the_loop_reserves_the_fit_check_s_answer_room(tmp_path: Path) -> None:
    # 28,000 is under the 0.9 cap (29,491) but over 27,034: only the reserve compacts it.
    provider = _Provider(tokens=28_000, window=32_768)
    loop = _loop(provider, tmp_path)
    loop.session.messages.extend(_history(5))
    notice = asyncio.run(loop._maybe_compact())
    assert notice is not None and notice.startswith("context near the window"), notice
    assert provider.summaries == 1


def test_the_loop_leaves_a_count_under_the_threshold_alone(tmp_path: Path) -> None:
    provider = _Provider(tokens=27_000, window=32_768)
    loop = _loop(provider, tmp_path)
    loop.session.messages.extend(_history(5))
    assert asyncio.run(loop._maybe_compact()) is None
    assert provider.summaries == 0


def test_a_model_s_output_cap_moves_its_threshold(tmp_path: Path) -> None:
    # 175,000 of a 200k window: under the 0.9 cap, over the 173,616 a 16,384 reserve allows.
    capped = _Provider(tokens=175_000, window=200_000, max_output=64_000)
    loop = _loop(capped, tmp_path)
    loop.session.messages.extend(_history(5))
    assert asyncio.run(loop._maybe_compact()) is not None
    # The same count on a model with no declared cap reserves only 4,096: no compaction.
    uncapped = _Provider(tokens=175_000, window=200_000)
    loop = _loop(uncapped, tmp_path)
    loop.session.messages.extend(_history(5))
    assert asyncio.run(loop._maybe_compact()) is None


def test_the_anchored_count_floors_what_arrived_since_at_the_clamp_density(
    tmp_path: Path,
) -> None:
    # The anchor measured the prompt through the first two messages; a 30,000-character tool
    # result arrived since. The estimate calls it 100 tokens; the count floors it at 3 chars
    # per token, 10,000, the density the threshold's margin is sized for.
    provider = _Provider(tokens=100, window=131_072)
    loop = _loop(provider, tmp_path)
    loop.session.messages.extend([Message.user("go"), Message.assistant_text("reading")])
    loop._anchor_prompt(50_000)
    loop.session.messages.extend(
        [
            Message(role="assistant", blocks=[ToolUseBlock(id="t1", name="read", input={})]),
            Message.tool_results([ToolResultBlock(tool_use_id="t1", output="x" * 30_000)]),
        ]
    )
    assert loop._count_tokens_anchored(loop.session.messages) == 60_000


# --- the bench's probe --------------------------------------------------------------------------


def test_the_bench_probe_accepts_every_keyword_should_compact_takes() -> None:
    """``bench/run_task.py`` swaps ``Compactor.should_compact`` for a recording probe. A probe
    missing one of its keywords raises TypeError on the loop's first check and kills every
    compaction-arm run. Read from the source, as the other bench guards do: bench/ stays
    outside the runtime gates."""
    tree = ast.parse(BENCH_RUNNER.read_text(encoding="utf-8"), filename=str(BENCH_RUNNER))
    probes = [
        node
        for outer in ast.walk(tree)
        if isinstance(outer, ast.FunctionDef) and outer.name == "_instrument_compaction"
        for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef) and node.name == "probe"
    ]
    assert len(probes) == 1, "expected one probe inside _instrument_compaction"
    probe_keywords = {arg.arg for arg in probes[0].args.kwonlyargs}
    engine_keywords = {
        name
        for name, param in inspect.signature(Compactor.should_compact).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert engine_keywords, "a positive control: should_compact takes keyword-only arguments"
    assert engine_keywords <= probe_keywords, sorted(engine_keywords - probe_keywords)
