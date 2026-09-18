"""ADR-0193: the per-call plan reminder must not defeat the provider's prompt cache.

Measured 2026-09-18 (gpt-5.6-luna / -terra): the tier reuses only a previously sent FULL
prompt that is a prefix of the new one. The plan reminder rides the END of every call and
is never persisted, so no call's prompt was ever a prefix of the next — a served turn's
cache reads sat at 28,079 tokens for 241 calls while its prompt grew to 360k.

The loop changes nothing on a guess: a flat cache read buys ONE tail-less call, and only a
next-call read of that whole prompt makes the reminder ride every other call. Four caches
are modelled here, each after something measured or documented — the loop is tested against
their behaviour, not against a mock of its own code:

* ``checkpoint`` — luna/terra: only a whole earlier prompt is reused.
* ``prefix`` — the longest shared prefix in 128-token blocks (gpt-5-mini's documented shape).
* ``system`` — only the system block is ever cached, tail or no tail (a single system
  breakpoint): flat forever, and resting the tail cannot help it.
* ``silent`` — no cache reads reported (most local pods): the small-model guarantee.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from zakcode.agent.loop import (
    _CACHE_FLAT_MIN_GROWTH,
    _CACHE_PROBE_LIMIT,
    _CACHE_PROBE_SLACK,
    AgentLoop,
)
from zakcode.messages import Message
from zakcode.permissions import PermissionTier
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.session.store import Session
from zakcode.tasks import Task
from zakcode.tools.base import (
    ConcurrencyClass,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from zakcode.tools.builtins.update_plan import UpdatePlanTool
from zakcode.usage import Usage

CHECKPOINT, PREFIX, SYSTEM, SILENT = "checkpoint", "prefix", "system", "silent"


def _text(message: Message) -> str:
    parts = [message.role]
    for block in message.blocks:
        parts.append(
            str(getattr(block, "text", "") or getattr(block, "output", "") or block.model_dump())
        )
    return "\x1f".join(parts)


class _Work(Tool):
    """A step of real work: ~2.3k tokens of result, so two of them outgrow the growth bar."""

    spec = ToolSpec(
        name="work",
        description="do one unit of work",
        required_permission=PermissionTier.READ_ONLY,
        concurrency=ConcurrencyClass.READ_ONLY_SAFE,
    )

    async def execute(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(f"unit {args.get('n')} done. " + "detail " * 1300)


class _CachingProvider(Provider):
    """Scripted completions over a MODEL of the provider's prompt cache."""

    def __init__(self, script: Callable[[int], LLMResult], style: str) -> None:
        self._script = script
        self._style = style
        self.name = f"fake/{style}"
        self.main_calls = 0
        self.sent: list[tuple[str, ...]] = []
        self.tails: list[bool] = []
        self.reads: list[int] = []
        self.prompts: list[int] = []

    def model_id(self) -> str:
        return self.name

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=400_000)

    def count_tokens(self, messages: list[Message], *, system: str | None = None) -> int:
        return (len(system or "") + sum(len(_text(m)) for m in messages)) // 4

    def _cache_read(self, texts: tuple[str, ...]) -> int:
        if self._style == SILENT or not self.sent:
            return 0
        if self._style == SYSTEM:
            return len(texts[0]) // 4
        if self._style == CHECKPOINT:
            # Only a WHOLE earlier prompt that is a prefix of this one is reused; the system
            # prompt is its own checkpoint.
            best = len(texts[0]) // 4
            for earlier in self.sent:
                if texts[: len(earlier)] == earlier:
                    best = max(best, sum(len(t) for t in earlier) // 4)
            return best
        joined = "\x1e".join(texts)
        best = 0
        for earlier in self.sent:
            other = "\x1e".join(earlier)
            n = 0
            for a, b in zip(joined, other, strict=False):
                if a != b:
                    break
                n += 1
            best = max(best, n // 4)
        return best // 128 * 128  # a prefix cache advances in 128-token blocks

    async def acomplete(
        self, messages: list[Message], *, system: str | None = None, tools: Any = None, **kw: Any
    ) -> LLMResult:
        texts = (system or "", *(_text(m) for m in messages))
        prompt = sum(len(t) for t in texts) // 4
        if not tools:  # a side call (a judge): answered, never part of the measurement
            return LLMResult(text="ok", usage=Usage(prompt_tokens=prompt, total_tokens=prompt))
        self.main_calls += 1
        read = self._cache_read(texts)
        self.sent.append(texts)
        self.tails.append(bool(messages) and "[plan]" in _text(messages[-1])[:40])
        self.reads.append(read)
        self.prompts.append(prompt)
        usage = Usage(
            prompt_tokens=prompt,
            completion_tokens=5,
            total_tokens=prompt + 5,
            cache_read_tokens=read,
        )
        return self._script(self.main_calls).model_copy(update={"usage": usage})


def _plan(*statuses: str) -> LLMResult:
    tasks = [{"title": f"unit {k}", "status": s} for k, s in enumerate(statuses, 1)]
    return LLMResult(tool_calls=[ToolCall(id="p", name="update_plan", arguments={"tasks": tasks})])


def _work(n: int) -> LLMResult:
    return LLMResult(tool_calls=[ToolCall(id=f"w{n}", name="work", arguments={"n": n})])


def _turn(work_calls: int) -> Callable[[int], LLMResult]:
    """Plan, ``work_calls`` units of work, close the plan, answer."""

    def script(n: int) -> LLMResult:
        if n == 1:
            return _plan("in_progress", "pending", "pending")
        if n <= work_calls + 1:
            return _work(n)
        if n == work_calls + 2:
            return _plan("done", "done", "done")
        return LLMResult(text="All three units are done.")

    return script


def _loop(provider: Provider, tmp_path: Path, session: Session | None = None) -> AgentLoop:
    registry = ToolRegistry()
    registry.register(UpdatePlanTool())
    registry.register(_Work())
    return AgentLoop(
        provider,
        registry,
        session or Session(cwd=str(tmp_path), model="test"),
        workspace_root=tmp_path,
        max_iterations=30,
    )


T, F = True, False

# ── the four caches, through a whole turn ─────────────────────


def test_a_checkpoint_cache_is_suspected_probed_confirmed_and_then_spared(tmp_path: Path) -> None:
    provider = _CachingProvider(_turn(work_calls=8), CHECKPOINT)
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("do the three units"))
    assert result.stop_reason == "completed"
    # 1: no plan yet. 2-4: tailed, the read holds at call 1's prompt while the prompt grows
    # past the bar. 5: the probe. 6: reads the probe's whole prompt — confirmed. Then every
    # other call, until 11: it follows a tailed call but carries the finished plan's
    # "answer now" line, which always rides.
    assert provider.tails == [F, T, T, T, F, T, F, T, F, T, T]
    assert provider.reads[1] == provider.reads[2] == provider.reads[3] > 0
    assert provider.prompts[3] - provider.prompts[1] >= _CACHE_FLAT_MIN_GROWTH
    assert provider.reads[5] == provider.prompts[4]  # the measurement
    assert loop.session.tail_sparse_models == ["fake/checkpoint"]
    assert loop.session.tail_probe_misses == {}
    # The cache works from here on: each tailed call reuses the tail-less call before it.
    assert provider.reads[7] == provider.prompts[6] > provider.reads[5]
    assert provider.reads[9] == provider.prompts[8] > provider.reads[7]
    notes = [e for e in loop._trace.events if e.data.get("kind") == "cache_friendly_tail"]
    assert len(notes) == 1
    assert notes[0].data["probe_prompt_tokens"] == provider.prompts[4]
    assert notes[0].data["flat_cache_read_tokens"] == provider.reads[3]


@pytest.mark.parametrize("style", [PREFIX, SILENT])
def test_a_working_cache_and_a_silent_backend_are_never_even_probed(
    tmp_path: Path, style: str
) -> None:
    """The small-model guarantee: nothing changes where the tail costs nothing (a prefix
    cache) or where the backend reports no cache reads at all (most local pods)."""
    provider = _CachingProvider(_turn(work_calls=8), style)
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("do the three units"))
    assert result.stop_reason == "completed"
    assert provider.tails == [F] + [T] * 10
    assert loop.session.tail_sparse_models == []
    assert loop.session.tail_probe_misses == {}
    if style == PREFIX:  # positive control: this fake CAN report a growing cache
        assert provider.reads[-1] > provider.reads[1] > 0


def test_a_cache_the_tail_does_not_hold_costs_two_probes_and_is_left_alone(
    tmp_path: Path,
) -> None:
    """Only the system block is ever cached: flat forever, and resting the tail cannot help.
    The flat read is suspected like the checkpoint tier's — the probe is what tells them
    apart. Two misses, then the reminder rides every call for the rest of the session."""
    provider = _CachingProvider(_turn(work_calls=12), SYSTEM)
    loop = _loop(provider, tmp_path)
    result = asyncio.run(loop.arun_turn("do the three units"))
    assert result.stop_reason == "completed"
    assert provider.tails == [F, T, T, T, F, T, T, T, T, F, T, T, T, T, T]
    assert provider.tails.count(F) == 1 + _CACHE_PROBE_LIMIT
    assert loop.session.tail_sparse_models == []
    assert loop.session.tail_probe_misses == {"fake/system": _CACHE_PROBE_LIMIT}
    assert not [e for e in loop._trace.events if e.data.get("kind") == "cache_friendly_tail"]


# ── both verdicts outlive the loop that reached them ──────────


def _resume(tmp_path: Path, style: str, **fields: Any) -> tuple[_CachingProvider, AgentLoop]:
    """A NEW loop on a session an earlier one left mid-plan — a served mind builds one per
    turn, and must not pay the measurement again each turn."""
    session = Session(cwd=str(tmp_path), model="test", **fields)
    session.task_network.replace_from_author(
        [Task(title="unit 1", status="in_progress"), Task(title="unit 2"), Task(title="unit 3")]
    )

    def script(n: int) -> LLMResult:
        if n <= 6:
            return _work(n)
        return _plan("done", "done", "done") if n == 7 else LLMResult(text="Done.")

    provider = _CachingProvider(script, style)
    loop = _loop(provider, tmp_path, session)
    loop._prev_call_had_tail = True  # whatever the last turn left: a turn's first call is tailed
    loop._cache_probe_rests_next = True
    asyncio.run(loop.arun_turn("continue"))
    return provider, loop


def test_a_confirmed_model_is_spared_from_the_next_turns_second_call(tmp_path: Path) -> None:
    provider, _ = _resume(tmp_path, CHECKPOINT, tail_sparse_models=["fake/checkpoint"])
    assert provider.tails == [T, F, T, F, T, F, T, T]
    assert provider.reads[2] == provider.prompts[1]  # the tailed call reuses the tail-less one


def test_a_settled_miss_is_never_probed_again_and_one_miss_leaves_one_probe(
    tmp_path: Path,
) -> None:
    provider, loop = _resume(tmp_path, SYSTEM, tail_probe_misses={"fake/system": 2})
    assert provider.tails == [T] * 8
    provider, loop = _resume(tmp_path, SYSTEM, tail_probe_misses={"fake/system": 1})
    # Call 1 is cold (nothing cached to read), so the flat run is calls 2-4 and 5 is the probe.
    assert provider.tails == [T, T, T, T, F, T, T, T]
    assert loop.session.tail_probe_misses == {"fake/system": 2}


# ── the rule, driven directly with the measured numbers ───────


def _observe(loop: AgentLoop, prompt: int, read: int, *, tail: bool = True) -> None:
    loop._call_has_tail = tail
    loop._observe_prompt_cache(
        Usage(prompt_tokens=prompt, total_tokens=prompt, cache_read_tokens=read)
    )


def _bare_loop(tmp_path: Path) -> tuple[AgentLoop, _CachingProvider]:
    provider = _CachingProvider(_turn(work_calls=1), SILENT)
    return _loop(provider, tmp_path), provider


def test_what_does_not_raise_a_suspicion(tmp_path: Path) -> None:
    loop, provider = _bare_loop(tmp_path)
    # gpt-5-mini, live loop on main, 2026-09-18: one read held for six calls — 969 tokens of
    # growth. A healthy cache that advances in lumps.
    for prompt in (15_399, 15_699, 15_765, 16_078, 16_144, 16_368):
        _observe(loop, prompt, 14_848)
    assert not loop._cache_probe_rests_next
    for prompt, read in ((20_000, 8_000), (25_000, 20_000), (30_000, 25_000), (35_000, 30_000)):
        _observe(loop, prompt, read)  # the cache advances every call: it works
    assert not loop._cache_probe_rests_next
    for prompt in (40_000, 45_000, 9_000, 12_000):  # a compaction shrank the prompt mid-run
        _observe(loop, prompt, 8_000)
    assert not loop._cache_probe_rests_next
    for prompt in (50_000, 55_000, 60_000, 65_000):  # no cache reads reported
        _observe(loop, prompt, 0)
    assert not loop._cache_probe_rests_next
    _observe(loop, 70_000, 8_000)
    _observe(loop, 75_000, 8_000, tail=False)  # a tail-less call is no evidence about the tail
    _observe(loop, 80_000, 8_000)
    _observe(loop, 85_000, 8_000)
    assert not loop._cache_probe_rests_next
    provider.name = "fake/another-model"  # a route change: the run starts over
    _observe(loop, 90_000, 8_000)
    _observe(loop, 95_000, 8_000)
    assert not loop._cache_probe_rests_next
    _observe(loop, 99_000, 8_000)  # third tailed call on one model, 9k of growth, read flat
    assert loop._cache_probe_rests_next
    assert loop.session.tail_sparse_models == [] and loop.session.tail_probe_misses == {}


def _suspect(loop: AgentLoop, base: int = 30_000) -> None:
    """The served turn's shape: the read sits at the system block while the prompt grows."""
    for prompt in (base, base + 3_000, base + 6_000):
        _observe(loop, prompt, 28_079)
    assert loop._cache_probe_rests_next


def test_the_probe_confirms_only_on_a_read_of_its_whole_prompt(tmp_path: Path) -> None:
    loop, _ = _bare_loop(tmp_path)
    _suspect(loop)
    _observe(loop, 38_000, 28_079, tail=False)  # the probe itself reads the old checkpoint
    assert not loop._cache_probe_rests_next
    _observe(loop, 40_000, 38_000 - _CACHE_PROBE_SLACK)  # luna reads the whole prompt less 3
    assert loop.session.tail_sparse_models == ["fake/silent"]

    loop, _ = _bare_loop(tmp_path)
    _suspect(loop)
    _observe(loop, 38_000, 28_079, tail=False)
    _observe(loop, 40_000, 38_000 - _CACHE_PROBE_SLACK - 1)  # a lagging prefix cache: a miss
    assert loop.session.tail_sparse_models == []
    assert loop.session.tail_probe_misses == {"fake/silent": 1}


def test_a_probe_that_measured_nothing_is_not_a_miss(tmp_path: Path) -> None:
    loop, provider = _bare_loop(tmp_path)
    _suspect(loop)
    _observe(loop, 38_000, 28_079, tail=True)  # the "answer now" line rode the probe call
    assert loop._cache_probe_rested is None and not loop._cache_probe_rests_next
    _suspect(loop, base=50_000)
    _observe(loop, 58_000, 28_079, tail=False)
    _observe(loop, 20_000, 9_000)  # a compaction rewrote the history before the reading
    assert loop.session.tail_probe_misses == {} and loop.session.tail_sparse_models == []
    _suspect(loop, base=21_000)
    _observe(loop, 29_000, 28_079, tail=False)
    provider.name = "fake/another-model"  # another model answered the reading call
    _observe(loop, 31_000, 0)
    assert loop.session.tail_probe_misses == {} and loop.session.tail_sparse_models == []
