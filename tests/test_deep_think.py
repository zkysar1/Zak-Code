"""``deep_think`` — best-of-N self-fusion deliberation tool.

Hermetic: the Sampler is a fake (no network). Covers the tool's synthesis + degradation paths,
the Sampler seam, and end-to-end wiring through a real Agent (the model calls deep_think, the
tool samples the agent's provider, and the spend is attributed in the session usage).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import zakcode
from zakcode.providers.base import Capabilities, LLMResult, Provider, ToolCall
from zakcode.tools.base import SampleCutOff, Sampler, ToolContext
from zakcode.tools.builtins.deep_think import (
    _CANDIDATE_SYSTEM,
    _CUT_OFF_MARK,
    _MAX_SAMPLES,
    _SYNTH_SYSTEM,
    DeepThinkTool,
)
from zakcode.usage import Usage


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZAKCODE_HOME", str(tmp_path / "confighome"))
    for var in ("ZAKCODE_FALLBACK_MODEL", "ZAKCODE_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    yield


def _ctx(sampler: Sampler | None) -> ToolContext:
    return ToolContext(workspace_root=Path("."), sampler=sampler)


# ── the tool ─────────────────────────────────────────────────────────────────────


async def test_best_of_n_synthesizes(tmp_path: Path) -> None:
    calls = {"candidate": 0, "synth": 0, "temps": []}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _SYNTH_SYSTEM:
            calls["synth"] += 1
            return "THE FUSED ANSWER"
        calls["candidate"] += 1
        calls["temps"].append(temperature)
        return f"candidate {calls['candidate']}"

    result = await DeepThinkTool().execute(
        {"question": "best design?", "samples": 3}, _ctx(sampler)
    )
    assert not result.is_error
    assert result.output == "THE FUSED ANSWER"
    assert result.data == {"samples": 3, "synthesized": True}
    assert calls["candidate"] == 3 and calls["synth"] == 1  # N candidates + 1 synthesis
    assert all(t > 0 for t in calls["temps"])  # candidates sampled with diversity temperature


async def test_single_sample_skips_synthesis() -> None:
    async def sampler(prompt, *, system=None, temperature=0.0):
        return "only answer"

    result = await DeepThinkTool().execute({"question": "q", "samples": 1}, _ctx(sampler))
    assert not result.is_error
    assert result.output == "only answer"
    assert result.data["synthesized"] is False  # nothing to fuse across


async def test_samples_clamped_to_max() -> None:
    n = {"count": 0}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _CANDIDATE_SYSTEM:
            n["count"] += 1
        return "x"

    await DeepThinkTool().execute({"question": "q", "samples": 99}, _ctx(sampler))
    assert n["count"] == _MAX_SAMPLES  # capped


async def test_no_sampler_is_graceful_error() -> None:
    result = await DeepThinkTool().execute({"question": "q"}, _ctx(None))
    assert result.is_error and "unavailable" in result.output


async def test_empty_question_is_error() -> None:
    async def sampler(prompt, *, system=None, temperature=0.0):
        return "x"

    result = await DeepThinkTool().execute({"question": "   "}, _ctx(sampler))
    assert result.is_error and "question" in result.output


async def test_all_samples_fail_is_error() -> None:
    async def sampler(prompt, *, system=None, temperature=0.0):
        raise RuntimeError("provider down")

    result = await DeepThinkTool().execute({"question": "q", "samples": 3}, _ctx(sampler))
    assert result.is_error and "no candidate" in result.output


async def test_synthesis_failure_falls_back_to_fullest_candidate() -> None:
    seen = {"n": 0}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _SYNTH_SYSTEM:
            raise RuntimeError("synth down")
        seen["n"] += 1
        return "tiny" if seen["n"] == 1 else "a much longer and fuller candidate answer here"

    result = await DeepThinkTool().execute({"question": "q", "samples": 2}, _ctx(sampler))
    assert not result.is_error  # degrades, doesn't fail
    assert result.output == "a much longer and fuller candidate answer here"  # fullest candidate
    assert result.data["synthesized"] is False


async def test_empty_synthesis_falls_back_and_labels_honestly() -> None:
    # Synthesis SUCCEEDS but returns whitespace → fall back to the fullest candidate and label it
    # synthesized=False (not True), consistent with the exception path.
    seen = {"n": 0}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _SYNTH_SYSTEM:
            return "   "  # empty/whitespace synthesis
        seen["n"] += 1
        return "tiny" if seen["n"] == 1 else "the fuller candidate answer"

    result = await DeepThinkTool().execute({"question": "q", "samples": 2}, _ctx(sampler))
    assert not result.is_error
    assert result.output == "the fuller candidate answer"
    assert result.data["synthesized"] is False
    assert result.data["synthesis_error"] == "empty"


# ── output-limit cut-offs ───────────────────────────────────────────────────────────


async def test_cut_off_candidate_is_marked_for_the_synthesis_and_counted() -> None:
    # A candidate that stopped at the output limit is a fragment: it reaches the synthesis marked
    # incomplete, and the result says how many candidates ran out of room.
    seen = {"n": 0, "synth_prompt": ""}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _SYNTH_SYSTEM:
            seen["synth_prompt"] = prompt
            return "THE FUSED ANSWER"
        seen["n"] += 1
        if seen["n"] == 1:
            raise SampleCutOff("the start of an answer")
        return f"a finished answer {seen['n']}"

    result = await DeepThinkTool().execute({"question": "q", "samples": 3}, _ctx(sampler))
    assert not result.is_error
    assert result.output == "THE FUSED ANSWER"
    assert result.data == {"samples": 3, "synthesized": True, "cut_off": 1}
    assert "the start of an answer" + _CUT_OFF_MARK in seen["synth_prompt"]
    assert "1 of 3 candidate answers were cut off at the output limit" in (result.hint or "")


async def test_every_candidate_cut_off_before_answering_is_an_honest_error() -> None:
    # A reasoning model can spend the whole output limit thinking and deliver no answer at all.
    # When every candidate did, the error names the limit and says the same question will end
    # the same way, instead of a bare "no answers produced" that invites the costly retry.
    calls = {"synth": 0}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _SYNTH_SYSTEM:
            calls["synth"] += 1
        raise SampleCutOff("")

    result = await DeepThinkTool().execute({"question": "q", "samples": 3}, _ctx(sampler))
    assert result.is_error
    assert "all 3 candidates were cut off at the output limit" in result.output
    assert "narrower question" in (result.fix or "")
    assert calls["synth"] == 0  # nothing to fuse


async def test_fallback_prefers_a_finished_candidate_over_a_longer_fragment() -> None:
    # The synthesis itself ran out of room, so the fallback picks a candidate. The longest one is
    # a fragment; the shorter one finished, and a finished answer wins.
    seen = {"n": 0}

    async def sampler(prompt, *, system=None, temperature=0.0):
        if system == _SYNTH_SYSTEM:
            raise SampleCutOff("half a fus")
        seen["n"] += 1
        if seen["n"] == 1:
            raise SampleCutOff("a long fragment " * 20)
        return "short but finished"

    result = await DeepThinkTool().execute({"question": "q", "samples": 2}, _ctx(sampler))
    assert not result.is_error
    assert result.output == "short but finished"
    assert result.data["synthesized"] is False
    assert "cut off at the output limit" in result.data["synthesis_error"]
    assert result.data["cut_off"] == 1


def test_deep_think_in_default_registry() -> None:
    from zakcode.tools.builtins.default_registry import default_registry

    reg = default_registry()
    assert reg.get("deep_think") is not None
    assert reg.get("deliberate") is not None  # alias


# ── Agent + loop wiring ────────────────────────────────────────────────────────────


class _Responder(Provider):
    """Prompt-aware scripted provider: the main loop calls deep_think once, then finishes;
    deep_think's own sample/synthesis calls (distinguished by their system prompt) return
    candidates / a fused answer."""

    def __init__(self) -> None:
        self.main_calls = 0
        self.deliberation_calls = 0

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        if system in (_CANDIDATE_SYSTEM, _SYNTH_SYSTEM):
            self.deliberation_calls += 1
            text = "fused answer" if system == _SYNTH_SYSTEM else "a candidate"
            return LLMResult(text=text, usage=Usage(total_tokens=2, cost_usd=0.005))
        self.main_calls += 1
        if self.main_calls == 1:
            return LLMResult(
                tool_calls=[
                    ToolCall(
                        id="t1", name="deep_think", arguments={"question": "hard?", "samples": 2}
                    )
                ],
                usage=Usage(total_tokens=1, cost_usd=0.001),
            )
        return LLMResult(
            text="done after deliberating", usage=Usage(total_tokens=1, cost_usd=0.001)
        )

    def count_tokens(self, messages, *, system=None) -> int:
        return 10

    def capabilities(self) -> Capabilities:
        return Capabilities(supports_tools=True, context_window=8192)

    def model_id(self) -> str:
        return "test/model"


def test_agent_wires_sampler_and_records_usage(tmp_path: Path) -> None:
    # The Agent's sampler samples its provider and attributes the spend to the session (so it
    # shows in /cost). Use the buffered run for determinism.
    import asyncio

    agent = zakcode.Agent(workspace_root=tmp_path, provider=_Responder())
    text = asyncio.run(agent._deep_think_sample("ponder this", system=_CANDIDATE_SYSTEM))
    assert text == "a candidate"
    # the deliberation's usage was recorded, tagged with the model
    by_model = agent.session.usage_by_model()
    assert "test/model" in by_model
    assert by_model["test/model"].cost_usd == pytest.approx(0.005)
    # ADR-0243: a deliberation is a side call; no reply of the conversation pairs with it
    assert {u.side_call for u in agent.session.usages} == {"deep_think"}


class _CutOffResponder(_Responder):
    """Every candidate call stops at the output limit, spelled the way the backend spells it."""

    def __init__(self, finish_reason: str) -> None:
        super().__init__()
        self.finish_reason = finish_reason

    async def acomplete(self, messages, *, system=None, tools=None, response_format=None, **kw):
        if system == _CANDIDATE_SYSTEM:
            return LLMResult(
                text="the start of an answer",
                finish_reason=self.finish_reason,
                usage=Usage(total_tokens=2, cost_usd=0.005),
            )
        return await super().acomplete(
            messages, system=system, tools=tools, response_format=response_format, **kw
        )


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_agent_sampler_raises_cut_off_after_recording_usage(
    tmp_path: Path, finish_reason: str
) -> None:
    # The sampler has no continuation path, so a cut-off completion is raised, never returned
    # as if it were a finished answer. Its spend was real and is still counted.
    import asyncio

    agent = zakcode.Agent(workspace_root=tmp_path, provider=_CutOffResponder(finish_reason))
    with pytest.raises(SampleCutOff) as info:
        asyncio.run(agent._deep_think_sample("ponder this", system=_CANDIDATE_SYSTEM))
    assert info.value.text == "the start of an answer"
    assert agent.session.usage_by_model()["test/model"].cost_usd == pytest.approx(0.005)


def test_full_turn_invokes_deep_think(tmp_path: Path) -> None:
    import asyncio

    responder = _Responder()
    agent = zakcode.Agent(workspace_root=tmp_path, provider=responder)
    result = asyncio.run(agent.arun_turn("think hard about this"))
    assert result.stop_reason == "completed"
    assert "done after deliberating" in "\n".join(m.text for m in result.assistant_messages)
    # the model's deep_think call ran the deliberation: 2 candidates + 1 synthesis
    assert responder.deliberation_calls == 3
    # a deep_think tool result is in the turn
    assert any("fused answer" in (tr.output or "") for tr in result.tool_results)
