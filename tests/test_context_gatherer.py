"""Tests for the deterministic context-gatherer (zakcode.context)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

from zakcode import Agent
from zakcode.config import Settings
from zakcode.context import (
    SmallModelClassifier,
    default_gatherer,
    mentioned_files_collector,
    recent_files_collector,
    score_relevance,
)
from zakcode.context.gatherer import (
    Candidate,
    ContextGatherer,
    _fit_budget,
    heuristic_classifier,
)
from zakcode.hooks import HookManager, LLMContextPayload
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.usage import Usage


class _Provider(Provider):
    """Scripted in-memory provider: returns a fixed text as the model response."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self._text, usage=Usage(total_tokens=5, cost_usd=0.0))

    def count_tokens(self, messages, *, system=None) -> int:  # noqa: ANN001
        return 0

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def model_id(self) -> str:
        return "classify/test"


class _BoomProvider(_Provider):
    """A provider whose call raises -- exercises the classifier's fail-soft path."""

    async def acomplete(  # noqa: ANN001
        self, messages, *, system=None, tools=None, response_format=None, **kwargs: Any
    ) -> LLMResult:
        raise RuntimeError("provider down")


def _payload(user_text: str = "", cwd: str = "") -> LLMContextPayload:
    return LLMContextPayload(user_text=user_text, cwd=cwd)


def _cand(ref: str, score: float, text: str = "x") -> Candidate:
    return Candidate(source="t", ref=ref, text=text, cheap_score=score)


def _scores(pairs: dict[str, float]) -> str:
    return json.dumps({"scores": [{"ref": r, "score": s} for r, s in pairs.items()]})


# --- ranking + budget ------------------------------------------------------
def test_heuristic_classifier_sorts_by_score_desc() -> None:
    ranked = heuristic_classifier("task", [_cand("a", 0.1), _cand("b", 0.9), _cand("c", 0.5)])
    assert [c.ref for c in ranked] == ["b", "c", "a"]


def test_fit_budget_respects_top_k() -> None:
    cands = [_cand(str(i), 1.0, "x" * 10) for i in range(20)]
    assert len(_fit_budget(cands, budget_chars=10_000, top_k=5)) == 5


def test_fit_budget_respects_char_budget() -> None:
    cands = [_cand(str(i), 1.0, "x" * 100) for i in range(20)]
    assert len(_fit_budget(cands, budget_chars=250, top_k=100)) == 2


def test_fit_budget_always_takes_at_least_one() -> None:
    picked = _fit_budget([_cand("big", 1.0, "x" * 5000)], budget_chars=100, top_k=10)
    assert len(picked) == 1


# --- gatherer behaviour (async) --------------------------------------------
async def test_gatherer_none_when_no_candidates() -> None:
    def empty(_p: LLMContextPayload) -> list[Candidate]:
        return []

    assert await ContextGatherer([empty])(_payload()) is None


async def test_gatherer_renders_picked() -> None:
    def col(_p: LLMContextPayload) -> list[Candidate]:
        return [_cand("notes.md", 0.5, "hello world")]

    out = await ContextGatherer([col])(_payload())
    assert out is not None
    assert "notes.md" in out and "hello world" in out


async def test_gatherer_skips_throwing_collector() -> None:
    def boom(_p: LLMContextPayload) -> list[Candidate]:
        raise RuntimeError("collector blew up")

    def good(_p: LLMContextPayload) -> list[Candidate]:
        return [_cand("ok", 0.5, "survived")]

    out = await ContextGatherer([boom, good])(_payload())
    assert out is not None and "survived" in out


async def test_gatherer_falls_back_when_classifier_throws() -> None:
    def col(_p: LLMContextPayload) -> list[Candidate]:
        return [_cand("a", 0.2, "low"), _cand("b", 0.9, "high")]

    def bad_classifier(_task: str, _cands: Sequence[Candidate]) -> list[Candidate]:
        raise RuntimeError("classifier blew up")

    out = await ContextGatherer([col], classifier=bad_classifier)(_payload())
    assert out is not None
    assert out.index("high") < out.index("low")


# --- collectors ------------------------------------------------------------
def test_recent_files_collector_finds_recent_skips_old(tmp_path) -> None:
    (tmp_path / "fresh.py").write_text("print('fresh')", encoding="utf-8")
    old = tmp_path / "old.py"
    old.write_text("print('old')", encoding="utf-8")
    stale = time.time() - 40 * 86400
    os.utime(old, (stale, stale))
    refs = {c.ref for c in recent_files_collector(_payload(cwd=str(tmp_path)), max_age_days=14.0)}
    assert "fresh.py" in refs
    assert "old.py" not in refs


def test_recent_files_collector_skips_skip_dirs(tmp_path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("x", encoding="utf-8")
    (tmp_path / "real.py").write_text("print('x')", encoding="utf-8")
    refs = {c.ref for c in recent_files_collector(_payload(cwd=str(tmp_path)))}
    assert "real.py" in refs
    assert not any(".git" in r for r in refs)


def test_mentioned_files_collector_top_prior(tmp_path) -> None:
    (tmp_path / "target.py").write_text("print('target')", encoding="utf-8")
    cands = mentioned_files_collector(
        _payload(user_text="please fix the bug in target.py", cwd=str(tmp_path))
    )
    assert any(c.ref == "target.py" and c.cheap_score == 1.0 for c in cands)


def test_collectors_safe_on_missing_cwd() -> None:
    p = _payload(user_text="x.py", cwd="/no/such/dir/anywhere")
    assert recent_files_collector(p) == []
    assert mentioned_files_collector(p) == []


# --- small-model classifier (step 2) ---------------------------------------
async def test_score_relevance_parses_scores() -> None:
    cands = [_cand("a", 0.0), _cand("b", 0.0)]
    scores, usage = await score_relevance(_Provider(_scores({"a": 0.2, "b": 0.9})), "task", cands)
    assert scores == {"a": 0.2, "b": 0.9}
    assert usage.total_tokens == 5


async def test_score_relevance_fail_soft_on_provider_error() -> None:
    scores, usage = await score_relevance(_BoomProvider(""), "task", [_cand("a", 0.0)])
    assert scores == {} and usage.total_tokens == 0


async def test_small_model_classifier_ranks_by_model_score() -> None:
    # Heuristic prior says a(0.9) > b(0.1); the model flips it (b is essential).
    cands = [_cand("a", 0.9, "A"), _cand("b", 0.1, "B")]
    ranked = await SmallModelClassifier(_Provider(_scores({"a": 0.1, "b": 0.95})))("task", cands)
    assert [c.ref for c in ranked] == ["b", "a"]


async def test_small_model_classifier_falls_back_to_heuristic() -> None:
    # On provider failure, ranking falls back to the deterministic prior (a > b).
    cands = [_cand("a", 0.9, "A"), _cand("b", 0.1, "B")]
    ranked = await SmallModelClassifier(_BoomProvider(""))("task", cands)
    assert [c.ref for c in ranked] == ["a", "b"]


async def test_small_model_classifier_reports_usage() -> None:
    seen: list[Usage] = []
    clf = SmallModelClassifier(_Provider(_scores({"a": 0.5})), on_usage=seen.append)
    await clf("task", [_cand("a", 0.0)])
    assert seen and seen[0].total_tokens == 5


# --- end-to-end through the real hook seam ---------------------------------
async def test_default_gatherer_integrates_with_hook_manager(tmp_path) -> None:
    (tmp_path / "notes.md").write_text("the important notes", encoding="utf-8")
    hm = HookManager()
    hm.register_context(default_gatherer())
    texts = await hm.gather_context(LLMContextPayload(user_text="see notes.md", cwd=str(tmp_path)))
    assert texts
    assert "notes.md" in texts[0]


async def test_default_gatherer_with_model_classifier(tmp_path) -> None:
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    clf = SmallModelClassifier(_Provider(_scores({"a.md": 0.9, "b.md": 0.1})))
    hm = HookManager()
    hm.register_context(default_gatherer(clf))
    texts = await hm.gather_context(LLMContextPayload(user_text="recent", cwd=str(tmp_path)))
    assert texts
    assert texts[0].index("a.md") < texts[0].index("b.md")  # model-favoured ranks first


# --- agent wiring (the opt-in flags) ---------------------------------------
def test_agent_flag_registers_gatherer(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    on = Agent(settings=settings, enable_context_gathering=True)
    off = Agent(settings=settings)
    assert len(on.hook_manager.context_hooks) == len(off.hook_manager.context_hooks) + 1


def test_agent_model_classifier_flag(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    agent = Agent(settings=settings, enable_context_gathering=True, context_classifier="model")
    assert len(agent.hook_manager.context_hooks) == 1
