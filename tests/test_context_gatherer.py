"""Tests for the deterministic context-gatherer (zakcode.context)."""

from __future__ import annotations

import os
import time

from zakcode import Agent
from zakcode.config import Settings
from zakcode.context import (
    default_gatherer,
    mentioned_files_collector,
    recent_files_collector,
)
from zakcode.context.gatherer import (
    Candidate,
    ContextGatherer,
    _fit_budget,
    heuristic_classifier,
)
from zakcode.hooks import HookManager, LLMContextPayload


def _payload(user_text: str = "", cwd: str = "") -> LLMContextPayload:
    return LLMContextPayload(user_text=user_text, cwd=cwd)


def _cand(ref: str, score: float, text: str = "x") -> Candidate:
    return Candidate(source="t", ref=ref, text=text, cheap_score=score)


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
    # A single oversize candidate is still injected (better some than none).
    picked = _fit_budget([_cand("big", 1.0, "x" * 5000)], budget_chars=100, top_k=10)
    assert len(picked) == 1


# --- gatherer behaviour ----------------------------------------------------
def test_gatherer_none_when_no_candidates() -> None:
    def empty(_p: LLMContextPayload) -> list[Candidate]:
        return []

    assert ContextGatherer([empty])(_payload()) is None


def test_gatherer_renders_picked() -> None:
    def col(_p: LLMContextPayload) -> list[Candidate]:
        return [_cand("notes.md", 0.5, "hello world")]

    out = ContextGatherer([col])(_payload())
    assert out is not None
    assert "notes.md" in out and "hello world" in out


def test_gatherer_skips_throwing_collector() -> None:
    def boom(_p: LLMContextPayload) -> list[Candidate]:
        raise RuntimeError("collector blew up")

    def good(_p: LLMContextPayload) -> list[Candidate]:
        return [_cand("ok", 0.5, "survived")]

    out = ContextGatherer([boom, good])(_payload())
    assert out is not None and "survived" in out


def test_gatherer_falls_back_when_classifier_throws() -> None:
    def col(_p: LLMContextPayload) -> list[Candidate]:
        return [_cand("a", 0.2, "low"), _cand("b", 0.9, "high")]

    def bad_classifier(_task: str, _cands: list[Candidate]) -> list[Candidate]:
        raise RuntimeError("classifier blew up")

    # Falls back to the heuristic order (high score first), never crashes.
    out = ContextGatherer([col], classifier=bad_classifier)(_payload())
    assert out is not None
    assert out.index("high") < out.index("low")


# --- collectors ------------------------------------------------------------
def test_recent_files_collector_finds_recent_skips_old(tmp_path) -> None:
    (tmp_path / "fresh.py").write_text("print('fresh')", encoding="utf-8")
    old = tmp_path / "old.py"
    old.write_text("print('old')", encoding="utf-8")
    stale = time.time() - 40 * 86400  # 40 days
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


# --- end-to-end through the real hook seam ---------------------------------
async def test_default_gatherer_integrates_with_hook_manager(tmp_path) -> None:
    (tmp_path / "notes.md").write_text("the important notes", encoding="utf-8")
    hm = HookManager()
    hm.register_context(default_gatherer())
    texts = await hm.gather_context(LLMContextPayload(user_text="see notes.md", cwd=str(tmp_path)))
    assert texts
    assert "notes.md" in texts[0]


# --- agent wiring (the opt-in flag) ----------------------------------------
def test_agent_flag_registers_gatherer(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    on = Agent(settings=settings, enable_context_gathering=True)
    off = Agent(settings=settings)
    # The flag adds exactly one in-process context hook (the gatherer); default ships none.
    assert len(on.hook_manager.context_hooks) == len(off.hook_manager.context_hooks) + 1
