"""Tests for the deterministic context-gatherer (zakcode.context)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Awaitable, Sequence
from typing import Any

from zakcode import Agent
from zakcode.config import Settings
from zakcode.context import (
    ModelUsedDetector,
    RelevanceModel,
    SignalLogger,
    SmallModelClassifier,
    TrainedClassifier,
    default_gatherer,
    judge_used,
    mentioned_files_collector,
    recent_files_collector,
    reference_used,
    score_relevance,
    train_relevance,
)
from zakcode.context.gatherer import (
    Candidate,
    ContextGatherer,
    _fit_budget,
    heuristic_classifier,
)
from zakcode.evals.harness import ScriptedProvider, reply
from zakcode.hooks import HookManager, LLMContextPayload, TurnEndPayload
from zakcode.messages import Message, ToolUseBlock
from zakcode.permissions import PermissionPolicy
from zakcode.providers.base import Capabilities, LLMResult, Provider
from zakcode.session.store import Session
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

    def bad_classifier(
        _task: str, _cands: Sequence[Candidate]
    ) -> list[Candidate] | Awaitable[list[Candidate]]:
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


# --- signal logging (step 3) -----------------------------------------------
async def test_signal_logger_records_offered_and_used(tmp_path) -> None:
    g = ContextGatherer([])
    g.last_offer = [_cand("used.py", 0.8), _cand("ignored.py", 0.3)]
    g.last_task = "fix the bug"
    session = Session(
        cwd=str(tmp_path),
        model="test",
        messages=[
            Message(
                role="assistant",
                blocks=[ToolUseBlock(id="t1", name="read_file", input={"path": "used.py"})],
            )
        ],
    )
    log = tmp_path / "signal.jsonl"
    await SignalLogger(g, session, log).on_turn_end(TurnEndPayload(stop_reason="completed"))
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["n_offered"] == 2 and rec["n_used"] == 1
    assert {o["ref"]: o["used"] for o in rec["offered"]} == {"used.py": True, "ignored.py": False}
    assert rec["task"] == "fix the bug"


async def test_signal_logger_detects_use_in_assistant_text(tmp_path) -> None:
    g = ContextGatherer([])
    g.last_offer = [_cand("notes.md", 0.5)]
    session = Session(cwd=str(tmp_path), model="test", messages=[])
    log = tmp_path / "s.jsonl"
    await SignalLogger(g, session, log).on_turn_end(
        TurnEndPayload(stop_reason="completed", last_assistant_message="see notes.md for details")
    )
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["n_used"] == 1


async def test_signal_logger_skips_when_no_offer(tmp_path) -> None:
    g = ContextGatherer([])  # last_offer defaults to []
    session = Session(cwd=str(tmp_path), model="test", messages=[])
    log = tmp_path / "none.jsonl"
    await SignalLogger(g, session, log).on_turn_end(TurnEndPayload(stop_reason="completed"))
    assert not log.exists()


def test_agent_signal_log_registers_observer(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    log = str(tmp_path / "sig.jsonl")
    on = Agent(settings=settings, enable_context_gathering=True, context_signal_log=log)
    off = Agent(settings=settings, enable_context_gathering=True)
    # Registered as an OBSERVE-ONLY turn-end hook so it fires regardless of the veto budget.
    assert len(on.hook_manager.turn_end_observers) == len(off.hook_manager.turn_end_observers) + 1


# --- trained classifier (step 4) -------------------------------------------
def _signal_log(tmp_path, records: list[dict]) -> str:
    p = tmp_path / "sig.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return str(p)


def test_train_relevance_learns_overlap(tmp_path) -> None:
    # 'used' perfectly correlates with task<->ref name overlap; cheap_score is non-discriminative.
    records = [
        {
            "task": "fix the alpha module",
            "offered": [
                {"ref": "alpha.py", "score": 0.5, "source": "file", "used": True},
                {"ref": "zzz.py", "score": 0.5, "source": "file", "used": False},
            ],
        }
    ] * 20
    model = train_relevance(_signal_log(tmp_path, records), epochs=300)
    assert model.n_examples == 40
    # For a task overlapping 'alpha', the learned ranker puts alpha.py above zzz.py.
    ranked = TrainedClassifier(model)("alpha", [_cand("zzz.py", 0.5), _cand("alpha.py", 0.5)])
    assert [c.ref for c in ranked] == ["alpha.py", "zzz.py"]


def test_trained_default_ranks_by_prior() -> None:
    ranked = TrainedClassifier(RelevanceModel())("t", [_cand("a", 0.2), _cand("b", 0.9)])
    assert [c.ref for c in ranked] == ["b", "a"]


def test_train_relevance_empty_log_is_default(tmp_path) -> None:
    model = train_relevance(str(tmp_path / "missing.jsonl"))
    assert model.n_examples == 0


def test_relevance_model_save_load(tmp_path) -> None:
    RelevanceModel(weights=[0.1, 0.2, 0.3], n_examples=7).save(tmp_path / "w.json")
    assert RelevanceModel.load(tmp_path / "w.json").weights == [0.1, 0.2, 0.3]


def test_agent_classifier_weights_wires_trained(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    weights = tmp_path / "w.json"
    RelevanceModel(weights=[0.0, 1.0, 0.5]).save(weights)
    agent = Agent(
        settings=settings, enable_context_gathering=True, context_classifier_weights=str(weights)
    )
    assert len(agent.hook_manager.context_hooks) == 1


def test_agent_bad_weights_falls_back(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    # A missing weights file must not crash construction -- fail-soft to the heuristic gatherer.
    agent = Agent(
        settings=settings,
        enable_context_gathering=True,
        context_classifier_weights=str(tmp_path / "nope.json"),
    )
    assert len(agent.hook_manager.context_hooks) == 1


async def test_signal_log_written_through_real_turn(tmp_path) -> None:
    # The gap that hid the dead-by-default bug: drive a REAL turn and assert the signal logger
    # fires with the DEFAULT turn_end_veto_budget (0). It must run as an observe-only hook.
    (tmp_path / "target.py").write_text("x = 1\n", encoding="utf-8")
    log = tmp_path / "sig.jsonl"
    agent = Agent(
        provider=ScriptedProvider([reply("looked at it")]),
        permission_policy=PermissionPolicy("allow"),
        default_model="scripted/e2e",
        workspace_root=str(tmp_path),
        enable_context_gathering=True,
        context_signal_log=str(log),
    )
    result = await agent.arun_turn("please look at target.py")
    assert result.stop_reason == "completed", result.stop_reason
    assert log.is_file(), "signal log not written — the observe-only turn-end hook never fired"
    rec = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["n_offered"] >= 1
    assert any(o["ref"] == "target.py" for o in rec["offered"])


# --- used-signal judge: the in-context-use label-bias fix ------------------
def test_reference_used_normalizes_paths() -> None:
    # A Windows/absolute tool arg still matches the posix-relative offered ref.
    used = reference_used(
        "t", [_cand("src/app.py", 0.5), _cand("z.py", 0.5)], r"open C:\p\src\app.py"
    )
    assert used == {"src/app.py"}


async def test_judge_used_parses_and_filters_hallucinations() -> None:
    offered = [_cand("a.py", 0.5), _cand("b.py", 0.5)]
    used, usage = await judge_used(_Provider('{"used": ["a.py", "ghost.py"]}'), "t", offered, "ev")
    assert used == {"a.py"}  # ghost.py was not offered -> dropped
    assert usage.total_tokens == 5


async def test_judge_used_fail_soft() -> None:
    used, usage = await judge_used(_BoomProvider("x"), "t", [_cand("a.py", 0.5)], "ev")
    assert used == set()
    assert usage.total_tokens == 0


async def test_model_used_detector_catches_in_context_use() -> None:
    # THE FIX: a.py is never referenced in the evidence, but the judge says the response drew on it.
    offered = [_cand("a.py", 0.5), _cand("b.py", 0.5)]
    used = await ModelUsedDetector(_Provider('{"used": ["a.py"]}'))("t", offered, "unrelated reply")
    assert used == {"a.py"}


async def test_model_used_detector_unions_reference_proxy() -> None:
    # The judge returns nothing, but b.py is explicitly referenced -> the proxy still catches it.
    offered = [_cand("a.py", 0.5), _cand("b.py", 0.5)]
    used = await ModelUsedDetector(_Provider('{"used": []}'))("t", offered, "I edited b.py")
    assert used == {"b.py"}


async def test_signal_logger_judge_labels_in_context_use(tmp_path) -> None:
    # End to end: a candidate consumed IN-CONTEXT (judge yes) but never referenced is used=True --
    # exactly the label the reference proxy gets wrong.
    g = ContextGatherer([])
    g.last_offer = [_cand("helper.py", 0.5)]
    g.last_task = "do the thing"
    session = Session(cwd=str(tmp_path), model="test", messages=[])  # no tool call, no mention
    log = tmp_path / "judge.jsonl"
    detector = ModelUsedDetector(_Provider('{"used": ["helper.py"]}'))
    await SignalLogger(g, session, log, used_detector=detector).on_turn_end(
        TurnEndPayload(stop_reason="completed", last_assistant_message="all done")
    )
    rec = json.loads(log.read_text(encoding="utf-8").strip())
    assert rec["n_used"] == 1
    assert rec["offered"][0]["used"] is True


def test_agent_signal_judge_wires_observer(tmp_path) -> None:
    settings = Settings(
        default_model="scripted/test", workspace_root=tmp_path, permission_mode="allow"
    )
    agent = Agent(
        settings=settings,
        enable_context_gathering=True,
        context_signal_log=str(tmp_path / "s.jsonl"),
        context_signal_judge=True,
    )
    assert len(agent.hook_manager.turn_end_observers) == 1
