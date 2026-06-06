"""Live reliability harness for Omni's Job 1 (the Vinheim hosted-agent substrate).

The smoke test (``%TEMP%\\zak_smoke.py``) proved ONE fact survives ONE restart. Real Vinheim
usage is many turns: a customer drips facts over a conversation, comes back later, and expects
the agent to remember. This harness measures whether the *substrate* (a mind-seeded
``zakcode.Agent`` on a small local model, the same one ``zakcode serve`` builds) holds up over
that realistic shape — it tests the runtime, not any particular mind.

It runs entirely against a local Ollama model ($0, no cloud) and reports:

  * remember_call_rate   — over N seed turns, how often the model actually emitted a
                           ``remember`` tool call (did the use-memory rule land?).
  * facts_persisted      — distinct facts that produced a row in <env>/.zakcode/memory.db.
  * recall_after_restart — a FRESH Agent (new history, same env/DB) correctly answers each
                           recall question from stored memory.
  * false_recall         — asked about a fact never given, the agent does NOT assert one of
                           the OTHER stored values (a crude confabulation check).
  * latency              — wall-clock per turn (informs the hosting cost/UX bar).

Usage:
    uv run python scripts/job1_memory_reliability.py
    uv run python scripts/job1_memory_reliability.py --model ollama_chat/qwen2.5:3b --json

Exit code is 0 if the headline thresholds are met, 1 otherwise — so it can gate a release.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import zakcode
from zakcode.memory.sqlite_store import SqliteMemoryProvider
from zakcode.messages import ToolUseBlock

DEFAULT_MODEL = "ollama_chat/qwen2.5:3b"

# Distinctive, easy-to-keyword-check facts. ``expect`` is matched case-insensitively against
# the recall answer; any one hit counts as recalled.
FACTS: list[dict] = [
    {
        "key": "language",
        "statement": "Remember that my favorite programming language is Rust.",
        "question": "What is my favorite programming language?",
        "expect": ["rust"],
    },
    {
        "key": "city",
        "statement": "Remember that I live in Tucson, Arizona.",
        "question": "Which city do I live in?",
        "expect": ["tucson"],
    },
    {
        "key": "pet",
        "statement": "Remember that my dog's name is Mochi.",
        "question": "What is my dog's name?",
        "expect": ["mochi"],
    },
    {
        "key": "project",
        "statement": "Remember that I am building a product called Vinheim.",
        "question": "What product am I building?",
        "expect": ["vinheim"],
    },
    {
        "key": "number",
        "statement": "Remember that my lucky number is 47.",
        "question": "What is my lucky number?",
        "expect": ["47"],
    },
    {
        "key": "drink",
        "statement": "Remember that I always order oat milk lattes.",
        "question": "What do I always order to drink?",
        "expect": ["oat milk", "oat-milk", "latte"],
    },
]

# A fact never told — the recall answer must not confabulate one of the stored values.
FALSE_QUESTION = "What is my favorite movie?"

# Per-turn iteration cap for the live agents. A remember/recall turn needs <=3 iterations;
# this bounds any runaway turn far below the default 50 so a slow local run stays observable.
MAX_ITERATIONS = 16


def _progress(msg: str) -> None:
    """Emit a live progress line to stderr (the run only prints its summary at the end)."""
    print(f"[harness] {msg}", file=sys.stderr, flush=True)


# Headline pass/fail thresholds (the Job-1 connection bar — quality is secondary, but the
# substrate must reliably persist and recall).
THRESHOLDS = {"remember_call_rate": 0.8, "facts_persisted": 0.8, "recall_after_restart": 0.8}


def _seed_mind(env: Path) -> None:
    """Drop a minimal Vinheim 'mind' into the env (identity + use-memory rule).

    Mirrors Omni's seed shape so the harness is self-contained and reproducible.
    """
    (env / ".zakcode" / "rules").mkdir(parents=True, exist_ok=True)
    (env / "self.md").write_text(
        "You are Vinheim, a friendly, concise guide in the user's environment. You remember "
        "what users tell you and use it to help them later.",
        encoding="utf-8",
    )
    (env / ".zakcode" / "rules" / "use-memory.md").write_text(
        "When the user shares something worth keeping (a preference or a fact about them), call "
        "the remember tool to save it. Recall relevant memories before answering.",
        encoding="utf-8",
    )


def _agent(env: Path, model: str) -> zakcode.Agent:
    """An Agent matching what ``zakcode serve`` builds for Job 1 (acceptEdits so the

    WORKSPACE_WRITE ``remember`` tool auto-allows without a REST prompter).
    """
    return zakcode.Agent(
        workspace_root=env,
        default_model=model,
        permission_mode="acceptEdits",
        max_iterations=MAX_ITERATIONS,
        enable_skills=True,
        enable_rules=True,
        enable_memory=True,
    )


def _called_remember(result) -> bool:
    """True if the turn emitted at least one ``remember`` tool call."""
    return any(
        isinstance(b, ToolUseBlock) and b.name == "remember"
        for m in result.assistant_messages
        for b in m.blocks
    )


def _answer_text(result) -> str:
    return "\n".join(m.text for m in result.assistant_messages if m.text)


@dataclass
class FactOutcome:
    key: str
    remembered_call: bool = False
    persisted: bool = False  # the DB grew on this fact's seed turn
    recalled: bool = False
    recall_answer: str = ""
    seed_latency_s: float = 0.0
    recall_latency_s: float = 0.0


@dataclass
class Report:
    model: str
    outcomes: list[FactOutcome] = field(default_factory=list)
    db_rows_total: int = 0
    false_recall_clean: bool = True
    false_recall_answer: str = ""

    def _rate(self, attr: str) -> float:
        n = len(self.outcomes) or 1
        return sum(1 for o in self.outcomes if getattr(o, attr)) / n

    def summary(self) -> dict:
        return {
            "model": self.model,
            "facts": len(self.outcomes),
            "remember_call_rate": round(self._rate("remembered_call"), 3),
            "facts_persisted": round(self._rate("persisted"), 3),
            "recall_after_restart": round(self._rate("recalled"), 3),
            "false_recall_clean": self.false_recall_clean,
            "db_rows_total": self.db_rows_total,
            "per_fact": [
                {
                    "key": o.key,
                    "remember_call": o.remembered_call,
                    "persisted": o.persisted,
                    "recalled": o.recalled,
                    "answer": o.recall_answer[:120],
                    "seed_s": round(o.seed_latency_s, 2),
                    "recall_s": round(o.recall_latency_s, 2),
                }
                for o in self.outcomes
            ],
            "false_recall_answer": self.false_recall_answer[:160],
        }

    def passed(self) -> bool:
        s = self.summary()
        return all(s[k] >= v for k, v in THRESHOLDS.items())


async def run(env: Path, model: str) -> Report:
    _seed_mind(env)
    report = Report(model=model)
    db_path = env / ".zakcode" / "memory.db"

    def db_count() -> int:
        return SqliteMemoryProvider(str(db_path)).count() if db_path.exists() else 0

    # ── Phase 1: seed facts over many turns (one agent = one "session") ──────────
    n = len(FACTS)
    _progress(f"seeding {n} facts on {model} (acceptEdits, max_iter={MAX_ITERATIONS}) ...")
    seeder = _agent(env, model)
    for i, fact in enumerate(FACTS, 1):
        outcome = FactOutcome(key=fact["key"])
        before = db_count()
        _progress(f"seed {i}/{n} {fact['key']} ...")
        t0 = time.perf_counter()
        result = await seeder.arun_turn(fact["statement"])
        outcome.seed_latency_s = time.perf_counter() - t0
        outcome.remembered_call = _called_remember(result)
        outcome.persisted = db_count() > before
        report.outcomes.append(outcome)
        _progress(
            f"  -> remember_call={outcome.remembered_call} persisted={outcome.persisted} "
            f"({outcome.seed_latency_s:.1f}s, db_rows={db_count()})"
        )
    await seeder.aclose()
    report.db_rows_total = db_count()

    # ── Phase 2: RESTART — a fresh agent recalls each fact from the on-disk DB ────
    _progress(f"restart: recalling {n} facts with fresh agents ...")
    by_key = {o.key: o for o in report.outcomes}
    for i, fact in enumerate(FACTS, 1):
        recaller = _agent(env, model)  # fresh history, same env/DB
        _progress(f"recall {i}/{n} {fact['key']} ...")
        t0 = time.perf_counter()
        result = await recaller.arun_turn(fact["question"])
        latency = time.perf_counter() - t0
        await recaller.aclose()
        answer = _answer_text(result)
        o = by_key[fact["key"]]
        o.recall_answer = answer
        o.recall_latency_s = latency
        o.recalled = any(e.lower() in answer.lower() for e in fact["expect"])
        _progress(f"  -> recalled={o.recalled} ({latency:.1f}s) {answer[:80]!r}")

    # ── Phase 3: false-recall — must not confabulate another stored value ────────
    _progress("false-recall check ...")
    fr_agent = _agent(env, model)
    fr_result = await fr_agent.arun_turn(FALSE_QUESTION)
    await fr_agent.aclose()
    fr_answer = _answer_text(fr_result)
    report.false_recall_answer = fr_answer
    # Word-boundary match so a stored value is only a "confabulation" when it appears as a
    # whole word/phrase — not as a substring (e.g. "rust" inside "trust", "47" inside "1947",
    # "latte" inside "later"), which would false-positive the check.
    other_values = [e for f in FACTS for e in f["expect"]]
    report.false_recall_clean = not any(
        re.search(rf"\b{re.escape(v)}\b", fr_answer, re.IGNORECASE) for v in other_values
    )

    return report


async def main() -> int:
    parser = argparse.ArgumentParser(description="Job-1 memory reliability harness (live, local).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="litellm model string.")
    parser.add_argument("--workspace", default=None, help="Reuse an env dir (default: temp).")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    args = parser.parse_args()

    if args.workspace:
        env = Path(args.workspace)
        env.mkdir(parents=True, exist_ok=True)
        report = await run(env, args.model)
    else:
        with TemporaryDirectory() as d:
            env = Path(d) / "test-env"
            env.mkdir()
            report = await run(env, args.model)

    summary = report.summary()
    ok = report.passed()
    if args.json:
        print(json.dumps({"ok": ok, **summary}))
    else:
        print(f"\nJob-1 memory reliability — {summary['model']}")
        print(f"  remember_call_rate   {summary['remember_call_rate']:.0%}")
        print(f"  facts_persisted      {summary['facts_persisted']:.0%}")
        print(f"  recall_after_restart {summary['recall_after_restart']:.0%}")
        print(f"  false_recall_clean   {summary['false_recall_clean']}")
        print(f"  false_recall_answer  {summary['false_recall_answer']!r}")
        print(f"  db_rows_total        {summary['db_rows_total']}")
        print("  per-fact:")
        for pf in summary["per_fact"]:
            mark = "ok " if pf["recalled"] else "MISS"
            print(
                f"    [{mark}] {pf['key']:<9} call={pf['remember_call']!s:<5} "
                f"persist={pf['persisted']!s:<5} -> {pf['answer']!r}"
            )
        print(f"\n  {'PASS' if ok else 'FAIL'} (thresholds {THRESHOLDS})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
