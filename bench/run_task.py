#!/usr/bin/env python
"""Lane-D benchmark runner: run ONE zak-code task headless against Groq, capture cost.

Usage (from the Zak-Code repo root, with the repo venv):
    ./.venv/Scripts/python.exe bench/run_task.py bench/tasks/01-wordfreq
    ./.venv/Scripts/python.exe bench/run_task.py --preflight bench/tasks/01-wordfreq

A task dir contains:
    task.json   {id, title, prompt, max_iterations?, max_cost_usd?, verify_timeout_s?}
    workspace/  (optional) seed files copied into a fresh temp workspace
    verify.py   exits 0 on success; run with cwd = the temp workspace (held-out oracle)

The runner sets up an isolated temp workspace, drives the agent to completion with
permission_mode=autonomous (no prompts; dangerous ops hard-DENY), runs verify.py, and
emits a single JSON result to stdout. --preflight constructs the agent and checks the
cost/permission API WITHOUT any LLM call (cheap API smoke).
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _ensure_interpreter_on_path() -> None:
    """Put THIS runner's interpreter dir first on PATH for the agent's SUBPROCESS tools.

    The agent verifies its work by shelling out (`python -m pytest ...`) via bash/powershell,
    which resolve `python` from PATH. The isolated temp workspace has no project venv, so a bare
    `python` hits the *system* interpreter — which typically lacks pytest. The model's correct
    test-suite verification then fails with "No module named pytest" (and `pip install` is
    hard-denied in autonomous mode), so the recipe gate stalls a turn whose code is actually
    fine — while the held-out oracle, run with THIS interpreter (the repo venv, which HAS
    pytest), passes. That mismatch is exactly the bench's `recipe_stalled`-but-oracle-passes
    artifact. A real user runs zakcode inside an activated project venv, so the representative
    fix is to expose the runner's own (pytest-capable) environment to the subprocess. Idempotent.
    """
    scripts_dir = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    if scripts_dir not in path.split(os.pathsep):
        os.environ["PATH"] = scripts_dir + os.pathsep + path


def _build_agent(workspace: Path, spec: dict):
    """Mirror the CLI's canonical construction, adapted for headless reproducible runs."""
    from zakcode import Agent
    from zakcode.config import load_settings

    # Make `python`/`pytest` resolve to this runner's (pytest-capable) interpreter for the
    # agent's subprocess verification, mirroring an activated project venv (see the helper).
    _ensure_interpreter_on_path()
    # load_settings() (cwd = repo root) loads the repo .env -> GROQ_API_KEY into os.environ,
    # which litellm reads directly for groq/ models. api_base stays None (commented out in .env).
    base = load_settings()
    update = {
        # A Path, NOT str(workspace): `base.model_copy(update=...)` on line ~120 does NOT
        # re-validate, so a str here defeats the `workspace_root: Path` annotation on
        # Settings and survives all the way to load_settings_permissions(), which does
        # `workspace_root / ".claude"` and dies with TypeError: unsupported operand
        # type(s) for /: 'str' and 'str' — before a single model call. Every test passes
        # a real tmp_path, so the suite stays green while every bench task crashes.
        "workspace_root": workspace,
        "default_model": "zakpick",        # Groq per-category routing
        "permission_mode": "autonomous",   # headless: never prompts, dangerous=hard-DENY
        "max_cost_usd": spec.get("max_cost_usd", 1.0),
        "max_iterations": spec.get("max_iterations", 50),
        "api_base": None,                  # belt-and-suspenders: never a local base
    }
    # Model-comparison override (Lane D): ZBENCH_DEEP_MODEL swaps the deep_code category's
    # model so the same tasks can be benchmarked across candidates (e.g. llama-3.3-70b-versatile
    # vs the default openai/gpt-oss-120b). tool_calling_mode flows from ZAKCODE_TOOL_CALLING_MODE
    # via base settings (text|native|auto) — no override needed here.
    deep_model = os.environ.get("ZBENCH_DEEP_MODEL")
    if deep_model:
        from zakcode.providers.routing import ZakpickModel

        # ZBENCH_DEEP_SOURCE picks the provider/runtime: "groq" (default), "openai",
        # "gemini", "deepseek", "fireworks_ai", "together_ai", "local"/"ollama", etc.
        # The litellm string is "<source>/<model>" so any litellm-supported supplier
        # can be benchmarked for the deep_code category by config alone.
        deep_source = os.environ.get("ZBENCH_DEEP_SOURCE", "groq")
        existing = dict(getattr(base, "zakpick_models", None) or {})
        existing["deep_code"] = ZakpickModel(model=deep_model, source=deep_source)
        existing["delegate"] = ZakpickModel(model=deep_model, source=deep_source)
        update["zakpick_models"] = existing
    # Rules arm (g-016-91). By default this runner builds with enable_rules=False AND a temp
    # workspace holding no rules — so a ZAKCODE_LEAN_RULES A/B is structurally incapable of
    # varying its own variable: both arms render the empty string. Measured 2026-07-29 at
    # 8e59d49: rules_discovered=0, full=0 chars, index=0 chars, sha1-identical. That matters
    # because the lean_rules decision rule is "flip the default if NO regression" — a dead
    # instrument returns exactly the null result that authorizes the flip.
    #
    # Opt in with ZBENCH_RULES_ROOT=<mind-repo>, which seeds that mind's .claude/rules into the
    # temp workspace (a path `default_rule_dirs` searches) and turns rule injection on. Mirrors
    # the ZBENCH_DEEP_MODEL / ZBENCH_DEEP_SOURCE override pattern above. Unset — the default —
    # reproduces the previous behaviour byte-for-byte.
    #
    # Fails LOUD on a bad root rather than degrading to zero rules: a silent fallback would
    # recreate the very dead-instrument failure this flag exists to remove.
    rules_root = os.environ.get("ZBENCH_RULES_ROOT")
    enable_rules = False
    if rules_root:
        src_rules = Path(rules_root) / ".claude" / "rules"
        if not src_rules.is_dir():
            raise SystemExit(f"ZBENCH_RULES_ROOT={rules_root!r}: no .claude/rules directory")
        dst_rules = workspace / ".claude" / "rules"
        dst_rules.mkdir(parents=True, exist_ok=True)
        seeded = 0
        for rule_file in sorted(src_rules.glob("*.md")):
            shutil.copy2(rule_file, dst_rules / rule_file.name)
            seeded += 1
        if seeded == 0:
            raise SystemExit(f"ZBENCH_RULES_ROOT={rules_root!r}: .claude/rules holds no *.md files")
        enable_rules = True
        lean = os.environ.get("ZAKCODE_LEAN_RULES", "false")
        print(
            f"[bench] rules arm ON: seeded {seeded} rule(s) from {src_rules} "
            f"(ZAKCODE_LEAN_RULES={lean})",
            file=sys.stderr,
        )
    settings = base.model_copy(update=update)
    agent = Agent(
        settings=settings,
        enable_compaction=True,   # needed so long tasks survive the context window
        enable_rules=enable_rules,  # default OFF; ZBENCH_RULES_ROOT turns it on (see above)
        enable_skills=False,      # OFF: no skill dirs
        enable_subagents=False,   # OFF: keep the baseline single-agent (enable later for a "full" run)
        enable_mcp=False,
        enable_plugins=False,
    )
    # ZBENCH_COMPACT_FRACTION moves the compaction THRESHOLD without touching anything else.
    # The obvious way to force compaction -- declare a smaller context_window -- is CONFOUNDED:
    # _window() feeds three consumers, so shrinking it also cuts the seam clamp in
    # _clamp_result (window * 0.25 * 3 chars: 98,304 -> 24,576, every tool result 4x smaller)
    # and changes _refuse_oversized_body. A task failing under that arm could be lost grep/read
    # output rather than compaction, and the failure would be unattributable -- the same
    # confound that wrecked two prior attempts to separate engine behaviour from model
    # behaviour. Moving threshold_fraction on the FULL window changes ONE variable.
    # Unset (the default) leaves the shipped 0.8 byte-unchanged.
    # ZBENCH_TOOL_DENY narrows the ADVERTISED tool surface via the registry's own
    # set_exposure_filter -- documented least-privilege, already shipped, operator-set. Measured
    # across 22 recorded runs: the agent called EIGHT distinct tools; the other 17 cost 3,907 tok
    # (58% of the tool surface, 44% of the 8,956-token fixed floor) and were never called once.
    # That floor is 6.8% of a 131,072 window and 27.3% of a 32,768 one, so it scales badly toward
    # exactly the models this bench exists to compare.
    deny = os.environ.get("ZBENCH_TOOL_DENY")
    if deny:
        agent.registry.set_exposure_filter(deny=[x.strip() for x in deny.split(",") if x.strip()])
        print(f"[bench] tool deny filter: {deny}", file=sys.stderr)
    frac = os.environ.get("ZBENCH_COMPACT_FRACTION")
    if frac and agent.compactor is not None:
        agent.compactor.config.threshold_fraction = float(frac)
        print(f"[bench] compaction threshold_fraction={frac}", file=sys.stderr)
    return agent


def _usage_snapshot(agent) -> dict:
    sess = agent.session.cumulative_usage()
    per_model = {
        m: {
            "cost_usd": round(u.cost_usd, 6),
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        for m, u in agent.session.usage_by_model().items()
    }
    return {
        "session_cost_usd": round(sess.cost_usd, 6),
        "session_prompt_tokens": sess.prompt_tokens,
        "session_completion_tokens": sess.completion_tokens,
        "session_total_tokens": sess.total_tokens,
        "cache_read_tokens": sess.cache_read_tokens,
        "per_model": per_model,
    }


def preflight(task_dir: Path) -> int:
    """Construct the agent and exercise the cost/permission API with NO LLM call."""
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ws = Path(tempfile.mkdtemp(prefix=f"zbench-pf-{spec['id']}-"))
    try:
        agent = _build_agent(ws, spec)
        snap = _usage_snapshot(agent)  # must work on a fresh session -> all zero
        perm = agent.settings.permission_mode
        print(
            json.dumps(
                {
                    "preflight": "ok",
                    "permission_mode": str(perm),
                    "default_model": agent.settings.default_model,
                    "max_cost_usd": agent.settings.max_cost_usd,
                    "max_iterations": agent.settings.max_iterations,
                    "fresh_session_cost_usd": snap["session_cost_usd"],
                    "has_run_turn": hasattr(agent, "run_turn"),
                },
                indent=2,
            )
        )
        return 0
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def run(task_dir: Path) -> int:
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    ws = Path(tempfile.mkdtemp(prefix=f"zbench-{spec['id']}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)

    # Armed BEFORE the agent is built: the probe patches Compactor.should_compact at class
    # level, and the Agent's compactor is constructed inside _build_agent.
    compaction = _instrument_compaction()
    agent = _build_agent(ws, spec)
    # The surface the model was ACTUALLY offered. Without this a deny filter that silently failed
    # to apply is indistinguishable from one that applied and changed nothing -- the same
    # "instrument that stopped measuring" shape this bench keeps finding (ADR-0145).
    _defs = agent.registry.definitions()
    tool_surface = {"exposed": len(_defs), "schema_chars": sum(len(json.dumps(x)) for x in _defs)}
    # EVERY EXPERIMENTAL KNOB RECORDS ITS VALUE HERE, because an arm LABEL kept outside the
    # artifact is an assumption about which env var was set when this process started. Measured
    # 2026-09-12: a 6-pass temperature experiment lost every pass to a filename error, and the one
    # surviving file could only be assigned to an arm by MTIME ORDERING -- the data itself did not
    # say. compaction records threshold_tokens and the tool filter records tool_surface.exposed,
    # and both of those let an inert knob be told apart from a genuine null result; temperature had
    # no such field. `temperature` is read off the RESOLVED Settings (what actually took effect),
    # not off the env string (what was requested).
    knobs = {
        "temperature": getattr(getattr(agent, "settings", None), "temperature", None),
        "compact_fraction": os.environ.get("ZBENCH_COMPACT_FRACTION"),
        "tool_deny": os.environ.get("ZBENCH_TOOL_DENY"),
        "rules_root": os.environ.get("ZBENCH_RULES_ROOT"),
    }

    err = None
    stop_reason = iterations = routed_category = routed_escalated = degraded = None
    turn_error = None
    turn_cost = turn_tokens = 0
    tool_calls: dict = {}
    tool_errors = 0
    trace_events: dict = {}
    trace_interventions: dict = {}
    t0 = time.perf_counter()
    try:
        result = agent.run_turn(spec["prompt"])
        stop_reason = result.stop_reason
        iterations = result.iterations
        routed_category = result.routed_category
        routed_escalated = result.routed_escalated
        degraded = result.degraded
        turn_error = result.error  # TurnResult.error: the provider/loop error detail behind stop_reason
        turn_cost = result.usage.cost_usd
        turn_tokens = result.usage.total_tokens
        # TurnResult already carries three signals this bench used to throw away: WHICH tools
        # were called (assistant_messages -> ToolUseBlock.name), which RESULTS errored, and the
        # engine's own decision trace (every gate/recovery intervention it fired, empty on a
        # clean turn). Recording what the engine already reports beats bolting on more probes,
        # and it makes "which robustness paths actually ran?" answerable natively rather than
        # by monkeypatch. Tool-name counts also answer whether the 25-tool / 6,735-token schema
        # surface is earning its place on a small model -- nothing measured that before.
        counts: collections.Counter = collections.Counter()
        for msg in result.assistant_messages:
            for blk in msg.blocks:
                if getattr(blk, "type", None) == "tool_use":
                    counts[blk.name] += 1
        tool_calls = dict(counts.most_common())
        tool_errors = sum(1 for r in result.tool_results if r.is_error)
        trace_events = dict(collections.Counter(e.kind for e in result.trace.events).most_common())
        # An "intervention" event's IDENTITY lives in its payload, not its kind: TurnTrace.note
        # is `note("intervention", "...", kind="doom_loop")`, where the leading positional is the
        # EVENT kind and the keyword `kind` lands in `data`. Counting only e.kind therefore
        # reports `intervention: 5` while saying nothing about WHICH five gates fired -- and
        # "which robustness paths ran?" is the entire question this recording exists to answer.
        # Measured 2026-09-11 on 04-todo-cli: intervention=5 beside compaction fired=3, leaving
        # two interventions unidentifiable at exactly the moment a failure needed explaining.
        trace_interventions = dict(
            collections.Counter(
                (e.data or {}).get("kind") or (e.detail or "?")[:40]
                for e in result.trace.events
                if e.kind == "intervention"
            ).most_common()
        )
    except Exception as e:  # noqa: BLE001 - a crash is a result (a bug to file), not a runner failure
        err = f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - t0

    snap = _usage_snapshot(agent)

    verify_rc = None
    verify_out = ""
    vf = task_dir / "verify.py"
    if vf.is_file() and err is None:
        try:
            proc = subprocess.run(
                [sys.executable, str(vf)],
                cwd=ws,
                capture_output=True,
                text=True,
                timeout=spec.get("verify_timeout_s", 120),
            )
            verify_rc = proc.returncode
            verify_out = (proc.stdout + proc.stderr).strip()[-800:]
        except subprocess.TimeoutExpired:
            verify_rc = 124
            verify_out = "verify timed out"

    success = verify_rc == 0
    report = {
        "id": spec["id"],
        "title": spec.get("title", ""),
        "success": success,
        "stop_reason": stop_reason,
        "iterations": iterations,
        "routed_category": routed_category,
        "routed_escalated": routed_escalated,
        "degraded": degraded,
        "turn_error": turn_error,
        "elapsed_s": round(elapsed, 1),
        "tok_per_s": round((snap["session_completion_tokens"] / elapsed), 1) if elapsed > 0 else None,
        "turn_cost_usd": round(turn_cost, 6),
        "turn_tokens": turn_tokens,
        **snap,
        "compaction": compaction,
        "tool_surface": tool_surface,
        "knobs": knobs,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "trace_events": trace_events,
        "trace_interventions": trace_interventions,
        "verify_rc": verify_rc,
        "verify_out": verify_out,
        "error": err,
        "workspace": str(ws) if not success else "(cleaned)",
    }
    print(json.dumps(report, indent=2))

    # Keep the workspace for inspection on failure/crash; clean it on success.
    if success:
        shutil.rmtree(ws, ignore_errors=True)
    return 0


def _instrument_compaction() -> dict:
    """Record what the compactor SAW on every check, so a threshold that NEVER TRIPS is visible.

    ``enable_compaction=True`` is set honestly, a ``Compactor`` is attached honestly, and
    ``_maybe_compact()`` runs before every provider call honestly -- and none of that is
    evidence the compaction path ever EXECUTED. ``should_compact`` fires at
    ``threshold_fraction`` (0.8) of the window, and every ``_podenv*.sh`` slot declares
    ``context_window: 131072``, so the threshold is 104,857 tokens on all three pod variants
    (baseline 3.6-35b, weak 3.8-27b, older 3.5-35b alike). The hardest task in this suite
    (``05-ledger``) peaks near 52k -- HALF the threshold. So every pass measured so far
    returned False on every call, and a compaction path that never runs reports
    byte-identically to one that works perfectly.

    Peak figures are from ``weak-pass-{1,2}.json`` -- 2 passes x 10 tasks on
    ``zds-qwen3.8-27b`` -- derived by fitting ``input(i) = floor + k*i`` to the per-task
    billed prompt totals, NOT read off a per-call record (the bench persists no transcript,
    so no per-call context size exists for any prior run; that is what this probe adds).

    That matters most for exactly the models this bench exists to compare. The threshold is a
    FRACTION of the window, so a 32,768-window model compacts at 26,214 -- which
    ``04-todo-cli`` (23-33k) and ``05-ledger`` (32-52k) both cross. ``trim_tail``, ``_split_index``,
    ``_adopt_compacted`` and the summarizer call would first execute on the smallest models,
    having never been exercised by a single benchmark run.

    Reports ``peak_context_tokens`` beside ``fired``: the peak alone cannot say whether the
    mechanism is healthy or dead, and ``fired: 0`` alone cannot say whether the context stayed
    small or the check is broken. The pair is readable; either number alone is not.

    Calls the ORIGINAL for the verdict rather than re-deriving ``n > threshold`` here -- a
    duplicated predicate would drift from the engine's and report a confident wrong "never
    fired". ``count_tokens`` is chars/4 (a local string op, no API call), so counting a second
    time for the peak costs nothing measurable.
    """
    from zakcode.agent.compact import Compactor

    stats: dict = {
        "checks": 0,
        "fired": 0,
        "peak_context_tokens": 0,
        "threshold_tokens": None,
        "context_window": None,
    }
    original = Compactor.should_compact

    def probe(self, messages, *, context_window, count_tokens):
        verdict = original(
            self, messages, context_window=context_window, count_tokens=count_tokens
        )
        stats["checks"] += 1
        stats["context_window"] = context_window
        if context_window:
            stats["threshold_tokens"] = int(context_window * self.config.threshold_fraction)
        stats["peak_context_tokens"] = max(stats["peak_context_tokens"], count_tokens(messages))
        if verdict:
            stats["fired"] += 1
        return verdict

    Compactor.should_compact = probe
    return stats


def _enable_engine_warnings() -> None:
    """Route the engine's WARNING log records to stderr, so the bench can SEE them.

    ``zakcode/__init__`` installs a ``NullHandler`` on the package logger -- correct library
    hygiene, and it also means ``logging.lastResort`` never fires, because lastResort only
    engages when NO handler is found anywhere in the chain. A NullHandler counts as one. So
    ``loop.py``'s ``logger.warning("provider rejected a malformed tool call; retrying ...")``
    -- and its rate-limit and timeout siblings -- were emitted correctly and went NOWHERE in
    every bench run this file has ever driven.

    Measured 2026-09-11: a bare ``logger.warning`` on a fresh interpreter DOES reach stderr
    via lastResort; the same call after ``import zakcode`` does not. The package logger
    propagates (``propagate=True``), so a root handler receives the record past the
    NullHandler -- which is what this installs.

    Why it matters: those warnings are the ONLY signal that a provider retry, a malformed
    tool call, or a rate limit occurred. None of them reach ``TurnResult`` -- a successful
    retry ends the turn ``completed`` with ``degraded=False`` and ``error=""`` -- so with the
    log swallowed, a run that fought through three malformed tool calls is byte-identical to
    a clean one in the report. That is the hazard the suite exists to detect, in the suite.
    """
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="[engine] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main(argv: list[str]) -> int:
    _enable_engine_warnings()
    args = [a for a in argv if a != "--preflight"]
    if not args:
        print("usage: run_task.py [--preflight] <task_dir>", file=sys.stderr)
        return 2
    task_dir = Path(args[0]).resolve()
    if "--preflight" in argv:
        return preflight(task_dir)
    return run(task_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
