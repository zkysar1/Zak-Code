#!/usr/bin/env python
"""Cross-agent determinism arm: run ONE bench task through Claude Code N times, compare outputs.

WHY THIS AXIS. The parity question has two blockers -- suite saturation and the model confound.
Saturation is closed by measurement (ADR-0151/0156: both arms pass everything, including three
tasks purpose-built to reject wrong answers). The model confound looked equally fatal, because
holding the model fixed on this box is impossible.

IT IS NOT FATAL HERE. Determinism is a property of the LOOP, not of the model: each agent is
measured on the RUN-TO-RUN VARIANCE OF ITS OWN OUTPUTS. Nothing is compared across models, so
nothing is confounded by them. zakcode's side is already measured (ADR-0152: m05 at cap=64
reproduced 3/3 byte-identically with ZBENCH_PIN_IDENTITY set). Claude Code's side is not, and
without it "zakcode is more deterministic" is an assertion.

THE HONEST FRAMING, which must travel with every number this prints: zakcode's 3/3 was measured
WITH identity pinning, a deliberate intervention. Claude Code gets no equivalent because none
exists to apply. So this is "zakcode with its determinism feature ON" vs "Claude Code as it
ships" -- a claim about an AVAILABLE CAPABILITY, not about two loops under identical treatment.

Pre-registered in bench/results/determinism-arm-preregistration.log BEFORE any run.

Usage:  ./.venv/bin/python bench/determinism_cc.py bench/tasks/02-median-bug [N]
"""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
MIN_AVAIL_MB = 800  # the box has 4 GB; a prior pass OOM-killed a 57-minute run (ADR-0156).


def _avail_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return -1


def _digest_tree(ws: Path) -> dict[str, str]:
    """SHA-256 of every file the agent left. Paths are ws-relative so the temp-dir suffix --
    which differs by construction every run -- cannot leak into the comparison."""
    out: dict[str, str] = {}
    for p in sorted(ws.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            out[str(p.relative_to(ws))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


_SOURCE_SUFFIXES = (".py", ".md", ".txt", ".csv", ".json", ".toml", ".yaml", ".yml", ".cfg", ".ini")


def _capture_sources(ws: Path) -> dict[str, str]:
    """Text of the task's own small output files. A digest says THAT two runs differ; only the
    content says HOW, and 'differs in a comment' and 'differs in the algorithm' are not the same
    finding. Bounded at 4 KB/file so this can never become the reason a run is unaffordable.

    Captured Python only until ADR-0164: an m-task whose output is a ``count.md`` produced a
    within-cell digest difference that could not be inspected afterwards (both sources empty).
    Now every small text file the task can write; binaries are skipped by the decode error."""
    out: dict[str, str] = {}
    for p in sorted(ws.rglob("*")):
        if not p.is_file() or p.suffix not in _SOURCE_SUFFIXES or "__pycache__" in p.parts:
            continue
        if any(part.startswith(".") for part in p.relative_to(ws).parts):
            continue  # zakcode's own markers / VCS dirs are not task output
        with contextlib.suppress(OSError, UnicodeDecodeError):
            out[str(p.relative_to(ws))] = p.read_text(encoding="utf-8")[:4096]
    return out


_RUN_COUNTER = itertools.count(1)  # per-process run index, for per-run request dumps


def one_run_zakcode(task_dir: Path, spec: dict, timeout_s: int = 1800, pin: bool = True) -> dict:
    """zakcode arm, WITH ZBENCH_PIN_IDENTITY -- its determinism feature switched ON.

    run_task.py under the pin uses a CONSTANT workspace path (/tmp/zbench-pinned-<id>), wiped
    and re-seeded at the start of each run, so the workspace survives the process and can be
    digested here by the same _digest_tree the reference arm uses. Identical instrument on both
    sides is the whole point: two different digest functions would make any difference between
    the arms unattributable."""
    import os
    # KEEP_WORKSPACE is load-bearing, not convenience: run_task.py deletes the workspace on
    # success, and digesting a deleted directory yields an empty set that compares equal to any
    # other empty set -- a vacuous "identical" verdict. The guard in main() refuses that, but the
    # capture has to actually work for the arm to produce a result at all.
    env = dict(os.environ, ZBENCH_KEEP_WORKSPACE="1")
    if pin:
        env["ZBENCH_PIN_IDENTITY"] = os.environ.get("ZBENCH_PIN_MODE", "1")
    # ZBENCH_DUMP_REQUESTS (run_task.py) writes call-NNN.json into ONE directory, so N children
    # sharing the arm's environment would overwrite each other and the diff the pre-registered
    # protocol calls for ("dump requests and diff before attributing", ADR-0157) would compare a
    # run with itself. Each run gets its own subdirectory.
    k = next(_RUN_COUNTER)
    if os.environ.get("ZBENCH_DUMP_REQUESTS"):
        env["ZBENCH_DUMP_REQUESTS"] = str(Path(os.environ["ZBENCH_DUMP_REQUESTS"]) / f"run-{k}")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(BENCH / "run_task.py"), str(task_dir)],
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
        rc, stdout, stderr_tail = proc.returncode, proc.stdout, proc.stderr.strip()[-300:]
    except subprocess.TimeoutExpired:
        rc, stdout, stderr_tail = 124, "", f"exceeded {timeout_s}s"
    elapsed = time.perf_counter() - t0

    rep: dict = {}
    try:
        txt = stdout.strip()
        start = txt.rfind("\n{")
        rep = json.loads(txt if txt.startswith("{") else txt[start:].strip())
    except (json.JSONDecodeError, ValueError):
        rep = {}

    # Unpinned runs get an mkdtemp suffix, so the path must come from the REPORT, not from a
    # constant. Guessing a constant path here would silently digest nothing.
    reported = rep.get("workspace") or ""
    ws = (Path(reported) if reported.startswith("/")
          else Path(tempfile.gettempdir()) / f"zbench-pinned-{spec['id']}")
    digests = _digest_tree(ws) if ws.is_dir() else {}
    sources = _capture_sources(ws) if ws.is_dir() else {}
    if not pin:
        shutil.rmtree(ws, ignore_errors=True)  # unpinned runs leave a fresh dir each time
    return {
        "cli_rc": rc,
        "workspace": str(ws),
        "pin": pin,
        "temperature_env": os.environ.get("ZAKCODE_TEMPERATURE", "unset"),
        "elapsed_s": round(elapsed, 1),
        "num_turns": rep.get("iterations"),
        "total_cost_usd": rep.get("session_cost_usd"),
        "verify_rc": 0 if rep.get("success") else 1,
        # The runner's report carries the verifier's own tail under `verify_out`; until 2026-09-12
        # this stored the run's stop_reason there instead, so a GAP row could never name its
        # verify reason (the CC arm below always stored the real tail). stop_reason keeps its own key.
        "verify_out": str(rep.get("verify_out") or "")[:300],
        "stop_reason": str(rep.get("stop_reason")),
        # The census (intervention_coverage.py) tallies this per row; until 2026-09-13 only
        # run_task's single-run report carried it, so no arm cell could show a kind as recorded.
        "trace_interventions": rep.get("trace_interventions") or {},
        # A report has a `success` key even when the task failed; its absence means the child
        # never got as far as running the agent (crash, config refusal, import error).
        "no_report": "success" not in rep,
        "stderr_tail": stderr_tail,
        "digests": digests,
        "py_digests": {k: v for k, v in digests.items() if k.endswith(".py")},
        "sources": sources,
    }


def one_run(task_dir: Path, spec: dict, timeout_s: int = 900) -> dict:
    ws = Path(tempfile.mkdtemp(prefix=f"zbench-det-{spec['id']}-"))
    seed = task_dir / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)

    # IDENTICAL to bench/run_claude_code.py's invocation. Any divergence here would measure a
    # different code path than the arm this is being compared against (probe-with-canonical-
    # code-path.md: canonical BINARY is not canonical INVOCATION).
    cmd = [
        "claude", "-p", spec["prompt"],
        "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep",
        "--output-format", "json",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ws, capture_output=True, text=True, timeout=timeout_s)
        rc, stdout, stderr_tail = proc.returncode, proc.stdout, proc.stderr.strip()[-300:]
    except subprocess.TimeoutExpired:
        rc, stdout, stderr_tail = 124, "", f"exceeded {timeout_s}s"
    elapsed = time.perf_counter() - t0

    meta: dict = {}
    try:
        meta = json.loads(stdout) if stdout.strip().startswith("{") else {}
    except json.JSONDecodeError:
        meta = {}

    verify_rc, verify_out = None, ""
    vf = task_dir / "verify.py"
    if vf.is_file():
        try:
            vp = subprocess.run(
                [sys.executable, str(vf)], cwd=ws, capture_output=True, text=True,
                timeout=spec.get("verify_timeout_s", 120),
            )
            verify_rc, verify_out = vp.returncode, (vp.stdout + vp.stderr).strip()[-300:]
        except subprocess.TimeoutExpired:
            verify_rc, verify_out = 124, "verify timed out"

    digests = _digest_tree(ws)
    sources = _capture_sources(ws)
    shutil.rmtree(ws, ignore_errors=True)
    return {
        "cli_rc": rc,
        "elapsed_s": round(elapsed, 1),
        "num_turns": meta.get("num_turns"),
        "total_cost_usd": meta.get("total_cost_usd"),
        "verify_rc": verify_rc,
        "verify_out": verify_out,
        "stderr_tail": stderr_tail,
        "digests": digests,
        # Secondary view: the task's own source. Reported BESIDE the pre-registered primary
        # (every file), never instead of it.
        "py_digests": {k: v for k, v in digests.items() if k.endswith(".py")},
        "sources": sources,
    }


def main(argv: list[str]) -> int:
    arm = "claude-code"
    pin = "--no-pin" not in argv
    argv = [a for a in argv if a != "--no-pin"]
    if "--arm" in argv:
        i = argv.index("--arm")
        arm = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    task_dir = Path(argv[0]).resolve()
    n = int(argv[1]) if len(argv) > 1 else 3
    runner = ((lambda td, sp: one_run_zakcode(td, sp, pin=pin)) if arm == "zakcode" else one_run)
    cell = (f"pin{(os.environ.get('ZBENCH_PIN_MODE','1') if pin else 'OFF')}-temp{os.environ.get('ZAKCODE_TEMPERATURE','default')}"
            if arm == "zakcode" else "asships")
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    print(f"determinism arm: {spec['id']} x{n} through {arm}"
          + (f" (pin={'ON' if pin else 'OFF'}, ZAKCODE_TEMPERATURE={os.environ.get('ZAKCODE_TEMPERATURE','unset')})"
             if arm == "zakcode" else " (as it ships)") + "\n")

    runs = []
    for i in range(n):
        avail = _avail_mb()
        if avail < MIN_AVAIL_MB:
            print(f"ABORT before run {i+1}: only {avail}MB available (<{MIN_AVAIL_MB}MB). "
                  "Refusing to repeat the OOM that killed a 57-minute pass.")
            return 3
        print(f"  run {i+1}/{n}  (mem available {avail}MB) ...", flush=True)
        r = runner(task_dir, spec)
        runs.append(r)
        print(f"    verify_rc={r['verify_rc']} turns={r['num_turns']} "
              f"cost=${r['total_cost_usd'] or 0:.4f} {r['elapsed_s']}s "
              f"files={len(r['digests'])}", flush=True)

    print("\n" + "=" * 78)
    print(f"RESULTS  {spec['id']}  N={n}  arm={arm}"
          + ("  (determinism feature ON)" if arm == "zakcode" else "  (as it ships, no pinning available)"))
    print("=" * 78)

    passes = sum(1 for r in runs if r["verify_rc"] == 0)
    turns = [r["num_turns"] for r in runs]
    print(f"P2 outcome stability : {passes}/{n} verify_rc==0")
    print(f"P3 turn variance     : {turns}  -> spread {max(t for t in turns if t is not None) - min(t for t in turns if t is not None)}"
          if all(t is not None for t in turns) else f"P3 turn variance     : {turns}")

    # An EMPTY digest set compares equal to another empty digest set, so a runner that captured
    # nothing prints "IDENTICAL across all runs" -- the exact shape this campaign keeps finding:
    # an instrument that stopped measuring reports the same thing as a clean world. Measured here
    # 2026-09-12: the zakcode arm digested a workspace run_task.py had already deleted, scored
    # 0 files, and rendered a perfect determinism verdict. It was caught by the `files=` count
    # printed beside the verdict, not by the verdict looking wrong. Refuse instead.
    empty = [i + 1 for i, r in enumerate(runs) if not r["digests"]]
    # The same vacuity one level up (measured 2026-09-12, first cross-box run): the child
    # crashed at startup on a config check, produced NO report, and left the SEEDED task files
    # untouched -- a non-empty digest set that is identical across runs because nothing ran.
    # "IDENTICAL across all runs" printed over three 1.7s crashes. A run that produced no report
    # has not measured the agent, so it cannot contribute to an identity verdict either way.
    # (0/N verify_rc==0 with reports IS a result -- deterministic wrong output -- and stays.)
    noreport = [i + 1 for i, r in enumerate(runs) if r.get("no_report")]
    if empty or noreport:
        if empty:
            print(f"\nINSTRUMENT FAILURE: run(s) {empty} captured ZERO files. An empty digest set "
                  "compares equal to an empty digest set, so a determinism verdict here would be "
                  "vacuous. Refusing to render one. Fix the capture, then re-run.")
        if noreport:
            print(f"\nINSTRUMENT FAILURE: run(s) {noreport} produced NO report -- the child exited "
                  "before the agent ran, so the digested files are the untouched seed. Refusing to "
                  "render an identity verdict over runs that never ran. Last stderr line of each:")
            for i in noreport:
                print(f"    run {i}: {runs[i - 1].get('stderr_tail', '').strip().splitlines()[-1:] or ['(no stderr)']}")
        out = BENCH / "results" / f"determinism-{arm}-{cell}-{spec['id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"task": spec["id"], "arm": arm, "cell": cell, "n": n,
                                   "instrument_failure": (f"runs {empty} captured zero files; " if empty else "")
                                   + (f"runs {noreport} produced no report" if noreport else ""),
                                   "runs": runs}, indent=2), encoding="utf-8")
        return 4 if empty else 5

    for label, key in (("PRIMARY (every file)", "digests"), ("secondary (*.py only)", "py_digests")):
        sets = [json.dumps(r[key], sort_keys=True) for r in runs]
        identical = len(set(sets)) == 1
        print(f"\n{label}: {'IDENTICAL across all runs' if identical else f'{len(set(sets))} DISTINCT output states across {n} runs'}")
        # Name the files that actually differ -- a bare "not identical" does not say what moved.
        allnames = sorted({k for r in runs for k in r[key]})
        for name in allnames:
            vals = [r[key].get(name, "(absent)")[:12] for r in runs]
            if len(set(vals)) > 1:
                print(f"    DIFFERS  {name:28} {vals}")

    out = BENCH / "results" / f"determinism-{arm}-{cell}-{spec['id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": spec["id"], "arm": arm, "cell": cell, "n": n, "runs": runs}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    if not sys.argv[1:]:
        print("usage: determinism_arm.py [--arm claude-code|zakcode] <task_dir> [N]", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
