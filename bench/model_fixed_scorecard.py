#!/usr/bin/env python3
"""MODEL-FIXED head-to-head scorecard (CC-loop vs zakcode-loop, MODEL HELD FIXED at qwen-35B).

Unlike head_to_head_determinism.py -- which compares Claude Code AS-SHIPS (a Claude model, Fable)
against zakcode (qwen) and is therefore MODEL-CONFOUNDED -- this reader consumes the model-fixed
cell: BOTH arms run on zds-qwen3.6-35b via the pod's native Anthropic endpoint, BOTH with a fresh
mkdtemp workspace per run (no pin), N each. Only the LOOP varies. So a difference here is a claim
about the two AGENT LOOPS, not the two models.

Arm files per task T (in the results dir passed as argv[1]):
  CC-on-qwen   : cc-qwen-<T>.json                       (cc_on_pod_census.py; stores pass + distinct)
  zakcode-qwen : determinism-zakcode-pinOFF-temp0-<T>.json (determinism_arm.py --no-pin --arm zakcode
                 ZAKCODE_TEMPERATURE=0; stores only runs -> compute here)

Two axes, reported side by side:
  (1) CORRECTNESS  -- pass/N (verify_rc==0). The saturated axis on the m0x/06 parity suite; the
      question broadening asks is whether parity HOLDS on the greenfield coding tasks.
  (2) DETERMINISM  -- distinct output-tree states across N runs (1 => byte-deterministic). Asymmetric-
      honest per head_to_head_determinism.py: NONDET proven by a single counterexample; DET is only
      "not yet observed to vary" and flagged ~ at N<3; an empty-output arm is ?empty, never DET.

VACUITY GUARDS (this campaign's recurring lesson: a stopped instrument reads like a clean world):
  - zakcode arm with an "instrument_failure" key (rc 4/5: empty digests / no report) is INVALID -> skip.
  - any arm whose runs ALL captured zero agent-visible files is ?empty and awarded no determinism verdict.
Both are printed explicitly, never silently dropped.

Usage: ./model_fixed_scorecard.py <results_dir> [task ...]
"""
import json, os, sys
from collections import defaultdict

CACHE = (".pytest_cache", "__pycache__")

def _load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None

def _sig(run):
    dig = run.get("digests") or run.get("py_digests") or {}
    return tuple(sorted((f, h) for f, h in dig.items() if not any(c in f for c in CACHE)))

def _score(arm):
    """Return dict from an arm json's runs, or None. Marks invalid/empty explicitly."""
    if arm is None:
        return None
    if arm.get("instrument_failure"):
        return {"invalid": arm["instrument_failure"]}
    runs = arm.get("runs") or []
    if not runs:
        return {"invalid": "no runs"}
    npass = sum(1 for r in runs if r.get("verify_rc") == 0)
    sigs = [_sig(r) for r in runs]
    empty_any = any(len(s) == 0 for s in sigs)
    return {"n": len(runs), "pass": npass, "distinct": len(set(sigs)), "empty_any": empty_any}

def _det_label(s):
    if s is None or "invalid" in (s or {}):
        return "-"
    if s["empty_any"] and s["distinct"] == 1:
        return f"?empty({s['n']})"
    if s["distinct"] == 1:
        return f"DET({s['n']})" + ("~" if s["n"] < 3 else "")
    return f"NONDET({s['distinct']}/{s['n']})"

def _det_verdict(cc, zk):
    if not cc or not zk or "invalid" in cc or "invalid" in zk:
        return "(pending)"
    cc_det = cc["distinct"] == 1 and not cc["empty_any"]
    zk_det = zk["distinct"] == 1 and not zk["empty_any"]
    cc_nd, zk_nd = cc["distinct"] > 1, zk["distinct"] > 1
    if zk_det and cc_nd:  return "ZAKCODE-WINS"
    if cc_det and zk_nd:  return "CC-WINS"
    if zk_det and cc_det: return "TIE-DET"
    if zk_nd and cc_nd:   return "TIE-NONDET"
    return "INCONCLUSIVE"

def _corr_verdict(cc, zk):
    if not cc or not zk or "invalid" in cc or "invalid" in zk:
        return "(pending)"
    if zk["pass"] > cc["pass"]:  return "ZAK>CC"
    if cc["pass"] > zk["pass"]:  return "CC>ZAK"
    return "PARITY"

def main(d, tasks):
    files = os.listdir(d) if os.path.isdir(d) else []
    if not tasks:
        # discover every task that has a cc-qwen arm
        tasks = sorted(f[len("cc-qwen-"):-len(".json")] for f in files
                       if f.startswith("cc-qwen-") and f.endswith(".json") and "." not in f[len("cc-qwen-"):-len(".json")])
    print("MODEL-FIXED HEAD-TO-HEAD  (CC-loop vs zakcode-loop, both on zds-qwen3.6-35b, fresh-workspace/run)")
    print("correctness = pass/N (verify_rc==0);  determinism = distinct output states/N (1=byte-det)")
    print(f"{'task':26} {'CC pass':>8} {'ZAK pass':>9} {'corr':>8} | {'CC det':>12} {'ZAK det':>12} {'det verdict':>14}")
    print("-" * 100)
    ct = defaultdict(int); dt = defaultdict(int); invalids = []
    for t in tasks:
        cc = _score(_load(os.path.join(d, f"cc-qwen-{t}.json")))
        zk = _score(_load(os.path.join(d, f"determinism-zakcode-pinOFF-temp0-{t}.json")))
        for name, s in (("cc-qwen", cc), ("zak", zk)):
            if s and "invalid" in s:
                invalids.append(f"{t} [{name}]: {s['invalid']}")
        cc_p = f"{cc['pass']}/{cc['n']}" if cc and "invalid" not in cc else ("INVALID" if cc else "-")
        zk_p = f"{zk['pass']}/{zk['n']}" if zk and "invalid" not in zk else ("INVALID" if zk else "-")
        cv, dv = _corr_verdict(cc, zk), _det_verdict(cc, zk)
        if cv != "(pending)": ct[cv] += 1
        if dv != "(pending)": dt[dv] += 1
        print(f"{t:26} {cc_p:>8} {zk_p:>9} {cv:>8} | {_det_label(cc):>12} {_det_label(zk):>12} {dv:>14}")
    print("-" * 100)
    print("correctness tally:", ", ".join(f"{k}={v}" for k, v in sorted(ct.items())) or "(none)")
    print("determinism tally:", ", ".join(f"{k}={v}" for k, v in sorted(dt.items())) or "(none)")
    if invalids:
        print("\nINVALID/vacuous arms (excluded from tallies):")
        for i in invalids: print("  ", i)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bench/results", sys.argv[2:])
