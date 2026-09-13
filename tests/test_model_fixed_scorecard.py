"""model_fixed_scorecard.py scores the MODEL-FIXED head-to-head (CC-loop vs zakcode-loop, both on
qwen-35B) from the two arm-json shapes -- CC's cc_on_pod_census output (stores pass + distinct) and
zakcode's determinism_arm output (stores only ``runs``). This test pins the cross-arm scoring: it
must compute pass/distinct from ``runs`` for BOTH shapes, honour the campaign's vacuity guards (an
``instrument_failure`` arm is INVALID, not a clean 0; an all-empty-digest arm is ``?empty`` (never
scoring a determinism verdict), and emits the asymmetric-honest correctness/determinism verdicts.

The .md/.py-agnostic determinism logic and the empty-arm rule are shared in spirit with
test_head_to_head_determinism.py; this module pins the MODEL-FIXED reader's own additions (the two
arm shapes, the correctness verdict, and the instrument_failure guard). ``bench/`` is outside the
runtime gates, so the module loads by path and its pure functions run -- the CLI is never invoked.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MOD = Path(__file__).resolve().parent.parent / "bench" / "model_fixed_scorecard.py"


def _load():
    spec = importlib.util.spec_from_file_location("mfs_under_test", MOD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cc_run(files: dict) -> dict:
    return {"digests": dict(files), "verify_rc": 0}


def test_score_computes_pass_and_distinct_from_runs() -> None:
    # zakcode-shape arm (only `runs`): 2 pass, 2 distinct output states across 3 runs.
    mod = _load()
    arm = {"runs": [
        {"digests": {"a.py": "h1"}, "verify_rc": 0},
        {"digests": {"a.py": "h1"}, "verify_rc": 0},
        {"digests": {"a.py": "h2"}, "verify_rc": 1},
    ]}
    s = mod._score(arm)
    assert s == {"n": 3, "pass": 2, "distinct": 2, "empty_any": False}


def test_score_recomputes_cc_shape_from_runs_not_stored_field() -> None:
    # CC-shape arm carries a stored pass/distinct, but _score derives from runs (robust to either).
    mod = _load()
    arm = {"pass": 99, "distinct_output_states": 99,
           "runs": [_cc_run({"x.py": "h"}), _cc_run({"x.py": "h"})]}
    s = mod._score(arm)
    assert s["pass"] == 2 and s["distinct"] == 1 and s["n"] == 2


def test_instrument_failure_arm_is_invalid_not_zero() -> None:
    mod = _load()
    s = mod._score({"instrument_failure": "runs [1,2] captured zero files", "runs": []})
    assert "invalid" in s
    # an INVALID arm yields a pending verdict and a "-" label -- never a spurious win/loss
    ok = {"n": 6, "pass": 6, "distinct": 1, "empty_any": False}
    assert mod._corr_verdict(s, ok) == "(pending)"
    assert mod._det_label(s) == "-"


def test_empty_and_missing_arms() -> None:
    mod = _load()
    assert mod._score(None) is None
    assert mod._score({"runs": []}) == {"invalid": "no runs"}
    empty = mod._score({"runs": [_cc_run({}), _cc_run({})]})
    assert empty["empty_any"] and empty["distinct"] == 1
    assert mod._det_label(empty) == "?empty(2)"


def test_correctness_verdicts() -> None:
    mod = _load()
    cc = {"n": 6, "pass": 2, "distinct": 6, "empty_any": False}
    zk = {"n": 6, "pass": 5, "distinct": 6, "empty_any": False}
    assert mod._corr_verdict(cc, zk) == "ZAK>CC"
    assert mod._corr_verdict(zk, cc) == "CC>ZAK"
    assert mod._corr_verdict(cc, dict(cc)) == "PARITY"
    assert mod._corr_verdict(None, zk) == "(pending)"


def test_determinism_verdicts_asymmetric_and_empty_inconclusive() -> None:
    mod = _load()
    det = {"n": 6, "pass": 6, "distinct": 1, "empty_any": False}
    nondet = {"n": 6, "pass": 6, "distinct": 4, "empty_any": False}
    empty = {"n": 6, "pass": 0, "distinct": 1, "empty_any": True}
    assert mod._det_verdict(nondet, det) == "ZAKCODE-WINS"   # cc nondet, zak det
    assert mod._det_verdict(det, nondet) == "CC-WINS"
    assert mod._det_verdict(det, det) == "TIE-DET"
    assert mod._det_verdict(nondet, nondet) == "TIE-NONDET"
    assert mod._det_verdict(empty, det) == "INCONCLUSIVE"    # an empty arm is never DET/NONDET
    assert mod._det_verdict(det, empty) == "INCONCLUSIVE"


def test_det_label_weak_flag_and_nondet() -> None:
    mod = _load()
    assert mod._det_label({"n": 2, "pass": 2, "distinct": 1, "empty_any": False}) == "DET(2)~"
    assert mod._det_label({"n": 3, "pass": 3, "distinct": 1, "empty_any": False}) == "DET(3)"
    assert mod._det_label({"n": 6, "pass": 6, "distinct": 5, "empty_any": False}) == "NONDET(5/6)"


def test_cache_files_excluded_from_signature() -> None:
    mod = _load()
    arm = {"runs": [
        {"digests": {"a.py": "h", ".pytest_cache/x": "c1"}, "verify_rc": 0},
        {"digests": {"a.py": "h", "__pycache__/y": "c2"}, "verify_rc": 0},
    ]}
    s = mod._score(arm)
    assert s["distinct"] == 1  # only a.py counts; cache noise excluded
