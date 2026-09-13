"""head_to_head_determinism.py must measure byte-determinism over ALL agent-visible files, not just
``.py``, and its cross-agent verdict must be asymmetric-honest.

The ``.md`` regression is real (2026-09-13): an earlier hand analyzer scored determinism over
``*.py`` only, so a task whose deliverable is ``report.md`` / ``count.md`` / ``finding.md`` read
BYTE-DETERMINISTIC while the agent's actual output varied every run -- and Claude Code varies
exactly there (m01/m02/m04 in the H2H suite). ``determinism()`` counts every agent-visible file;
this test pins that, plus the asymmetric verdict (non-determinism is proven by a counterexample; an
empty-output arm never counts as deterministic).

``bench/`` is outside the runtime gates (testpaths = ["tests"]), so this test loads the module by
path and exercises its pure functions -- it never runs the CLI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

MOD = Path(__file__).resolve().parent.parent / "bench" / "head_to_head_determinism.py"


def _load():
    spec = importlib.util.spec_from_file_location("h2h_det_under_test", MOD)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run(files: dict) -> dict:
    return {"digests": dict(files), "verify_rc": 0}


def test_identical_runs_are_deterministic() -> None:
    mod = _load()
    r = _run({"a.py": "h1", "b.md": "h2"})
    d = mod.determinism([dict(r), dict(r), dict(r)])
    assert d["distinct"] == 1 and d["n"] == 3 and d["varying"] == [] and not d["empty_any"]


def test_a_varying_md_is_caught_not_just_py() -> None:
    # The regression: the deliverable is a .md and it varies -> NON-DET. .md must not be exempt.
    mod = _load()
    d = mod.determinism([_run({"report.md": "x1"}), _run({"report.md": "x2"})])
    assert d["distinct"] == 2 and d["varying"] == ["report.md"]


def test_a_missing_file_in_one_run_is_nondeterminism() -> None:
    mod = _load()
    d = mod.determinism([_run({"a.py": "h", "extra.py": "e"}), _run({"a.py": "h"})])
    assert d["distinct"] == 2 and "extra.py" in d["varying"]


def test_cache_files_are_ignored() -> None:
    mod = _load()
    d = mod.determinism(
        [_run({"a.py": "h", ".pytest_cache/x": "c1"}), _run({"a.py": "h", ".pytest_cache/x": "c2"})]
    )
    assert d["distinct"] == 1 and d["varying"] == []


def test_empty_output_never_counts_as_deterministic() -> None:
    mod = _load()
    empty = mod.determinism([_run({}), _run({})])
    assert empty["empty_any"] and empty["distinct"] == 1
    nondet = {"distinct": 2, "n": 2, "varying": ["x"], "empty_any": False}
    # an empty arm is neither DET nor NONDET, so no verdict is awarded either way
    assert mod.verdict(empty, nondet) == "INCONCLUSIVE"
    assert mod.verdict(nondet, empty) == "INCONCLUSIVE"


def test_verdict_matrix() -> None:
    mod = _load()
    det = {"distinct": 1, "n": 3, "varying": [], "empty_any": False}
    nondet = {"distinct": 2, "n": 2, "varying": ["x"], "empty_any": False}
    assert mod.verdict(nondet, det) == "ZAKCODE-WINS"  # cc nondet, zak det
    assert mod.verdict(det, nondet) == "CC-WINS"
    assert mod.verdict(det, det) == "TIE-DET"
    assert mod.verdict(nondet, nondet) == "TIE-NONDET"
    assert mod.verdict(None, det) == "(pending)"


def test_weak_det_flag_at_small_n() -> None:
    mod = _load()
    assert mod._label({"distinct": 1, "n": 2, "varying": [], "empty_any": False}) == "DET(2)~"
    assert mod._label({"distinct": 1, "n": 3, "varying": [], "empty_any": False}) == "DET(3)"
    assert (
        mod._label({"distinct": 3, "n": 6, "varying": ["x"], "empty_any": False}) == "NONDET(3/6)"
    )
