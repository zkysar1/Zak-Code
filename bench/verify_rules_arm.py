#!/usr/bin/env python
"""Two-way positive control for the bench's lean_rules arm (g-016-91).

Run this BEFORE trusting any ZAKCODE_LEAN_RULES A/B. It exercises the real
``run_task._build_agent`` in two configurations and asserts that the instrument has
discriminating power in BOTH directions:

    must-DIFFER   ZBENCH_RULES_ROOT=<rules-heavy mind>  ->  the two lean arms differ
    must-MATCH    ZBENCH_RULES_ROOT unset (the default) ->  the two lean arms are identical

Why both directions are required. A bench whose arms are identical returns "no
difference"; the lean_rules decision rule is "flip the default if NO regression"; so a
DEAD instrument emits exactly the null result that authorizes a production change. The
must-DIFFER arm proves the instrument is alive. The must-MATCH arm proves it is measuring
*the variable* rather than incidental run-to-run noise — and that the default bench path
is unchanged.

Measured at 8e59d49 before the arm existed: every one of the 7 bench runners hardcoded
``enable_rules=False`` and the temp workspace discovered 0 rules, so both arms rendered
the empty string (``full=0 chars, index=0 chars``, sha1-identical).

Usage (from the Zak-Code repo root, with the repo venv):
    ./.venv/bin/python bench/verify_rules_arm.py                    # default rules root
    ./.venv/bin/python bench/verify_rules_arm.py --rules-root /path/to/mind

Exit 0 when both directions hold; exit 1 (loud, with the observed values) otherwise.
No LLM call is made — construction only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_task import _build_agent  # noqa: E402

SPEC = {"id": "rules-arm-control", "max_iterations": 1, "max_cost_usd": 0.0}


def rules_text_for(lean: bool, rules_root: str | None) -> tuple[str, int]:
    """Build an agent exactly as the bench does; return (rules block, rules discovered)."""
    os.environ["ZAKCODE_LEAN_RULES"] = "true" if lean else "false"
    if rules_root:
        os.environ["ZBENCH_RULES_ROOT"] = rules_root
    else:
        os.environ.pop("ZBENCH_RULES_ROOT", None)
    workspace = Path(tempfile.mkdtemp(prefix="zbench-control-"))
    agent = _build_agent(workspace, SPEC)
    builder = getattr(getattr(agent, "loop", None), "prompt_builder", None)
    text = getattr(builder, "rules", None) if builder is not None else None
    registry = getattr(agent, "rule_registry", None)
    return (text or ""), (len(registry) if registry is not None else 0)


def report(label: str, rules_root: str | None, want_differ: bool) -> bool:
    full, n_full = rules_text_for(False, rules_root)
    lean, n_lean = rules_text_for(True, rules_root)
    differ = full != lean
    ok = differ is want_differ
    print(f"{label}")
    print(f"   rules discovered      : {n_full} (lean arm: {n_lean})")
    print(f"   lean=False  chars={len(full):>6}  sha1={hashlib.sha1(full.encode()).hexdigest()[:10]}")
    print(f"   lean=True   chars={len(lean):>6}  sha1={hashlib.sha1(lean.encode()).hexdigest()[:10]}")
    print(f"   expected              : {'DIFFER' if want_differ else 'IDENTICAL'}")
    print(f"   observed              : {'DIFFER' if differ else 'IDENTICAL'}")
    if differ and len(lean):
        print(f"   index is {len(full) / len(lean):.2f}x smaller")
    print(f"   => {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rules-root",
        default=os.environ.get("ZBENCH_CONTROL_RULES_ROOT", "/opt/zds-mind"),
        help="A rules-heavy mind (must contain .claude/rules/*.md).",
    )
    args = ap.parse_args()

    src = Path(args.rules_root) / ".claude" / "rules"
    if not src.is_dir() or not any(src.glob("*.md")):
        print(f"FAIL: --rules-root {args.rules_root!r} has no .claude/rules/*.md to seed", file=sys.stderr)
        return 1

    print(f"two-way positive control for the bench lean_rules arm\nrules root: {args.rules_root}\n")
    a = report("A  must-DIFFER  (ZBENCH_RULES_ROOT set — the live arm)", args.rules_root, want_differ=True)
    b = report("B  must-MATCH   (ZBENCH_RULES_ROOT unset — bench default)", None, want_differ=False)

    if a and b:
        print("CONTROL PASSED — the instrument discriminates in both directions.")
        print("A lean_rules A/B run with ZBENCH_RULES_ROOT set is now meaningful.")
        return 0
    print("CONTROL FAILED — do NOT trust a lean_rules A/B until this passes.", file=sys.stderr)
    if not a:
        print("  A failed: the arms did not differ where they must — the instrument is DEAD.", file=sys.stderr)
    if not b:
        print("  B failed: the arms differed with no rules seeded — the probe is measuring noise.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
