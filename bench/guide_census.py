#!/usr/bin/env python3
"""Fleet census for a fold change: fold every guide with two prompt.py versions and report, per
guide over the cap, the fold size and the rules-like headings (Constraints / Caveats / gotchas /
MANDATORY / Rules / Non-negotiable / Conventions) each fold DROPS, plus a one-line tally. A
deterministic regression for the guide fold (ADR-0170 … ADR-0176): run it on every real CLAUDE.md /
AGENTS.md you hold before shipping a fold change — the guides stay where they are (private guides
never need to enter the repo) and only headings and sizes are printed.

usage: guide_census.py <shipped prompt.py> <candidate prompt.py> <guide.md> ...
"""

import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RULEISH = re.compile(
    r"^#{1,6}\s+.*(constraint|caveat|gotcha|mandatory|rules?\b|non-negotiable|convention)", re.I
)


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path} (give it a .py name)")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fold_of(mod, content: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        ws.mkdir()
        (ws / "CLAUDE.md").write_text(content + "\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(ws), "init", "-q"], check=True)
        found = mod.discover_context(ws, include_readme=False)
        return [c for p, c in found if p.name == "CLAUDE.md"][0]


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    mods = {"shipped": load(argv[0], "pm_shipped"), "cand": load(argv[1], "pm_cand")}
    cap = int(getattr(mods["cand"], "MAX_CONTEXT_FILE_CHARS", 8_192))
    tot = {"over": 0, "under": 0}
    miss = {"shipped": 0, "cand": 0}
    for g in argv[2:]:
        path = Path(g)
        content = path.read_text(encoding="utf-8").strip()
        if len(content) <= cap:
            tot["under"] += 1
            continue
        tot["over"] += 1
        ruleish = [ln.lstrip("#").strip() for ln in content.splitlines() if RULEISH.match(ln)]
        h1 = sum(1 for ln in content.splitlines() if re.match(r"^#\s+\S", ln))
        res = {}
        for name, mod in mods.items():
            fold = fold_of(mod, content)
            kept = {ln.lstrip("#").strip() for ln in fold.splitlines() if ln.startswith("#")}
            dropped = [r for r in ruleish if r not in kept]
            abridged = fold.count("[abridged:")
            res[name] = (len(fold), dropped, abridged)
            if dropped:
                miss[name] += 1
        label = (path.parent.name or path.stem)[:38]
        s, c = res["shipped"], res["cand"]
        print(
            f"{len(content):>6} h1={h1:<2} {label:<38} shipped {s[0]:>5} drops {s[1]!s:<60} "
            f"| cand {c[0]:>5} drops {c[1]}" + (f" ({c[2]} abridged)" if c[2] else "")
        )
    print(
        f"\nguides: {tot['over']} over the {cap}-char cap, {tot['under']} under (pass through). "
        f"Over-cap guides whose rules-like headings the fold DROPS: "
        f"shipped {miss['shipped']}/{tot['over']}, candidate {miss['cand']}/{tot['over']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
