#!/usr/bin/env python3
"""Build the 15c-realguide-mind-timestamp-core workspace — thrust 23, a COMPOSITION cell for the ADR-0172 cell-15 cost.

CORE — the ten sections the ADR-0171 and ADR-0172 folds of cell 15 SHARE (Mode System, Universal Conventions, Naming Rules, Priority Values, Status Values, Pipeline Rules, Skill Invocation Rules, Code Change Verification (MANDATORY), Enforcement Rules, Autonomous Loop Rules) plus the preamble: 5,307 chars, md5 559afae30623 — NEITHER the six orientation sections ADR-0171 carried NOR the seven convention sections ADR-0172 added. Under the 8,192 cap: the shipped fold passes it through and the model sees exactly these bytes.

The guide is a COMPOSED fold of the real 47K CLAUDE.md (cell 15's guide-CLAUDE.md): whole sections in document
order, preamble and omission note in the fold's own output format, built by the campaign's compose15.py from the
same fence-aware section split `_fit_sections` uses — the ADR-0171 and ADR-0172 folds rebuilt that way are
byte-identical to the live folds (md5 ef3a7a2063d4 / 9b02c28eb12c), which validates the composition. Everything
else — the stub, the modules, the tests, verify.py, the prompt — is cell 15's, imported from the sibling dir.
"""
import hashlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "15-realguide-mind-timestamp"))
import build_workspace as base  # noqa: E402  — cell 15's builder: stub, modules, tests, rule markers

GUIDE_ASSET = _HERE / "guide-CLAUDE.md"
FOLD_CAP = base.FOLD_CAP  # 8_192 == zakcode prompt.py MAX_CONTEXT_FILE_CHARS
GUIDE_OVER_CAP = False  # False: under the cap, passed through unfolded; True: needs MAX_CONTEXT_FILE_CHARS raised to pass unfolded


def _selfcheck(root: Path, guide_file: str) -> None:
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    n = len(files)
    assert n <= 40, f"{n} files exceeds the small-repo budget (<=40)"
    stub = (root / "app/journal.py").read_text(encoding="utf-8")
    assert "def journal_stamp(" in stub and "NotImplementedError" in stub, "stub missing/short"
    carriers = sorted(str(p.relative_to(root)) for p in files if base.RULE_MARK in p.read_text(encoding="utf-8", errors="ignore"))
    assert carriers == [guide_file], f"format string must be in exactly {guide_file}, found in {carriers}"
    content = (root / guide_file).read_text(encoding="utf-8")
    assert content == GUIDE_ASSET.read_text(encoding="utf-8") and content == content.strip(), "installed guide must be the composed asset, byte for byte"
    off = content.find(base.RULE_HEADER)
    assert off > 0 and base.RULE_PHRASE in content[off:off + 800] and base.RULE_MARK in content[off:off + 800], "rule not in its section"
    assert content.startswith("# CLAUDE.md") and "\n[... CLAUDE.md: 47511 characters, " in content, "composed fold format (preamble + note)"
    assert (len(content) > FOLD_CAP) == GUIDE_OVER_CAP, f"guide {len(content)} chars vs cap {FOLD_CAP}: over={GUIDE_OVER_CAP} expected"
    md5 = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
    headings = sum(1 for ln in content.splitlines() if ln.startswith("#"))
    where = "OVER the cap (the arm raises MAX_CONTEXT_FILE_CHARS to pass it unfolded)" if GUIDE_OVER_CAP else "under the cap (passed through unfolded)"
    print(f"selfcheck OK: {n} files (<=40); composed guide {len(content)} chars md5 {md5}, {headings} headings, rule header at char {off}; {where}; format string only in the guide")


def main(target=_HERE / "workspace", guide_file=base.DEFAULT_GUIDE) -> None:
    base.GUIDE_ASSET = GUIDE_ASSET
    base._selfcheck = _selfcheck
    base.main(target, guide_file)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args) if args else main()
