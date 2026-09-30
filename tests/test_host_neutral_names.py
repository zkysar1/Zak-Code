"""Zak Code names its host the way Claude Code does: generically.

A host framework drives Zak Code through the same surfaces Claude Code offers (hooks, skills,
settings, the served API). Nothing in ``src``, ``tests`` or ``docs`` should name one particular
host, its machines or its internal record ids, so this test fails when one comes back.

The words are stored as hashes, so this file does not itself spell out the names it keeps out.
A token is a run of letters, digits, ``-`` and ``_``, lower-cased; a compound token is also
checked part by part. Record ids and machine names are shapes rather than words, so they are
matched by pattern.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "tests", "docs")
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

# sha256 of each kept-out word, first 16 hex digits. The last entry is a canary that names
# nothing: the positive control below plants it to prove the scan can see.
BLOCKED = frozenset(
    {
        "ddff93cc887d1c20",
        "4f62a64634107ba8",
        "ea0d4d6415ec3f60",
        "04ff3e02e61460cb",
        "f78c81cd83165fcc",
        "bfc0e42966d7e056",
        "a977a1bf7f22b1ea",
        "656ba9a950169a74",
        "04bb89a726067d7a",
        "8092825faa15e456",
        "3f9fea909e21e7df",
        "d06387c22d0770d8",
        "5faf316d15b3ac52",
        "2402136c3cb55cb6",
        "1ac9eb18dd2f08db",
        "0b1c4cfb78cf16c6",
        "1cab2f756a3cad41",
        "3389c66bfe02bacb",
        "77c1d4492afac7f7",
        "2e1190f397549e2a",
        "7c4327279cb82108",
        "5f4870bc644c66a3",
        "e4a7832e3042c604",
        "79b50deb7e2b1fc1",
        "0c53ebdd405ce990",
        "471686599953e0e7",
        "aede461bbd8d0f77",
        "5a2c997f707919b5",
        "e809d01bbaaa5bf6",
        "76b58615bf264a5a",
        "1ac2f0a60cf6e5b4",
        "91b192e7f05bddb2",
        "4996c68e8dd7f526",
        "75e6d39dcaf99262",
        "584baf18f79ef70c",
        "8b5dbf88a3e16e2a",
        "4d443e7a57383271",
        "26e3b2a5dc54dd58",
        "cfb12585da56e4c0",
        "77bd5dd94c37c5b1",
        "c04b347d824bd044",
        "e4ff23a43add67fd",
    }
)
CANARY = "hostneutral" + "canary"
PATTERNS = {
    "a record id": re.compile(r"\b(?:g-\d{3}-\d+|guard-\d{3,}|rb-\d{3,}|asp-\d{3})\b"),
    "a machine name": re.compile(r"\b(?:zc|cc)-\d{2}\b"),
}
TOKEN = re.compile(r"[a-z0-9][a-z0-9_-]*")

# Files that still carry one host's stop and signal behaviour: the code, its documentation, and
# the tests whose planted scripts read the variables that code sets. They leave this list when
# that behaviour moves behind a generic hook; the ratchet test below fails once one no longer
# needs its place here.
EXEMPT = {
    "docs/CONFIG.md",
    "src/zakcode/session/framework_signal.py",
    "src/zakcode/session/framework_stop.py",
    "tests/test_framework_stop.py",
    "tests/test_framework_stop_seen.py",
    "tests/test_framework_stop_wiring.py",
    "tests/test_observe_perception_wake.py",
    "tests/test_run_stop_awaits_framework_stop.py",
    "tests/test_server_sidecar.py",
    "tests/test_settings_env_block_reaches_children.py",
}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


def findings(text: str) -> list[tuple[int, str]]:
    """(line number, what) for each line that names the host."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        for tok in TOKEN.findall(low):
            parts = [tok, *re.split(r"[-_]", tok)] if ("-" in tok or "_" in tok) else [tok]
            if any(_digest(p) in BLOCKED for p in parts):
                out.append((n, "a blocked word"))
                break
        out.extend((n, what) for what, rx in PATTERNS.items() if rx.search(low))
    return out


def _scanned_files() -> list[Path]:
    me = Path(__file__).resolve()
    files = []
    for top in SCANNED:
        for p in sorted((ROOT / top).rglob("*")):
            if p.is_file() and p.suffix in TEXT_SUFFIXES and p.resolve() != me:
                files.append(p)
    return files


def test_no_file_names_the_host() -> None:
    bad = []
    for p in _scanned_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        bad.extend(f"{rel}:{n}: {what}" for n, what in findings(text))
    assert not bad, "name the host generically:\n" + "\n".join(bad[:50])


def test_each_exemption_is_still_needed() -> None:
    for rel in sorted(EXEMPT):
        p = ROOT / rel
        assert p.is_file(), f"{rel} is gone; drop it from EXEMPT"
        assert findings(p.read_text(encoding="utf-8", errors="replace")), (
            f"{rel} no longer names the host; drop it from EXEMPT"
        )


def test_the_scan_sees_a_planted_name(tmp_path: Path) -> None:
    assert findings(f"a line with {CANARY} in it") == [(1, "a blocked word")]
    assert findings(f"compound {CANARY}_file too") == [(1, "a blocked word")]
    assert findings("see g-000-00 there") == [(1, "a record id")]
    assert findings("ran on zc-00") == [(1, "a machine name")]
    assert findings("a host framework, a worker machine, ADR-0236, PR #706") == []
