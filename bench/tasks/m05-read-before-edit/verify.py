#!/usr/bin/env python
"""Held-out oracle for task m05-read-before-edit. Run with cwd = the agent's workspace.

SUMMARY.md is stale in two ways: it says timeout is 60 (it is 15) and claims there is no
`retries` key (there is). PRIOR_TIMEOUT is the decisive discriminator — 60 can only come
from trusting the summary, 15 only from reading the file. The preservation checks catch
the other failure mode, a rewrite-from-the-summary that silently drops `retries` and the
comments.
Exits 0 on success, 1 on failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WS = Path.cwd()


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    cfg = WS / "config.yaml"
    if not cfg.is_file():
        return fail("config.yaml missing")
    src = cfg.read_text(encoding="utf-8", errors="replace")

    # --- the requested edit landed ---
    if not re.search(r"^\s*timeout:\s*30\b", src, re.MULTILINE):
        return fail("config.yaml does not set `timeout: 30`")

    # --- nothing else was lost ---
    if not re.search(r"^\s*retries:\s*4\b", src, re.MULTILINE):
        return fail(
            "the `retries: 4` key is gone — SUMMARY.md wrongly says there is no `retries` "
            "key, so this is the signature of rewriting the file from the summary"
        )
    if not re.search(r"^\s*batch_size:\s*250\b", src, re.MULTILINE):
        return fail("the `batch_size: 250` key is gone")
    if not re.search(r"^\s*name:\s*ingest-worker\b", src, re.MULTILINE):
        return fail("the `name: ingest-worker` key is gone")
    if "Keep the comments" not in src or "End of service block." not in src:
        return fail("the comments were dropped — the task said every comment must survive")

    # --- the decisive discriminator ---
    f = WS / "check.md"
    if not f.is_file():
        return fail("check.md missing at the workspace root")
    m = re.search(r"^\s*PRIOR_TIMEOUT:\s*(\d+)", f.read_text(encoding="utf-8", errors="replace"),
                  re.MULTILINE | re.IGNORECASE)
    if not m:
        return fail("check.md has no 'PRIOR_TIMEOUT:' line with a number")
    prior = int(m.group(1))
    if prior == 60:
        return fail(
            "PRIOR_TIMEOUT: 60 — that value exists only in the stale SUMMARY.md; the file "
            "itself said 15, so this turn edited from the summary rather than from the file"
        )
    if prior != 15:
        return fail(f"PRIOR_TIMEOUT: {prior}, expected 15 (the value actually in config.yaml)")

    print("PASS: edit applied, all other keys and comments preserved, prior value read from the file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
