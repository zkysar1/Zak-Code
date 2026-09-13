#!/usr/bin/env python3
"""How long the model spends in the hole after a harness verify, per run, from the wire dumps.

Usage: harness_verify_holes.py <cell-dir>...   (each dir holds run-*/wire-NNNN.json)

For every run: the call at which the harness's "[harness] I ran the file to verify it:" message
first appears, the KIND of its body (RUNPY-WARNING / bare-exit / other), the number of calls the
model made after it, and the model's first tool call in response. Built for ADR-0166 (arm H): the
injection is universal on 06-plugin-conventions, so the calls AFTER it are the quantity a fix moves.
"""

from __future__ import annotations

import glob
import json
import sys

HARNESS = "[harness] I ran the file to verify it:"


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("messages") or []


def _first_tool_call(msg: dict) -> str | None:
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (TypeError, ValueError):
            args = {}
        cmd = args.get("command") or args.get("path") or args.get("file_path") or ""
        return f"{fn.get('name')}({str(cmd)[:70]})"
    return None


def _kind(body: str) -> str:
    if "RuntimeWarning" in body:
        return "RUNPY-WARNING"
    stripped = body.strip()
    if stripped.startswith("[exit code:") and "\n" not in stripped:
        return "bare-exit"
    return "other"


def report(cell: str) -> None:
    for rd in sorted(glob.glob(cell.rstrip("/") + "/run-*")):
        wires = sorted(glob.glob(rd + "/wire-*.json"))
        n = len(wires)
        tag = "/".join(rd.rsplit("/", 2)[-2:])
        hit: tuple[int, int] | None = None
        body = ""
        for i, wf in enumerate(wires, 1):
            for j, m in enumerate(_load(wf)):
                c = m.get("content") if isinstance(m.get("content"), str) else ""
                if m.get("role") == "user" and c.startswith(HARNESS):
                    hit = (i, j)
                    body = c.split("\n", 1)[1] if "\n" in c else ""
                    break
            if hit:
                break
        if hit is None:
            print(f"{tag}: calls {n}, no harness verify message")
            continue
        i, j = hit
        nxt = None
        if i < n:  # the response to the injection is the next assistant message (call i+1)
            for m in _load(wires[i])[j + 1 :]:
                if m.get("role") == "assistant":
                    text = (m.get("content") or "")[:60].replace("\n", " ")
                    nxt = _first_tool_call(m) or f"text:{text}"
                    break
        print(
            f"{tag}: calls {n}, harness verify at call {i} [{_kind(body)}], "
            f"calls after it {n - i}, first response {nxt}"
        )


if __name__ == "__main__":
    for cell in sys.argv[1:]:
        report(cell)
