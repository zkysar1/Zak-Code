#!/usr/bin/env python3
"""How often does the pod hand the loop undecodable tool arguments? (review lever L7, measure first)

ADR-0081's step 0b bounces a call whose argument string did not decode (``{"_raw": ...}``) with a
``Fix:`` message naming the cause: cut off by the output limit, or an unescaped character. The
bounce is model-free to repair instead of bounce -- but only worth building if it fires. This
counts it over the provider request dumps (``ZBENCH_DUMP_REQUESTS``: ``<root>/<cell>/run-N/
wire-NNNN.json``), which carry every tool result the model saw.

Per run the LARGEST wire dump is the conversation: messages accumulate, so the main
conversation's last request is the biggest file, while the turn-end side requests (structured
output, system + user only) are a few KB and are often the LAST file. Taking the last file
undercounts 14% of runs as "no tool calls at all" (measured 2026-09-13, 18 of 127 runs).

Prints counts only, never prompt content, and prints its own positive control beside the
result: tool results carrying ``[exit code:`` (bash.py appends it to every shell result), so a
zero bounce count is self-refuting if the parser is not seeing tool text.

    python3 undecodable_bounces.py /root/zb-dumps
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import sys

BOUNCE = "were not valid JSON, so the call was not executed"
CUT = "cut off by the output limit"
TOOL_RE = re.compile(r"Fix: the arguments for '([a-z_]+)' were not valid JSON")
CHARS_RE = re.compile(r"\((\d+) characters;")


def conversation_dump(run_dir: str) -> str | None:
    wires = glob.glob(os.path.join(run_dir, "wire-*.json"))
    return max(wires, key=os.path.getsize) if wires else None


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return ""


def main(root: str) -> int:
    totals: collections.Counter = collections.Counter()
    by_tool: collections.Counter = collections.Counter()
    sizes: list[int] = []
    rows = []
    for cell in sorted(os.listdir(root)):
        cell_dir = os.path.join(root, cell)
        if not os.path.isdir(cell_dir):
            continue
        counts: collections.Counter = collections.Counter()
        for run_dir in sorted(glob.glob(os.path.join(cell_dir, "run-*"))):
            dump = conversation_dump(run_dir)
            if dump is None:
                continue
            with open(dump, encoding="utf-8") as fh:
                request = json.load(fh)
            counts["runs"] += 1
            hit = False
            for message in request.get("messages") or []:
                if message.get("role") == "assistant":
                    counts["calls"] += len(message.get("tool_calls") or [])
                if message.get("role") != "tool":
                    continue
                text = text_of(message)
                counts["results"] += 1
                counts["exit_code_results"] += "[exit code:" in text
                if BOUNCE not in text:
                    continue
                hit = True
                match = TOOL_RE.search(text)
                by_tool[match.group(1) if match else "?"] += 1
                chars = CHARS_RE.search(text)
                if chars:
                    sizes.append(int(chars.group(1)))
                counts["cut_off" if CUT in text else "unescaped"] += 1
            counts["runs_with_bounce"] += hit
        rows.append((cell, counts))
        totals.update(counts)

    print(f"{'cell':36} {'runs':>4} {'calls':>6} {'results':>7} {'exit':>5} {'cut':>4} {'esc':>4}")
    for cell, c in rows:
        flag = "  <-- bounced" if c["cut_off"] + c["unescaped"] else ""
        print(
            f"{cell:36} {c['runs']:>4} {c['calls']:>6} {c['results']:>7} "
            f"{c['exit_code_results']:>5} {c['cut_off']:>4} {c['unescaped']:>4}{flag}"
        )
    print(
        f"\nruns {totals['runs']}  tool calls {totals['calls']}  tool results {totals['results']}"
        f"  runs with a bounce {totals['runs_with_bounce']}"
    )
    print(
        f"bounces: cut off {totals['cut_off']}  unescaped {totals['unescaped']}"
        f"  by tool {dict(by_tool)}"
    )
    if sizes:
        sizes.sort()
        print(
            f"bounced argument sizes: n={len(sizes)} min={sizes[0]}"
            f" median={sizes[len(sizes) // 2]} max={sizes[-1]}"
        )
    verdict = "parser sees tool text" if totals["exit_code_results"] else "PARSER BLIND"
    print(
        f"positive control: {totals['exit_code_results']} tool results carry '[exit code:'"
        f" ({verdict})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/root/zb-dumps"))
