#!/usr/bin/env python
"""Diagnostic: run ONE bench task and dump the tool-call transcript.

Prints every assistant tool_use (name + command/path) and each tool_result's is_error flag
plus a short output tail, then the final stop_reason. Use it to see exactly what the model
ran — e.g. to diagnose why the recipe verify-before-finish gate fires. NOT part of the suite
or the lint/type gate (bench/ is excluded); run it manually like run_task.py.

    ./.venv/Scripts/python.exe bench/diag_task.py bench/tasks/01-wordfreq
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import json  # noqa: E402

from run_task import _build_agent  # noqa: E402 — reuse the canonical headless construction


def main(task_dir: str) -> int:
    td = Path(task_dir).resolve()
    spec = json.loads((td / "task.json").read_text(encoding="utf-8"))
    ws = Path(tempfile.mkdtemp(prefix=f"zdiag-{spec['id']}-"))
    seed = td / "workspace"
    if seed.is_dir():
        shutil.copytree(seed, ws, dirs_exist_ok=True)
    agent = _build_agent(ws, spec)
    result = agent.run_turn(spec["prompt"])

    print(f"=== {spec['id']} ===")
    print(
        f"STOP_REASON={result.stop_reason} degraded={result.degraded} "
        f"iters={result.iterations} routed={result.routed_category}"
    )
    print("--- transcript (assistant text / tool_use / tool_result) ---")
    for msg in agent.session.messages:
        for block in msg.blocks:
            kind = getattr(block, "type", "")
            if kind == "tool_use":
                detail = block.input.get("command") or block.input.get("path") or ""
                print(f"  CALL {block.name}: {str(detail)[:150]}")
            elif kind == "tool_result":
                tail = (block.output or "").strip().replace("\n", " ")[-110:]
                print(f"    -> is_error={block.is_error}  {tail}")
            elif kind == "text" and msg.role == "assistant":
                # Surface the model's prose too — a text-mode model that "answers" with no tool
                # call (and no work) shows up here, which is the 04/llama-text stall shape.
                txt = (block.text or "").strip().replace("\n", " ")
                if txt:
                    print(f"  TEXT: {txt[:240]}")
    shutil.rmtree(ws, ignore_errors=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: diag_task.py <task_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
