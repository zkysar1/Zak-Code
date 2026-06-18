#!/usr/bin/env python
"""Diagnostic probe: capture the RAW Groq ``tool_use_failed`` body (``failed_generation``)
that the deep_code model (gpt-oss-120b) emits, to root-cause the malformed tool call.

Calls ``litellm.acompletion`` DIRECTLY (bypassing ``_map_error``, which scrubs the body)
with the agent's REAL tool schema + a file-creation prompt that reliably triggers a tool
call. Runs N attempts (the failure is semi-intermittent) and prints each failed_generation.

Usage: ./.venv/Scripts/python.exe bench/probe_tool_use_failed.py [N]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path


def build_agent():
    from zakcode import Agent
    from zakcode.config import load_settings

    base = load_settings()
    ws = Path(tempfile.mkdtemp(prefix="zprobe-"))
    settings = base.model_copy(
        update={
            "workspace_root": str(ws),
            "default_model": "zakpick",
            "permission_mode": "autonomous",
            "api_base": None,
        }
    )
    return Agent(
        settings=settings,
        enable_compaction=False,
        enable_memory=False,
        enable_rules=False,
        enable_skills=False,
        enable_subagents=False,
        enable_mcp=False,
        enable_plugins=False,
    )


def extract_failed_generation(text: str) -> str | None:
    m = re.search(r'"failed_generation"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not m:
        return None
    try:
        return json.loads('"' + m.group(1) + '"')
    except Exception:  # noqa: BLE001
        return m.group(1)


async def main(n: int) -> int:
    from zakcode.messages import Message

    agent = build_agent()
    provider = agent._main_provider_for("deep_code")
    tools = agent.registry.definitions()
    names = [t.get("function", {}).get("name") for t in tools]
    print(f"deep provider: {type(provider).__name__} mode={getattr(provider, 'mode', '?')}")
    print(f"inner model: {provider.model_id()}  | #tools: {len(tools)}")
    print(f"tool names: {names}\n")
    system = (
        "You are a coding agent operating headlessly in a workspace. Use the available "
        "tools to accomplish the task. When asked to create a file, call the file-writing "
        "tool with the full file contents."
    )
    messages = [
        Message.user(
            "Create a file named lru.py in the workspace. It must define a class LRUCache "
            "with __init__(self, capacity), get(self, key) returning -1 if absent, and "
            "put(self, key, value) that evicts the least-recently-used entry when over "
            "capacity. Write the file now using your file-creation tool."
        )
    ]

    rejects = 0
    no_tool = 0
    for i in range(n):
        try:
            # acomplete on the RESOLVED provider: native mode raises ModelOutputRejected on a
            # malformed call; text mode returns text the wrapper tried to parse into tool_calls.
            result = await provider.acomplete(messages, system=system, tools=tools)
            tcs = result.tool_calls or []
            txt = result.text or ""
            print(f"[{i}] OK  parsed_tool_calls={len(tcs)}  text_len={len(txt)}")
            if tcs:
                for c in tcs:
                    print(f"     -> {c.name}({json.dumps(c.arguments)[:120]})")
            else:
                no_tool += 1
                print(f"     NO TOOL CALL. raw model text [:1800]:\n     " + txt[:1800].replace("\n", "\n     "))
        except Exception as e:  # noqa: BLE001
            raw = str(e)
            if "tool_use_failed" in raw:
                rejects += 1
            fg = extract_failed_generation(raw)
            print(f"[{i}] RAISED {type(e).__name__}  tool_use_failed={'tool_use_failed' in raw}")
            if fg is not None:
                print(f"     failed_generation ({len(fg)} chars):\n     " + fg[:1800].replace("\n", "\n     "))
            else:
                print("     raw[:900]:\n     " + raw[:900].replace("\n", "\n     "))
    print(f"\n{rejects}/{n} raised tool_use_failed | {no_tool}/{n} returned NO tool call")
    return 0


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sys.exit(asyncio.run(main(count)))
