#!/usr/bin/env python3
"""Reasoning effort on the route the PRODUCT actually uses for the gpt-5.6 tier.

Why this exists. The effort bench (effort_decision_points.py) compared three hand-built requests:
A = /v1/chat/completions at effort none, B = /v1/responses at low, C = /v1/responses at none, and it
called A "what the product sends today". That label was never measured. The product does not build
its own HTTP request: it calls litellm, and litellm 1.86.2 moves a gpt-5.4+ chat call onto the
Responses API whenever function tools ride with a reasoning effort that is not Python ``None`` —
the string ``"none"`` included (``litellm.main.responses_api_bridge_check``). The provider's own
rule sends exactly that string with every tool call on this tier, so today's product may sit on C's
route, not A's. This bench measures it instead of assuming it:

* P-none  the product's provider class, built as a served app builds it: tools force effort none.
* P-low   the same class with a configured depth of low and the forced-none latch released: what
          the provider would send if a configured depth were allowed to ride with tools.
* A, B, C the three hand-built requests of the first bench, unchanged, as same-batch controls.

Every P call records WHERE it went and what effort rode on the wire (URL path and the body's effort
field, read inside an httpx send hook; never a header). The decision-point cells, the scorer and
the raw request builders are imported from the first bench so they cannot drift.

    python bench/effort_product_route.py --selftest            # no network
    python bench/effort_product_route.py --out batch.jsonl     # needs OPENAI_API_KEY
    python bench/effort_product_route.py --report batch.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import effort_decision_points as edp  # noqa: E402 — sibling bench script: cells, scorer, builders

RAW_ARMS = ("A-chat-none", "B-resp-low", "C-resp-none")
PRODUCT_ARMS: dict[str, str] = {"P-none": "none", "P-low": "low"}
ARMS = (*RAW_ARMS, *PRODUCT_ARMS)
CELLS = ("order", "refusal")
BRIDGE_PATH = "/v1/responses"

# ── the pre-registered reading (passes per 20; fixed before the first scored call) ───────────
GAIN = 7  # P-low - P-none >= GAIN on a cell supports a change
HARM = 4  # P-low <= P-none - HARM on any cell is HARM
FLAT = 2  # |P-low - P-none| <= FLAT reads FLAT
CEILING = 17  # P-none >= CEILING: no room for a GAIN on that cell
NEAR = 4  # |P-none - X| <= NEAR: today's product sits with control X on that cell
APART = 5  # |A - C| >= APART: the two controls are far enough apart to place P-none
MIN_SCORED = 18  # fewer scored calls than this in an arm x cell: NOT MEASURED


def effect_line(p_none: float, p_low: float) -> str:
    """What a configured depth of low does on the product's own route, on one cell."""
    diff = p_low - p_none
    if diff <= -HARM:
        return "HARM"
    if p_none >= CEILING:
        return "NO ROOM"
    if diff >= GAIN:
        return "GAIN"
    if abs(diff) <= FLAT:
        return "FLAT"
    return "BETWEEN"


def placement_line(p_none: float, a: float, c: float) -> str:
    """Which hand-built control today's product behaves like, on one cell."""
    if abs(a - c) < APART:
        return "NOT SEPARABLE"
    near_a, near_c = abs(p_none - a) <= NEAR, abs(p_none - c) <= NEAR
    if near_a and near_c:
        return "BETWEEN THE CONTROLS"
    if near_c:
        return "SITS WITH C"
    if near_a:
        return "SITS WITH A"
    return "NEITHER"


def verdict_line(effects: list[str]) -> str:
    if not effects or "NOT MEASURED" in effects:
        return "NOT MEASURED"
    if "HARM" in effects:
        return "HARM"
    if "GAIN" in effects:
        return "CHANGE-CANDIDATE"
    if all(e in ("FLAT", "NO ROOM") for e in effects):
        return "NO CHANGE"
    return "INCONCLUSIVE"


def per20(rows: list[dict[str, Any]], cell: str, arm: str) -> tuple[float | None, int, int]:
    """(passes per 20, scored calls, rows) for one arm x cell; the rate is None under MIN_SCORED."""
    mine = [r for r in rows if r.get("cell") == cell and r.get("arm") == arm]
    scored = [r for r in mine if r.get("pass") is not None]
    if len(scored) < MIN_SCORED:
        return None, len(scored), len(mine)
    return 20 * sum(1 for r in scored if r["pass"]) / len(scored), len(scored), len(mine)


def wire_line(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the product arms' calls went, and whether the effort on the wire was the arm's own."""
    out: dict[str, Any] = {}
    for arm, effort in PRODUCT_ARMS.items():
        mine = [r for r in rows if r.get("arm") == arm and r.get("pass") is not None]
        on_bridge = [r for r in mine if r.get("wire") and r["wire"][-1].get("path") == BRIDGE_PATH]
        own_effort = [r for r in on_bridge if r["wire"][-1].get("effort") == effort]
        out[arm] = {"scored": len(mine), "on_bridge": len(on_bridge), "own_effort": len(own_effort)}
    out["all_on_bridge"] = all(
        v["scored"] > 0 and v["scored"] == v["on_bridge"] == v["own_effort"]
        for v in out.values()
        if isinstance(v, dict)
    )
    return out


def read_batch(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wire = wire_line(rows)
    cells: dict[str, Any] = {}
    effects: list[str] = []
    for cell in CELLS:
        rates = {arm: per20(rows, cell, arm) for arm in ARMS}
        line: dict[str, Any] = {
            "per20": {a: (None if v[0] is None else round(v[0], 1)) for a, v in rates.items()},
            "scored": {a: v[1] for a, v in rates.items()},
        }
        p_none, p_low = rates["P-none"][0], rates["P-low"][0]
        a, c = rates["A-chat-none"][0], rates["C-resp-none"][0]
        if p_none is None or p_low is None:
            line["effect"] = "NOT MEASURED"
        else:
            line["effect"] = effect_line(p_none, p_low)
        if p_none is None or a is None or c is None:
            line["placement"] = "NOT MEASURED"
        else:
            line["placement"] = placement_line(p_none, a, c)
        effects.append(line["effect"])
        cells[cell] = line
    verdict = verdict_line(effects) if wire["all_on_bridge"] else "NOT THE BRIDGE: NOT READ"
    return {"verdict": verdict, "wire": wire, "cells": cells}


# ── the live batch ──────────────────────────────────────────────────────────────────────────
WIRE: list[dict[str, Any]] = []


def _install_wire_hook() -> None:
    """Record the URL path and the effort field of every request httpx sends. Never a header."""
    import httpx

    def note(request: Any) -> None:
        row: dict[str, Any] = {"path": request.url.path}
        try:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        if isinstance(body, dict):
            reasoning = body.get("reasoning")
            row["effort"] = (
                reasoning.get("effort") if isinstance(reasoning, dict) else None
            ) or body.get("reasoning_effort")
            row["has_instructions"] = "instructions" in body
            row["n_tools"] = len(body.get("tools") or [])
        WIRE.append(row)

    sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def send(self: Any, request: Any, **kw: Any) -> Any:
        note(request)
        return sync_send(self, request, **kw)

    async def asend(self: Any, request: Any, **kw: Any) -> Any:
        note(request)
        return await async_send(self, request, **kw)

    httpx.Client.send = send  # type: ignore[method-assign]
    httpx.AsyncClient.send = asend  # type: ignore[method-assign]


def _messages(history: edp.History) -> list[Any]:
    from zakcode.messages import Message, ToolResultBlock, ToolUseBlock

    out: list[Any] = []
    for step in history:
        if step[0] == "user":
            out.append(Message.user(step[1]))
        elif step[0] == "call":
            _, cid, name, args = step
            out.append(
                Message(role="assistant", blocks=[ToolUseBlock(id=cid, name=name, input=args)])
            )
        else:
            _, cid, output = step
            out.append(Message.tool_results([ToolResultBlock(tool_use_id=cid, output=output)]))
    return out


def _providers(model: str) -> dict[str, Any]:
    from zakcode.providers.litellm_provider import LiteLLMProvider

    today = LiteLLMProvider(model=f"openai/{model}")
    depth = LiteLLMProvider(model=f"openai/{model}", reasoning_effort="low")
    # The latch is what forces effort none whenever tools ride along. Released on THIS instance
    # only: the arm is "what the provider sends when a configured depth may ride with tools".
    depth.tools_require_effort_none = False
    return {"P-none": today, "P-low": depth}


async def _product_call(provider: Any, cell: str) -> dict[str, Any]:
    WIRE.clear()
    tools = [{"type": "function", "function": t} for t in edp.TOOLS]
    res = await provider.acomplete(_messages(edp.CELLS[cell]), system=edp.SYSTEM, tools=tools)
    raw_usage = ((res.raw or {}).get("usage") or {}) if isinstance(res.raw, dict) else {}
    details = raw_usage.get("completion_tokens_details") or {}
    resp = {
        "calls": [{"name": tc.name, "args": dict(tc.arguments)} for tc in res.tool_calls],
        "text_chars": len((res.text or "").strip()),
        "reasoning_tokens": details.get("reasoning_tokens") if isinstance(details, dict) else None,
        "tokens_in": res.usage.prompt_tokens,
        "tokens_out": res.usage.completion_tokens,
    }
    ok, label = edp.score(cell, resp)
    return {**resp, "pass": ok, "label": label, "served_model": res.usage.model, "wire": list(WIRE)}


async def run_batch(model: str, n: int, out: str, key: str) -> int:
    _install_wire_hook()
    providers = _providers(model)
    total = done = 0
    with open(out, "a", encoding="utf-8") as fh:
        # Interleave arms inside every repetition, so provider-side drift hits all arms alike.
        for rep in range(n):
            for cell in CELLS:
                for arm in ARMS:
                    t0 = time.monotonic()
                    row: dict[str, Any] = {"rep": rep, "cell": cell, "arm": arm}
                    total += 1
                    if arm in PRODUCT_ARMS:
                        try:
                            row.update(await _product_call(providers[arm], cell))
                            row["status"] = 200
                            done += 1
                        except Exception as exc:  # noqa: BLE001 — a failed call is a row, not a crash
                            row.update(
                                {
                                    "status": getattr(exc, "status_code", 0) or 0,
                                    "pass": None,
                                    "label": "provider-error",
                                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                    "wire": list(WIRE),
                                }
                            )
                    else:
                        route, effort = edp.ARMS[arm]
                        build = edp.chat_payload if route == "chat" else edp.resp_payload
                        url = edp.CHAT_URL if route == "chat" else edp.RESP_URL
                        status, body = edp.post(url, build(model, effort, edp.CELLS[cell]), key)
                        row["status"] = status
                        if status == 200 and isinstance(body, dict):
                            resp = edp.normalise(route, body)
                            ok, label = edp.score(cell, resp)
                            row.update(resp)
                            row.update(
                                {"pass": ok, "label": label, "served_model": body.get("model")}
                            )
                            done += 1
                        else:
                            row.update({"pass": None, "label": "http-error", "error": body})
                    row["seconds"] = round(time.monotonic() - t0, 2)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
            print(f"rep {rep + 1}/{n}: {done}/{total} calls scored", flush=True)
    return 0


def report(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    reading = read_batch(rows)
    print(json.dumps(reading, indent=1))
    return 0


# ── self-test: no network ───────────────────────────────────────────────────────────────────
def _rows(cell: str, arm: str, passes: int, n: int = 20, **extra: Any) -> list[dict[str, Any]]:
    return [{"cell": cell, "arm": arm, "pass": i < passes, "label": "x", **extra} for i in range(n)]


def _bridge(effort: str) -> dict[str, Any]:
    return {"wire": [{"path": BRIDGE_PATH, "effort": effort}]}


def selftest() -> int:
    bad: list[str] = []

    def check(name: str, got: Any, want: Any) -> None:
        if got != want:
            bad.append(f"{name}: got {got!r}, want {want!r}")

    # Every boundary is pinned by a LITERAL case, never by the constant it tests.
    for p_none, p_low, want in (
        (5, 12, "GAIN"),
        (5, 11, "BETWEEN"),
        (5, 8, "BETWEEN"),
        (5, 7, "FLAT"),
        (5, 3, "FLAT"),
        (5, 2, "BETWEEN"),
        (5, 1, "HARM"),
        (16, 20, "BETWEEN"),
        (13, 20, "GAIN"),
        (17, 20, "NO ROOM"),
        (16, 23, "GAIN"),
        (17, 13, "HARM"),
        (20, 17, "NO ROOM"),
        (20, 16, "HARM"),
    ):
        check(f"effect({p_none},{p_low})", effect_line(p_none, p_low), want)
    for p_none, a, c, want in (
        (7, 18, 7, "SITS WITH C"),
        (11, 18, 7, "SITS WITH C"),
        (12, 18, 7, "NEITHER"),
        (14, 18, 7, "SITS WITH A"),
        (13, 18, 7, "NEITHER"),
        (18, 18, 7, "SITS WITH A"),
        (9, 12, 7, "BETWEEN THE CONTROLS"),
        (9, 11, 7, "NOT SEPARABLE"),
        (3, 12, 7, "SITS WITH C"),
        (2, 12, 7, "NEITHER"),
        (7, 7, 18, "SITS WITH A"),
    ):
        check(f"placement({p_none},{a},{c})", placement_line(p_none, a, c), want)
    for effects, want in (
        (["GAIN", "FLAT"], "CHANGE-CANDIDATE"),
        (["GAIN", "HARM"], "HARM"),
        (["FLAT", "NO ROOM"], "NO CHANGE"),
        (["FLAT", "BETWEEN"], "INCONCLUSIVE"),
        (["GAIN", "NOT MEASURED"], "NOT MEASURED"),
        ([], "NOT MEASURED"),
    ):
        check(f"verdict({effects})", verdict_line(effects), want)

    rows: list[dict[str, Any]] = []
    for cell, (a, b, c, pn, pl) in {
        "order": (18, 19, 7, 8, 18),
        "refusal": (4, 20, 0, 1, 20),
    }.items():
        rows += _rows(cell, "A-chat-none", a) + _rows(cell, "B-resp-low", b)
        rows += _rows(cell, "C-resp-none", c)
        rows += _rows(cell, "P-none", pn, **_bridge("none")) + _rows(
            cell, "P-low", pl, **_bridge("low")
        )
    got = read_batch(rows)
    check("batch verdict", got["verdict"], "CHANGE-CANDIDATE")
    check("order effect", got["cells"]["order"]["effect"], "GAIN")
    check("order placement", got["cells"]["order"]["placement"], "SITS WITH C")
    check("refusal placement", got["cells"]["refusal"]["placement"], "NOT SEPARABLE")
    check("wire", got["wire"]["all_on_bridge"], True)

    # A product arm that did not ride the bridge, or rode it with another effort, is not read.
    off = [
        dict(r, wire=[{"path": "/v1/chat/completions", "effort": "none"}])
        if r["arm"] == "P-none"
        else r
        for r in rows
    ]
    check("off-bridge verdict", read_batch(off)["verdict"], "NOT THE BRIDGE: NOT READ")
    wrong = [
        dict(r, wire=[{"path": BRIDGE_PATH, "effort": "none"}]) if r["arm"] == "P-low" else r
        for r in rows
    ]
    check("wrong-effort verdict", read_batch(wrong)["verdict"], "NOT THE BRIDGE: NOT READ")
    # A re-issued call is read by its LAST request.
    twice = [
        dict(r, wire=[{"path": "/v1/chat/completions", "effort": "low"}, *r["wire"]])
        if r["arm"] == "P-low"
        else r
        for r in rows
    ]
    check("re-issued verdict", read_batch(twice)["verdict"], "CHANGE-CANDIDATE")

    # Coverage: 18 scored calls is read, 17 is not; an errored row is not a scored call.
    thin = [r for r in rows if not (r["arm"] == "P-low" and r["cell"] == "order")]
    thin += _rows("order", "P-low", 18, n=18, **_bridge("low"))
    check("18 scored", read_batch(thin)["cells"]["order"]["effect"], "GAIN")
    thinner = [r for r in rows if not (r["arm"] == "P-low" and r["cell"] == "order")]
    thinner += _rows("order", "P-low", 17, n=17, **_bridge("low"))
    thinner += [{"cell": "order", "arm": "P-low", "pass": None, "label": "provider-error"}] * 3
    check("17 scored", read_batch(thinner)["cells"]["order"]["effect"], "NOT MEASURED")
    check("17 scored verdict", read_batch(thinner)["verdict"], "NOT MEASURED")

    # The request builders the controls use are the first bench's own, byte for byte.
    check("cells", [c in edp.CELLS for c in CELLS], [True, True])
    check("raw arms", [a in edp.ARMS for a in RAW_ARMS], [True, True, True])

    summary = f"selftest: {14 + 11 + 6} literal cases, 12 batch cases, {len(bad)} mismatches"
    print(summary)
    for line in bad:
        print("  " + line)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="effort-product-route.jsonl")
    ap.add_argument("--report", default="", help="read a finished batch and print the reading")
    ns = ap.parse_args()
    if ns.selftest:
        return selftest()
    if ns.report:
        return report(ns.report)
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 2
    return asyncio.run(run_batch(ns.model, ns.n, ns.out, key))


if __name__ == "__main__":
    raise SystemExit(main())
