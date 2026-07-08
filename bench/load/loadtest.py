"""loadtest.py — zds-inference-server load-sweep harness (M2 of the local-inference plan).

Measures how a pod's llama-server scales with concurrent agent-shaped load, to feed the
agents-per-GPU capacity model:

    agents_per_gpu = aggregate_tok_s(N) / (duty_cycle * per_stream_tok_s_demand)

Design notes (verified against llama.cpp c198af4 on zakpod1, 2026-07-08):
- The engine runs --parallel N --kv-unified: slots share ONE CTX-token pool; a single
  request may use up to full CTX; concurrent big-prompt streams queue on pool exhaustion.
- llama-server assigns slots by longest-common-prefix similarity, so giving each stream a
  DISTINCT nonce prefix simulates N distinct agents (each stream "owns" a slot's prompt
  cache), while rounds within a stream share their prefix like a real multi-turn agent
  (round 1 = cold prompt eval, round 2+ = prefix-cache hit).
- The :9090 proxy passes llama-server's `timings` through: prompt_ms / prompt_per_second /
  predicted_ms / predicted_per_second per request. Queue wait is estimated as
  wall_ms - (prompt_ms + predicted_ms).

Usage (from the Zak-Code venv):
    python bench/load/loadtest.py --base-url http://10.0.0.204:9090/v1 \
        --streams 1,2,4,8 --profiles short,agent --rounds 3 --gen-tokens 160 \
        --out bench/load/results/sweep.json

Excluded from the package gate like the rest of bench/ — experiment code, run manually.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

PARA = (
    "You are a careful coding agent. You inspect files, plan edits, and verify results "
    "with tests. Prefer minimal diffs and clear reports. When a task is ambiguous, state "
    "the assumption you chose and continue rather than stalling. "
)  # ~37 tokens per copy (measured via llama-server usage passthrough)

PROFILES = {
    # name: (approx prompt tokens, copies of PARA)
    "short": 300,
    "agent": 9000,  # matches the measured real Zak-Code request (~9.4k tokens)
}


def build_prompt(profile: str, stream_id: int) -> str:
    target = PROFILES[profile]
    copies = max(1, int(target / 37))
    # Distinct head per stream -> distinct slot affinity (simulates N different agents).
    nonce = f"[agent-{stream_id:02d} session context]\n"
    return nonce + PARA * copies


async def one_request(
    client: httpx.AsyncClient, model: str, system: str, user: str, gen_tokens: int
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": gen_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        r = await client.post("/chat/completions", json=body)
        wall_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "error": r.text[:200], "wall_ms": wall_ms}
        d = r.json()
        usage = d.get("usage") or {}
        timings = d.get("timings") or {}
        server_ms = (timings.get("prompt_ms") or 0) + (timings.get("predicted_ms") or 0)
        return {
            "ok": True,
            "wall_ms": wall_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_tps": timings.get("prompt_per_second"),
            "gen_ms": timings.get("predicted_ms"),
            "gen_tps": timings.get("predicted_per_second"),
            "queue_ms": max(0.0, wall_ms - server_ms) if server_ms else None,
        }
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "wall_ms": (time.perf_counter() - t0) * 1000}


async def stream_worker(
    client: httpx.AsyncClient, model: str, profile: str, stream_id: int, rounds: int, gen_tokens: int
) -> list[dict[str, Any]]:
    system = build_prompt(profile, stream_id)
    results = []
    for rnd in range(rounds):
        user = f"Round {rnd}: write a short paragraph about efficient GPU batching. Vary the wording."
        res = await one_request(client, model, system, user, gen_tokens)
        res["stream"] = stream_id
        res["round"] = rnd
        results.append(res)
    return results


async def gpu_poller(status_url: str, samples: list[dict[str, Any]], stop: asyncio.Event) -> None:
    async with httpx.AsyncClient(timeout=5) as c:
        while not stop.is_set():
            try:
                r = await c.get(status_url)
                if r.status_code == 200:
                    gpus = r.json().get("gpus") or [{}]
                    gpu = gpus[0]  # one llama-server per GPU; index 0 is the serving card
                    samples.append(
                        {
                            "t": time.time(),
                            "vram_used_mb": gpu.get("memory_used_mib"),
                            "util_pct": gpu.get("utilization_pct"),
                            "power_w": gpu.get("power_w"),
                            "temp_c": gpu.get("temperature_c"),
                        }
                    )
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except TimeoutError:
                pass


def summarize(config: dict[str, Any], results: list[dict[str, Any]], gpu: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.get("ok")]
    errs = [r for r in results if not r.get("ok")]
    total_gen = sum(r.get("completion_tokens") or 0 for r in ok)
    cold = [r for r in ok if r["round"] == 0]
    warm = [r for r in ok if r["round"] > 0]

    def mean(vals: list) -> float | None:
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 1) if vals else None

    def med(vals: list) -> float | None:
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 1) if vals else None

    return {
        **config,
        "requests_ok": len(ok),
        "requests_err": len(errs),
        "errors": [e.get("error") or str(e.get("status")) for e in errs][:3],
        "wall_s": round(wall_s, 1),
        "aggregate_gen_tps": round(total_gen / wall_s, 1) if wall_s > 0 else None,
        "per_req_gen_tps_mean": mean([r.get("gen_tps") for r in ok]),
        "prompt_tps_cold_mean": mean([r.get("prompt_tps") for r in cold]),
        "prompt_tokens_warm_mean": mean([r.get("prompt_tokens") for r in warm]),
        "prompt_ms_cold_med": med([r.get("prompt_ms") for r in cold]),
        "queue_ms_med": med([r.get("queue_ms") for r in ok]),
        "queue_ms_max": round(max((r.get("queue_ms") or 0 for r in ok), default=0), 1),
        "vram_max_mb": max((s.get("vram_used_mb") or 0 for s in gpu), default=None),
        "gpu_util_mean_pct": mean([s.get("util_pct") for s in gpu]),
        "power_max_w": max((s.get("power_w") or 0 for s in gpu), default=None),
    }


async def run_config(
    base_url: str, status_url: str, model: str, profile: str, streams: int, rounds: int, gen_tokens: int
) -> dict[str, Any]:
    gpu_samples: list[dict[str, Any]] = []
    stop = asyncio.Event()
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as client:
        poller = asyncio.create_task(gpu_poller(status_url, gpu_samples, stop))
        t0 = time.perf_counter()
        worker_results = await asyncio.gather(
            *(stream_worker(client, model, profile, i, rounds, gen_tokens) for i in range(streams))
        )
        wall_s = time.perf_counter() - t0
        stop.set()
        await poller
    flat = [r for stream in worker_results for r in stream]
    config = {"profile": profile, "streams": streams, "rounds": rounds, "gen_tokens": gen_tokens}
    summary = summarize(config, flat, gpu_samples, wall_s)
    summary["_raw"] = flat
    return summary


def md_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("profile", "profile"), ("streams", "N"), ("aggregate_gen_tps", "agg gen tok/s"),
        ("per_req_gen_tps_mean", "per-req gen tok/s"), ("prompt_tps_cold_mean", "prompt tok/s (cold)"),
        ("prompt_tokens_warm_mean", "warm prompt toks"), ("queue_ms_med", "queue ms (med)"),
        ("queue_ms_max", "queue ms (max)"), ("vram_max_mb", "VRAM max MiB"),
        ("gpu_util_mean_pct", "util %"), ("requests_err", "errs"),
    ]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://10.0.0.204:9090/v1")
    ap.add_argument("--status-url", default=None, help="default: <base-url host>/status")
    ap.add_argument("--model", default="zds-qwen3.5-9b")
    ap.add_argument("--streams", default="1,2,4,8")
    ap.add_argument("--profiles", default="short,agent")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gen-tokens", type=int, default=160)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    status_url = args.status_url or args.base_url.rsplit("/v1", 1)[0] + "/status"
    stream_counts = [int(s) for s in args.streams.split(",")]
    profiles = [p.strip() for p in args.profiles.split(",")]

    rows = []
    for profile in profiles:
        for n in stream_counts:
            print(f"[loadtest] profile={profile} streams={n} rounds={args.rounds} ...", flush=True)
            row = await run_config(
                args.base_url, status_url, args.model, profile, n, args.rounds, args.gen_tokens
            )
            printable = {k: v for k, v in row.items() if k != "_raw"}
            print(f"[loadtest]   -> {json.dumps(printable, default=str)}", flush=True)
            rows.append(row)

    print("\n## Sweep results\n")
    print(md_table(rows))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"ran_at": datetime.now().isoformat(timespec="seconds"),
                        "base_url": args.base_url, "rows": rows}, indent=2),
            encoding="utf-8",
        )
        print(f"\n[loadtest] raw results -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
