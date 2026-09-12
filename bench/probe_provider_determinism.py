#!/usr/bin/env python
"""Is the SERVING STACK reproducible at temperature 0? -- the control ADR-0150 never ran.

ADR-0150 partitions run-to-run variance as ``sampler + engine`` and, having pinned
``ZAKCODE_TEMPERATURE=0``, attributes the whole residual (9% / 36% / 0% iteration spread) to the
engine: "That is variance the engine owns." That is an attribution by elimination over a candidate
list with TWO members, and there is a third. A batching inference server is not bitwise
reproducible just because the sampler is greedy -- batch composition changes the reduction order in
the kernels, so byte-identical input can yield different logits and, at a near-tie, a different
token. If that happens here, part of the residual is the PROVIDER's and "36% is a defect, not a
floor" is an over-claim.

``04-todo-cli`` running 17/17/17 does not settle it: iteration count is coarse, and a provider can
return different prose on every call while still converging in the same number of steps.

PRE-REGISTERED DECISION RULE (written before the first request):
  * All temp-0 repeats byte-identical  -> the provider reproduces for this shape; the third
    candidate is eliminated for single-call reproduction and ADR-0150's attribution survives.
  * Any temp-0 repeat differs          -> provider non-determinism at temp 0 is REAL; ADR-0150
    over-attributes and the 36% target's OWNERSHIP is unresolved until partitioned again.
  * The SHORT prompt agreeing proves little about a 15-iteration task, so both a short and a long
    generation are run and reported separately. Divergence probability grows with token count.

INSTRUMENT POSITIVE CONTROL, and the reason this script cannot report a false "deterministic":
  a temp-1.0 arm must DIFFER. If even temp 1.0 returns identical text, something between here and
  the model is serving one cached answer, and the temp-0 agreement would be an artifact of that
  cache rather than evidence of determinism. An instrument that cannot say "different" has not
  said "same".

Usage:  source bench/results/_podenv.sh && ./.venv/bin/python bench/probe_provider_determinism.py
"""
from __future__ import annotations

import json
import os
import sys

import httpx

SHORT = "Reply with exactly the word OK and nothing else."
LONG = (
    "Write a complete Python implementation of an LRU cache class named Cache with get(key) and "
    "put(key, value) methods, capacity fixed at 3, using only the standard library. Include a "
    "short docstring on each method and a brief comment explaining the eviction order. Output only "
    "the code."
)


def one(client: httpx.Client, base: str, model: str, prompt: str, temperature: float, max_tokens: int) -> tuple[str, dict]:
    """Return (comparable payload, usage).

    The payload is the WHOLE assistant message, canonically serialized -- never `content` alone.
    A thinking model puts its tokens in `reasoning_content` and can hit `max_tokens` before it ever
    emits `content`, so a `content`-only comparison compares two empty strings and reports
    `identical=True` having measured nothing. That happened on this probe's first run.
    """
    r = client.post(
        f"{base}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        timeout=300,
    )
    r.raise_for_status()
    body = r.json()
    msg = body["choices"][0]["message"]
    payload = json.dumps({k: v for k, v in msg.items() if k != "role"}, sort_keys=True, ensure_ascii=False)
    return payload, body.get("usage") or {}


def arm(client, base, model, label: str, prompt: str, temperature: float, n: int, max_tokens: int) -> bool:
    outs: list[str] = []
    usages: list[dict] = []
    for _ in range(n):
        txt, usage = one(client, base, model, prompt, temperature, max_tokens)
        outs.append(txt)
        usages.append(usage)
    distinct = {o for o in outs}
    identical = len(distinct) == 1
    first_div = -1
    if not identical:
        a = outs[0]
        for b in outs[1:]:
            for i, (ca, cb) in enumerate(zip(a, b)):
                if ca != cb:
                    first_div = i if first_div < 0 else min(first_div, i)
                    break
            else:
                if len(a) != len(b):
                    i = min(len(a), len(b))
                    first_div = i if first_div < 0 else min(first_div, i)
    fields = sorted({k for o in outs for k, v in json.loads(o).items() if v})
    print(
        f"{label:34} n={n} temp={temperature:<4} distinct={len(distinct)}/{n} "
        f"identical={identical}  chars={[len(o) for o in outs]}  "
        f"completion_tokens={[u.get('completion_tokens') for u in usages]}"
        + f"  nonempty_fields={fields or 'NONE'}"
        + (f"  first_divergence_at_char={first_div}" if first_div >= 0 else "")
    )
    if not fields:
        print("    ^^ NOTHING WAS COMPARED: every message field is empty. An instrument with no")
        print("       content can only ever say 'same'. This arm is VOID, not deterministic.")
    if not identical and first_div >= 0:
        a = outs[0]
        lo = max(0, first_div - 60)
        print(f"    ...{a[lo:first_div]}>>>HERE<<<{a[first_div:first_div + 60]}...")
    return identical


def main() -> int:
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if not base:
        print("OPENAI_BASE_URL unset -- source bench/results/_podenv.sh first", file=sys.stderr)
        return 2
    models = json.loads(os.environ["ZAKCODE_ZAKPICK_MODELS"])
    model = models["deep_code"]["model"]
    print(f"endpoint={base}  model={model}\n")

    with httpx.Client() as client:
        print("--- INSTRUMENT POSITIVE CONTROL (must be identical=False, or every result below is void)")
        control_identical = arm(client, base, model, "temp1.0 long (control)", LONG, 1.0, 3, 4000)
        print("\n--- TEMP-0 ARMS (the measurement)")
        short_same = arm(client, base, model, "temp0 short", SHORT, 0.0, 5, 16)
        long_same = arm(client, base, model, "temp0 long", LONG, 0.0, 5, 4000)

    print("\n--- VERDICT (against the pre-registered rule)")
    if control_identical:
        print("VOID: the temp-1.0 control came back IDENTICAL. Something is serving one cached answer;")
        print("      the temp-0 readings cannot distinguish determinism from caching. Fix the")
        print("      instrument before citing anything above.")
        return 3
    print("instrument OK: temp 1.0 diverges, so 'identical' below is a real reading.")
    if long_same and short_same:
        print("PROVIDER REPRODUCES at temp 0 for both shapes -> third candidate eliminated for")
        print("single-call reproduction; ADR-0150's engine attribution SURVIVES this control.")
        return 0
    print("PROVIDER IS NON-DETERMINISTIC at temp 0 -> ADR-0150 over-attributes the residual to the")
    print("engine. The 9%/36%/0% spread is sampler-pinned but NOT provider-pinned, so ownership of")
    print("the 36% is UNRESOLVED and must not be cited as an engine defect until re-partitioned.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
