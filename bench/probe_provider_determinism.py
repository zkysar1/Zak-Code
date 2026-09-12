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

import ast
import asyncio
import json
import os
import sys
from pathlib import Path

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


def replay(dump: Path, n: int, temperature: float, max_tokens: int) -> int:
    """Replay a dumped provider request verbatim, N times, and report whether it reproduces.

    WHY THIS EXISTS, and it is a correction to this file's own earlier claim. The arms above send a
    tool-free single-user-message request and found the pod byte-reproducible 5/5 at temperature 0.
    That reading was then used to exonerate the serving stack -- and it does not generalize. A
    dumped `03-lru` call-1 (9,975-char system prompt, 25 tool schemas, a completion that emits tool
    calls) produced two DIFFERENT implementations from byte-identical wire input across two runs.
    One request shape reproducing is not evidence that another does; the only honest test is to
    replay the shape that actually diverged.

    Reconstructs the wire payload from a ZBENCH_DUMP_REQUESTS file. Only handles a dump whose
    messages are plain user/system text -- a mid-conversation dump carries tool_use/tool_result
    blocks whose wire translation lives in the provider, and hand-rolling a second translator here
    would measure a payload the engine never sends. Refuses instead of guessing.
    """
    d = json.loads(dump.read_text(encoding="utf-8"))
    wire: list[dict] = []
    if d.get("system"):
        wire.append({"role": "system", "content": d["system"]})
    for m in d.get("messages") or []:
        blocks = m.get("blocks") or []
        kinds = {b.get("type") for b in blocks}
        if kinds - {"text"}:
            print(f"REFUSING: {dump.name} carries non-text blocks {sorted(kinds)}.", file=sys.stderr)
            print("Its wire translation lives in the provider; reconstructing it here would", file=sys.stderr)
            print("measure a payload the engine never sends. Replay a call-0001 dump.", file=sys.stderr)
            return 2
        text = "".join(b.get("text") or "" for b in blocks)
        wire.append({"role": m.get("role", "user"), "content": text})
    tools = d.get("tools")
    # Replay what the ENGINE SENDS, not a plausible approximation of it. The first version of this
    # mode omitted `extra_body` and `prompt_cache_key` because the dump did not carry them, and its
    # 5/5 "REPRODUCES" was therefore a reading on a request zakcode never makes: `deep_code` is a
    # thinking entry, so the real call carries chat_template_kwargs.enable_thinking=True, and a
    # thinking completion is several times longer with correspondingly more near-ties. Running the
    # canonical binary with a non-canonical argument shape is the defect this rebuilds around.
    # Values come back from the dump as repr() strings; ast.literal_eval is the safe inverse.
    extra_body = None
    cache_key = None
    state = d.get("provider_state") or {}
    kwargs = d.get("kwargs") or {}
    if state.get("extra_body") and state["extra_body"] != "None":
        extra_body = ast.literal_eval(state["extra_body"])
    if kwargs.get("prompt_cache_key"):
        cache_key = ast.literal_eval(kwargs["prompt_cache_key"])
    if state.get("temperature") and state["temperature"] != "None":
        temperature = float(ast.literal_eval(state["temperature"]))
    print(f"  extra_body={extra_body}  prompt_cache_key={cache_key}  temperature={temperature}")
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    model = json.loads(os.environ["ZAKCODE_ZAKPICK_MODELS"])["deep_code"]["model"]
    print(f"replaying {dump} -- {len(wire)} wire messages, {len(tools or [])} tools, "
          f"n={n} temp={temperature} max_tokens={max_tokens}")
    payloads: list[str] = []
    with httpx.Client() as client:
        for _ in range(n):
            body: dict = {
                "model": model,
                "messages": wire,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            merged = dict(extra_body or {})
            if cache_key:
                merged["prompt_cache_key"] = cache_key
            if merged:
                body["extra_body"] = merged
            r = client.post(
                f"{base}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                timeout=600,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            # Server-minted tool-call ids are normalized out: they are random by construction and
            # say nothing about whether the model chose the same tokens.
            calls = [
                {"name": (c.get("function") or {}).get("name"),
                 "arguments": (c.get("function") or {}).get("arguments")}
                for c in (msg.get("tool_calls") or [])
            ]
            payloads.append(json.dumps(
                {"content": msg.get("content"), "reasoning": msg.get("reasoning_content"),
                 "tool_calls": calls},
                sort_keys=True, ensure_ascii=False))
    distinct = len(set(payloads))
    print(f"distinct completions: {distinct}/{n}   lengths={[len(x) for x in payloads]}")
    if distinct == 1:
        print("REPRODUCES for this shape.")
        return 0
    a = payloads[0]
    for b in payloads[1:]:
        if b != a:
            i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"first divergence at char {i}:")
            print(f"  A ...{a[max(0, i - 70):i]}>>>{a[i:i + 70]}")
            print(f"  B ...{b[max(0, i - 70):i]}>>>{b[i:i + 70]}")
            break
    print("DOES NOT REPRODUCE for this shape -- byte-identical wire input, different completion.")
    print("The serving stack is the source; nothing downstream of it can be held responsible.")
    return 1


def replay_wire(dump: Path, n: int) -> int:
    """Replay a `wire-NNNN.json` body verbatim through litellm, N times.

    This is the faithful mode, and it exists because the reconstruction mode above is NOT faithful.
    `--replay` rebuilds a request from the engine-level dump and, measured, reproduced 5/5 with
    itself while matching NEITHER of the two runs it was meant to explain (3,757 chars against the
    real call's 5,085). A self-consistent answer to the wrong question is the most expensive kind:
    it looks like a result.

    `wire-NNNN.json` is the exact kwargs dict handed to `litellm.acompletion`, so there is nothing
    to reconstruct -- including the keys the reconstruction missed entirely (`response_format`,
    `drop_params`, `num_retries`, `timeout`, and a per-call `enable_thinking` that is False on the
    classifier call and True on the agent's).
    """
    import litellm  # noqa: PLC0415

    body = json.loads(dump.read_text(encoding="utf-8"))
    body.pop("api_key", None)
    print(f"replaying {dump} verbatim -- {len(body.get('messages') or [])} messages, "
          f"{len(body.get('tools') or [])} tools, n={n}")
    print(f"  keys: {sorted(body)}")

    async def _go() -> list[str]:
        out = []
        for _ in range(n):
            r = await litellm.acompletion(**body)
            msg = r.choices[0].message
            calls = [
                {"name": c.function.name, "arguments": c.function.arguments}
                for c in (getattr(msg, "tool_calls", None) or [])
            ]
            out.append(json.dumps(
                {"content": msg.content,
                 "reasoning": getattr(msg, "reasoning_content", None),
                 "tool_calls": calls},
                sort_keys=True, ensure_ascii=False))
        return out

    payloads = asyncio.run(_go())
    distinct = len(set(payloads))
    print(f"distinct completions: {distinct}/{n}   lengths={[len(x) for x in payloads]}")
    if distinct == 1:
        print("REPRODUCES -- byte-identical wire body, byte-identical completion.")
        return 0
    print("DOES NOT REPRODUCE -- byte-identical wire body, different completion. The serving")
    print("stack is the source, and nothing downstream of it can be held responsible.")
    return 1


def main() -> int:
    if "--replay-wire" in sys.argv:
        i = sys.argv.index("--replay-wire")
        return replay_wire(Path(sys.argv[i + 1]), int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else 5)
    if "--replay" in sys.argv:
        i = sys.argv.index("--replay")
        n = int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else 5
        return replay(Path(sys.argv[i + 1]), n, 0.0, 8192)
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
