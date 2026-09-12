#!/usr/bin/env python3
"""Coach replication of the ADR-0155 choosability arms. Runs ON zc-03, stdlib only.

Coach's skill text never leaves the box: this prints SCORES and COUNTS only, never a catalogue,
a description or a query. Pod credentials are sourced from /etc/zakcode/.env by the caller and
read from the environment here -- never printed.

Identical extraction, rendering, scoring, clustering and bootstrap as the 420-query ayoai-mind
run, so the two are directly comparable.
"""
import json, os, random, re, urllib.request
from collections import Counter
from pathlib import Path

ROOTS = [Path("/opt/coach-mind/.claude/skills"), Path("/opt/coach-mind/.zakcode/skills")]
VOID_BELOW = 0.45


def entries():
    out, seen = [], set()
    for r in ROOTS:
        for sk in sorted(r.glob("*/SKILL.md")):
            if sk.parent.name in seen:
                continue
            h = sk.read_text(encoding="utf-8", errors="replace")[:6000]
            m = re.search(r"^description:\s*(.+)$", h, re.M)
            if m:
                seen.add(sk.parent.name)
                out.append((sk.parent.name, m.group(1).strip(), sk))
    return out


def render(cat, arm):
    lines = []
    for name, desc, _ in cat:
        if arm == "NAMES":
            lines.append("- %s" % name)
        elif arm == "FIRST":
            first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
            lines.append("- %s: %s" % (name, first))
        else:
            lines.append("- %s: %s" % (name, desc))
    return "\n".join(lines)


def queries_for(path, name, k=3):
    body = re.sub(r"^---.*?\n---\n", "", path.read_text(encoding="utf-8", errors="replace"), flags=re.S)
    body = re.sub(r"^#.*$", "", body, flags=re.M)
    body = re.sub(r"`[^`]*`", "", body)
    paras = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 200]
    out = []
    for chunk in paras[1:1 + k]:
        c = re.sub(r"\s+", " ", chunk)[:600]
        out.append(c.replace(name, "this") if name in c else c)
    return out


def ask(base, key, model, catalogue, query):
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content":
             "You choose exactly one skill from a catalogue. Reply with the skill name only, "
             "nothing else.\n\nCatalogue:\n" + catalogue},
            {"role": "user", "content":
             "Which single skill from the catalogue does this text belong to?\n\n" + query},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=300) as r:
        ch = json.loads(r.read())["choices"][0]
    content = (ch["message"].get("content") or "").strip().strip("`/ ")
    if not content:
        raise SystemExit("INSTRUMENT FAILURE: empty content (finish=%r). Fix the request shape "
                         "before reading any score." % ch.get("finish_reason"))
    return content


def boot(per_skill, a, b, iters=40000):
    names = list(per_skill)
    diffs = [per_skill[n][a] - per_skill[n][b] for n in names]
    pt = sum(diffs) / len(diffs)
    rng = random.Random(20260912)
    reps = sorted(sum(s) / len(s) for s in
                  ([diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))] for _ in range(iters)))
    return pt, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps))]


def _dotenv(path="/etc/zakcode/.env"):
    """Parse the env file directly rather than sourcing it in bash.

    `. /etc/zakcode/.env` STRIPS the inner double-quotes from a JSON-valued var, so
    ZAKCODE_ZAKPICK_MODELS arrives as {classify:{model:...}} and json.loads fails. zakcode reads
    this file with a real dotenv parser, not a shell. Values are returned in-process and never
    printed -- the API key does not reach stdout, a log, or the wire back to the caller.
    """
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def main():
    env = _dotenv()
    base = env["ZAKCODE_API_BASE"]
    key = env["ZAKCODE_API_KEY"]
    # Pinned to the SAME model the ayoai-mind runs used, so the replication is comparable. Read
    # from config where possible, but never let a config-parse difference silently change the
    # model under a comparison -- that would make the two runs incomparable without saying so.
    model = "zds-qwen3.6-35b"
    cfg_model = None
    try:
        cfg_model = json.loads(env["ZAKCODE_ZAKPICK_MODELS"])["deep_code"]["model"]
    except Exception:
        pass
    if cfg_model and cfg_model != model:
        raise SystemExit("MODEL MISMATCH: config says %r, the ayoai-mind runs used %r. Refusing "
                         "to report a comparison across different models." % (cfg_model, model))
    cat = entries()
    qmap = {n: qs for n, _, p in cat if len(qs := queries_for(p, n)) == 3}
    total = sum(len(v) for v in qmap.values())
    print("coach skills=%d  with 3 queries=%d  queries=%d  requests=%d  model=%s"
          % (len(cat), len(qmap), total, total * 3, model))
    fam = Counter(n.split("-")[0] for n, _, _ in cat)
    print("families>=3: %d covering %d skills (%.0f%%)"
          % (sum(1 for v in fam.values() if v >= 3),
             sum(v for v in fam.values() if v >= 3),
             sum(v for v in fam.values() if v >= 3) / len(cat) * 100))

    per_skill = {n: {} for n in qmap}
    for arm in ("FULL", "FIRST", "NAMES"):
        rendered = render(cat, arm)
        hits = 0
        for name, qs in qmap.items():
            got = [ask(base, key, model, rendered, q) for q in qs]
            s = sum(1 for g in got if g == name)
            per_skill[name][arm] = s / 3
            hits += s
        print("%-6s %d/%d = %.1f%%   catalogue %d chars" % (arm, hits, total, hits / total * 100, len(rendered)))

    full = sum(v["FULL"] for v in per_skill.values()) / len(per_skill)
    print("\n--- verdict against the pre-registered rules ---")
    if full < VOID_BELOW:
        print("VOID: FULL control %.1f%% < %.0f%%." % (full * 100, VOID_BELOW * 100))
        return 3
    print("control OK: FULL %.1f%% (VOID below %.0f%%)." % (full * 100, VOID_BELOW * 100))
    for arm in ("FIRST", "NAMES"):
        pt, lo, hi = boot(per_skill, "FULL", arm)
        print("FULL minus %-5s: %+.1f%%  95%% CI (%+.1f%%, %+.1f%%)  excludes zero: %s"
              % (arm, pt * 100, lo * 100, hi * 100, lo > 0 or hi < 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
