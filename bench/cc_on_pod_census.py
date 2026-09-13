#!/usr/bin/env python3
# Model-fixed CC arm: runs Claude Code (`claude -p`) on the pod (native Anthropic endpoint)
# against zds-qwen3.6-35b, so the LOOP can be compared with the MODEL HELD FIXED vs zakcode.
# Produced bench/results/cc-qwen-06-{census,surf}.json; see 06-model-fixed-head-to-head.log.
"""Model-fixed CC-on-qwen census runner (task-agnostic). Runs `claude -p` on qwen-35B via the pod's
native Anthropic endpoint against ONE bench task, N times, capturing verify pass/fail, the tool-call
trace (which files Read/Edited/Grepped), and the output-tree digest. --surf prepends the task's
CONTRIBUTING.md (if present) to the prompt to mimic zakcode's CONVENTION_FILENAMES fold.

Usage:
  python3 cc-qwen-task-census.py --task /opt/zak-code-bench/bench/tasks/06-plugin-conventions \
      --n 6 --out /root/cc-qwen-06-census.json [--surf]
"""
import argparse, os, json, subprocess, tempfile, shutil, time, hashlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--out", required=True)
ap.add_argument("--surf", action="store_true", help="prepend CONTRIBUTING.md to the prompt (fold)")
ap.add_argument("--model", default="zds-qwen3.6-35b")
ap.add_argument("--timeout", type=int, default=700)
A = ap.parse_args()

key = None
for line in open("/etc/zakcode/.env"):
    if line.strip().startswith("ZAKCODE_API_KEY="):
        key = line.split("=", 1)[1].strip().strip('"').strip("'")
env = dict(os.environ)
env.update({
    "ANTHROPIC_BASE_URL": "http://10.0.0.250:9090",
    "ANTHROPIC_API_KEY": key, "ANTHROPIC_AUTH_TOKEN": key,
    "ANTHROPIC_MODEL": A.model, "ANTHROPIC_SMALL_FAST_MODEL": A.model,
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": A.model, "ANTHROPIC_DEFAULT_SONNET_MODEL": A.model,
    "ANTHROPIC_DEFAULT_OPUS_MODEL": A.model,
    "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_FAST_MODE_STRICT": "0",
})
task = Path(A.task)
tid = task.name
spec = json.loads((task / "task.json").read_text())
prompt = spec["prompt"]
contrib_p = task / "workspace" / "CONTRIBUTING.md"
if A.surf and contrib_p.is_file():
    prompt = prompt + "\n\nProject conventions (hard rules you MUST follow):\n" + contrib_p.read_text()

def digest_tree(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if ".pytest_cache" in rel or "__pycache__" in rel or rel.startswith(".git/"):
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out

def analyze_stream(text: str) -> dict:
    tools, reads, edits, greps, globs = {}, set(), set(), set(), set()
    result_obj = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") == "result":
            result_obj = obj
        msg = obj.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    name = blk.get("name", "?")
                    tools[name] = tools.get(name, 0) + 1
                    inp = blk.get("input") or {}
                    fp = os.path.basename(str(inp.get("file_path") or inp.get("path") or ""))
                    pat = str(inp.get("pattern") or "")
                    if name == "Read": reads.add(fp)
                    elif name in ("Edit", "Write"): edits.add(fp)
                    elif name == "Grep": greps.add(pat[:40])
                    elif name == "Glob": globs.add(pat[:40])
    m = lambda coll, n: any(n.lower() in x.lower() for x in coll)
    return {"tools": tools, "reads": sorted(reads), "edits": sorted(edits),
            "greps": sorted(greps), "globs": sorted(globs),
            "read_contributing": m(reads, "contributing") or m(greps, "contributing") or m(globs, "contributing"),
            "result": result_obj}

arm = f"cc-on-qwen-35B{'-surf' if A.surf else ''}"
print(f"=== CC-on-qwen CENSUS: {tid} arm={arm} N={A.n} ===", time.strftime("%H:%M:%S"), flush=True)
runs = []
for k in range(1, A.n + 1):
    ws = Path(tempfile.mkdtemp(prefix=f"cc-{tid}-r{k}-"))
    shutil.copytree(task / "workspace", ws, dirs_exist_ok=True)
    t0 = time.time()
    p = subprocess.run(
        ["claude", "-p", prompt, "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep",
         "--output-format", "stream-json", "--verbose", "--model", A.model],
        cwd=ws, env=env, capture_output=True, text=True, timeout=A.timeout)
    elapsed = round(time.time() - t0, 1)
    a = analyze_stream(p.stdout)
    res = a.get("result") or {}
    dig = digest_tree(ws)
    v = subprocess.run(["/opt/zak-code-bench/.venv/bin/python", str(task / "verify.py")],
                       cwd=ws, capture_output=True, text=True, timeout=120)
    vmsg = (v.stdout or v.stderr).strip()
    rec = {"run": k, "cc_rc": p.returncode, "elapsed_s": elapsed,
           "terminal_reason": res.get("terminal_reason"), "num_turns": res.get("num_turns"),
           "verify_rc": v.returncode, "verify_msg": vmsg[:200],
           "read_contributing": a["read_contributing"], "tools": a["tools"],
           "reads": a["reads"], "edits": a["edits"], "greps": a["greps"], "globs": a["globs"],
           "digests": dig}
    runs.append(rec)
    print(f"run {k}: verify_rc={v.returncode} turns={res.get('num_turns')} "
          f"read_CONTRIB={a['read_contributing']} elapsed={elapsed}s | {vmsg[:90]}", flush=True)
    shutil.rmtree(ws, ignore_errors=True)

npass = sum(1 for r in runs if r["verify_rc"] == 0)
distinct = len({tuple(sorted(r["digests"].items())) for r in runs})
summary = {"task": tid, "arm": arm, "n": A.n, "pass": npass,
           "read_contributing": sum(1 for r in runs if r["read_contributing"]),
           "distinct_output_states": distinct, "runs": runs}
Path(A.out).write_text(json.dumps(summary, indent=2))
print(f"=== DONE: {tid} {arm} PASS {npass}/{A.n} | distinct_outputs {distinct}/{A.n} -> {A.out} ===",
      time.strftime("%H:%M:%S"), flush=True)
