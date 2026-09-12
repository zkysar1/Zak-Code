"""Where does a divergence between two dumped request streams ENTER?

Dumps (ZBENCH_DUMP_REQUESTS) are the INPUT of every provider call. So for the first call whose
content differs, the first differing MESSAGE tells the story:
  * role == tool / user  -> the environment handed the two runs different inputs (a tool result,
                            a listing, a timing line). The loop was deterministic given its inputs.
  * role == assistant    -> the previous call had identical input (or it would have been the first
                            differing call) and the model answered differently: the PROVIDER diverged.
Server-minted ids and the dump's created_at are normalised away first (see _diffcalls.py).
Usage: _firstdiff.py <run-A-dir> <run-B-dir>
"""
import json, re, sys
from pathlib import Path

ID_RE = re.compile(r'"(id|tool_use_id|tool_call_id)": "[^"]+"')
UUID_RE = re.compile(r'[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}')


def load(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "created_at"}
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o
    return json.loads(UUID_RE.sub('<uuid>', ID_RE.sub('"ID": "<n>"', json.dumps(strip(d), sort_keys=True))))


def main(a: Path, b: Path) -> int:
    ca = sorted(x.name for x in a.glob("call-*.json")); cb = sorted(x.name for x in b.glob("call-*.json"))
    print(f"calls: A={len(ca)} B={len(cb)}")
    for name in ca:
        if name not in cb:
            print(f"first divergence: {name} exists only in A (call counts differ, all shared calls identical)"); return 0
        da, db = load(a / name), load(b / name)
        if da == db:
            continue
        ma, mb = da.get("messages", []), db.get("messages", [])
        for i, (x, y) in enumerate(zip(ma, mb)):
            if x != y:
                role = x.get("role") or y.get("role")
                print(f"first divergence: {name}, message {i} of {len(ma)}/{len(mb)}, role={role}")
                sx, sy = json.dumps(x, ensure_ascii=False), json.dumps(y, ensure_ascii=False)
                j = next((k for k in range(min(len(sx), len(sy))) if sx[k] != sy[k]), min(len(sx), len(sy)))
                print(f"  A: ...{sx[max(0, j-80):j+160]}")
                print(f"  B: ...{sy[max(0, j-80):j+160]}")
                if role == "tool" or (role == "user" and i == 0 and name == "call-0001.json"):
                    verdict = "ENVIRONMENT (tool result / task prompt differs)"
                elif role == "user" and len(ma) == 1:
                    verdict = "PROVIDER upstream (single-message auxiliary call built by the loop from earlier model output)"
                elif role == "assistant":
                    verdict = "PROVIDER (assistant output differed on identical input)"
                elif role == "user":
                    verdict = "PROVIDER upstream (loop-injected user-role text built from earlier model output)"
                else:
                    verdict = f"role {role}"
                print("  ENTERS THROUGH:", verdict)
                return 0
        if len(ma) != len(mb):
            print(f"first divergence: {name}, message count {len(ma)} vs {len(mb)}, shared prefix identical"); return 0
        other = {k for k in set(da) | set(db) if da.get(k) != db.get(k)}
        # Not a message-level divergence: note it once and keep scanning for the first one that is.
        if not getattr(main, '_noted', False):
            main._noted = True
            print(f"  note: {name} differs outside messages in {sorted(other)} (after uuid/id normalisation); scanning on")
        continue
    print("no divergence in the shared calls" + ("" if len(ca) == len(cb) else f"; call counts differ {len(ca)} vs {len(cb)}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
