"""Classify what differs between two dumped request streams.

Three classes, because they carry OPPOSITE verdicts and lumping them together is how a
determinism claim goes wrong:
  * created_at          -- a dump-only field; `_translate_messages` emits role/content/
                           tool_call_id and nothing else, so this never reaches the wire.
  * tool-call ids       -- minted by the SERVER on its response and echoed back by the client on
                           the next turn. Different here means the server is not deterministic in
                           its id generation; it does NOT mean the engine chose differently.
  * everything else     -- real input divergence. This is the only class that is evidence about
                           the engine.
"""
import json, re, sys
from pathlib import Path

A, B = Path(sys.argv[1]), Path(sys.argv[2])
ID_RE = re.compile(r'"(id|tool_use_id|tool_call_id)": "[^"]+"')

def norm(p, drop_ids):
    d = json.loads(p.read_text(encoding="utf-8"))
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "created_at"}
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o
    txt = json.dumps(strip(d), sort_keys=True, indent=1, ensure_ascii=False)
    return ID_RE.sub('"ID": "<normalized>"', txt) if drop_ids else txt

calls = sorted(x.name for x in A.glob("call-*.json"))
other = sorted(x.name for x in B.glob("call-*.json"))
print(f"calls: A={len(calls)} B={len(other)}")
verdict_content_clean = True
for name in calls:
    a, b = A / name, B / name
    if not b.exists():
        print(f"  {name}: ONLY IN A -- the runs diverged in CALL COUNT"); verdict_content_clean = False; continue
    raw_same = norm(a, False) == norm(b, False)
    id_same = norm(a, True) == norm(b, True)
    if raw_same:
        print(f"  {name}: IDENTICAL (modulo created_at)")
    elif id_same:
        print(f"  {name}: differs ONLY in server-minted ids")
    else:
        print(f"  {name}: CONTENT DIFFERS")
        verdict_content_clean = False
        x, y = norm(a, True).split("\n"), norm(b, True).split("\n")
        for i, (p_, q_) in enumerate(zip(x, y)):
            if p_ != q_:
                print(f"      first differing line {i}:")
                print(f"        A: {p_.strip()[:200]}")
                print(f"        B: {q_.strip()[:200]}")
                break
if len(calls) != len(other):
    print(f"  CALL COUNT DIFFERS: {len(calls)} vs {len(other)}"); verdict_content_clean = False
print("\nVERDICT:", "no content divergence -- every difference is a dump field or a server-minted id"
      if verdict_content_clean else "CONTENT DIVERGED -- see the first differing line above")
