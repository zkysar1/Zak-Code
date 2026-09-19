"""Is the SERVED code path identical between two zakcode builds?

Drives the planning surface the served loop actually uses — the doom-loop resend, a never-worked
resend, an edit, and a completed plan — through a Settings-wired context (which is what a served
run constructs) and prints every output, hint and data dict. Run under PYTHONPATH pinned to each
build and diff. Offline: no model calls.
"""
import asyncio, json, os, sys
from pathlib import Path
import zakcode
from zakcode.config import Settings
from zakcode.tasks import TaskNetwork
from zakcode.tools.base import ToolContext
from zakcode.tools.builtins.update_plan import UpdatePlanTool

PLAN = [
    {"title": "write the module", "note": "file exists"},
    {"title": "export it", "note": "symbol imports"},
    {"title": "add tests", "note": "tests pass"},
]

def served_ctx(net):
    """What a served loop builds: a ToolContext wired from Settings, as loop.py does."""
    s = Settings()
    kw = {"workspace_root": Path("/tmp"), "task_network": net}
    if "plan_autoadvance" in ToolContext.model_fields:      # the old build wires it from Settings
        kw["plan_autoadvance"] = s.plan_autoadvance
    return ToolContext(**kw)

def show(tag, r):
    print(f"[{tag}] output={r.output!r}")
    print(f"[{tag}] hint={r.hint!r}")
    print(f"[{tag}] data={json.dumps(r.data, sort_keys=True) if r.data else None}")
    print(f"[{tag}] is_error={r.is_error}")

async def main():
    print("build_src:", os.path.dirname(zakcode.__file__))
    T = UpdatePlanTool()

    # A. the doom-loop shape: lay out, work it, resend unchanged three times (frontier walk)
    net = TaskNetwork(); ctx = served_ctx(net)
    show("A0", await T.execute({"tasks": PLAN}, ctx))
    net.attach_evidence(net.current(), "wrote /tmp/utils/duration.py")
    for i in (1, 2, 3):
        show(f"A{i}", await T.execute({"tasks": PLAN}, ctx))
    print("A_final_statuses:", [t.status for t in net.tasks], "complete:", net.is_complete())

    # B. never worked: the rail, not an advance
    net = TaskNetwork(); ctx = served_ctx(net)
    await T.execute({"tasks": PLAN}, ctx)
    show("B1", await T.execute({"tasks": PLAN}, ctx))

    # C. outcome recorded, status left pending
    net = TaskNetwork(); ctx = served_ctx(net)
    plan = [dict(PLAN[0], outcome="wrote the module"), PLAN[1], PLAN[2]]
    await T.execute({"tasks": plan}, ctx)
    show("C1", await T.execute({"tasks": plan}, ctx))

    # D. a real edit is an update, not an advance
    net = TaskNetwork(); ctx = served_ctx(net)
    await T.execute({"tasks": PLAN}, ctx)
    net.attach_evidence(net.current(), "wrote it")
    show("D1", await T.execute({"tasks": [dict(PLAN[0], status="done"), PLAN[1], PLAN[2]]}, ctx))

    # E. a finished plan resent
    net = TaskNetwork(); ctx = served_ctx(net)
    done = [{"title": "a", "status": "done"}, {"title": "b", "status": "done"}]
    await T.execute({"tasks": done}, ctx)
    show("E1", await T.execute({"tasks": done}, ctx))

asyncio.run(main())
