#!/usr/bin/env python3
"""Offline proof that measurement build K draws the way served sample 7 says it does.

K is ``served_builds/k.patch`` on main: NEVER merged, because it draws lots. Once a refused stop
has named a skill re-entry, each plan that FINISHES later in that turn gets one draw, held for as
long as that plan stays finished: its "answer now" line is sent, or kept silent. Draws come in
pairs, one of each side in random order, and no pair spans two turns.

This runs the REAL agent, the REAL shell Stop hook and the real plan machinery of the tree it is
started in, on a scripted model, in the veto-door bench's synthesized workspace, with the first
draw of each pair forced each way in turn. No network. Run it in K's tree, and in an unpatched
tree to see it fail (a proof that cannot fail proves nothing):

    PYTHONPATH=<tree>/src python bench/served_coin_build_proof.py --base /srv/zc-coin

``--base`` must sit outside every repository: the product folds project guides up to the
repository root.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import veto_door as door  # noqa: E402  (the bench's scripted model, workspace and served posture)

LINE = "[plan] Plan complete"  # how the product opens the finished plan's "answer now" line
STEPS = ("Open cycle {n}", "Run the preflight checks of cycle {n}", "Close cycle {n}")


class _Forced:
    """Stands in for the build's random source: the first draw of every pair comes up as told."""

    def __init__(self, silent_first: bool) -> None:
        self.value = 0.0 if silent_first else 0.9

    def random(self) -> float:
        return self.value


def _lap(tag: str, *, linger: bool = False) -> list[Any]:
    """One planned pass of the loop, its plan closed at the end. ``linger``: one harmless command
    more while the plan is still the finished one, so the same plan meets a second request. Each
    pass names its own cycle in its plan, as a model does: the same plan sent four times over
    reads to the product's stuck ladder as no progress, and it steps in."""
    titles = [step.format(n=ord(tag) - ord("a") + 1) for step in STEPS]
    opened = [{"title": t, "status": "in_progress" if i == 0 else "pending"}
              for i, t in enumerate(titles)]  # fmt: skip
    steps = [
        door._do("update_plan", f"{tag}0", tasks=opened),
        door._do("Bash", f"{tag}1", command="bash scripts/cycle-open.sh"),
        door._do("Bash", f"{tag}2", command="bash scripts/cycle-close.sh"),
        door._do("update_plan", f"{tag}3", tasks=[{"title": t, "status": "done"} for t in titles]),
    ]
    if linger:
        steps.append(door._do("Bash", f"{tag}4", command="echo still here"))
    return steps


def _recording(scripted: Any) -> Any:
    class Recording(scripted):  # type: ignore[misc, valid-type]
        """The scripted model, remembering whether each request carried the line."""

        def __init__(self, script: list[Any]) -> None:
            super().__init__(script)
            self.lines: list[bool] = []

        async def acomplete(self, messages: list[Any], **kw: Any) -> Any:
            self.lines.append(
                any(m.get("rail") == "[plan:complete]" for m in door._tail_view(messages))
            )
            return await super().acomplete(messages, **kw)

    return Recording


def _run(base: Path, silent_first: bool) -> dict[str, Any]:
    """Four passes in one turn. Pass 1 ends in words and is refused (the turn is governed from
    there); passes 2, 3 and 4 each finish a plan in the governed turn; pass 2's plan lingers."""
    workspace = door.make_workspace(base)
    (workspace / "state").mkdir(exist_ok=True)
    (workspace / "state" / "veto-budget").write_text("12\n", encoding="utf-8")
    os.environ["ZAKCODE_HOME"] = str(door._registry(base) / f"home-{workspace.name}")
    os.chdir(workspace)
    script = [
        *_lap("a"),
        door._say("Cycle 1 closed."),  # refused: from here a hook governs the turn's end
        *_lap("b", linger=True),
        *_lap("c"),
        *_lap("d"),
    ]
    model = _recording(door._scripted_class())(script)
    agent = door.served_agent(workspace, max_iterations=60, cost_cap=0.01, provider=model)
    if hasattr(agent.loop, "_coin"):
        agent.loop._coin = _Forced(silent_first)
    asyncio.run(door._run_turn(agent))
    events = list(agent.loop._trace.events)
    usage = [e.data for e in events if e.kind == "usage"]
    notes = [e.data for e in events if e.kind == "intervention"]
    # A SECOND turn on the same agent. The first ended with pair 2 half drawn (its other side
    # still owed). A new turn owes nothing: its first governed plan opens pair 1 with a fair draw.
    model.script = [*_lap("e"), door._say("Cycle 5 closed."), *_lap("f")]
    asyncio.run(door._run_turn(agent))
    second_turn = [
        (e.data["pair"], e.data["slot"], e.data["side"])
        for e in agent.loop._trace.events
        if e.kind == "intervention" and e.data.get("kind") == "answer_now_coin"
    ]
    return {
        "second_turn": second_turn,
        "lines": model.lines[: len(script)],
        "labels": [
            "sent"
            if "plan_complete" in (u.get("rails") or [])
            else "silent"
            if "plan_complete" in (u.get("rails_silenced") or [])
            else "-"
            for u in usage[: len(script)]
        ],  # fmt: skip
        "coins": [
            (n["pair"], n["slot"], n["side"]) for n in notes if n.get("kind") == "answer_now_coin"
        ],  # fmt: skip
        "order": [
            n.get("kind")
            for n in notes
            if n.get("kind") in ("turn_end_governed", "answer_now_coin")
        ],  # fmt: skip
        "has_build": hasattr(agent.loop, "_coin"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base", type=Path, default=Path("/srv/zc-coin"))
    args = parser.parse_args()
    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(door._check(name, ok, detail))

    # Request indices (0-based; one request per scripted completion, plus the one the delivery
    # at the refused stop adds nothing to): pass 1 is 0-3, its words are 4; pass 2 is 5-8 and its
    # lingering command 9; pass 3 is 10-13; pass 4 is 14-17. A plan marked done by request n is
    # first SEEN finished by request n + 1.
    after_a, after_b, b_lingers, after_c, after_d = 4, 9, 10, 14, 18
    for silent_first in (True, False):
        first, second = ("silent", "sent") if silent_first else ("sent", "silent")
        print(f"build K, first draw of each pair forced {first}")
        seen = _run(args.base, silent_first)
        lines, labels = seen["lines"], seen["labels"]
        check("this tree is build K (it has a draw to force)", seen["has_build"])
        check(
            "the plan finished BEFORE the refused stop drew nothing and was sent its line",
            len(lines) > after_a and lines[after_a] and labels[after_a] == "sent",
            f"{lines[after_a : after_a + 1]} {labels[after_a : after_a + 1]}",
        )
        check(
            "three plans finished in the governed turn: three draws, pair 1 then pair 2",
            [(p, s) for p, s, _ in seen["coins"]] == [(1, 1), (1, 2), (2, 1)],
            str(seen["coins"]),
        )
        check(
            "no draw came before the refusal that named the skill",
            seen["order"][:1] == ["turn_end_governed"],
            str(seen["order"]),
        )
        sides = [side for _, _, side in seen["coins"]]
        check(f"the pair's sides are {first} then {second}, and pair 2 opens {first}",
              sides == [first, second, first], str(sides))  # fmt: skip
        for name, at, side in (
            ("pass 2", after_b, first),
            ("pass 2, the same plan one request later", b_lingers, first),
            ("pass 3", after_c, second),
        ):
            got = (lines[at], labels[at]) if len(lines) > at else None
            check(
                f"{name}: the line is {side}, on the wire and in the trace label",
                got == (side == "sent", side),
                str(got),
            )
        # Pass 4's plan is first seen by the request AFTER the script ran out, which the recording
        # does not cover; its draw is in the notes above.
        check("the script was long enough for every draw to be seen", len(lines) >= after_d - 1)
        check(
            f"a new turn starts its own pair 1 with a fair draw ({first}), owing nothing",
            seen["second_turn"][:1] == [(1, 1, first)],
            str(seen["second_turn"]),
        )
    failed = results.count(False)
    print(f"build proof: {len(results) - failed} ok, {failed} failed ({door.build_identity()})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
