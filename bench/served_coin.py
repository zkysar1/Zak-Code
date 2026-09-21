"""Served sample 7's reader: what a finished plan's "answer now" line does to a served loop.

usage: python bench/served_coin.py <world-dir> [<world-dir> ...]     one JSON reading
       python bench/served_coin.py --selftest                        known answers, synthetic traces

Labels, counts and draws only: no prompt, completion or command text is read or printed.

EPISODES. A finished plan stays on the board until the model plans again, and while it does
every request says which form its reminder took (`served_stops.py`): the line SENT, kept SILENT,
or a RESTED tail. An EPISODE is one such stretch inside one turn: it opens at the first request
that names a finished plan and closes at the first that names an open plan or none. It STOPPED
IN WORDS if any request in it was answered with a stop the model made itself (words a turn-end
hook refused, or words that ended the turn); a stop the stuck ladder made is counted apart. It
is GOVERNED if a refused stop had already delivered a skill earlier in the same turn, which is
where arm L would keep the line silent. This part reads any build since ADR-0204.

DRAWS (build K only, `served_builds/k.patch`). K gives each plan that finishes in a governed turn
one draw, in pairs of one silent and one sent in random order, and writes each draw to the trace
(`answer_now_coin`: pair, slot, side). A draw's episode is the one its note opens. The run is
REFUSED if a draw and the requests after it disagree: a silent draw followed by a sent line, or
a sent draw whose first unrested request did not carry it.

THE READING (fixed in results/served-luna-preregistration.log before any run). Over the COMPLETE
pairs of every world given: s and h are the shares of sent and of silent episodes that stopped in
words, and p is the exact two-sided sign test over the pairs whose two episodes differ.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import served_stops as stops  # noqa: E402  (the request states and the door, read one way)

COIN = "answer_now_coin"
OWN_STOPS = ("words_refused", "turn_end")  # a stop the MODEL made, refused or not
#: The registered rule. Shares are exact fractions; points are hundredths.
RULE = {
    "min_pairs": 12,  # complete pairs the reading needs
    "min_pairs_a_run": 3,  # a run with fewer is not read
    "floor": Fraction(30, 100),  # s under this: nothing to reduce
    "gain_ratio": Fraction(1, 2),  # GAIN: h is at most this share of s, and...
    # ...at least this far under it. HARM: h this far OVER s. FLAT: the two are nearer than this.
    "points": Fraction(20, 100),
    "p": 0.05,
}


def read_turns(world: Path) -> list[tuple[str, list[dict]]]:
    """Each turn's trace events, turns in order (turn_2 before turn_10)."""
    files = sorted(
        (world / "logs" / "traces").rglob("turn_*.jsonl"),
        key=lambda p: (str(p.parent), int(re.sub(r"\D", "", p.stem) or 0)),
    )
    turns = []
    for path in files:
        events = []
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        turns.append((f"{path.parent.name}/{path.stem}", events))
    return turns


def _requests(events: list[dict]) -> list[dict]:
    """Every request of one turn: its reminder state, what answered it, whether a refused stop
    had delivered a skill before it in this turn, and the draw (if any) made while building it."""
    usage = [i for i, e in enumerate(events) if e.get("kind") == "usage"]
    rows, governed, start = [], False, 0
    for n, i in enumerate(usage):
        end = usage[n + 1] if n + 1 < len(usage) else len(events)
        # The draw is made while the request is built, so its note sits BEFORE the usage event.
        draws = [
            e.get("data") or {}
            for e in events[start:i]
            if e.get("kind") == "intervention" and (e.get("data") or {}).get("kind") == COIN
        ]
        tools, answer, door = False, None, False
        for event in events[i + 1 : end]:
            if event.get("kind") == "tool":
                tools = True
            elif stops._is_door(event):
                door = True
                if answer is None:
                    answer = "harness_stop_refused" if tools else "words_refused"
            elif event.get("kind") == "stop" and answer is None and not tools:
                answer = "turn_end"
        rows.append(
            {
                "state": stops._state(events[i].get("data") or {}),
                "answer": answer or ("tools" if tools else "other"),
                "governed": governed,
                "draw": draws[-1] if draws else None,
            }
        )
        governed = governed or door
        start = i + 1
    return rows


def read_episodes(turns: list[tuple[str, list[dict]]]) -> list[dict]:
    """Every finished-plan episode of the run, in order."""
    episodes: list[dict] = []
    for name, events in turns:
        current: dict | None = None
        for row in _requests(events):
            finished = row["state"] in ("sent", "silent")
            if row["draw"] is not None or (finished and current is None):
                current = {
                    "turn": name,
                    "governed": row["governed"],
                    "draw": row["draw"],
                    "states": [],
                    "answers": [],
                }
                episodes.append(current)
            elif not finished and row["state"] != "rested":
                current = None
            if current is not None:
                current["states"].append(row["state"])
                current["answers"].append(row["answer"])
    for episode in episodes:
        answers = episode["answers"]
        own = [i for i, a in enumerate(answers) if a in OWN_STOPS]
        episode["stopped_in_words"] = bool(own)
        episode["requests_to_the_stop"] = own[0] + 1 if own else None
        episode["ladder_stop"] = "harness_stop_refused" in answers
    return episodes


def _witness(episode: dict) -> str | None:
    """Why a drawn episode does not show its draw, or None."""
    side = (episode["draw"] or {}).get("side")
    unrested = [s for s in episode["states"] if s != "rested"]
    if not episode["governed"]:
        # K draws only once a refusal has named a skill re-entry in the turn. A draw this reader
        # places before any such refusal means the reader and the build disagree on the scope.
        return "a draw before any refused stop had delivered a skill in its turn"
    if side == "silent" and "sent" in unrested:
        return "a silent draw, and the line was sent"
    if side == "sent" and unrested[:1] != ["sent"]:
        return "a sent draw, and its first unrested request did not carry the line"
    if side not in ("sent", "silent"):
        return f"a draw with no side ({side!r})"
    return None


def _pair_faults(drawn: list[dict]) -> list[str]:
    """What is wrong with the draws AS PAIRS. K numbers pairs from one in each turn and gives a
    pair slot 1, then slot 2 on the other side. Anything else is a build or a reader defect, and
    a pair read through it would not be one silent and one sent plan in random order."""
    faults = []
    by_pair: dict[tuple[str, int], list[dict]] = {}
    for e in drawn:
        by_pair.setdefault((e["turn"], int(e["draw"].get("pair") or 0)), []).append(e["draw"])
    for (turn, pair), draws in sorted(by_pair.items()):
        slots = [d.get("slot") for d in draws]
        sides = [d.get("side") for d in draws]
        if slots != [1, 2][: len(slots)] or len(slots) > 2:
            faults.append(f"{turn} pair {pair}: slots {slots}, not 1 then 2")
        elif len(set(sides)) != len(sides):
            faults.append(f"{turn} pair {pair}: both draws came up {sides[0]}")
    return faults


def read_world(world: Path) -> dict:
    turns = read_turns(world)
    usage = [e for _, events in turns for e in events if e.get("kind") == "usage"]
    if usage and not any("rails" in (e.get("data") or {}) for e in usage):
        return {
            "world": world.name,
            "refused": "no usage event carries `rails`: older than ADR-0204",
        }
    episodes = read_episodes(turns)
    table: Counter[tuple[str, str, bool]] = Counter()
    for e in episodes:
        first = next((s for s in e["states"] if s != "rested"), "rested")
        table[("governed" if e["governed"] else "ungoverned", first, e["stopped_in_words"])] += 1
    drawn = [e for e in episodes if e["draw"] is not None]
    violations = [w for w in map(_witness, drawn) if w] + _pair_faults(drawn)
    by_pair: dict[tuple[str, int], dict[str, dict]] = {}
    for e in drawn:
        by_pair.setdefault((e["turn"], int(e["draw"]["pair"])), {})[e["draw"]["side"]] = e
    pairs = [
        {
            "turn": turn,
            "pair": pair,
            "sent_stopped": sides["sent"]["stopped_in_words"],
            "silent_stopped": sides["silent"]["stopped_in_words"],
        }
        for (turn, pair), sides in sorted(by_pair.items())
        if set(sides) == {"sent", "silent"}
    ]
    return {
        "world": world.name,
        "turns": len(turns),
        "requests": len(usage),
        "episodes": len(episodes),
        "episodes_by_scope_first_state": {
            f"{scope}/{first}": {
                "stopped_in_words": table[(scope, first, True)],
                "went_on": table[(scope, first, False)],
            }
            for scope in ("governed", "ungoverned")
            for first in ("sent", "silent", "rested")
            if table[(scope, first, True)] or table[(scope, first, False)]
        },
        # On which request of its episode each stop in words came (1 = the first request that
        # named the finished plan), over every episode, drawn or not.
        "requests_to_the_stop": sorted(
            e["requests_to_the_stop"] for e in episodes if e["stopped_in_words"]
        ),
        "ladder_stops_in_episodes": sum(e["ladder_stop"] for e in episodes),
        "draws": {
            side: {
                "episodes": sum(e["draw"]["side"] == side for e in drawn),
                "stopped_in_words": sum(
                    e["draw"]["side"] == side and e["stopped_in_words"] for e in drawn
                ),
                "requests_to_the_stop": sorted(
                    e["requests_to_the_stop"]
                    for e in drawn
                    if e["draw"]["side"] == side and e["stopped_in_words"]
                ),
                # Secondaries, deciding nothing: how many requests each episode lasted before the
                # model planned again (or the turn ended), and what answered its FIRST request.
                "requests_in_episode": sorted(
                    len(e["states"]) for e in drawn if e["draw"]["side"] == side
                ),
                "requests_in_episode_that_went_on": sorted(
                    len(e["states"])
                    for e in drawn
                    if e["draw"]["side"] == side and not e["stopped_in_words"]
                ),
                "first_answers": dict(
                    sorted(
                        Counter(e["answers"][0] for e in drawn if e["draw"]["side"] == side).items()
                    )
                ),
            }
            for side in ("sent", "silent")
        },
        "governed_finished_plans_never_drawn": sum(
            e["governed"] and e["draw"] is None for e in episodes
        ),
        "complete_pairs": pairs,
        "unpaired_draws": len(drawn) - 2 * len(pairs),
        "witness_violations": violations,
    }


def sign_test(plus: int, minus: int) -> float:
    """Exact two-sided sign test: of the pairs whose two episodes differ, ``plus`` lean one way
    and ``minus`` the other. With none that differ there is nothing to test: p = 1."""
    k, lead = plus + minus, abs(plus - minus)
    if k == 0:
        return 1.0
    # X = (#plus - #minus) under fair signs; count the outcomes at least as lopsided as seen.
    hits = sum(comb(k, j) for j in range(k + 1) if abs(2 * j - k) >= lead)
    return hits / 2**k


def verdict(s: Fraction, h: Fraction, p: float) -> str:
    """The silent side against the sent side, both as stopped-in-words shares (lower is
    better). Read in this order, so a result that meets GAIN is never called FLAT."""
    if h - s >= RULE["points"] and p < RULE["p"]:
        return "HARM"
    if h <= RULE["gain_ratio"] * s and s - h >= RULE["points"] and p < RULE["p"]:
        return "GAIN"
    if abs(s - h) < RULE["points"]:
        return "FLAT"
    return "MIXED"


def _read_pairs(pairs: list[dict]) -> dict:
    n = len(pairs)
    s = Fraction(sum(p["sent_stopped"] for p in pairs), n)
    h = Fraction(sum(p["silent_stopped"] for p in pairs), n)
    plus = sum(p["sent_stopped"] and not p["silent_stopped"] for p in pairs)
    minus = sum(p["silent_stopped"] and not p["sent_stopped"] for p in pairs)
    p_value = sign_test(plus, minus)
    return {
        "pairs": n,
        "sent_stopped": str(s),
        "silent_stopped": str(h),
        "pairs_only_sent_stopped": plus,
        "pairs_only_silent_stopped": minus,
        "p": round(p_value, 4),
        "verdict": verdict(s, h, p_value),
        "_s": s,
    }


def reading(worlds: list[dict]) -> dict:
    """The registered reading over every world given. A run is left out, and named, if it was
    refused, shows a draw its requests contradict, or holds too few complete pairs."""
    out: dict = {"rule": {k: str(v) if isinstance(v, Fraction) else v for k, v in RULE.items()}}
    read, left_out = [], {}
    for w in worlds:
        if w.get("refused"):
            left_out[w["world"]] = w["refused"]
        elif w["witness_violations"]:
            left_out[w["world"]] = f"{len(w['witness_violations'])} draw(s) contradicted"
        elif len(w["complete_pairs"]) < RULE["min_pairs_a_run"]:
            left_out[w["world"]] = f"{len(w['complete_pairs'])} complete pair(s)"
        else:
            read.append(w)
    out["runs_read"], out["runs_left_out"] = [w["world"] for w in read], left_out
    pairs = [p for w in read for p in w["complete_pairs"]]
    out["pairs"] = len(pairs)
    if len(pairs) < RULE["min_pairs"]:
        out["batch"] = "NOT MEASURED"
        return out
    whole = _read_pairs(pairs)
    sent_share = whole.pop("_s")
    # A GAIN or a HARM must keep its SIZE with any one pair left out (the p is not re-taken:
    # losing a pair costs any test power and says nothing about whether one pair carried it).
    rests_on = []
    if whole["verdict"] in ("GAIN", "HARM"):
        for i in range(len(pairs)):
            rest = pairs[:i] + pairs[i + 1 :]
            s = Fraction(sum(p["sent_stopped"] for p in rest), len(rest))
            h = Fraction(sum(p["silent_stopped"] for p in rest), len(rest))
            if verdict(s, h, whole["p"]) != whole["verdict"]:
                rests_on.append(f"{pairs[i]['turn']}#{pairs[i]['pair']}")
    if rests_on:
        whole["verdict"] = "MIXED"
    # The floor: where the sent side itself seldom stops there is nothing for silence to reduce,
    # so no GAIN, FLAT or MIXED is read there. A HARM is: silence can still make things worse.
    if sent_share < RULE["floor"] and whole["verdict"] != "HARM":
        out["batch"] = "NOT DISCRIMINATING"
        return out | {k: whole[k] for k in ("sent_stopped", "silent_stopped", "p")}
    return out | {"batch": "READ", "rests_on": rests_on} | whole


# ── known answers ────────────────────────────────────────────────────────────────────────────────


def _use(state: str) -> dict:
    data: dict = {"rails": [], "rails_silenced": [], "rails_rested": False}
    if state == "sent":
        data["rails"] = ["plan_complete"]
    elif state == "silent":
        data["rails_silenced"] = ["plan_complete"]
    elif state == "open":
        data["rails"] = ["plan"]
    elif state == "rested":
        data["rails_rested"] = True
    return {"kind": "usage", "data": data}


_TOOL = {"kind": "tool", "detail": "Bash"}
_DOOR = {"kind": "intervention", "data": {"kind": "turn_end_skill", "skill": "loop"}}


def _coin(pair: int, slot: int, side: str) -> dict:
    return {
        "kind": "intervention",
        "data": {"kind": COIN, "pair": pair, "slot": slot, "side": side},
    }


def _write(root: Path, name: str, turns: list[list[dict]]) -> Path:
    world = root / name
    traces = world / "logs" / "traces" / "s1"
    traces.mkdir(parents=True)
    for n, events in enumerate(turns, 1):
        (traces / f"turn_{n}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return world


def _episode(side: str, pair: int, slot: int, stops_in_words: bool) -> list[dict]:
    """An open plan, then the finished one with its draw. A sent episode that stops is refused
    and its plan goes silent (ADR-0205); a silent one that stops is refused the same way."""
    events = [_use("open"), _TOOL, _coin(pair, slot, side), _use(side)]
    if stops_in_words:
        events += [_DOOR, _use("silent"), _TOOL]
    else:
        events += [_TOOL]
    return events


def selftest() -> int:
    results: list[bool] = []

    def check(name: str, got: object, want: object) -> None:
        ok = got == want
        results.append(ok)
        print(
            f"  {'ok  ' if ok else 'FAIL'} {name}{'' if ok else f' -- got {got!r}, want {want!r}'}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # One turn: an UNGOVERNED finished plan that stops (the first door), then two pairs. Pair
        # 1: sent stops, silent goes on. Pair 2: both stop. Then a draw with no partner.
        head = [_use("open"), _TOOL, _use("sent"), _DOOR, _use("silent"), _TOOL]
        turn = [
            *head,
            *_episode("sent", 1, 1, True),
            *_episode("silent", 1, 2, False),
            *_episode("silent", 2, 1, True),
            *_episode("sent", 2, 2, True),
            *_episode("sent", 3, 1, False),
            {"kind": "stop"},
        ]
        got = read_world(_write(root, "k", [turn]))
        check("requests", got["requests"], 16)
        check("episodes (one ungoverned, five drawn)", got["episodes"], 6)
        check(
            "the ungoverned plan was sent its line and stopped",
            got["episodes_by_scope_first_state"].get("ungoverned/sent"),
            {"stopped_in_words": 1, "went_on": 0},
        )
        check(
            "sent draws: 3 episodes, 2 stopped",
            (got["draws"]["sent"]["episodes"], got["draws"]["sent"]["stopped_in_words"]),
            (3, 2),
        )
        check(
            "silent draws: 2 episodes, 1 stopped",
            (got["draws"]["silent"]["episodes"], got["draws"]["silent"]["stopped_in_words"]),
            (2, 1),
        )
        check(
            "two complete pairs, one draw unpaired",
            (len(got["complete_pairs"]), got["unpaired_draws"]),
            (2, 1),
        )
        check(
            "pair 1: only the sent episode stopped",
            (got["complete_pairs"][0]["sent_stopped"], got["complete_pairs"][0]["silent_stopped"]),
            (True, False),
        )
        check("no draw is contradicted", got["witness_violations"], [])
        check(
            "every stop came on its episode's first request", got["requests_to_the_stop"], [1] * 4
        )
        check(
            "a stop came on the first request of each stopped episode",
            got["draws"]["sent"]["requests_to_the_stop"],
            [1, 1],
        )

        check(
            "sent episodes lasted 1, 2 and 2 requests; silent ones 1 and 2",
            (
                got["draws"]["sent"]["requests_in_episode"],
                got["draws"]["silent"]["requests_in_episode"],
            ),
            ([1, 2, 2], [1, 2]),
        )
        check(
            "the episodes that went on lasted 1 request on each side",
            (
                got["draws"]["sent"]["requests_in_episode_that_went_on"],
                got["draws"]["silent"]["requests_in_episode_that_went_on"],
            ),
            ([1], [1]),
        )
        check(
            "what answered each sent episode's first request",
            got["draws"]["sent"]["first_answers"],
            {"tools": 1, "words_refused": 2},
        )

        # The same pair in two TURNS is two pairs: pair numbers restart with the turn.
        two = read_world(
            _write(
                root,
                "k2",
                [
                    [*head, *_episode("sent", 1, 1, True), *_episode("silent", 1, 2, False)],
                    [*head, *_episode("silent", 1, 1, False), *_episode("sent", 1, 2, True)],
                ],
            )
        )
        check("pair 1 of turn 1 and pair 1 of turn 2 are two pairs", len(two["complete_pairs"]), 2)

        # A draw its requests contradict: silent drawn, the line sent.
        bad = read_world(
            _write(
                root,
                "bad",
                [[*head, _use("open"), _TOOL, _coin(1, 1, "silent"), _use("sent"), _TOOL]],
            )
        )
        check("a silent draw with the line sent is a violation", len(bad["witness_violations"]), 1)
        late = read_world(
            _write(
                root,
                "late",
                [
                    [
                        *head,
                        _use("open"),
                        _TOOL,
                        _coin(1, 1, "sent"),
                        _use("rested"),
                        _TOOL,
                        _use("sent"),
                        _TOOL,
                    ]
                ],
            )
        )
        check(
            "a sent draw whose first request rested, then carried the line, is sound",
            late["witness_violations"],
            [],
        )

        # A rested request inside an episode belongs to it: a stop made there is the episode's.
        rest = read_world(
            _write(
                root,
                "rest",
                [
                    [
                        *head,
                        _use("open"),
                        _TOOL,
                        _coin(1, 1, "sent"),
                        _use("sent"),
                        _TOOL,
                        _use("rested"),
                        _DOOR,
                        _use("silent"),
                        _TOOL,
                    ]
                ],
            )
        )
        check(
            "a stop on a rested request inside an episode is that episode's, on its request 2",
            (rest["draws"]["sent"]["stopped_in_words"], rest["requests_to_the_stop"]),
            (1, [1, 2]),
        )

        # Draws that are not a pair of one silent and one sent plan, slot 1 then slot 2.
        twice = read_world(
            _write(
                root,
                "twice",
                [[*head, *_episode("sent", 1, 1, True), *_episode("sent", 1, 2, True)]],
            )
        )
        check(
            "a pair whose two draws came up the same side is a violation, and is no pair",
            (len(twice["witness_violations"]), len(twice["complete_pairs"])),
            (1, 0),
        )
        order = read_world(
            _write(
                root,
                "order",
                [[*head, *_episode("sent", 1, 2, True), *_episode("silent", 1, 1, False)]],
            )
        )
        check("a pair drawn slot 2 then slot 1 is a violation", len(order["witness_violations"]), 1)
        early = read_world(
            _write(root, "early", [[_use("open"), _TOOL, *_episode("silent", 1, 1, False)]])
        )
        check(
            "a draw before any refused stop delivered a skill is a violation",
            len(early["witness_violations"]),
            1,
        )

        # The stuck ladder's stop is not the model's.
        ladder = read_world(
            _write(
                root,
                "ladder",
                [
                    [
                        *head,
                        _use("open"),
                        _TOOL,
                        _coin(1, 1, "sent"),
                        _use("sent"),
                        _TOOL,
                        _DOOR,
                        _use("silent"),
                    ]
                ],
            )
        )
        check(
            "a stop after tool events is the ladder's, not a stop in words",
            (ladder["draws"]["sent"]["stopped_in_words"], ladder["ladder_stops_in_episodes"]),
            (0, 1),
        )

        # Words that END the turn, which no hook refused, are the model's stop too.
        ended = read_world(
            _write(
                root,
                "ended",
                [[*head, _use("open"), _TOOL, _coin(1, 1, "sent"), _use("sent"), {"kind": "stop"}]],
            )
        )
        check(
            "words that ended the turn are a stop in words",
            ended["draws"]["sent"]["stopped_in_words"],
            1,
        )

        # A build with no draws is described and never read.
        plain = read_world(
            _write(
                root,
                "plain",
                [[*head, _use("open"), _TOOL, _use("sent"), _DOOR, _use("silent"), _TOOL]],
            )
        )
        check(
            "an undrawn run: a governed plan that was sent its line and stopped",
            plain["episodes_by_scope_first_state"].get("governed/sent"),
            {"stopped_in_words": 1, "went_on": 0},
        )
        check(
            "  and it counts as a governed plan never drawn",
            plain["governed_finished_plans_never_drawn"],
            1,
        )
        old = read_world(_write(root, "old", [[{"kind": "usage", "data": {"cost_usd": 0.1}}]]))
        check("a trace older than ADR-0204 is refused", "refused" in old, True)

    print("the sign test (worked by hand: k pairs all one way give 2 / 2**k)")
    check("8 of 8 one way", sign_test(8, 0), 2 / 2**8)
    check("5 of 5 one way cannot reach 0.05", sign_test(5, 0), 2 / 2**5)
    check("6 against 1", sign_test(6, 1), 16 / 2**7)
    check("no pair differs", sign_test(0, 0), 1.0)
    check("even", sign_test(3, 3), 1.0)

    print("the reading")

    def read(worlds: list[dict]) -> dict:
        """The reading, or the name of what it raised: a crash is a failed check, by name."""
        try:
            return reading(worlds)
        except Exception as exc:  # noqa: BLE001
            return {"raised": type(exc).__name__}

    def world_of(name: str, rows: list[tuple[bool, bool]]) -> dict:
        return {
            "world": name,
            "witness_violations": [],
            "complete_pairs": [
                {"turn": "t", "pair": i, "sent_stopped": a, "silent_stopped": b}
                for i, (a, b) in enumerate(rows)
            ],
        }

    gain = read([world_of("a", [(True, False)] * 6), world_of("b", [(True, False)] * 6)])
    check(
        "12 pairs, the sent side always stops, the silent never: GAIN",
        (gain.get("batch"), gain.get("verdict"), gain.get("p")),
        ("READ", "GAIN", round(2 / 2**12, 4)),
    )
    check(
        "  and every pair is one where only the SENT episode stopped",
        (gain.get("pairs_only_sent_stopped"), gain.get("pairs_only_silent_stopped")),
        (12, 0),
    )
    flat = read([world_of("a", [(True, True)] * 12)])
    check(
        "both sides always stop: FLAT", (flat.get("batch"), flat.get("verdict")), ("READ", "FLAT")
    )
    half = read([world_of("a", [(True, False)] * 5 + [(True, True)] * 7)])
    check(
        "5 of 12 pairs differ: the p is out of reach, MIXED",
        (half.get("verdict"), half.get("p")),
        ("MIXED", round(2 / 2**5, 4)),
    )
    harm = read([world_of("a", [(False, True)] * 8 + [(True, True)] * 4)])
    check("the silent side stops where the sent side did not: HARM", harm.get("verdict"), "HARM")
    check(
        "  and its pairs are ones where only the SILENT episode stopped",
        (harm.get("pairs_only_sent_stopped"), harm.get("pairs_only_silent_stopped")),
        (0, 8),
    )
    low = read([world_of("a", [(False, True)] * 10 + [(False, False)] * 2)])
    check(
        "a HARM is read even where the sent side never stops (under the floor)",
        (low.get("batch"), low.get("verdict")),
        ("READ", "HARM"),
    )
    few = read([world_of("a", [(True, False)] * 11)])
    check("11 pairs is NOT MEASURED", few.get("batch"), "NOT MEASURED")
    calm = read([world_of("a", [(False, False)] * 9 + [(True, False)] * 3)])
    check(
        "the sent side stops 3 times in 12: NOT DISCRIMINATING",
        calm.get("batch"),
        "NOT DISCRIMINATING",
    )
    thin = read([world_of("a", [(True, False)] * 12), world_of("b", [(True, False)] * 2)])
    check(
        "a run with two complete pairs is left out, and named",
        (thin.get("pairs"), list(thin.get("runs_left_out") or [])),
        (12, ["b"]),
    )
    wrong = read([world_of("a", [(True, False)] * 12) | {"witness_violations": ["x"]}])
    check("a run with a contradicted draw is left out", wrong.get("batch"), "NOT MEASURED")
    check(
        "a threshold met exactly is met (h is half of s, 20 points under it)",
        (verdict(Fraction(1), Fraction(1, 2), 0.03), verdict(Fraction(2, 5), Fraction(1, 5), 0.03)),
        ("GAIN", "GAIN"),
    )
    check(
        "exactly 20 points apart is not FLAT; 19 points is",
        (
            verdict(Fraction(1), Fraction(4, 5), 0.5),
            verdict(Fraction(1), Fraction(81, 100), 0.5),
        ),
        ("MIXED", "FLAT"),
    )
    check(
        "a GAIN or a HARM needs p under 0.05, not at it",
        (verdict(Fraction(1), Fraction(0), 0.05), verdict(Fraction(0), Fraction(1), 0.05)),
        ("MIXED", "MIXED"),
    )
    edge = read([world_of("a", [(True, False)] * 6 + [(True, True)] * 6)])
    check(
        "but a GAIN that any one pair can undo is MIXED, and the pairs are named",
        (edge.get("verdict"), len(edge.get("rests_on") or [])),
        ("MIXED", 6),
    )
    frail = read([world_of("a", [(False, True)] * 6 + [(True, True)] * 24)])
    check(
        "and so is a HARM: 30 pairs, exactly 20 points over, any one of 6 pairs undoes it",
        (frail.get("verdict"), len(frail.get("rests_on") or [])),
        ("MIXED", 6),
    )
    firm = read([world_of("a", [(True, False)] * 8 + [(True, True)] * 4)])
    check(
        "8 of 12 pairs differ, all one way: GAIN, resting on no pair",
        (firm.get("verdict"), firm.get("rests_on"), firm.get("p")),
        ("GAIN", [], round(2 / 2**8, 4)),
    )
    every = Counter()
    for plus in range(13):
        for minus in range(13 - plus):
            for both in range(13 - plus - minus):
                rows = (
                    [(True, False)] * plus
                    + [(False, True)] * minus
                    + [(True, True)] * both
                    + [(False, False)] * (12 - plus - minus - both)
                )
                every[read([world_of("a", rows)]).get("verdict", "unread")] += 1
    check(
        "every table of 12 pairs lands in one named branch",
        set(every) <= {"GAIN", "HARM", "FLAT", "MIXED", "unread"},
        True,
    )
    print(f"    ({dict(every)})")
    failed = results.count(False)
    print(f"selftest: {len(results) - failed} ok, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    if not sys.argv[1:]:
        print(__doc__)
        return 2
    worlds = [read_world(Path(arg)) for arg in sys.argv[1:]]
    print(json.dumps({"worlds": worlds, "reading": reading(worlds)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
