#!/usr/bin/env python3
"""The veto-door bench: what a small model does after a refused stop, under each candidate build.

THE DOOR. A Stop hook refuses the stop and names a skill re-entry ("your FIRST action MUST be
Skill('cycle') with args='loop'"). The harness delivers that skill itself (ADR-0187) and the model
gets the next completion. Measured on served gpt-5.6-luna at reasoning effort none: it never went
straight to work, it called the skill tool again about half the time (and got the "[already
loaded]" pointer, ADR-0063), and it answered in words the rest. Words end the turn, the hook
refuses again, and three such refusals with no skill run end the turn ``veto_stall`` behind a
600 s wake-up. One served run spent 73% of its wall clock in those rests.

WHAT IS MEASURED. One segment: from the harness's delivery at a refused stop to the first of
(a) the loop RESUMED: one of the loop's own commands (``scripts/*.sh``) ran and succeeded,
(b) the next refused stop or the turn's own end, (c) a cap on completions. Success is read off
the product's own tool result for that call (a call the product withheld or refused is an error
result, as a failed command is), with the product's work counter kept beside it. Reading a file
is not resuming and an ``echo`` is not work: both keep the segment open, neither closes it. A
single completion cannot carry this: the product's first-load rendering of a long body asks for a
plan FIRST, and the product nudges after some text-only completions, so "the next completion is a
work call" mis-scores exactly the arms under test.

HOW. Capture and fork. A capture is one live run of the real product (the served agent factory's
posture: zakpick on luna, autonomous, lean rules, skills, rules, compaction, streamed turns) in the
synthesized workspace of ``veto_door_world``, recorded call by call and stopped at the first
delivery: the fork point. A rollout rebuilds that workspace at the same path, runs the product
again with every recorded completion served from the tape (no network, and checked message by
message against what was recorded), and goes live from the fork. Every arm therefore starts from
the same conversation, tool state, plan, fence count and hook budget, because the product rebuilt
them itself.

ARMS ARE BUILDS. An arm is this repository at one commit plus the patches under
``veto_door_arms/``, checked out as a worktree and run with ``PYTHONPATH`` on its ``src``. Nothing
is switched at run time and nothing is monkeypatched: what is measured is code that could ship.
Each row carries the build it ran (the imported package's path, HEAD, the hash of its source
diff) and a witness only that build can produce (the door answer it gave, the ``tool_choice`` it
sent), so a row from the wrong tree cannot pass as its arm.

WHAT NEVER LEAVES THE MACHINE. Tapes, request snapshots and completion text go to ``--out`` (a
scratch directory). The ledger carries labels, counts, hashes and usage only. The wire hook
records the URL path and four body fields, never a header.

Subcommands: ``selftest`` (offline, the real Agent on a scripted model), ``preflight`` (offline,
what each category would send), ``arms`` (build the worktrees), ``capture`` / ``rollouts``
(orchestrators, live), ``capture-one`` / ``rollout-one`` (one run, in the arm's own tree),
``probe`` (one live call: does a forced tool choice reach the wire and get accepted), ``report``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import fcntl
import functools
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import veto_door_world as world  # noqa: E402  (a sibling, importable once HERE is on the path)

# ── the served configuration, as the results log already records it ─────────────────────────────

LUNA = "gpt-5.6-luna"
#: Every task category on luna with its served window; classification on the small model.
ROUTING: dict[str, dict[str, Any]] = {
    **{
        category: {"model": LUNA, "source": "openai", "context_window": 163_840}
        for category in ("deep_code", "plan", "quick_code", "summarize", "delegate")
    },
    "classify": {"model": "gpt-5-nano", "source": "openai"},
}
FALLBACK = "openai/gpt-5-mini"

#: A capture that has not stopped after this many iterations is a loop that kept working: the
#: healthy outcome, recorded as such, and not a fork point.
CAPTURE_MAX_ITERATIONS = 60
#: Live completions one rollout may spend before it is scored ``cap``.
LIVE_CAP = 6
#: Iteration bound for a rollout: the replayed prefix counts, so it sits well above both.
ROLLOUT_MAX_ITERATIONS = 200
#: Per-run cost ceilings handed to the product (its own budget stop). The bench's spend bound is
#: the sum the orchestrators enforce across runs.
CAPTURE_COST_CAP = 0.40
ROLLOUT_COST_CAP = 0.60

#: The arms and the patches each one is built from (``veto_door_arms/<name>.patch``). A letter
#: in an arm's name is a patch it carries, which is how the selftest knows what to expect of it.
ARMS: dict[str, list[str]] = {
    "A": [],  # this build
    "A2": [],  # the placebo: this build again, read against A like any arm (see ``report``)
    "R": ["r"],  # a finished plan's "answer now" reminder is silent once a stop was refused on it
    "P": ["p"],  # a finished plan leaves the board when a stop is refused (R, and the board too)
    "C": ["c"],  # the pointer carries the skill's first step
    "F": ["f"],  # the engine binds: a tool call is required until the model has acted
    "B": ["b"],  # the body again at the veto door (the fallback the served log named in advance)
    "RC": ["r", "c"],
    "RF": ["r", "f"],
}

#: The arm that is the baseline under another name. It can never be selected or shipped.
PLACEBO = "A2"

# ── scoring one completion (pure; known answers in the selftest) ─────────────────────────────────

#: The loop's own commands. A segment is a PASS when one of these ran and succeeded.
RESUME_LABELS = ("first-step", "script")
#: Labels that count as the model acting at all (the secondary reading). ``trivial`` is a work
#: call to the product and is deliberately not one here: an ``echo`` after a refused stop is the
#: spin, not the work.
WORK_LABELS = (*RESUME_LABELS, "other-work")
#: The harness's own notes (trace interventions) a row tallies. Five of them are written by one
#: patch each, so they say which build ran: a field only that build can write.
NOTE_KINDS = (
    "turn_end_skill",
    "skill_pointer",
    "skill_reentry",
    "plan_first",
    "veto_plan_silenced",  # R
    "veto_plan_retired",  # P
    "veto_forced_tool",  # F
    "skill_redelivered",  # B
)
#: Best label first: a completion carrying several calls is labelled by the best of them.
LABEL_ORDER = (
    "first-step",
    "script",
    "other-work",
    "trivial",
    "plan",
    "skill-other",
    "skill-again",
    "wakeup",
    "text-only",
)

_TRIVIAL_WORDS = frozenset({"echo", "printf", "true", ":", "sleep", "pwd", "date", "whoami"})
#: Any mention of one of them, run or not: what the audit counts beside the labels.
_NAMED_RE = re.compile("|".join(re.escape(name) for name in sorted(world.SCRIPTS)))
_SHELLS = frozenset({"bash", "sh", "dash", "zsh"})
#: Words that run whatever follows them: shell keywords, and commands that wrap a command
#: (not ``command`` itself: ``command -v x`` only looks ``x`` up).
_PREFIX_WORDS = frozenset(
    {"if", "then", "elif", "else", "do", "while", "until", "!", "{", "}"}
    | {"env", "time", "nohup", "exec", "nice", "stdbuf"}
)
_ASSIGN_RE = re.compile(r"[A-Za-z_]\w*=.*", re.DOTALL)
_DURATION_RE = re.compile(r"\d+(?:\.\d+)?[smhd]?")


#: One of the loop's scripts as a PATH: its file name under any directory, and nothing else in
#: the word (``X='bash scripts/x.sh'`` is one shell word that ends in a script's name, and so is
#: the redirection ``>scripts/x.sh``).
_LOOP_SCRIPT_RE = re.compile(rf"(?:[^\s=<>]*/)?(?:{_NAMED_RE.pattern})")


def _is_loop_script(word: str) -> bool:
    """Is this word one of the loop's scripts, by file name? ``scripts/x.sh``, ``./x.sh`` after
    a ``cd scripts``, an absolute path: a wrong path costs nothing, because a command that does
    not find its script fails, and only a call that SUCCEEDED counts as resuming."""
    return _LOOP_SCRIPT_RE.fullmatch(word) is not None


def _simple_commands(command: str) -> list[str]:
    """The simple commands a shell line STARTS, in order.

    It splits where the shell does (``;``, ``&&``, ``||``, ``|``, a new line, the opening of a
    subshell or of a ``$(...)``) and nowhere else: not inside quotes (a ``$(...)`` inside double
    quotes still runs, so it still opens one), not inside a comment, and not past a
    here-document's ``<<``, where what follows is a file's text. What comes after a closing
    ``)`` continues the command around it (``cat $(pwd)/scripts/x.sh``) and starts nothing.

    Every choice here errs one way: a rollout ENDS at the first successful call read as running
    a loop script, so a mention read as a run is a wrong PASS nothing can take back, while a run
    read as a mention only lets the rollout go on (a loop that resumed runs its next script, and
    the tape keeps the command for a re-score). Backticks, ``case`` arms and function bodies
    are left unread for that reason."""
    command = command.replace("\\\n", "")  # a line continuation joins two lines, as in the shell
    out: list[str] = []
    current: list[str] = []
    starts = True  # does ``current`` sit where a command starts?
    quote = ""  # the quote we are inside: "", "'" or '"'
    saved: list[str] = []  # the quote state around each open parenthesis
    i, n = 0, len(command)

    def close(next_starts: bool) -> None:
        nonlocal current, starts
        if starts and "".join(current).strip():
            out.append("".join(current))
        current, starts = [], next_starts

    while i < n:
        ch, two = command[i], command[i : i + 2]
        if quote == "'":  # nothing is special inside single quotes
            quote = "" if ch == "'" else quote
        elif ch == "\\" and i + 1 < n:
            current.append(two)
            i += 2
            continue
        elif two == "$(":  # runs inside double quotes too
            saved.append(quote)
            quote = ""
            close(True)
            i += 2
            continue
        elif quote == '"':
            quote = "" if ch == '"' else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or command[i - 1].isspace()):
            newline = command.find("\n", i)
            i = n if newline < 0 else newline
            continue
        elif command[i : i + 3] == "<<<":  # a here-string is an argument, not a document
            current.append("<<<")
            i += 3
            continue
        elif two == "<<":
            break
        elif two in ("&&", "||"):
            close(True)
            i += 2
            continue
        elif ch in ";|\n(":
            if ch == "(":
                saved.append(quote)
            close(True)
            i += 1
            continue
        elif ch == ")":
            quote = saved.pop() if saved else ""
            close(False)
            i += 1
            continue
        current.append(ch)
        i += 1
    close(True)
    return out


def _words(simple: str) -> list[str]:
    """One simple command as words, quotes honoured."""
    try:
        return shlex.split(simple)
    except ValueError:  # an unbalanced quote: the model's, not ours; read it loosely
        return simple.split()


def _scripts_run(command: str) -> list[str]:
    """The loop scripts a shell command RUNS, as opposed to names: ``bash scripts/x.sh``,
    ``./scripts/x.sh``, ``source scripts/x.sh``, the same behind ``timeout 60``, ``env``, ``if``
    or a ``VAR=value`` prefix, and the same inside ``bash -c "..."`` or ``$(...)``.
    ``cat scripts/x.sh`` names a script and runs nothing, and ``bash -n`` only checks syntax:
    looking at the loop's commands is not resuming the loop."""
    ran: list[str] = []
    for simple in _simple_commands(command):
        words = _words(simple)
        while words and not _is_loop_script(words[0]):
            head = words[0].rsplit("/", 1)[-1]
            if head == "timeout":  # its options and its durations go with it
                words.pop(0)
                while words and (words[0].startswith("-") or _DURATION_RE.fullmatch(words[0])):
                    words.pop(0)
            elif head in _PREFIX_WORDS or _ASSIGN_RE.fullmatch(words[0]):
                words.pop(0)
                while words and words[0].startswith("-"):  # ``env -i``
                    words.pop(0)
            else:
                break
        if not words:
            continue
        head = words[0].rsplit("/", 1)[-1]
        if _is_loop_script(words[0]):
            ran.append(words[0])
        elif head in _SHELLS:
            options = list(itertools.takewhile(lambda word: word.startswith("-"), words[1:]))
            rest = words[1 + len(options) :]
            flags = "".join(option[1:] for option in options if not option.startswith("--"))
            if "n" in flags or not rest:
                continue
            if "c" in flags:
                ran += _scripts_run(rest[0])
            elif _is_loop_script(rest[0]):
                ran.append(rest[0])
        elif head in ("source", ".") and len(words) > 1 and _is_loop_script(words[1]):
            ran.append(words[1])
    return ran


def _command(arguments: dict[str, Any]) -> str:
    """The command line of a shell call, under either spelling of its argument."""
    return str(arguments.get("command") or arguments.get("cmd") or "")


def _shell_label(command: str) -> str:
    """Label one shell command: the loop's first step, another of its scripts, other work, or
    a command that does nothing (every simple command in it is an echo, a sleep, and so on)."""
    ran = _scripts_run(command)
    first = world.FIRST_STEP.rsplit("/", 1)[-1]
    if any(script.rsplit("/", 1)[-1] == first for script in ran):
        return "first-step"
    if ran:
        return "script"
    heads = [_words(simple)[:1] for simple in _simple_commands(command)]
    real = [head[0] for head in heads if head and head[0] != "cd"]
    if real and all(word in _TRIVIAL_WORDS for word in real):
        return "trivial"
    return "other-work"


def named_not_run(calls: list[dict[str, Any]], labels: list[str]) -> int:
    """How many of a completion's shell calls NAME a loop script without running one: the
    count a reader audits the labels by (the commands themselves stay in the local tape)."""
    return sum(
        1
        for call, label in zip(calls, labels, strict=True)
        if label not in RESUME_LABELS
        and _NAMED_RE.search(_command(dict(call.get("arguments") or {})))
    )


@functools.cache
def _canonicalizer() -> Callable[[str], str]:
    """Tool names the way the loop canonicalizes them before anything reads them (ADR-0190):
    the registry's aliases first (``TodoWrite`` is the plan tool), then the pre-0190 spellings
    for a name the registry does not hold (the skill tool is registered per agent)."""
    from zakcode.tool_names import canonical_tool_name
    from zakcode.tools.builtins.default_registry import default_registry

    registry = default_registry()

    def canonical(name: str) -> str:
        resolved = registry.canonical(name)
        if resolved == name and registry.get(resolved) is None:
            return canonical_tool_name(name)
        return resolved

    return canonical


def call_label(name: str, arguments: dict[str, Any]) -> str:
    """Label one tool call on the PRODUCT's own sets (plan, skill and wake-up tools are not work;
    names are canonicalized the way the loop does it, so an alias cannot change a label)."""
    from zakcode.agent.loop import _PLAN_TOOLS, _SKILL_TOOLS, _WAKEUP_TOOLS

    canonical = _canonicalizer()(name)
    if canonical in _SKILL_TOOLS:
        asked = str(arguments.get("skill") or arguments.get("name") or "").strip().lstrip("/")
        return "skill-again" if asked.lower() == world.SKILL else "skill-other"
    if canonical in _PLAN_TOOLS:
        return "plan"
    if canonical in _WAKEUP_TOOLS:
        return "wakeup"
    if canonical in ("Bash", "PowerShell"):
        return _shell_label(_command(arguments))
    return "other-work"


def completion_label(calls: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """The completion's label (the best of its calls, ``text-only`` for none) and each call's."""
    labels = [call_label(str(c.get("name", "")), dict(c.get("arguments") or {})) for c in calls]
    if not labels:
        return "text-only", []
    return min(labels, key=LABEL_ORDER.index), labels


# ── fingerprints: is a replayed request the recorded one? ────────────────────────────────────────


def _message_view(message: Any) -> tuple[list[Any], list[Any]]:
    """What the model sees of one message (strict) and its skeleton (shape). Timestamps and the
    structured ``data`` a tool result carries are the product's bookkeeping, not the prompt."""
    strict: list[Any] = [message.role]
    shape: list[Any] = [message.role]
    for block in message.blocks:
        kind = block.type
        if kind == "tool_use":
            strict.append([kind, block.id, block.name, block.input])
            shape.append([kind, block.id, block.name])
        elif kind == "tool_result":
            strict.append([kind, block.tool_use_id, block.output, block.is_error])
            shape.append([kind, block.tool_use_id])
        else:
            strict.append([kind, getattr(block, "text", "")])
            shape.append([kind])
    return strict, shape


def fingerprint(
    messages: list[Any], system: str | None, tools: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """``(strict, shape)`` hashes of one request. ``shape`` (roles, block kinds, tool names and
    call ids) must match on replay or the run has left the recorded trajectory; ``strict`` adds
    every character the model is shown, and a mismatch there is counted and reported."""
    names = sorted(str((t.get("function") or {}).get("name", "")) for t in tools or [])
    views = [_message_view(m) for m in messages]

    def digest(payload: Any) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    return (
        digest([system or "", names, [v[0] for v in views]]),
        digest([names, [v[1] for v in views]]),
    )


def _tail_view(messages: list[Any], last: int = 3) -> list[dict[str, Any]]:
    """The last few messages of a request, without their content: role, size, and the tag of
    a harness rail (``[plan]``, ``[harness]``: the product's own words, which open with one).
    The plan reminder is EPHEMERAL: it rides the request and is never in the session file, so
    this is the only place a run records that it was there, and in which of its two forms."""
    view = []
    for message in messages[-last:]:
        text = "".join(
            getattr(b, "text", "") or getattr(b, "output", "") or "" for b in message.blocks
        )
        head = text.lstrip()
        tag = head[: head.index("]") + 1] if head.startswith("[") and "]" in head[:40] else None
        if tag == "[plan]":
            tag = "[plan:complete]" if head.startswith("[plan] Plan complete") else "[plan:open]"
        view.append({"role": message.role, "chars": len(text), "rail": tag})
    return view


# ── the tape: record a run, replay its prefix, go live after it ──────────────────────────────────

#: Set while a live call is in flight, so a provider whose ``astream`` is built on its own
#: ``acomplete`` (the base class's default) is recorded once, not twice.
_INSIDE: contextvars.ContextVar[bool] = contextvars.ContextVar("veto_door_inside", default=False)

Decide = Callable[["Tape", dict[str, Any], list[Any], "str | None", "list[dict[str, Any]] | None"],
                  "str | None"]  # fmt: skip


class Tape:
    """Every provider call of one run, in order.

    With ``fork_at=None`` every call is live and recorded (a capture). With ``fork_at=N`` calls
    ``0..N-1`` are answered from ``entries`` without touching the provider, each checked against
    the recorded request, and calls from ``N`` on are live (a rollout). ``decide`` sees every
    live call before it is made and may end the run by returning a reason: the tape then raises
    ``CancelledError``, the loop's own control signal, which it persists behind and re-raises.
    """

    def __init__(
        self,
        *,
        entries: list[dict[str, Any]] | None = None,
        fork_at: int | None = None,
        decide: Decide | None = None,
    ) -> None:
        self.entries = list(entries or [])
        self.fork_at = fork_at
        self.decide: Decide = decide or (lambda *_: None)
        self.seq = 0
        self.recorded: list[dict[str, Any]] = []
        self.stop: str | None = None
        self.diverged: str | None = None
        self.strict_mismatches = 0
        self.replayed = 0

    def _begin(
        self,
        method: str,
        provider: Any,
        messages: list[Any],
        tools: list[dict[str, Any]] | None,
        system: str | None,
        kw: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        seq, self.seq = self.seq, self.seq + 1
        strict, shape = fingerprint(messages, system, tools)
        choice = kw.get("tool_choice")
        request: dict[str, Any] = {
            "seq": seq,
            "method": method,
            "model": provider.model_id(),
            "strict": strict,
            "shape": shape,
            "aux": not tools,  # a call with no tools is never the main loop's (classify, judge)
            "n_messages": len(messages),
            "tool_choice": choice if isinstance(choice, str | type(None)) else "function",
            "reasoning_effort": kw.get("reasoning_effort"),
            "tail": _tail_view(messages),
        }
        if self.fork_at is not None and seq < self.fork_at:
            return request, self._recorded(seq, request)
        verdict = self.decide(self, request, messages, system, tools)
        if verdict is not None:
            self.stop = verdict
            raise asyncio.CancelledError(f"veto-door bench: {verdict}")
        # The entry exists from the moment the call starts, so a call the loop abandons
        # mid-stream still holds its place: entries are found by position on replay.
        request["complete"] = False
        self.recorded.append(request)
        return request, None

    @staticmethod
    def _end(entry: dict[str, Any], started: float, **payload: Any) -> None:
        label, labels = completion_label(payload["calls"])
        entry.update(payload, label=label, labels=labels, complete=True)
        entry["latency_s"] = round(time.monotonic() - started, 2)

    def _recorded(self, seq: int, request: dict[str, Any]) -> dict[str, Any]:
        """The tape's entry for replayed call ``seq``, checked against the request in hand."""
        entry = self.entries[seq]
        wrong = [k for k in ("method", "model", "shape") if entry[k] != request[k]]
        if wrong or not entry.get("complete"):
            self.diverged = f"call {seq}: {', '.join(wrong) or 'the recorded call never finished'}"
            raise asyncio.CancelledError(self.diverged)
        self.strict_mismatches += entry["strict"] != request["strict"]
        self.replayed += 1
        return entry

    async def complete(self, provider: Any, original: Any, messages: list[Any], **kw: Any) -> Any:
        if _INSIDE.get():
            return await original(provider, messages, **kw)
        from zakcode.providers.base import LLMResult

        tools, system = kw.get("tools"), kw.get("system")
        rest = {k: v for k, v in kw.items() if k not in ("tools", "system")}
        entry, recorded = self._begin("acomplete", provider, messages, tools, system, rest)
        if recorded is not None:
            return LLMResult.model_validate(recorded["result"])
        started = time.monotonic()
        token = _INSIDE.set(True)
        try:
            result = await original(provider, messages, **kw)
        finally:
            _INSIDE.reset(token)
        self._end(
            entry,
            started,
            result=result.model_dump(mode="json"),
            calls=[
                {"id": c.id, "name": c.name, "arguments": dict(c.arguments)}
                for c in result.tool_calls
            ],
            text_chars=len((result.text or "").strip()),
            usage=result.usage.model_dump(mode="json"),
        )
        return result

    async def stream(
        self, provider: Any, original: Any, messages: list[Any], **kw: Any
    ) -> AsyncIterator[Any]:
        if _INSIDE.get():
            async for event in original(provider, messages, **kw):
                yield event
            return
        from pydantic import TypeAdapter

        from zakcode.providers.base import ProviderStreamEvent

        tools, system = kw.get("tools"), kw.get("system")
        rest = {k: v for k, v in kw.items() if k not in ("tools", "system")}
        entry, recorded = self._begin("astream", provider, messages, tools, system, rest)
        if recorded is not None:
            adapter: TypeAdapter[Any] = TypeAdapter(ProviderStreamEvent)
            for raw in recorded["events"]:
                yield adapter.validate_python(raw)
            return
        started = time.monotonic()
        events: list[dict[str, Any]] = []
        source = original(provider, messages, **kw).__aiter__()
        while True:
            # The mark is held only while the provider's own code runs (where a stream built on
            # ``acomplete`` would re-enter the tape), never across a ``yield``: the loop stops
            # reading at the ``done`` event, and a generator it has stopped reading never
            # reaches its ``finally``.
            token = _INSIDE.set(True)
            try:
                event = await source.__anext__()
            except StopAsyncIteration:
                break
            finally:
                _INSIDE.reset(token)
            events.append(event.model_dump(mode="json"))
            if events[-1].get("event") == "done":
                self._finish_stream(entry, started, events)
            yield event
        if not entry["complete"]:
            self._finish_stream(entry, started, events)

    def _finish_stream(
        self, entry: dict[str, Any], started: float, events: list[dict[str, Any]]
    ) -> None:
        text, calls, usage = _fold_stream(events)
        self._end(
            entry, started, events=events, calls=calls, text_chars=len(text.strip()), usage=usage
        )


def _fold_stream(events: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Text, tool calls and usage of one streamed completion, the way the loop assembles them:
    text deltas joined, call fragments gathered by index, the arguments parsed once complete."""
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    for event in events:
        kind = event.get("event")
        if kind == "text_delta":
            text.append(str(event.get("text", "")))
        elif kind == "tool_call_delta":
            slot = calls.setdefault(int(event["index"]), {"id": "", "name": "", "raw": ""})
            slot["id"] = event.get("id") or slot["id"]
            slot["name"] = event.get("name") or slot["name"]
            slot["raw"] += event.get("arguments_delta") or ""
        elif kind == "usage":
            usage = dict(event.get("usage") or {})
    folded = []
    for _, slot in sorted(calls.items()):
        try:
            arguments = json.loads(slot["raw"]) if slot["raw"].strip() else {}
        except ValueError:
            arguments = {"_raw": slot["raw"]}
        folded.append(
            {
                "id": slot["id"],
                "name": slot["name"],
                "arguments": arguments if isinstance(arguments, dict) else {"_raw": slot["raw"]},
            }
        )
    return "".join(text), folded, usage


def install(provider_class: Any, tape: Tape) -> Callable[[], None]:
    """Route every ``acomplete`` / ``astream`` of ``provider_class`` through ``tape``, for every
    instance (zakpick builds one provider per category). Returns the undo."""
    original_complete, original_stream = provider_class.acomplete, provider_class.astream

    async def acomplete(self: Any, messages: list[Any], **kw: Any) -> Any:
        return await tape.complete(self, original_complete, messages, **kw)

    async def astream(self: Any, messages: list[Any], **kw: Any) -> AsyncIterator[Any]:
        async for event in tape.stream(self, original_stream, messages, **kw):
            yield event

    provider_class.acomplete, provider_class.astream = acomplete, astream

    def undo() -> None:
        provider_class.acomplete, provider_class.astream = original_complete, original_stream

    return undo


# ── the wire: what actually left the process (path and four body fields, never a header) ────────

WIRE: list[dict[str, Any]] = []


def install_wire_hook() -> None:
    import httpx

    def note(request: Any) -> None:
        row: dict[str, Any] = {"path": request.url.path}
        try:
            body = json.loads(request.content.decode("utf-8")) if request.content else {}
        except (ValueError, UnicodeDecodeError):
            body = {}
        if isinstance(body, dict):
            reasoning = body.get("reasoning")
            row["effort"] = (
                reasoning.get("effort") if isinstance(reasoning, dict) else None
            ) or body.get("reasoning_effort")
            choice = body.get("tool_choice")
            row["tool_choice"] = choice if isinstance(choice, str | type(None)) else "function"
            row["n_tools"] = len(body.get("tools") or [])
            row["model"] = body.get("model")
        WIRE.append(row)

    sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def send(self: Any, request: Any, **kw: Any) -> Any:
        note(request)
        return sync_send(self, request, **kw)

    async def asend(self: Any, request: Any, **kw: Any) -> Any:
        note(request)
        return await async_send(self, request, **kw)

    httpx.Client.send = send  # type: ignore[method-assign]
    httpx.AsyncClient.send = asend  # type: ignore[method-assign]


# ── builds and workspaces ────────────────────────────────────────────────────────────────────────


def _git(*args: str, cwd: Path = REPO) -> str:
    done = subprocess.run(  # noqa: S603  (fixed argv, no shell)
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return done.stdout


def build_identity() -> dict[str, Any]:
    """Which zakcode this process imported, and what tree it is. ``own_tree`` is the gate: a
    rollout whose package is not the ``src`` beside this file is measuring some other build
    (an installed copy shadows a worktree unless ``PYTHONPATH`` says otherwise)."""
    import zakcode

    imported = Path(zakcode.__file__).resolve()
    diff = _git("diff", "HEAD", "--", "src")
    return {
        "zakcode": str(imported),
        "own_tree": imported == (REPO / "src" / "zakcode" / "__init__.py").resolve(),
        "head": _git("rev-parse", "HEAD").strip(),
        "src_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else "clean",
    }


def _assert_clear_above(base: Path) -> None:
    """The product folds project guides from the workspace up to its repository root. A
    workspace inside somebody's checkout would carry that checkout's guide into every prompt
    (and to the provider), so the base must sit under no repository and no guide."""
    from zakcode.agent.prompt import AGENT_GUIDE_FILENAMES

    for ancestor in [base, *base.parents]:
        for marker in (".git", ".hg", ".svn", *AGENT_GUIDE_FILENAMES):
            if (ancestor / marker).exists():
                raise SystemExit(f"{base}: {ancestor / marker} is above it; pick another --base")


def _registry(base: Path) -> Path:
    return base / ".veto-door"


def make_workspace(base: Path, *, at: Path | None = None) -> Path:
    """Build the world in a fresh directory under ``base`` (or again at ``at``, a path this
    bench registered earlier: the only directories it will ever remove)."""
    base.mkdir(parents=True, exist_ok=True)
    _assert_clear_above(base)
    _registry(base).mkdir(exist_ok=True)
    if at is None:
        at = Path(tempfile.mkdtemp(prefix="w", dir=base))
        (_registry(base) / at.name).write_text(world.digest() + "\n", encoding="utf-8")
    else:
        if at.parent != base or not (_registry(base) / at.name).is_file():
            raise SystemExit(f"{at} is not a workspace this bench created under {base}")
        shutil.rmtree(at, ignore_errors=True)
    world.build(at)
    return at


def served_agent(
    workspace: Path,
    *,
    max_iterations: int,
    cost_cap: float,
    provider: Any = None,
    session: dict[str, str] | None = None,
):
    """The served agent factory's posture (``server/app.py``), explicit field by field so that
    no ``.env`` and no stray variable can change what is measured. ``provider`` is the
    selftest's scripted model; with one injected the product routes nothing."""
    from zakcode import Agent
    from zakcode.config import Settings
    from zakcode.session.store import Session, SessionStore

    # ZAKCODE_HOME is the bench's own; ZAKCODE_SESSION is the harness marker every Agent exports.
    ours = ("ZAKCODE_HOME", "ZAKCODE_SESSION")
    stray = sorted(k for k in os.environ if k.startswith("ZAKCODE_") and k not in ours)
    if stray:
        raise SystemExit(f"unset {stray}: the bench fixes its own configuration")
    routed: dict[str, Any] = (
        {"default_model": "scripted/test", "context_window": 163_840}
        if provider is not None
        else {"default_model": "zakpick", "zakpick_models": ROUTING, "fallback_model": FALLBACK}
    )
    settings = Settings(
        workspace_root=workspace,
        permission_mode="autonomous",
        lean_rules=True,
        max_cost_usd=cost_cap,
        **routed,
    )
    # A rollout takes its capture's session id: the id is in the system prompt and in the
    # workspace survey, so a fresh one would make every replayed request differ from the
    # recorded one by two lines (and would scatter the provider's prompt cache across keys).
    # A capture lets the Agent open its own session, as the served factory's caller does.
    pinned = (
        Session(cwd=str(workspace), id=session["id"], model=session["model"]) if session else None
    )
    return Agent(
        settings=settings,
        provider=provider,
        session=pinned,
        session_store=SessionStore.for_workspace(workspace),
        enable_skills=True,
        enable_rules=True,
        enable_compaction=True,
        lean_rules=True,
        max_iterations=max_iterations,
    )


def _main_steps(tape: Tape) -> list[dict[str, Any]]:
    """The main loop's finished live completions: not the tool-less side calls (difficulty
    classification, a judge), and not a call the loop abandoned mid-stream."""
    return [e for e in tape.recorded if e["complete"] and not e["aux"]]


def deliveries(agent: Any) -> int:
    """How many refused stops the harness has answered by delivering the skill, this turn."""
    return sum(
        1
        for e in agent.loop._trace.events
        if e.kind == "intervention"
        and e.data.get("kind") == "turn_end_skill"
        and not e.data.get("refused")
    )


def notes(agent: Any) -> dict[str, int]:
    """How often the harness wrote each of ``NOTE_KINDS`` into this run's trace."""
    seen = Counter(
        str(e.data.get("kind")) for e in agent.loop._trace.events if e.kind == "intervention"
    )
    return {kind: seen[kind] for kind in NOTE_KINDS if seen[kind]}


def _call_results(agent: Any) -> dict[str, tuple[bool, str]]:
    """Tool-call id -> (the product answered it with an error result, the result's text). A call
    the product withheld (its plan-first gate) or refused gets an error result as a failed
    command does, so ``False`` here means the call ran and succeeded."""
    from zakcode.messages import ToolResultBlock

    return {
        block.tool_use_id: (bool(block.is_error), block.output or "")
        for message in agent.session.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    }


def settle(agent: Any, tape: Tape) -> dict[str, Any] | None:
    """Write down what the last finished live completion came to (once): which of its calls
    succeeded and which results show a loop script's own output (kept call by call, so a
    corrected scorer can be re-applied to a finished rollout), the labels that stand, and the
    product's work counter after it. Returns that completion.

    A call READ as running a loop script stands as one only if its result shows a line that
    script prints when it runs. The command's text cannot say whether it ran: ``false && bash
    scripts/x.sh; echo done`` reads as a run, runs nothing, and exits 0. Without that line the
    call is what it demonstrably was, a command that succeeded: ``other-work``."""
    done = _main_steps(tape)
    if not done:
        return None
    entry = done[-1]
    if "ok_labels" not in entry:
        answered = _call_results(agent)
        results = [answered.get(call["id"]) for call in entry["calls"]]
        entry["call_ok"] = [r is not None and not r[0] for r in results]
        entry["call_ran"] = [r is not None and bool(world.RAN_RE.search(r[1])) for r in results]
        entry["ok_labels"] = [
            label if ran or label not in RESUME_LABELS else "other-work"
            for label, ok, ran in zip(
                entry["labels"], entry["call_ok"], entry["call_ran"], strict=True
            )
            if ok
        ]
        entry["work_after"] = agent.loop.work_calls()
    return entry


async def _run_turn(agent: Any) -> str:
    """Open the loop the way a typed ``/cycle loop`` does and stream the turn, as the served
    path does. Returns the stop reason, or ``cancelled`` when the tape ended the run."""
    invocation = await agent.compose_skill_turn(world.SKILL, world.SKILL_ARGS)
    if not invocation.turn_text:
        raise SystemExit(f"/{world.SKILL} did not compose: {invocation}")
    stop = "no-done-event"
    try:
        async for event in agent.astream_turn(invocation.turn_text):
            if getattr(event, "event", "") == "done":
                stop = event.stop_reason
    except asyncio.CancelledError:
        stop = "cancelled"
    return stop


def _usage_sum(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total: Counter[str] = Counter()
    for row in rows:
        usage = row.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "cache_read_tokens", "reasoning_tokens"):
            total[key] += int(usage.get(key) or 0)
        total["cost_microusd"] += round(float(usage.get("cost_usd") or 0.0) * 1_000_000)
    return {**total, "cost_usd": total["cost_microusd"] / 1_000_000}


def _public(entry: dict[str, Any]) -> dict[str, Any]:
    """One recorded call without its content: what a committed ledger may carry."""
    keep = ("seq", "method", "model", "aux", "complete", "tool_choice", "reasoning_effort", "tail",
            "label", "labels", "call_ok", "call_ran", "ok_labels", "text_chars", "latency_s",
            "work_before",
            "work_after")  # fmt: skip
    row = {k: entry[k] for k in keep if k in entry}
    usage = entry.get("usage") or {}
    row["served_model"] = usage.get("model")
    row["prompt_tokens"] = usage.get("prompt_tokens")
    row["cache_read_tokens"] = usage.get("cache_read_tokens")
    row["reasoning_tokens"] = usage.get("reasoning_tokens")
    return row


def _first_live(row: dict[str, Any]) -> dict[str, Any]:
    """A rollout row's fork request: the first live call of the main loop (a row's ``steps``
    are its live calls only; the replayed prefix is never recorded twice)."""
    return next((s for s in row.get("steps", []) if not s.get("aux")), {})


def _append(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON line under a lock, opening the file for this one write: children of one
    batch share a ledger, and a file held open across a long run is what a sync layer orphans."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ── one capture ──────────────────────────────────────────────────────────────────────────────────


async def capture_one(
    base: Path, out: Path, *, provider: Any = None, provider_class: Any = None
) -> dict[str, Any]:
    """Run the product live in a fresh workspace until the harness has delivered the skill at
    the first refused stop, and keep the tape. A run that never stops is recorded as that."""
    workspace = make_workspace(base)
    os.environ["ZAKCODE_HOME"] = str(_registry(base) / f"home-{workspace.name}")
    os.chdir(workspace)
    holder: dict[str, Any] = {}

    def decide(tape: Tape, request: dict[str, Any], messages: list[Any], *_: Any) -> str | None:
        if request["aux"] or deliveries(holder["agent"]) < 1:
            return None
        network = holder["agent"].session.task_network
        holder["fork"] = {
            "seq": request["seq"],
            "strict": request["strict"],
            "shape": request["shape"],
            "n_messages": request["n_messages"],
            "tail": request["tail"],
            "plan": {"complete": network.is_complete(), "progress": list(network.progress())},
            "work_calls": holder["agent"].loop.work_calls(),
        }
        holder["request"] = [m.model_dump(mode="json") for m in messages]
        return "fork"

    tape = Tape(decide=decide)
    undo = install(provider_class or _litellm_class(), tape)
    WIRE.clear()
    started = time.time()
    try:
        agent = served_agent(
            workspace,
            max_iterations=CAPTURE_MAX_ITERATIONS,
            cost_cap=CAPTURE_COST_CAP,
            provider=provider,
        )
        holder["agent"] = agent
        stop = await _run_turn(agent)
        with contextlib.suppress(Exception):
            await agent.aclose()
    finally:
        undo()
    fork = holder.get("fork")
    target = out / "captures" / workspace.name
    target.mkdir(parents=True, exist_ok=True)
    facts = {
        "capture": workspace.name,
        "workspace": str(workspace),
        "session": {"id": agent.session.id, "model": agent.session.model},
        "world_digest": world.digest(),
        "build": build_identity(),
        "outcome": "fork" if fork else f"no-stop:{stop}",
        "fork": fork,
        "calls": len(tape.recorded),
        "labels": [e["label"] for e in _main_steps(tape)],
        "usage": _usage_sum(tape.recorded),
        "wire": list(WIRE),
        "seconds": round(time.time() - started, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (target / "capture.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    (target / "tape.json").write_text(json.dumps(tape.recorded), encoding="utf-8")
    if fork:
        (target / "fork-request.json").write_text(json.dumps(holder["request"]), encoding="utf-8")
    return facts


def _litellm_class() -> Any:
    from zakcode.providers.litellm_provider import LiteLLMProvider

    return LiteLLMProvider


# ── one rollout ──────────────────────────────────────────────────────────────────────────────────


async def rollout_one(
    base: Path,
    capture_dir: Path,
    arm: str,
    rep: int,
    *,
    provider: Any = None,
    provider_class: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the capture's workspace where it stood, replay its tape to the fork, go live.

    Returns ``(row, detail)``: the ledger row (no content) and the local-only detail (the live
    completions as recorded). The segment's outcome:

    ``pass``            the loop resumed: one of its own commands ran and succeeded
    ``stopped-again``   the model ended in words and the hook refused the stop again
    ``cap``             ``LIVE_CAP`` completions and the loop had not resumed (a plan or skill
                        spiral, or reading without end)
    ``ended:<reason>``  the turn ended by itself
    ``invalid:<why>``   the replay left the recorded trajectory, or the build is not its own
    """
    facts = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
    entries = json.loads((capture_dir / "tape.json").read_text(encoding="utf-8"))
    fork = facts["fork"]
    workspace = make_workspace(base, at=Path(facts["workspace"]))
    os.environ["ZAKCODE_HOME"] = str(_registry(base) / f"home-{workspace.name}")
    os.chdir(workspace)
    identity = build_identity()
    holder: dict[str, Any] = {}

    def resumed(tape: Tape) -> bool:
        last = settle(holder["agent"], tape)
        return last is not None and any(label in RESUME_LABELS for label in last["ok_labels"])

    def decide(tape: Tape, request: dict[str, Any], *_: Any) -> str | None:
        if request["aux"]:
            return None
        agent = holder["agent"]
        request["work_before"] = agent.loop.work_calls()
        if not _main_steps(tape):
            holder["fork_identical"] = request["strict"] == fork["strict"]
            holder["fence_at_fork"] = agent.loop._vetoes_without_skill
        if resumed(tape):
            return "pass"
        if deliveries(agent) >= 2:
            return "stopped-again"
        if len(_main_steps(tape)) >= LIVE_CAP:
            return "cap"
        return None

    tape = Tape(entries=entries, fork_at=fork["seq"], decide=decide)
    undo = install(provider_class or _litellm_class(), tape)
    WIRE.clear()
    started = time.time()
    try:
        agent = served_agent(
            workspace,
            max_iterations=ROLLOUT_MAX_ITERATIONS,
            cost_cap=ROLLOUT_COST_CAP,
            provider=provider,
            session=facts["session"],
        )
        holder["agent"] = agent
        stop = await _run_turn(agent)
        with contextlib.suppress(Exception):
            await agent.aclose()
    finally:
        undo()

    if not identity["own_tree"]:
        outcome = "invalid:not-own-tree"
    elif tape.diverged:
        outcome = "invalid:diverged"
    elif tape.stop:
        outcome = tape.stop
    elif resumed(tape):
        outcome = "pass"
    else:
        outcome = f"ended:{stop}"
    live = _main_steps(tape)
    path = [e["label"] for e in live]

    def first_with(labels: tuple[str, ...]) -> int | None:
        """The 1-based live completion on which a call with one of ``labels`` first succeeded."""
        hits = (i for i, e in enumerate(live, 1) if set(e.get("ok_labels", ())) & set(labels))
        return next(hits, None)

    row = {
        "capture": facts["capture"],
        "arm": arm,
        "rep": rep,
        "outcome": outcome,
        "pass": outcome == "pass",
        "path": path,
        "pass_at": first_with(RESUME_LABELS) if outcome == "pass" else None,
        "any_work_at": first_with(WORK_LABELS),
        "named_not_run": sum(named_not_run(e["calls"], e["labels"]) for e in live),
        "run_unproven": sum(
            ok and not ran and label in RESUME_LABELS
            for e in live
            for label, ok, ran in zip(
                e["labels"], e.get("call_ok", ()), e.get("call_ran", ()), strict=False
            )
        ),
        "notes": notes(agent),
        "plan_complete_at_fork": bool((fork.get("plan") or {}).get("complete")),
        "first_act": path[0] if path else None,
        "after_door": [b for a, b in itertools.pairwise(path) if a == "skill-again"],
        "doors": _door_answers(agent, live),
        "steps": [_public(e) for e in tape.recorded],
        "fence_at_fork": holder.get("fence_at_fork"),
        "fence_at_end": agent.loop._vetoes_without_skill,
        "deliveries": deliveries(agent),
        "fork_seq": fork["seq"],
        "replayed": tape.replayed,
        "strict_mismatches": tape.strict_mismatches,
        "fork_identical": holder.get("fork_identical"),
        "diverged": tape.diverged,
        "usage": _usage_sum(tape.recorded),
        "wire": list(WIRE),
        "build": identity,
        "world_digest": world.digest(),
        "seconds": round(time.time() - started, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    detail = {"capture": facts["capture"], "arm": arm, "rep": rep, "live": tape.recorded}
    return row, detail


def _door_answers(agent: Any, live: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What the skill door answered each time the model asked for the loop skill again: the
    witness of which build ran. A pointer is what the PRODUCT flags as one (ADR-0203); arm
    C's carries the first step; arm B's is the whole body.

    Batch 1 asked the text instead — does the answer OPEN with the pointer's tag — and the
    product frames a call's arguments ahead of that tag, so the flag read false on 137 door
    answers of 137 and arm C's witness never fired."""
    from zakcode.messages import ToolResultBlock

    asked = {
        call["id"]
        for entry in live
        for call, label in zip(entry["calls"], entry["labels"], strict=True)
        if label == "skill-again"
    }
    answers = []
    for message in agent.session.messages:
        for block in message.blocks:
            if isinstance(block, ToolResultBlock) and block.tool_use_id in asked:
                output = block.output or ""
                answers.append(
                    {
                        "chars": len(output),
                        "pointer": bool((block.data or {}).get("pointer")),
                        "first_step": world.FIRST_STEP in output,
                        "body": len(output) > len(world.loop_skill_body()) // 2,
                    }
                )
    return answers


# ── the offline proof: the real Agent, the real hook, a scripted model ──────────────────────────


def _scripted_class() -> Any:
    from zakcode.providers.base import Capabilities, LLMResult, Provider
    from zakcode.usage import Usage

    class Scripted(Provider):
        """Answers from a list, one entry per completion, and remembers what it was handed."""

        def __init__(self, script: list[LLMResult]) -> None:
            self.script = list(script)
            self.seen: list[dict[str, Any]] = []

        async def acomplete(self, messages: list[Any], **kw: Any) -> LLMResult:
            self.seen.append({k: v for k, v in kw.items() if k not in ("tools", "system")})
            if not self.script:
                return LLMResult(text="Nothing further.", usage=Usage(total_tokens=1))
            return self.script.pop(0)

        def count_tokens(self, messages: list[Any], *, system: str | None = None) -> int:
            return 1_000

        def capabilities(self) -> Capabilities:
            return Capabilities(supports_tools=True, context_window=163_840)

        def model_id(self) -> str:
            return "scripted/test"

    return Scripted


def _say(text: str) -> Any:
    from zakcode.providers.base import LLMResult
    from zakcode.usage import Usage

    return LLMResult(text=text, usage=Usage(total_tokens=1))


def _do(tool: str, call_id: str, **arguments: Any) -> Any:
    from zakcode.providers.base import LLMResult, ToolCall
    from zakcode.usage import Usage

    return LLMResult(
        tool_calls=[ToolCall(id=call_id, name=tool, arguments=arguments)],
        usage=Usage(total_tokens=1),
    )


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{(' -- ' + detail) if detail and not ok else ''}")
    return ok


#: The scorer's two-sided check. Commands that NAME a loop script and run none must never read as
#: a run (a rollout ends at the first successful run it sees, so that mistake cannot be re-scored);
#: the plain ways of running one must always read as a run; and the forms the scorer leaves unread
#: on purpose are listed so the selftest shows them reading as mentions, which is the safe side.
_NAMES_ONLY = (
    "cat scripts/cycle-open.sh",
    "less scripts/cycle-open.sh",
    "wc -l scripts/*.sh scripts/cycle-open.sh",
    "ls -la scripts/cycle-open.sh",
    "grep -rn 'bash scripts/cycle-open.sh' .",
    'grep -n "bash scripts/cycle-open.sh" SKILL.md; echo ok',
    'echo "run: bash scripts/cycle-open.sh"',
    "printf '%s\\n' 'bash scripts/cycle-open.sh'",
    "git log --oneline -- scripts/cycle-open.sh",
    "test -f scripts/cycle-open.sh && echo yes",
    "stat scripts/cycle-open.sh | head -3",
    "bash -n scripts/cycle-open.sh && echo ok",
    "sh -n scripts/cycle-close.sh",
    "cat $(pwd)/scripts/cycle-open.sh",
    "cat `pwd`/scripts/cycle-open.sh",
    "cat <<EOF\nbash scripts/cycle-open.sh\nEOF",
    "tee notes.txt <<'X'\nnext; bash scripts/cycle-open.sh\nX",
    "echo hi # ; bash scripts/cycle-open.sh",
    "# bash scripts/cycle-open.sh\nls",
    "sed -n '1,5p' scripts/cycle-open.sh",
    "diff scripts/cycle-open.sh scripts/cycle-close.sh",
    "cp scripts/cycle-open.sh /tmp/x.sh",
    "echo 'a; bash scripts/cycle-open.sh; b'",
    'echo "a && bash scripts/cycle-open.sh && b"',
    'echo "a | bash scripts/cycle-open.sh"',
    'echo "(bash scripts/cycle-open.sh)"',
    "echo '$(bash scripts/cycle-open.sh)'",
    "which bash; type scripts/cycle-open.sh",
    "bash --version; cat scripts/cycle-open.sh",
    "export X=scripts/cycle-open.sh; echo $X",
    "X='bash scripts/cycle-open.sh'; echo \"$X\"",
    "find . -name 'cycle-open.sh' -exec cat {} \\;",
    "cat scripts/cycle-open.sh | head -20",
    "cd scripts && cat cycle-open.sh",
    "bash -c 'echo bash scripts/cycle-open.sh'",
    "bash -c \"echo 'x; bash scripts/cycle-open.sh'\"",
    "echo \\; bash\\ scripts/cycle-open.sh",
    "true # && bash scripts/cycle-open.sh",
    "python3 -c \"print('bash scripts/cycle-open.sh')\"",
    "bash scripts/my-own-idea.sh",
    "command -v scripts/cycle-open.sh",
    "echo x >scripts/cycle-open.sh",
    ": >scripts/cycle-open.sh; >scripts/cycle-close.sh",
    "exec 3< scripts/cycle-open.sh",
    "while read l; do echo $l; done < scripts/cycle-open.sh",
    "diff <(cat scripts/cycle-open.sh) /tmp/x",
    "alias go='bash scripts/cycle-open.sh'",
    "[ -f scripts/cycle-open.sh ] && echo present",
)
_PLAIN_RUNS = (
    "bash scripts/cycle-open.sh",
    "cd /w && bash scripts/cycle-open.sh",
    "./scripts/cycle-open.sh",
    "sh scripts/cycle-open.sh",
    "bash scripts/cycle-open.sh 2>&1 | tail -n 20",
    "timeout 120 bash scripts/cycle-open.sh",
    "timeout -k 5 60 bash scripts/cycle-open.sh",
    'bash scripts/cycle-open.sh; echo "exit=$?"',
    'out=$(bash scripts/cycle-open.sh 2>&1); echo "$out"',
    'echo "$(bash scripts/cycle-open.sh)"',
    "(cd /w && bash scripts/cycle-open.sh)",
    'bash -c "bash scripts/cycle-open.sh"',
    "bash -lc 'cd /w && bash scripts/cycle-open.sh'",
    "set -e; bash scripts/cycle-open.sh",
    "bash -x scripts/cycle-open.sh",
    "bash ./scripts/cycle-open.sh",
    "bash /srv/zc-door/w1/scripts/cycle-open.sh",
    "source scripts/cycle-open.sh",
    ". scripts/cycle-open.sh",
    "exec bash scripts/cycle-open.sh",
    "time bash scripts/cycle-open.sh",
    "nohup bash scripts/cycle-open.sh &",
    "/usr/bin/env bash scripts/cycle-open.sh",
    "bash scripts/cycle-open.sh > /tmp/o.txt 2>&1; cat /tmp/o.txt",
    "bash scripts/cycle-open.sh <<< ''",
    "FOO=1 BAR=2 bash scripts/cycle-open.sh",
    "{ bash scripts/cycle-open.sh; }",
    "bash scripts/cycle-open.sh || echo failed",
    "echo start && bash scripts/cycle-open.sh && echo done",
    "bash scripts/cycle-open.sh # open the cycle",
    "if true; then bash scripts/cycle-open.sh; fi",
    "timeout 60 \\\n  bash scripts/cycle-open.sh",
    "cd scripts && ./cycle-open.sh",
)
_LEFT_UNREAD = (
    'bash "$(pwd)/scripts/cycle-open.sh"',
    "OUT=`bash scripts/cycle-open.sh`; echo $OUT",
    "cat scripts/cycle-open.sh | bash",
    "bash -euo pipefail scripts/cycle-open.sh",
    "sudo -u x bash scripts/cycle-open.sh",
    "case x in x) bash scripts/cycle-open.sh ;; esac",
    "f() { bash scripts/cycle-open.sh; }; f",
)


def _scripts_known_answers(check: Callable[..., None]) -> None:
    """``world.RAN_RE`` against the real scripts under a real shell: every one of the eight shows
    its line when it runs (and when it dies), none shows one when it is only read, and a
    command that reads as a run but runs nothing shows none either."""
    from zakcode._subprocess import find_bash

    bash = find_bash()
    check("there is a shell to run the scripts with", bash is not None)
    if bash is None:
        return
    root = Path(tempfile.mkdtemp(prefix="veto-door-scripts-"))
    shutil.rmtree(root)
    world.build(root)

    def run(command: str) -> str:
        done = subprocess.run(  # noqa: S603  (a fixed shell, the bench's own commands)
            [bash, "-c", command],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            check=False,
        )
        return done.stdout

    try:
        selected = run("bash scripts/cycle-open.sh >/dev/null && bash scripts/cycle-select.sh")
        lot = json.loads(selected.splitlines()[-1])
        delta = lot["recount"] - lot["ledger"]
        walk = {
            "cycle-open.sh": "bash scripts/cycle-close.sh >/dev/null; bash scripts/cycle-open.sh",
            "check-ledger.sh": "bash scripts/check-ledger.sh",
            "check-queue.sh": "bash scripts/check-queue.sh",
            "cycle-select.sh": "bash scripts/cycle-select.sh",
            "correct.sh": f"bash scripts/correct.sh {lot['lot']} {delta:+d} RC-01",
            "validate.sh": "bash scripts/validate.sh",
            "cycle-record.sh": f"bash scripts/cycle-record.sh {lot['lot']} corrected",
            "cycle-close.sh": "bash scripts/cycle-close.sh",
        }
        check("the walk covers every script", sorted(walk) == sorted(world.SCRIPTS))
        for name, command in walk.items():
            output = run(command)
            check(f"{name} shows its line when it runs", bool(world.RAN_RE.search(output)), output)
        died = run("bash scripts/cycle-open.sh; bash scripts/cycle-open.sh | tail -1")
        check("a script that dies behind a pipe still shows it ran", "E105" in died
              and bool(world.RAN_RE.search(died.splitlines()[-1])), died)  # fmt: skip
        source = run("cat scripts/*.sh .claude/skills/*/SKILL.md CLAUDE.md; grep -rh PASS scripts/")
        check("reading the scripts, the skills and the guide shows no such line",
              len(source) > 50_000 and not world.RAN_RE.search(source))  # fmt: skip
        skipped = run("false && bash scripts/cycle-close.sh; echo done")
        check("a run that never happened shows none", _shell_label("false && bash "
              "scripts/cycle-close.sh; echo done") == "script"
              and not world.RAN_RE.search(skipped), skipped)  # fmt: skip
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _known_answers() -> list[tuple[list[dict[str, Any]], str]]:
    """The scorer's known answers: one completion each, with the label it must get."""

    def call(tool: str, /, **arguments: Any) -> dict[str, Any]:
        return {"name": tool, "arguments": arguments}

    return [
        ([], "text-only"),
        ([call("Bash", command="bash scripts/cycle-open.sh")], "first-step"),
        ([call("bash", command="cd /w && bash scripts/check-queue.sh")], "script"),
        ([call("Bash", command="echo resuming; sleep 1")], "trivial"),
        ([call("Bash", command="cat queue/lots.tsv | head")], "other-work"),
        # Naming a script is not running it: reading the loop's commands is not resuming.
        ([call("Bash", command="cat scripts/cycle-open.sh")], "other-work"),
        (
            [call("Bash", command="sed -n 1,20p scripts/cycle-select.sh && ls scripts/")],
            "other-work",
        ),
        ([call("Bash", command="grep -n open scripts/cycle-open.sh; echo done")], "other-work"),
        ([call("Bash", command='bash -c "cat scripts/check-queue.sh"')], "other-work"),
        ([call("Bash", command="./scripts/cycle-open.sh")], "first-step"),
        (
            [call("Bash", command="timeout 60 bash scripts/cycle-open.sh 2>&1 | tail -5")],
            "first-step",
        ),
        (
            [call("Bash", command="(cd /w && LOG=1 bash /w/scripts/cycle-record.sh L-0001 held)")],
            "script",
        ),
        ([call("Bash", command='bash -c "bash scripts/check-queue.sh"')], "script"),
        ([call("Bash", command="bash scripts/cycle-select.sh; echo exit=$?")], "script"),
        ([call("Bash", command="timeout -k 5 60 bash scripts/cycle-select.sh")], "script"),
        ([call("Bash", command="/usr/bin/env bash scripts/validate.sh")], "script"),
        ([call("Bash", command="bash -lc 'cd /w && bash scripts/cycle-open.sh'")], "first-step"),
        (
            [call("Bash", command='OUT=$(bash scripts/cycle-open.sh 2>&1); echo "$OUT"')],
            "first-step",
        ),
        ([call("Bash", command='echo "$(bash scripts/cycle-open.sh)"')], "first-step"),
        (
            [
                call(
                    "Bash",
                    command="if bash scripts/check-queue.sh; then ./scripts/cycle-open.sh; fi",
                )
            ],
            "first-step",
        ),
        (
            [call("Bash", command="[ -x scripts/correct.sh ] && scripts/correct.sh L-0002")],
            "script",
        ),
        ([call("Bash", command="cd scripts && ./cycle-open.sh")], "first-step"),
        ([call("Bash", command="timeout 60 \\\n  bash scripts/cycle-open.sh")], "first-step"),
        # ... and every way of naming one without running it, which must never read as a run:
        # a rollout ends on the first run it sees, so this side of the scorer cannot be re-scored.
        ([call("Bash", command="cd scripts && cat cycle-open.sh")], "other-work"),
        ([call("Bash", command="bash scripts/my-own-idea.sh")], "other-work"),
        ([call("Bash", command="bash -n scripts/cycle-open.sh")], "other-work"),
        ([call("Bash", command="cat $(pwd)/scripts/cycle-open.sh")], "other-work"),
        ([call("Bash", command='head -5 "$(pwd)/scripts/cycle-open.sh"')], "other-work"),
        ([call("Bash", command="chmod +x scripts/cycle-open.sh && ls -la scripts/")], "other-work"),
        ([call("Bash", command="ls scripts/ # then: bash scripts/cycle-open.sh")], "other-work"),
        (
            [call("Bash", command="cat <<'EOF' > run.sh\nbash scripts/cycle-open.sh\nEOF")],
            "other-work",
        ),
        (
            [
                call(
                    "Bash", command="python3 -c \"print('x'); print('bash scripts/cycle-open.sh')\""
                )
            ],
            "other-work",
        ),
        ([call("Bash", command="echo 'next: bash scripts/cycle-open.sh'")], "trivial"),
        (
            [call("Bash", command='echo "plan: open; bash scripts/cycle-open.sh now" >> notes.md')],
            "trivial",
        ),
        ([call("Bash", command="echo next && echo bash scripts/cycle-open.sh")], "trivial"),
        ([call("Read", path="queue/lots.tsv")], "other-work"),
        ([call("update_plan", tasks=[])], "plan"),
        ([call("TodoWrite", todos=[])], "plan"),
        ([call("Skill", skill="cycle", args="loop")], "skill-again"),
        ([call("use_skill", name="cycle-check")], "skill-other"),
        ([call("ScheduleWakeup")], "wakeup"),
        (
            [call("update_plan", tasks=[]), call("Bash", command="bash scripts/cycle-select.sh")],
            "script",
        ),
    ]


def _suffixes(retired: bool) -> dict[str, tuple[list[Any], str, list[str]]]:
    """Scripted continuations from the fork: what the model does, the outcome, the path.

    Every passing script plans before it touches the workspace, as the product's plan-first gate
    asks of multi-step work, so one table holds for every arm. The last entry is the exception
    and the witness of patch P: with the finished plan ``retired`` at the refused stop the board
    is empty, the gate withholds an unplanned command, and the same script ends in words.
    """
    again = {"skill": world.SKILL, "args": world.SKILL_ARGS}
    plan = {"tasks": [{"title": "Open the cycle", "status": "in_progress"}]}
    first = _do("Bash", "r9", command=world.FIRST_STEP)
    return {
        "the skill again, a plan, then the first step": (
            [_do("Skill", "r1", **again), _do("update_plan", "r2", **plan), first],
            "pass",
            ["skill-again", "plan", "first-step"],
        ),
        "words": ([_say("The cycle is complete.")], "stopped-again", ["text-only"]),
        "an echo, then words": (
            [_do("Bash", "r1", command="echo resuming"), _say("Resumed.")],
            "stopped-again",
            ["trivial", "text-only"],
        ),
        "a plan, then a script": (
            [
                _do("update_plan", "r1", **plan),
                _do("Bash", "r2", command="bash scripts/check-queue.sh"),
            ],
            "pass",
            ["plan", "script"],
        ),
        # Reading is acting, and is not the loop resuming: the segment stays open through it.
        "a read, a plan, then the first step": (
            [_do("Read", "r1", path="queue/lots.tsv"), _do("update_plan", "r2", **plan), first],
            "pass",
            ["other-work", "plan", "first-step"],
        ),
        # The loop's own command, refused by the script itself (no cycle is open): not a pass.
        "a script that fails, then words": (
            [
                _do("update_plan", "r1", **plan),
                _do("Bash", "r2", command="bash scripts/cycle-select.sh"),
                _say("The cycle could not be selected."),
            ],
            "stopped-again",
            ["plan", "script", "text-only"],
        ),
        # Naming a loop script is not running it: the loop has not resumed on a ``cat``.
        "the first step's script read, then words": (
            [
                _do("update_plan", "r1", **plan),
                _do("Bash", "r2", command="cat scripts/cycle-open.sh"),
                _say("I have read the first step."),
            ],
            "stopped-again",
            ["plan", "other-work", "text-only"],
        ),
        # A command that reads as a run, exits 0 and ran nothing: no line of the script, no pass.
        "a run that never happened, then words": (
            [
                _do("update_plan", "r1", **plan),
                _do("Bash", "r2", command="false && bash scripts/cycle-close.sh; echo done"),
                _say("Closed."),
            ],
            "stopped-again",
            ["plan", "script", "text-only"],
        ),
        "the first step with no plan": (
            [first, _say("Opened.")],
            "stopped-again" if retired else "pass",
            ["first-step", "text-only"] if retired else ["first-step"],
        ),
        # Plans that never turn into work: nothing ends the turn, so the bench's cap does.
        "plans without end": (
            [
                _do(
                    "update_plan", f"r{i}", tasks=[{"title": f"Draft {i}", "status": "in_progress"}]
                )
                for i in range(LIVE_CAP + 2)
            ],
            "cap",
            ["plan"] * LIVE_CAP,
        ),
        # The same skill call over and over is the product's doom loop: ITS fence ends the
        # turn, the hook refuses again, and the segment is scored as any other refused stop.
        "the skill, over and over": (
            [_do("Skill", f"r{i}", **again) for i in range(LIVE_CAP + 2)],
            "stopped-again",
            ["skill-again"] * LIVE_CAP,
        ),
    }


def _fake_row(capture: str, arm: str, rep: int, passed: bool, **over: Any) -> dict[str, Any]:
    """A ledger row as a sound rollout of ``arm`` would write it, for the reading's known
    answers: every witness the arm's patches owe is present and no other."""
    altered = "R" in arm or "P" in arm
    forced = "F" in arm
    wrote = {"turn_end_skill": 1}
    for patch, kind in (
        ("R", "veto_plan_silenced"),
        ("P", "veto_plan_retired"),
        ("F", "veto_forced_tool"),
    ):
        if patch in arm:
            wrote[kind] = 1
    tail = [{"role": "user", "chars": 50_000, "rail": None}]
    if not altered:
        tail.append({"role": "user", "chars": 240, "rail": "[plan:complete]"})
    step = {
        "aux": False,
        "model": f"openai/{LUNA}",
        "tool_choice": "required" if forced else None,
        "tail": tail,
        "reasoning_tokens": 0,
    }
    sent = {
        "path": "/v1/responses",
        "effort": "none",
        "model": LUNA,
        "n_tools": 12,
        "tool_choice": "required" if forced else "auto",
    }
    row: dict[str, Any] = {
        "capture": capture,
        "arm": arm,
        "rep": rep,
        "outcome": "pass" if passed else "stopped-again",
        "pass": passed,
        "path": ["plan", "first-step"] if passed else ["text-only"],
        "pass_at": 2 if passed else None,
        "any_work_at": 2 if passed else None,
        "notes": wrote,
        "plan_complete_at_fork": True,
        "first_act": "plan" if passed else "text-only",
        "after_door": [],
        "doors": [],
        "steps": [step],
        "deliveries": 1 if passed else 2,
        "fork_seq": 14,
        "replayed": 14,
        "strict_mismatches": 0,
        "fork_identical": not altered,
        "usage": {"prompt_tokens": 40_000, "cost_usd": 0.004},
        "wire": [sent],
        "build": {"own_tree": True, "src_diff_sha256": "clean"},
    }
    return row | over


def _second_reading_known_answers(check: Callable[..., None]) -> None:
    """The second registration's rule on ledgers whose answer is known, each p worked by hand
    first: ``k`` forks that all lean one way give 2 / 2**k."""
    print("the second reading (known answers)")
    scratch = Path(tempfile.mkdtemp(prefix="veto-door-second-"))
    forks, reps = RULE2["min_forks"], 3

    def rows(
        missed: dict[str, dict[int, int]], n: int = forks, without: tuple[str, int] | None = None
    ) -> list[dict[str, Any]]:
        """``missed[arm][fork]`` rollouts of that arm on that fork do NOT resume. ``without``
        (arm, k) leaves that arm's first ``k`` forks unwritten, as dead children would."""
        return [
            _fake_row(f"w{fork:02d}", arm, rep, rep >= by_fork.get(fork, 0))
            for arm, by_fork in missed.items()
            for fork in range(n)
            for rep in range(reps)
            if not (without and without[0] == arm and fork < without[1])
        ]

    def read(ledger_rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        ledger = scratch / "ledger.jsonl"
        ledger.write_text("".join(json.dumps(r) + "\n" for r in ledger_rows), encoding="utf-8")
        return report(ledger, None, mode=mode, reps=reps)

    def read_by_command(mode: str, arms: str) -> dict[str, Any]:
        """The ledger ``read`` last wrote, through ``report`` as a batch runs it: the parser's
        choices and the printing are part of the reading (a rule echoed with a value JSON
        cannot hold reads fine in process and dies at the print)."""
        argv = [sys.executable, str(Path(__file__).resolve()), "report", "--mode", mode]
        argv += ["--ledger", str(scratch / "ledger.jsonl"), "--reps", str(reps), "--arms", arms]
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
        if done.returncode != 0:
            return {"batch": f"exit {done.returncode}: {done.stderr.strip()[-200:]}"}
        return dict(json.loads(done.stdout))

    def one_on(first: int, last: int) -> dict[int, int]:
        return dict.fromkeys(range(first, last), 1)

    def versus(read_out: dict[str, Any], arm: str) -> tuple[str, float | None]:
        seen = read_out["arms"][arm]["vs_baseline"]
        return seen["verdict"], seen.get("paired_p")

    try:
        alone = read(rows({"A": one_on(0, 5)}), "calibrate")
        got = (alone["batch"], alone["baseline_not_resumed"], alone["forks_with_a_miss"])
        check("the baseline alone, missing 5 of 72, says PROCEED", got == ("PROCEED", 0.0694, 5))
        by_command = read_by_command("calibrate", "A")
        check(
            "  and says so through the command line, rule and all",
            (by_command.get("batch"), (by_command.get("rule") or {}).get("calibrate_floor"))
            == ("PROCEED", "1/15"),
            str(by_command.get("batch")),
        )
        easy = read(rows({"A": one_on(0, 4)}), "calibrate")
        check(
            "missing 4 of 72 it says NOT DISCRIMINATING",
            easy["batch"] == "NOT DISCRIMINATING",
            str(easy["batch"]),
        )
        short = read(rows({"A": one_on(0, 5)}, n=forks - 1), "calibrate")
        check("one fork short it says NOT MEASURED", short["batch"] == "NOT MEASURED")

        base = one_on(0, 12)  # 12 of 72: 16.7%
        gain = read(rows({"A": base, "A2": base, "R": {}}), "refusal")
        got_gain = (versus(gain, "R"), versus(gain, "A2"), gain.get("next"))
        check(
            "an arm that never misses where the baseline missed on 12 forks is a GAIN",
            got_gain == (("GAIN", round(2 / 2**12, 4)), ("FLAT", 1.0), "ship R"),
            str(got_gain),
        )
        by_command = read_by_command("refusal", "A,A2,R")
        check(
            "  and through the command line it is the same reading",
            (by_command.get("batch"), by_command.get("next")) == ("READ", "ship R")
            and by_command["arms"]["R"]["vs_baseline"] == gain["arms"]["R"]["vs_baseline"],
            str(by_command.get("batch")),
        )
        flat = read(rows({"A": base, "R": one_on(12, 23)}), "refusal")
        check("one that misses about as often is FLAT", versus(flat, "R")[0] == "FLAT")
        harm = read(rows({"A": base, "R": one_on(0, 20)}), "refusal")
        check(
            "one that misses 11 points more often, on 8 forks, is a HARM",
            versus(harm, "R") == ("HARM", round(2 / 2**8, 4)),
            str(versus(harm, "R")),
        )
        few = read(rows({"A": {f: 3 for f in range(4)}, "R": {}}), "refusal")
        check(
            "the same 12 misses on only 4 forks cannot reach the p: MIXED",
            versus(few, "R") == ("MIXED", round(2 / 2**4, 4)),
            str(versus(few, "R")),
        )
        carried = read(rows({"A": {0: 3, **one_on(1, 12)}, "R": one_on(1, 7)}), "refusal")
        rests = carried["arms"]["R"]["vs_baseline"]
        check(
            "a gain one fork carries is MIXED, and the fork is named",
            (rests["verdict"], rests["rests_on"], carried.get("next"))
            == ("MIXED", ["w00"], "nothing ships from this bench"),
            str(rests),
        )
        edge = read(rows({"A": {0: 3, **one_on(1, 13)}, "R": one_on(1, 7)}), "refusal")
        check(
            "a threshold met exactly (6 of 69 against half of 12 of 69) is met",
            versus(edge, "R")[0] == "GAIN",
            str(edge["arms"]["R"]["vs_baseline"]),
        )
        ghost = read(rows({"A": base, "A2": {}, "R": {}}), "refusal")
        check(
            "a placebo that reads GAIN makes the batch NOT MEASURED",
            (ghost["batch"], ghost.get("why"), ghost.get("next"))
            == ("NOT MEASURED", "the placebo read GAIN", None),
            str((ghost["batch"], ghost.get("why"))),
        )
        calm = read(rows({"A": one_on(0, 5), "R": {}}), "refusal")
        check(
            "a baseline that misses 5 of 72 leaves nothing to reduce: NOT DISCRIMINATING",
            calm["batch"] == "NOT DISCRIMINATING" and "vs_baseline" not in calm["arms"]["R"],
        )
        thin = read(rows({"A": base, "A2": base, "R": {}}, without=("R", 3)), "refusal")
        got_thin = (versus(thin, "R")[0], versus(thin, "A2")[0], thin["complete_forks"])
        check(
            "an arm with 87.5% of its cells is VOID, and costs the others no fork",
            got_thin == ("VOID", "FLAT", forks),
            str(got_thin),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _paired_p_known_answers(check: Callable[..., None]) -> None:
    """The exact test counted by sum against the same test counted one assignment at a time,
    on every shape a batch can produce that is small enough to list."""
    print("the permutation test (exact, against the one-at-a-time count)")
    worst, cases = 0.0, 0
    for reps in (2, 3, 4):
        steps = [Fraction(k, reps) for k in range(-reps, reps + 1)]
        for n in range(1, 7):
            for combo in itertools.combinations_with_replacement(steps, n):
                diffs = [float(d) for d in combo]
                worst = max(worst, abs(paired_p(diffs) - _paired_p_one_at_a_time(diffs)))
                cases += 1
    check(f"  {cases} sets of per-fork differences agree exactly", worst == 0.0, str(worst))
    check("  no difference anywhere is p = 1", paired_p([0.0, 0.0]) == 1.0)
    check("  thirty forks leaning one way is 2 / 2**30", paired_p([1 / 3] * 30) == 2 / 2**30)


def _reading_known_answers(check: Callable[..., None]) -> None:
    """The reading rule on ledgers whose answer is known: every verdict the rule can give, the
    arm it selects, and the rows it must refuse to read."""
    print("the reading (known answers)")
    scratch = Path(tempfile.mkdtemp(prefix="veto-door-reading-"))

    def read(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        ledger = scratch / "ledger.jsonl"
        ledger.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return report(ledger, None, **kw)

    def batch(rates: dict[str, int], forks: int = 8, reps: int = 2) -> list[dict[str, Any]]:
        """``rates[arm]`` rollouts of that arm pass, filled fork by fork, first rep first."""
        rows = []
        for arm, passes in rates.items():
            cells = [(f, rep) for rep in range(reps) for f in range(forks)]
            rows += [_fake_row(f"w{f}", arm, rep, i < passes) for i, (f, rep) in enumerate(cells)]
        return rows

    def verdicts(read_out: dict[str, Any]) -> dict[str, str]:
        return {
            arm: s["vs_baseline"]["verdict"]
            for arm, s in read_out["arms"].items()
            if "vs_baseline" in s
        }

    try:
        screen = read(batch({"A": 8, "R": 16, "C": 8, "F": 0, "B": 10}))
        want = {"R": "GAIN", "C": "FLAT", "F": "HARM", "B": "MIXED"}
        check("a screen reads GAIN, FLAT, HARM and MIXED", verdicts(screen) == want, str(screen))
        check(
            "and names the arm to confirm",
            screen.get("next") == "confirm R",
            str(screen.get("next")),
        )
        near = read(batch({"A": 2, "R": 13, "B": 14}))
        check("the simplest arm close to the best is selected", near.get("next") == "confirm R")
        far = read(batch({"A": 2, "R": 10, "B": 16}))
        check("but not one far behind it", far.get("next") == "confirm B", str(far.get("next")))
        mixed = read(batch({"A": 4, "C": 6, "B": 7, "F": 4}))
        check("with no GAIN the MIXED arms go forward", mixed.get("next") == "confirm B,C")
        none = read(batch({"A": 4, "C": 4, "F": 0, "B": 2}))
        check(
            "with neither (FLAT, or MIXED but below the baseline), nothing does",
            none.get("next") == "no arm gains: nothing to confirm",
            str((verdicts(none), none.get("next"))),
        )
        few = read(batch({"A": 1, "R": 12}, forks=6))
        check("too few forks is NOT MEASURED", few["batch"] == "NOT MEASURED", few["batch"])
        high = read(batch({"A": 14, "R": 16}))
        check(
            "a baseline at the ceiling is NOT DISCRIMINATING", high["batch"] == "NOT DISCRIMINATING"
        )
        wrong = batch({"A": 4, "R": 16, "C": 16})
        for row in wrong:
            if row["arm"] == "C" and row["capture"] in ("w0", "w1"):
                row["doors"] = [{"pointer": True, "first_step": False, "body": False, "chars": 300}]
                row["path"] = ["skill-again", *row["path"]]
        void = read(wrong)
        got = (verdicts(void), void["complete_forks"], void["arms"]["C"]["unusable"])
        check(
            "an arm whose patch does not show is VOID, and costs the others no fork",
            got == ({"R": "GAIN", "C": "VOID"}, 8, {"witness:C-absent": 4}),
            str(got),
        )
        stray = batch({"A": 4, "R": 16})
        stray[0]["notes"] = {**stray[0]["notes"], "veto_plan_silenced": 1}
        stray[1]["wire"] = [{**stray[1]["wire"][0], "effort": "low"}]
        stray[2]["steps"] = [{**stray[2]["steps"][0], "model": "openai/gpt-5-mini"}]
        stray[3]["fork_identical"] = False
        seen = read(stray)["arms"]["A"]["unusable"]
        want_seen = {"witness:R-present": 1, "wire": 1, "not-luna": 1, "prefix": 1}
        check(
            "a patch showing in the wrong arm, a wrong wire, model or prefix is refused",
            seen == want_seen,
            str(seen),
        )
        gone = read(batch({"A": 4, "R": 16}), expected_arms=["A", "R", "F"], expected_forks=["w9"])
        got_gone = (verdicts(gone), gone["forks"], gone["complete_forks"])
        check(
            "an arm and a fork that wrote no row count as unfilled, not as absent",
            got_gone == ({"R": "GAIN", "F": "VOID"}, 9, 8),
            str(got_gone),
        )
        calm = read(batch({"A": 4, "A2": 5, "R": 16}))
        check(
            "a placebo that reads no verdict lets the batch be read",
            (calm["batch"], verdicts(calm)["A2"], calm.get("next"))
            == ("READ", "FLAT", "confirm R"),
        )
        loud = read(batch({"A": 0, "A2": 16, "R": 16}))
        check(
            "a placebo that reads a verdict voids the batch",
            (loud["batch"], loud.get("why")) == ("NOT MEASURED", "the placebo read GAIN"),
            str(loud.get("why")),
        )
        # One fork with thirty rollouts where the arm never fails and the baseline never
        # passes, six forks where the arm is a hair ahead: GAIN on the whole, and nothing like
        # one without the first fork.
        names = [f"w{i}" for i in range(7)]
        arm = {f: [2, 10] for f in names} | {"w0": [30, 30]}
        base = {f: [1, 10] for f in names} | {"w0": [0, 30]}
        thin = _against("screen", arm, base, names)
        check(
            "a GAIN that rests on one fork is MIXED, and names the fork",
            (thin["verdict"], thin["rests_on"], thin["paired_p"] < 0.05) == ("MIXED", ["w0"], True),
            str(thin),
        )
        even = _against("screen", {f: [8, 10] for f in names}, base | {"w0": [1, 10]}, names)
        check(
            "a GAIN every fork carries stands", (even["verdict"], even["rests_on"]) == ("GAIN", [])
        )
        yes = read(batch({"A": 4, "R": 16}, forks=6, reps=3), mode="confirm", reps=3)
        check(
            "a confirmation CONFIRMS and ships",
            (verdicts(yes), yes.get("next")) == ({"R": "CONFIRMED"}, "ship R"),
        )
        no = read(batch({"A": 4, "R": 6}, forks=6, reps=3), mode="confirm", reps=3)
        check(
            "or does not, and ships nothing",
            (verdicts(no), no.get("next"))
            == ({"R": "NOT CONFIRMED"}, "nothing ships from this bench"),
            str(no.get("next")),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _scripted_capture(
    base: Path, scripted: Any, check: Callable[..., None]
) -> tuple[Path, dict[str, Any]]:
    """One capture on a scripted model: a planned cycle, closed, then words. The turn a Stop hook
    refuses. Returns where it was kept and its facts."""
    print("capture (scripted)")
    # The product holds shell work until a plan exists (its plan-first gate), so the scripted
    # model plans, runs one cycle, closes its plan and reports.
    titles = ("Open the cycle", "Run the preflight checks", "Close the cycle")
    opened = [
        {"title": t, "status": "in_progress" if i == 0 else "pending"} for i, t in enumerate(titles)
    ]
    cycle = [
        _do("update_plan", "c0", tasks=opened),
        _do("Bash", "c1", command="bash scripts/cycle-open.sh"),
        _do("Skill", "c2", skill=world.SUB_SKILL),
        _do("Bash", "c3", command="bash scripts/cycle-close.sh"),
        _do("update_plan", "c4", tasks=[{"title": t, "status": "done"} for t in titles]),
        _say("Cycle 1 closed with nothing handled. Twenty lots remain."),
    ]
    out = _registry(base) / "selftest-out"
    shutil.rmtree(out, ignore_errors=True)
    model = scripted(list(cycle))
    facts = asyncio.run(capture_one(base, out, provider=model, provider_class=scripted))
    fork = facts["fork"] or {}
    tail = fork.get("tail") or [{}]
    body_chars = len(world.loop_skill_body())
    check("the run forked at the first delivery", facts["outcome"] == "fork", facts["outcome"])
    check("the fork is the call after the scripted cycle", fork.get("seq") == len(cycle))
    check("both scripts ran before it", fork.get("work_calls") == 2, str(fork.get("work_calls")))
    check(
        "the delivery is in the request's tail, whole",
        any(m.get("role") == "user" and m.get("chars", 0) > body_chars // 2 for m in tail),
        str(tail),
    )
    check(
        "the plan was finished when the stop was refused",
        (fork.get("plan") or {}).get("complete") is True,
    )
    return out / "captures" / facts["capture"], facts


def selftest(base: Path, expect: str, capture: Path | None) -> int:
    """Everything but the network. ``expect`` names the arm this tree is supposed to be, and the
    witnesses at the end are read against it: a patched tree that does not show its patch fails.

    The baseline's tree captures (no ``capture``) and keeps what it captured; an arm's tree is
    handed that directory and only rolls out from it. That is the bench's own division of labour
    (captures are always the baseline's), so the arm's selftest is also the offline proof that
    the baseline's tape replays under the arm's build, message for message.
    """
    from zakcode.agent.loop import skill_reentry_in
    from zakcode.tasks import _outline

    results: list[bool] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append(_check(name, ok, detail))

    print("scorer")
    for calls, want in _known_answers():
        got = completion_label(calls)[0]
        check(want, got == want, f"got {got}")
    # The audit's count: shell calls that name a loop script and were not read as running it.
    reads = [
        {"name": "Bash", "arguments": {"command": c}} for c in ("cat scripts/correct.sh", "ls")
    ]
    runs = [{"name": "Bash", "arguments": {"command": world.FIRST_STEP}}]
    check("a read of a loop script is counted as named, not run",
          named_not_run(reads, completion_label(reads)[1]) == 1)  # fmt: skip
    check("a run of one is not", named_not_run(runs, completion_label(runs)[1]) == 0)
    generous = [c for c in _NAMES_ONLY if _scripts_run(c)]
    check(f"none of {len(_NAMES_ONLY)} commands that only name a loop script reads as a run",
          not generous, "; ".join(generous))  # fmt: skip
    missed = [c for c in _PLAIN_RUNS if not _scripts_run(c)]
    check(f"each of {len(_PLAIN_RUNS)} plain ways of running one reads as a run",
          not missed, "; ".join(missed))  # fmt: skip
    read = [c for c in _LEFT_UNREAD if _scripts_run(c)]
    check(f"the {len(_LEFT_UNREAD)} forms left unread on purpose read as mentions", not read,
          "; ".join(read))  # fmt: skip

    _reading_known_answers(check)
    _second_reading_known_answers(check)
    _paired_p_known_answers(check)

    print("the loop's scripts, run for real")
    _scripts_known_answers(check)

    print("world and product agree")
    named = skill_reentry_in(world.HOOK_REASON)
    check("the hook's reason names the loop skill", named == (world.SKILL, world.SKILL_ARGS))
    first = _outline(world.loop_skill_body(), skill=world.SKILL).pages[0]
    check("the first step section carries the first step", world.FIRST_STEP in first.text)

    scripted = _scripted_class()
    if capture is None:
        capture_dir, facts = _scripted_capture(base, scripted, check)
    else:
        print(f"capture: the baseline's ({capture.name})")
        capture_dir = capture
        facts = json.loads((capture_dir / "capture.json").read_text(encoding="utf-8"))
        clean = facts["build"]["src_diff_sha256"] == "clean" and facts["outcome"] == "fork"
        check("it is a fork captured by the unpatched build", clean, str(facts["build"]))
    fork = facts["fork"] or {}
    print("rollouts (scripted)")
    # What this tree's patches must show. R and P change the fork request itself (the finished
    # plan's reminder is gone from it), so under either the fork is expected NOT to be
    # byte-identical; only P also empties the board.
    retired = "P" in expect
    rail_gone = retired or "R" in expect
    witness: dict[str, Any] = {}
    for name, (script, want, want_path) in _suffixes(retired).items():
        model = scripted(list(script))
        row, _ = asyncio.run(
            rollout_one(base, capture_dir, expect, 0, provider=model, provider_class=scripted)
        )
        # Words with a step still open are nudged by the product's plan gate before the turn
        # ends, so a run of closing ``text-only`` completions is read as one.
        got_path = list(row["path"])
        while got_path[-2:] == ["text-only", "text-only"]:
            got_path.pop()
        check(
            name, (row["outcome"], got_path) == (want, want_path), f"{row['outcome']} {row['path']}"
        )
        faithful = (
            row["replayed"] == fork["seq"]
            and not row["diverged"]
            and not row["strict_mismatches"]
            and row["fork_identical"] is (not rail_gone)
        )
        seen = {k: row[k] for k in ("replayed", "diverged", "fork_identical", "strict_mismatches")}
        check("  the prefix came off the tape, on the recorded trajectory", faithful, str(seen))
        # The rule a batch is READ by, on a row the real product wrote in this tree. Batch 1
        # checked each witness's ingredients by hand and ran the rule on hand-written rows
        # only, so a flag the rule needed and no row carried (the door's ``pointer``) cost
        # arm C every row in which its patch had acted.
        refused = _witness_failure(row, [step for step in row["steps"] if not step.get("aux")])
        check("  the usability rule takes it for a row of this arm", refused is None, str(refused))
        if not witness:
            witness = {"doors": row["doors"], "sent": model.seen, "row": row}
            check("  it resumed on its third completion", row["pass_at"] == 3, str(row["pass_at"]))
        if name.startswith("a read"):
            at = (row["any_work_at"], row["pass_at"])
            check("  the read counted as acting, the first step as resuming", at == (1, 3), str(at))
        audit = (row["named_not_run"], row["run_unproven"])
        if name.startswith("the first step's script read"):
            check(
                "  it is counted as naming a script without running one",
                audit == (1, 0),
                str(audit),
            )
        elif name.startswith("a run that never happened"):
            check("  it is counted as a run its result does not show", audit == (0, 1), str(audit))
        else:
            check("  the audit counts nothing", audit == (0, 0), str(audit))

    print("a tampered tape is caught")
    tape_path = capture_dir / "tape.json"
    kept = tape_path.read_text(encoding="utf-8")
    entries = json.loads(kept)
    entries[1]["shape"] = "0" * 64
    tape_path.write_text(json.dumps(entries), encoding="utf-8")
    try:
        row, _ = asyncio.run(
            rollout_one(
                base, capture_dir, expect, 0, provider=scripted([]), provider_class=scripted
            )
        )
    finally:
        tape_path.write_text(kept, encoding="utf-8")  # an arm's selftest reuses this capture
    check(
        "the rollout is invalid, not scored", row["outcome"] == "invalid:diverged", row["outcome"]
    )

    print(f"witnesses of arm {expect}")
    door = (witness.get("doors") or [{}])[0]
    sent = [s.get("tool_choice") for s in witness.get("sent", [])]
    body = "B" in expect
    first_step = body or "C" in expect
    forced = "F" in expect
    shown = witness.get("row", {})
    fences = (shown.get("fence_at_fork"), shown.get("fence_at_end"))
    wrote = shown.get("notes", {})
    rails = [m.get("rail") for m in _first_live(shown).get("tail", [])]
    check("the door answered the repeated skill call", bool(witness.get("doors")))
    check(
        f"the product flagged that answer a pointer: {not body}",
        door.get("pointer") is (not body),
        str(door),
    )
    # ...and the same row under every OTHER arm's name: the rule must refuse it unless that arm
    # is this very build (A and its placebo). "No patch shows another's", said by the rule.
    live = [step for step in shown.get("steps", []) if not step.get("aux")]
    for label, patches in ARMS.items():
        same_build = sorted(patches) == sorted(ARMS[expect])
        refusal = _witness_failure({**shown, "arm": label}, live)
        check(
            f"  read as arm {label} it is {'usable' if same_build else 'refused'}",
            (refusal is None) is same_build,
            str(refusal),
        )
    check(
        f"its answer carries the first step: {first_step}",
        door.get("first_step") is first_step,
        str(door),
    )
    check(f"its answer is the whole body: {body}", door.get("body") is body, str(door))
    check(
        f"the completion after the delivery was forced: {forced}",
        (sent[:1] == ["required"]) is forced,
        str(sent),
    )
    check(
        f"the finished plan's reminder rode the fork request: {not rail_gone}",
        ("[plan:complete]" in rails) is (not rail_gone),
        str(rails),
    )
    for kind, patch in (
        ("veto_plan_silenced", "R"),
        ("veto_plan_retired", "P"),
        ("veto_forced_tool", "F"),
        ("skill_redelivered", "B"),
    ):
        carried = patch in expect
        check(f"the trace carries {kind}: {carried}", (kind in wrote) is carried, str(wrote))
    check("the fence still counts this refused stop", fences == (1, 1), str(fences))
    failed = results.count(False)
    print(f"capture kept at: {capture_dir}")
    print(f"selftest: {len(results) - failed} ok, {failed} failed ({build_identity()})")
    return 1 if failed else 0


# ── offline: what each category would send ──────────────────────────────────────────────────────


def preflight(base: Path) -> int:
    """No network. Build the served agent and ask each category's provider what it would put
    on the wire with and without tools: the model, the effort, the tool choice, the endpoint."""
    workspace = make_workspace(base)
    os.environ["ZAKCODE_HOME"] = str(_registry(base) / f"home-{workspace.name}")
    agent = served_agent(workspace, max_iterations=1, cost_cap=0.01)
    tools = agent.registry.definitions()
    ok = True
    for category in ROUTING:
        routed, _ = agent._resolve_task_provider(category)
        # The loop's provider is the tool-calling adapter; the request is built by what it wraps.
        provider = getattr(routed, "inner", routed)
        wire = [{"role": "user", "content": "x"}]
        with_tools = provider._build_kwargs(list(wire), tools)
        without = provider._build_kwargs(list(wire), None)
        forced = provider._build_kwargs(list(wire), tools, tool_choice="required")
        row = {
            "category": category,
            "model": with_tools.get("model"),
            "api_base": with_tools.get("api_base"),
            "effort_with_tools": with_tools.get("reasoning_effort"),
            "effort_without": without.get("reasoning_effort"),
            "tool_choice": with_tools.get("tool_choice"),
            "tool_choice_forced": forced.get("tool_choice"),
            "store": (with_tools.get("extra_body") or {}).get("store"),
        }
        print(json.dumps(row))
        if category != "classify":
            ok &= (
                str(row["model"]).endswith(LUNA)
                and row["api_base"] is None
                and row["effort_with_tools"] == "none"
                and row["effort_without"] is None
                and row["tool_choice"] == "auto"
                and row["tool_choice_forced"] == "required"
                and row["store"] is False
            )
    print(
        "preflight:", "PASS" if ok else "FAIL", f"({len(tools)} tools, world {world.digest()[:12]})"
    )
    return 0 if ok else 1


# ── live: one call, does a required tool choice reach the wire and come back as a call? ─────────


async def _probe() -> dict[str, Any]:
    from zakcode.messages import Message
    from zakcode.providers.litellm_provider import LiteLLMProvider

    install_wire_hook()
    provider = LiteLLMProvider(model=f"openai/{LUNA}")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = [Message.user("Say hello in one short sentence. Do not run anything.")]
    rows = {}
    for choice in ("auto", "required"):
        WIRE.clear()
        kw: dict[str, Any] = {} if choice == "auto" else {"tool_choice": "required"}
        result = await provider.acomplete(messages, system="Be brief.", tools=tools, **kw)
        rows[choice] = {
            "calls": [c.name for c in result.tool_calls],
            "text_chars": len((result.text or "").strip()),
            "served_model": result.usage.model,
            "cost_usd": result.usage.cost_usd,
            "wire": list(WIRE),
        }
    return rows


# ── orchestration: children run in the arm's own tree ───────────────────────────────────────────


def _child(tree: Path, args: list[str], log: Path, timeout: int) -> int:
    """Run this file from ``tree`` with that tree's ``src`` first on the path. Output goes to a
    log, stdin is closed, and a hung child is killed: an exit code alone is never the verdict,
    the ledger row is."""
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "PYTHONUNBUFFERED": "1"}
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as sink:
        try:
            done = subprocess.run(  # noqa: S603
                [sys.executable, str(tree / "bench" / "veto_door.py"), *args],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return done.returncode
        except subprocess.TimeoutExpired:
            sink.write(f"\n[orchestrator] killed after {timeout}s: {' '.join(args)}\n")
            return 124


def _spent(ledgers: list[Path]) -> float:
    total = 0.0
    for path in ledgers:
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                total += float((json.loads(line).get("usage") or {}).get("cost_usd") or 0.0)
    return total


def build_arms(out: Path, arms: list[str]) -> dict[str, Any]:
    """One detached worktree per arm at HEAD, with the arm's patches applied. The manifest
    records each tree's source-diff hash: a rollout row must carry the same one. Only the
    arms a batch names are built: a patch cut against an older HEAD (batch 1's ``b`` and ``c``
    after ADR-0203) need not apply for a batch that does not use it."""
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"no such arm: {', '.join(unknown)} (known: {', '.join(ARMS)})")
    out.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD").strip()
    if _git("status", "--porcelain", "--", "src", "bench").strip():
        raise SystemExit("commit src/ and bench/ first: the arms are built from HEAD")
    manifest: dict[str, Any] = {"head": head, "arms": {}}
    for arm, patches in ((arm, ARMS[arm]) for arm in arms):
        tree = out / arm
        if tree.exists():
            _git("worktree", "remove", "--force", str(tree))
        _git("worktree", "add", "--detach", str(tree), head)
        applied = []
        for name in patches:
            patch = HERE / "veto_door_arms" / f"{name}.patch"
            _git("apply", "--whitespace=nowarn", str(patch), cwd=tree)
            applied.append(
                {"patch": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
            )
        diff = _git("diff", "HEAD", "--", "src", cwd=tree)
        manifest["arms"][arm] = {
            "tree": str(tree),
            "patches": applied,
            "src_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else "clean",
        }
    (out / "arms.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _forks(out: Path) -> list[Path]:
    """The captures of ``out`` that reached a fork point, in a stable order."""
    return sorted(
        p.parent
        for p in (out / "captures").glob("*/capture.json")
        if json.loads(p.read_text())["outcome"] == "fork"
    )


def run_captures(
    base: Path, out: Path, arms_dir: Path, want: int, runs: int, budget: float
) -> None:
    """Capture in the baseline's tree until ``want`` runs have forked, ``runs`` were tried, or
    the spend bound is reached, whichever comes first."""
    tree = Path(json.loads((arms_dir / "arms.json").read_text())["arms"]["A"]["tree"])
    ledger = out / "captures.jsonl"
    for index in range(runs):
        spent = _spent([ledger])
        if len(_forks(out)) >= want:
            break
        if spent >= budget:
            print(f"capture budget reached: ${spent:.4f} of ${budget:.2f}")
            break
        code = _child(
            tree,
            ["capture-one", "--base", str(base), "--out", str(out)],
            out / "logs" / f"capture-{index}.log",
            timeout=900,
        )
        print(f"capture {index}: exit {code}, spent so far ${_spent([ledger]):.4f}", flush=True)


def run_rollouts(
    base: Path,
    out: Path,
    arms_dir: Path,
    arms: list[str],
    reps: int,
    budget: float,
    workers: int,
    stage: str = "rollouts",
) -> None:
    """Every fork, every arm, ``reps`` times. Arms are interleaved inside a fork (so drift in
    the provider over the batch lands on all of them alike), starting one arm further along for
    each fork and each repetition (so no arm always goes first, on a cold prompt cache, or
    last). Forks run side by side, each at its own pinned path; two rollouts of one fork never
    overlap. ``stage`` names the ledger (``<stage>.jsonl``): a calibration's rows are kept out
    of the comparison's ledger by never being written to it."""
    manifest = json.loads((arms_dir / "arms.json").read_text())["arms"]
    ledger = out / f"{stage}.jsonl"
    forks = _forks(out)
    # A cell is settled by one usable row, or by two tries: an unusable run (the provider
    # failed, the replay left its tape, a witness is missing) is tried once more and no further.
    filled: set[tuple[str, str, int]] = set()
    tries: Counter[tuple[str, str, int]] = Counter()
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cell = (row["capture"], row["arm"], int(row["rep"]))
            tries[cell] += 1
            if _unusable(row, manifest[row["arm"]]["src_diff_sha256"]) is None:
                filled.add(cell)

    async def one_fork(capture_dir: Path, gate: asyncio.Semaphore) -> None:
        async with gate:
            for rep in range(reps):
                turn = (forks.index(capture_dir) + rep) % len(arms)
                for arm in arms[turn:] + arms[:turn]:
                    cell = (capture_dir.name, arm, rep)
                    if cell in filled or tries[cell] >= 2:
                        continue
                    if _spent([ledger]) >= budget:
                        print(f"rollout budget reached at ${_spent([ledger]):.4f}", flush=True)
                        return
                    args = ["rollout-one", "--base", str(base), "--out", str(out)]
                    args += ["--capture", str(capture_dir), "--arm", arm, "--rep", str(rep)]
                    args += ["--stage", stage]
                    log = out / "logs" / f"rollout-{capture_dir.name}.log"
                    code = await asyncio.to_thread(
                        _child, Path(manifest[arm]["tree"]), args, log, 420
                    )
                    print(f"{capture_dir.name} {arm} rep {rep}: exit {code}", flush=True)

    async def batch() -> None:
        gate = asyncio.Semaphore(workers)
        await asyncio.gather(*(one_fork(fork, gate) for fork in forks))

    asyncio.run(batch())


# ── the reading ──────────────────────────────────────────────────────────────────────────────────


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    z, p = 1.959964, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_p(differences: list[float]) -> float:
    """Two-sided sign-flip permutation test over per-fork differences, exact: the unit is the
    fork, because rollouts of one fork share a conversation and are not independent samples of
    the population of refused stops.

    Every assignment of signs is counted, but by SUM rather than one at a time: a fork's
    difference is a whole number of rollouts over its repetitions, so the distinct sums stay
    few while the assignments double with every fork (thirty forks are 2**30 of them). The
    selftest holds this against the one-at-a-time count on every input small enough to do."""
    live = [Fraction(d).limit_denominator(10**6) for d in differences if d != 0.0]
    if not live:
        return 1.0
    observed = abs(sum(live, Fraction(0)))
    sums: dict[Fraction, int] = {Fraction(0): 1}
    for d in live:
        spread: dict[Fraction, int] = {}
        for total, ways in sums.items():
            spread[total + d] = spread.get(total + d, 0) + ways
            spread[total - d] = spread.get(total - d, 0) + ways
        sums = spread
    return sum(ways for total, ways in sums.items() if abs(total) >= observed) / 2 ** len(live)


def _paired_p_one_at_a_time(differences: list[float]) -> float:
    """The same test by listing every assignment of signs: the check on :func:`paired_p`."""
    live = [Fraction(d).limit_denominator(10**6) for d in differences if d != 0.0]
    if not live:
        return 1.0
    observed = abs(sum(live, Fraction(0)))
    hits = sum(
        abs(sum((sign * d for sign, d in zip(signs, live, strict=True)), Fraction(0))) >= observed
        for signs in itertools.product((1, -1), repeat=len(live))
    )
    return hits / 2 ** len(live)


#: Outcomes that say nothing about the model: the replay left its tape, the process ran some
#: other build, the provider failed, the product's own budget stopped the run. Never scored;
#: the orchestrator runs such a cell once more.
EXCLUDED_OUTCOMES = ("invalid", "ended:provider_error", "ended:budget_exhausted")

#: The reading rule, as pre-registered in ``results/veto-door-preregistration.log``. Shares are
#: fractions of one (the log speaks in points: these times 100).
RULE: dict[str, Any] = {
    "min_forks": {"screen": 7, "confirm": 6},  # complete forks a batch needs to be read at all
    "min_filled": 0.90,  # an arm with fewer of its cells filled than this is VOID
    "ceiling": 0.85,  # a baseline this high leaves no room for a gain: not discriminating
    "gain": 0.20,  # screen: at least this far above the baseline, and
    "p": 0.05,  # ...a paired p below this
    "flat": 0.10,  # screen: closer to the baseline than this
    "harm": -0.15,  # screen: at least this far below it
    "near_best": 0.10,  # the simplest GAIN arm this close to the best one is the one selected
    "confirm": 0.15,  # confirm: at least this far above the baseline, with the same p
    "simplest_first": ("R", "P", "C", "RC", "F", "RF", "B"),
}


def _unusable(row: dict[str, Any], expected_diff: str | None) -> str | None:
    """Why a rollout row cannot be read as a sample of its arm, or ``None``. Every test is of
    the run, never of what the model did: the outcome plays no part."""
    arm, steps = str(row["arm"]), [s for s in row.get("steps", []) if not s.get("aux")]
    if str(row["outcome"]).startswith(EXCLUDED_OUTCOMES):
        return str(row["outcome"])
    build = row["build"]
    if not build["own_tree"] or (expected_diff and build["src_diff_sha256"] != expected_diff):
        return "wrong-build"
    # The fork request is the capture's own, byte for byte, unless the arm takes the finished
    # plan's reminder out of it (R, P), which it can only do where that reminder was.
    altered = ("R" in arm or "P" in arm) and row["plan_complete_at_fork"]
    if (
        row["strict_mismatches"]
        or row["replayed"] != row["fork_seq"]
        or row["fork_identical"] is altered
    ):
        return "prefix"
    if any(not str(s.get("model") or "").endswith(LUNA) for s in steps):
        return "not-luna"
    forced_ok = ("auto", "required") if "F" in arm else ("auto",)
    for sent in row["wire"]:
        if not sent.get("n_tools"):
            continue  # a side call (classification) carries no tools and is not the loop's
        if not (
            str(sent.get("path") or "").endswith("/responses")
            and sent.get("effort") == "none"
            and str(sent.get("model") or "").endswith(LUNA)
            and sent.get("tool_choice") in forced_ok
        ):
            return "wire"
    return _witness_failure(row, steps)


def _witness_failure(row: dict[str, Any], steps: list[dict[str, Any]]) -> str | None:
    """Each patch must show itself in a row of an arm that carries it, and must not show in a
    row of an arm that does not: a row from the wrong tree cannot pass as its arm."""
    arm, wrote, doors, path = str(row["arm"]), row["notes"], row["doors"], row["path"]
    rails = [m.get("rail") for m in (steps[0].get("tail", []) if steps else [])]
    at_the_door = bool(path) and path[0] == "skill-again" and bool(doors)
    for patch, kind in (("R", "veto_plan_silenced"), ("P", "veto_plan_retired")):
        if patch in arm:
            if row["plan_complete_at_fork"] and (kind not in wrote or "[plan:complete]" in rails):
                return f"witness:{patch}-absent"
        elif kind in wrote:
            return f"witness:{patch}-present"
    forced = [s.get("tool_choice") == "required" for s in steps]
    if "F" in arm:
        if not (forced[:1] == [True] and "veto_forced_tool" in wrote):
            return "witness:F-absent"
    elif any(forced) or "veto_forced_tool" in wrote:
        return "witness:F-present"
    if "C" in arm:
        if at_the_door and not (doors[0]["pointer"] and doors[0]["first_step"]):
            return "witness:C-absent"
    elif any(d["pointer"] and d["first_step"] for d in doors):
        return "witness:C-present"
    redelivered = wrote.get("skill_redelivered", 0)
    if "B" in arm:
        if (at_the_door and not redelivered) or redelivered > row["deliveries"]:
            return "witness:B-absent"
    elif redelivered:
        return "witness:B-present"
    return None


def _share(rows: list[dict[str, Any]], test: Callable[[dict[str, Any]], bool]) -> float | None:
    return round(sum(test(r) for r in rows) / len(rows), 4) if rows else None


def _tally(values: Any) -> dict[str, int]:
    return dict(Counter(str(v) for v in values))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def _arm_summary(cells: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """One arm, reduced. ``cells`` are its filled cells on the complete forks (one usable row
    each, the unit every rate is over); ``attempts`` are all of its rows, for the bookkeeping."""
    live = [s for r in cells for s in r["steps"] if not s.get("aux")]
    doors = [d for r in cells for d in r["doors"]]
    passed, n = sum(r["pass"] for r in cells), len(cells)
    low, high = wilson(passed, n)
    per_fork: dict[str, list[bool]] = {}
    for r in cells:
        per_fork.setdefault(r["capture"], []).append(bool(r["pass"]))
    return {
        "n": n,
        "pass": passed,
        "rate": round(passed / n, 4) if n else None,
        "wilson95": [round(low, 3), round(high, 3)],
        "attempts": len(attempts),
        "outcomes": _tally(r["outcome"] for r in cells),
        "pass_at": _tally(r["pass_at"] for r in cells if r["pass"]),
        "any_work_rate": _share(cells, lambda r: r["any_work_at"] is not None),
        "first_act": _tally(r["first_act"] for r in cells),
        "after_door": _tally(x for r in cells for x in r["after_door"]),
        "spiral_rate": _share(cells, lambda r: r["path"].count("skill-again") >= 2),
        "trivial_rate": _share(cells, lambda r: "trivial" in r["path"]),
        # Shell calls that named a loop script and were NOT read as running it, and the cells
        # they sit in: the scorer reads only plain forms as a run, and this is what it left out.
        "named_not_run": sum(int(r.get("named_not_run") or 0) for r in cells),
        "named_not_run_cells": sum(1 for r in cells if r.get("named_not_run")),
        # Successful calls read as running a loop script whose result showed no line of one: the
        # other side of the same audit (each was counted as other work, not as resuming).
        "run_unproven": sum(int(r.get("run_unproven") or 0) for r in cells),
        "mean_live_steps": _mean([float(len(r["path"])) for r in cells]),
        "mean_prompt_tokens": _mean([float(r["usage"].get("prompt_tokens") or 0) for r in cells]),
        "mean_cost_usd": round(sum(float(r["usage"].get("cost_usd") or 0) for r in cells) / n, 5)
        if n
        else None,
        "notes": dict(sum((Counter(r["notes"]) for r in cells), Counter())),
        "fork_rails": _tally(
            ",".join(str(m.get("rail")) for m in _first_live(r).get("tail", [])) for r in cells
        ),
        "door_answers": len(doors),
        "door_pointer": sum(d["pointer"] for d in doors),
        "door_first_step": sum(d["first_step"] for d in doors),
        "door_body": sum(d["body"] for d in doors),
        "live_calls": len(live),
        "forced_live_calls": sum(s.get("tool_choice") == "required" for s in live),
        "reasoning_tokens": sum(int(s.get("reasoning_tokens") or 0) for s in live),
        "per_fork": {c: [sum(v), len(v)] for c, v in sorted(per_fork.items())},
    }


#: The SECOND registration's rule (REGISTRATION 2 in the same log): one hypothesis, read on the
#: failure this world does produce — a rollout that does NOT resume — and every threshold a
#: function of the baseline's rate as MEASURED in the batch being read, never of a predicted
#: one (batch 1 read NOT DISCRIMINATING against an estimate that missed by 59 points).
RULE2: dict[str, Any] = {
    "min_forks": 24,  # complete forks the calibration and the comparison each need
    "min_filled": 0.90,  # as in the first rule: an arm under this share of its cells is VOID
    # The baseline ALONE must miss at least this often (1 rollout in 15) for the comparison to
    # be run at all. It is a gate on SPEND, set low on purpose: simulated before any data, a
    # world that misses 12% of the time is stopped here about 1 time in 9, one that misses 2%
    # about 9 times in 10. Whether a batch can be READ is the comparison's own floor, below.
    "calibrate_floor": Fraction(1, 15),
    "floor": Fraction(8, 100),  # ...and the comparison's own baseline this often, or no reading
    "gain_ratio": Fraction(1, 2),  # GAIN: the arm misses at most this share of the baseline, and
    "gain_points": Fraction(5, 100),  # ...at least this far below it, and
    "p": 0.05,  # ...a paired p below this
    "harm_points": Fraction(10, 100),  # HARM: the arm misses at least this far ABOVE it, same p
    "flat": Fraction(5, 100),  # FLAT: closer to the baseline than this
}


def _verdict2(base_rate: Fraction, arm_rate: Fraction, p: float) -> str:
    """One sound arm against the baseline under :data:`RULE2`, both as NOT-RESUMED rates (lower
    is better). Read in this order, so an arm that meets GAIN is never called FLAT. The rates
    are exact fractions of rollouts: a threshold met exactly (6 of 69 against half of 12 of 69)
    must not turn on the last bit of a float."""
    if arm_rate - base_rate >= RULE2["harm_points"] and p < RULE2["p"]:
        return "HARM"
    if (
        arm_rate <= RULE2["gain_ratio"] * base_rate
        and base_rate - arm_rate >= RULE2["gain_points"]
        and p < RULE2["p"]
    ):
        return "GAIN"
    if abs(base_rate - arm_rate) < RULE2["flat"]:
        return "FLAT"
    return "MIXED"


def _missed(per_fork: PerFork, over: list[str]) -> Fraction:
    """The share of rollouts over ``over`` that did NOT resume, exactly."""
    rollouts = sum(per_fork[f][1] for f in over)
    return Fraction(rollouts - sum(per_fork[f][0] for f in over), rollouts)


def _against2(arm: PerFork, base: PerFork, forks: list[str]) -> dict[str, Any]:
    """One arm against the baseline on how often a rollout did NOT resume, over ``forks``. The
    fork is the unit of the test; a GAIN or a HARM must keep its SIZE with any one fork left
    out (the p is not re-taken: losing a fork costs any test power and says nothing about
    whether one fork carried the result), or the arm reads MIXED and those forks are named."""
    diffs = [arm[f][0] / arm[f][1] - base[f][0] / base[f][1] for f in forks]
    p = paired_p(diffs)
    base_rate, arm_rate = _missed(base, forks), _missed(arm, forks)
    verdict = _verdict2(base_rate, arm_rate, p)
    rests_on: list[str] = []
    if verdict in ("GAIN", "HARM"):
        for left_out in forks:
            rest = [f for f in forks if f != left_out]
            if _verdict2(_missed(base, rest), _missed(arm, rest), p) != verdict:
                rests_on.append(left_out)
    if rests_on:
        verdict = "MIXED"
    return {
        "baseline_not_resumed": round(float(base_rate), 4),
        "arm_not_resumed": round(float(arm_rate), 4),
        "fewer_by": round(float(base_rate - arm_rate), 4),
        "share_of_baseline": round(float(arm_rate / base_rate), 4) if base_rate else None,
        "forks_better": sum(d > 0 for d in diffs),
        "forks_same": sum(d == 0 for d in diffs),
        "forks_worse": sum(d < 0 for d in diffs),
        "paired_p": round(p, 4),
        "rests_on": rests_on,
        "verdict": verdict,
    }


def _verdict(mode: str, difference: float, p: float) -> str:
    """One sound arm against the baseline. ``screen`` names every outcome; ``confirm`` has
    two. (An arm with too few of its cells filled is VOID before it gets here.)"""
    if mode == "confirm":
        return "CONFIRMED" if difference >= RULE["confirm"] and p < RULE["p"] else "NOT CONFIRMED"
    if difference <= RULE["harm"] and p < RULE["p"]:
        return "HARM"
    if difference >= RULE["gain"] and p < RULE["p"]:
        return "GAIN"
    if abs(difference) < RULE["flat"]:
        return "FLAT"
    return "MIXED"


PerFork = dict[str, list[int]]  # fork -> [passes, rollouts]


def _against(mode: str, arm: PerFork, base: PerFork, forks: list[str]) -> dict[str, Any]:
    """One arm against the baseline over ``forks``: the pooled difference, the paired p, the
    verdict, and whether the SIZE of the difference stands with any one fork left out. A rate
    over seven to ten forks can be carried by one of them, and only leaving each out in turn
    shows it: a GAIN, a HARM or a CONFIRMED whose difference falls back across its threshold
    without one fork is not given, and the forks it rested on are named. (The p is not re-taken
    on the smaller sets: losing a fork costs any test power, which says nothing about whether
    one fork carried the result.)"""

    def read(over: list[str]) -> tuple[float, float, list[float]]:
        diffs = [arm[f][0] / arm[f][1] - base[f][0] / base[f][1] for f in over]
        pooled = sum(arm[f][0] for f in over) / sum(arm[f][1] for f in over) - sum(
            base[f][0] for f in over
        ) / sum(base[f][1] for f in over)
        return pooled, paired_p(diffs), diffs

    pooled, p, diffs = read(forks)
    verdict = _verdict(mode, pooled, p)
    rests_on: list[str] = []
    if verdict in ("GAIN", "HARM", "CONFIRMED"):
        for left_out in forks:
            d, _, _ = read([f for f in forks if f != left_out])
            if _verdict(mode, d, p) != verdict:
                rests_on.append(left_out)
    if rests_on:
        verdict = "NOT CONFIRMED" if mode == "confirm" else "MIXED"
    return {
        "pooled_difference": round(pooled, 4),
        "mean_fork_difference": round(sum(diffs) / len(diffs), 4),
        "forks_up": sum(d > 0 for d in diffs),
        "forks_same": sum(d == 0 for d in diffs),
        "forks_down": sum(d < 0 for d in diffs),
        "paired_p": round(p, 4),
        "rests_on": rests_on,
        "verdict": verdict,
    }


def _simplest(arms: list[str]) -> list[str]:
    order = {arm: i for i, arm in enumerate(RULE["simplest_first"])}
    return sorted(arms, key=lambda arm: order.get(arm, len(order)))


def _next_step(mode: str, arms: dict[str, dict[str, Any]], baseline: str) -> str:
    """What the pre-registered rule says happens after this batch."""
    read = {a: s["vs_baseline"] for a, s in arms.items() if a not in (baseline, PLACEBO)}
    by_verdict: dict[str, list[str]] = {}
    for arm, versus in read.items():
        by_verdict.setdefault(versus["verdict"], []).append(arm)
    if mode == "confirm":
        confirmed = _simplest(by_verdict.get("CONFIRMED", []))
        if not confirmed:
            return "nothing ships from this bench"
        best = max(confirmed, key=lambda a: read[a]["pooled_difference"])
        lead = read[best]["pooled_difference"] - read[confirmed[0]]["pooled_difference"]
        return f"ship {best if lead > RULE['near_best'] else confirmed[0]}"
    gains = by_verdict.get("GAIN", [])
    if gains:
        best = max(read[a]["pooled_difference"] for a in gains)
        near = [a for a in gains if read[a]["pooled_difference"] >= best - RULE["near_best"]]
        return f"confirm {_simplest(near)[0]}"
    # Only an arm that read ABOVE the baseline is worth a second look.
    mixed = [a for a in _simplest(by_verdict.get("MIXED", [])) if read[a]["pooled_difference"] > 0]
    if mixed:
        mixed.sort(key=lambda a: -read[a]["pooled_difference"])  # stable: ties stay simplest first
        return "confirm " + ",".join(mixed[:2])
    return "no arm gains: nothing to confirm"


def report(
    ledger: Path,
    arms_json: Path | None,
    *,
    mode: str = "screen",
    reps: int = 2,
    baseline: str = "A",
    expected_arms: list[str] | None = None,
    expected_forks: list[str] | None = None,
) -> dict[str, Any]:
    """The pre-registered reading of one batch, and nothing else: which rows are usable, which
    forks are complete, every arm against the baseline over exactly those forks, the verdicts,
    and the step the rule names next. A batch is read alone, never pooled with another.

    ``expected_arms`` and ``expected_forks`` are what the batch was launched with. An arm or a
    fork whose every child died wrote no row at all, and must count as unfilled cells, not
    vanish from the reading.
    """
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]
    manifest = json.loads(arms_json.read_text())["arms"] if arms_json else {}
    arms = sorted(
        {r["arm"] for r in rows} | set(expected_arms or ()), key=lambda a: (a != baseline, a)
    )
    forks = sorted({r["capture"] for r in rows} | set(expected_forks or ()))
    # One usable row per cell (fork, arm, rep): the first in ledger order.
    filled: dict[tuple[str, str, int], dict[str, Any]] = {}
    unusable: dict[str, Counter[str]] = {arm: Counter() for arm in arms}
    for row in rows:
        why = _unusable(row, (manifest.get(row["arm"]) or {}).get("src_diff_sha256"))
        if why:
            unusable[row["arm"]][why] += 1
        else:
            filled.setdefault((row["capture"], row["arm"], int(row["rep"])), row)
    # A fork the BASELINE could not fill is a bad fork (the unpatched build replaying its own
    # capture), not a bad arm: an arm is judged on the forks the baseline filled. One with under
    # ``min_filled`` of its cells there is VOID, out of the reading, and must not take the other
    # arms' forks with it: a fork is complete when every arm still in the reading is whole on it.
    held = [f for f in forks if all((f, baseline, rep) in filled for rep in range(reps))]
    share_filled = {
        arm: sum((f, arm, rep) in filled for f in held for rep in range(reps))
        / max(1, len(held) * reps)
        for arm in arms
    }
    rule = RULE2 if mode in SECOND_MODES else RULE  # the one rule this batch is read by
    void = {arm: share_filled[arm] < rule["min_filled"] for arm in arms}
    sound = [arm for arm in arms if not void[arm]]
    complete = [
        f for f in held if all((f, arm, rep) in filled for arm in sound for rep in range(reps))
    ]
    spend = sum(float(r["usage"].get("cost_usd") or 0.0) for r in rows)
    out: dict[str, Any] = {
        "mode": mode,
        # Echoed as it was applied. The second rule's thresholds are exact fractions, which
        # JSON has no form for: they go out as "n/d" (a report that cannot print is a batch
        # that cannot be read, found only after the spend).
        "rule": {k: str(v) if isinstance(v, Fraction) else v for k, v in rule.items()},
        "rows": len(rows),
        "spend_usd": round(spend, 4),
        "forks": len(forks),
        "baseline_forks": len(held),
        "complete_forks": len(complete),
        "arms": {},
    }
    for arm in arms:
        cells = [
            filled[(f, arm, rep)]
            for f in complete
            for rep in range(reps)
            if (f, arm, rep) in filled
        ]
        summary = _arm_summary(cells, [r for r in rows if r["arm"] == arm])
        summary["filled_share"] = round(share_filled[arm], 4)
        summary["unusable"] = dict(unusable[arm])
        out["arms"][arm] = summary
    base = out["arms"].get(baseline)
    needed = rule["min_forks"] if mode in SECOND_MODES else rule["min_forks"][mode]
    if base is None or len(complete) < needed:
        out["batch"] = "NOT MEASURED"
        return out
    if mode in SECOND_MODES:
        return _read_second(mode, out, baseline, void, complete)
    if (base["rate"] or 0.0) >= RULE["ceiling"]:
        out["batch"] = "NOT DISCRIMINATING"
        return out
    out["batch"] = "READ"
    for arm, summary in out["arms"].items():
        if arm == baseline:
            continue
        if void[arm]:
            summary["vs_baseline"] = {"verdict": "VOID", "pooled_difference": 0.0}
            continue
        summary["vs_baseline"] = _against(mode, summary["per_fork"], base["per_fork"], complete)
    # The placebo is the baseline's own build under another name. Read like any arm, it says
    # what this batch's scale calls a verdict when nothing was changed: if that is anything
    # but no verdict, every other verdict in the batch is noise read as signal.
    placebo = (out["arms"].get(PLACEBO) or {}).get("vs_baseline", {}).get("verdict")
    if placebo in ("GAIN", "HARM", "CONFIRMED"):
        out["batch"] = "NOT MEASURED"
        out["why"] = f"the placebo read {placebo}"
        return out
    out["next"] = _next_step(mode, out["arms"], baseline)
    return out


#: The two readings of the second registration: the baseline alone, then the comparison.
SECOND_MODES = ("calibrate", "refusal")


def _read_second(
    mode: str, out: dict[str, Any], baseline: str, void: dict[str, bool], complete: list[str]
) -> dict[str, Any]:
    """The second registration's two readings, over forks already known to be complete.

    ``calibrate`` reads the baseline ALONE and says only whether the comparison is worth
    running: a world where the unpatched build almost always resumes has no failure for an arm
    to reduce. Its rows are never the comparison's (choosing to go on because an arm's own
    rows looked bad, and then comparing against those rows, would manufacture a difference).
    ``refusal`` is the comparison: each arm against the baseline on how often a rollout did
    not resume, the placebo read like any arm, and the one step the rule names next."""
    base = out["arms"][baseline]
    missed = _missed(base["per_fork"], complete)
    out["baseline_not_resumed"] = round(float(missed), 4)
    out["forks_with_a_miss"] = sum(passes < n for passes, n in base["per_fork"].values())
    floor = RULE2["calibrate_floor"] if mode == "calibrate" else RULE2["floor"]
    if missed < floor:
        out["batch"] = "NOT DISCRIMINATING"
        return out
    if mode == "calibrate":
        out["batch"] = "PROCEED"
        return out
    out["batch"] = "READ"
    for arm, summary in out["arms"].items():
        if arm == baseline:
            continue
        if void[arm]:
            summary["vs_baseline"] = {"verdict": "VOID"}
            continue
        summary["vs_baseline"] = _against2(summary["per_fork"], base["per_fork"], complete)
    placebo = (out["arms"].get(PLACEBO) or {}).get("vs_baseline", {}).get("verdict")
    if placebo in ("GAIN", "HARM"):
        out["batch"] = "NOT MEASURED"
        out["why"] = f"the placebo read {placebo}"
        return out
    gains = [
        arm
        for arm, summary in out["arms"].items()
        if arm not in (baseline, PLACEBO) and summary["vs_baseline"]["verdict"] == "GAIN"
    ]
    out["next"] = f"ship {_simplest(gains)[0]}" if gains else "nothing ships from this bench"
    return out


# ── command line ─────────────────────────────────────────────────────────────────────────────────


COMMANDS = (
    "selftest",
    "preflight",
    "arms",
    "capture",
    "rollouts",
    "capture-one",
    "rollout-one",
    "probe",
    "report",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--base", type=Path, default=Path("/srv/zc-door"), help="where workspaces are built"
    )
    parser.add_argument("--out", type=Path, help="scratch directory: tapes, ledgers, logs")
    parser.add_argument("--arms-dir", type=Path, help="where `arms` put the worktrees")
    parser.add_argument("--expect", default="A", help="selftest: the arm this tree should be")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--arm", default="A")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--forks", type=int, default=10, help="capture: fork points wanted")
    parser.add_argument("--runs", type=int, default=14, help="capture: runs tried at most")
    parser.add_argument("--reps", type=int, default=2, help="rollouts per arm per fork")
    parser.add_argument(
        "--mode", choices=("screen", "confirm", "calibrate", "refusal"), default="screen"
    )
    parser.add_argument(
        "--stage", default="rollouts", help="rollouts: the ledger's name (<stage>.jsonl)"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--budget", type=float, default=1.0, help="stop launching at this live spend (USD)"
    )
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()

    if args.command == "selftest":
        return selftest(args.base, args.expect, args.capture)
    if args.command == "preflight":
        return preflight(args.base)
    if args.command == "arms":
        print(json.dumps(build_arms(args.arms_dir, args.arms.split(",")), indent=1))
        return 0
    if args.command == "report":
        arms_json = args.arms_dir / "arms.json" if args.arms_dir else None
        read = report(
            args.ledger,
            arms_json,
            mode=args.mode,
            reps=args.reps,
            expected_arms=args.arms.split(","),
            expected_forks=[f.name for f in _forks(args.out)] if args.out else None,
        )
        print(json.dumps(read, indent=1))
        return 0
    if args.command == "probe":
        print(json.dumps(asyncio.run(_probe()), indent=1))
        return 0
    if args.command == "capture":
        run_captures(args.base, args.out, args.arms_dir, args.forks, args.runs, args.budget)
        return 0
    if args.command == "rollouts":
        run_rollouts(
            args.base,
            args.out,
            args.arms_dir,
            args.arms.split(","),
            args.reps,
            args.budget,
            args.workers,
            args.stage,
        )
        return 0

    # The two live children. Both record the wire and refuse a key-less or mis-built process.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    install_wire_hook()
    if args.command == "capture-one":
        facts = asyncio.run(capture_one(args.base, args.out))
        public = {k: v for k, v in facts.items() if k != "wire"} | {
            "wire_calls": len(facts["wire"])
        }
        _append(args.out / "captures.jsonl", public)
        print(json.dumps({k: public[k] for k in ("capture", "outcome", "calls", "seconds")}))
        return 0
    row, detail = asyncio.run(rollout_one(args.base, args.capture, args.arm, args.rep))
    _append(args.out / f"{args.stage}.jsonl", row)
    _append(args.out / f"{args.stage}-detail.jsonl", detail)
    print(json.dumps({k: row[k] for k in ("capture", "arm", "rep", "outcome", "path", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
