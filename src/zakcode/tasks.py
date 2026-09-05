"""The hierarchical task network — Zak Code's first-layer task decomposition substrate.

This is the engine's answer to "what am I doing, and what comes next?" — the near-term
planning layer (Claude Code's task list above the input bar), NOT a long-horizon multi-session
planner. A turn that is more than a single primitive action decomposes its goal into a small
**hierarchical task network** (HTN-flavored): every node is either

* **compound** — a goal that is not yet directly actionable and must be broken down into
  ``children`` before work proceeds (e.g. "add the /users endpoint"); or
* **primitive** — a directly-executable step the model can carry out with one short run of
  tools (e.g. "write the route handler in app/users.py").

The decomposition *discipline* lives here in the engine (a compound task is not "done" until
its children are; the frontier is the next actionable primitive; you don't finish while
actionable work remains). The decomposition *method knowledge* — HOW to break down a given
kind of task — is deliberately NOT here: that is domain knowledge and belongs in skills and
prompt guidance, keeping this substrate clean-room and domain-agnostic.

Design rules (mirroring :mod:`zakcode.agent.recipe` / :mod:`zakcode.agent.stuck`):

* **Pure and vendor-agnostic.** No provider, transport, or I/O — just the data structure and
  the deterministic logic over it. The *model* authors the plan (via the ``update_plan`` tool);
  the harness only tracks it, re-injects it, and enforces structural discipline.
* **Derived parent status.** A compound task's status is COMPUTED from its children, never set
  directly, so a parent can never be marked ``done`` while a child is still ``pending`` — the
  central HTN invariant, enforced rather than trusted.
* **Full-replace authoring.** The tool hands the whole tree each time (the proven TodoWrite
  pattern): robust for weak local models, and it lets the harness re-number ids and re-derive
  invariants from scratch every edit, so the stored network is always internally consistent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

#: The lifecycle of a single task. ``blocked`` is an explicit "cannot proceed yet" state (a
#: stated reason, e.g. an unmet dependency) distinct from ``pending`` (just not started).
TaskStatus = Literal["pending", "in_progress", "done", "blocked", "cancelled"]

#: A task is either a goal still to be broken down (``compound``) or a directly-executable
#: step (``primitive``). A compound task is "entered" (its children become the frontier),
#: never executed directly; a primitive task is the unit the model actually acts on.
TaskKind = Literal["compound", "primitive"]

#: Statuses that count a (primitive) task as finished — no more work is owed for it.
_TERMINAL: frozenset[str] = frozenset({"done", "cancelled"})

#: One-char glyphs for the rendered checklist (the live plan re-injected into context).
_GLYPH: dict[str, str] = {
    "pending": " ",
    "in_progress": "~",
    "done": "x",
    "blocked": "!",
    "cancelled": "-",
}

#: ADR-0110 — bounds on the plan's MEMORY: evidence lines kept per step, history events kept
#: before the oldest fold into ``TaskNetwork.log_folded``, characters kept of the request
#: anchor and of one evidence / outcome line. Constants, not knobs: the bounds are part of the
#: contract that the plan stays small enough to re-inject on every iteration.
MAX_EVIDENCE_PER_STEP = 12
MAX_LOG_EVENTS = 200
MAX_REQUEST_CHARS = 600
MAX_LINE_CHARS = 160


def clip(text: str, limit: int) -> str:
    """One line of ``text`` cut to ``limit`` characters (an ellipsis marks the cut)."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: max(0, limit - 1)] + "…"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class PlanEvent(BaseModel):
    """One entry of the plan's history (ADR-0110): what changed, when, and to which step.

    ``kind`` is one of ``authored`` (the model laid out or reshaped the plan), ``step`` (a
    step's status moved — ``detail`` carries ``old -> new`` and the outcome), ``seeded`` (the
    harness added steps: a skill skeleton, an investigation, the request anchor), ``cleared``
    (the model emptied the plan) or ``reset`` (the loop dropped a finished or abandoned plan at
    a turn start — ``detail`` summarises what it achieved). The log is what makes the plan a
    RECORD and not only a checklist: "what did I do a few steps ago" is answered here after the
    conversation that did it has been compacted away.
    """

    seq: int
    at: str
    kind: str
    step: str = ""
    title: str = ""
    detail: str = ""


class PlanContext(BaseModel):
    """What the plan is FOR (ADR-0110): the request that started it, verbatim and clipped.

    Set by the loop when a turn begins on an empty plan and carried unchanged while the plan
    lives, so a resumed or compacted session — or a small model twenty iterations in — can
    re-read the ask without scrolling for it. A full-replace by the model never touches it.
    """

    request: str = ""


class Task(BaseModel):
    """One node in the task network — a compound goal or a primitive step.

    ``id`` is assigned by the network (a stable, human-readable dotted path like ``2.1``),
    not by the author: the ``update_plan`` tool hands a tree by position and the network
    re-numbers it, so ids are always consistent with the current structure. ``children`` is
    non-empty only for compound tasks that have actually been decomposed; a ``compound`` task
    with no children is *under-decomposed* and is what the plan gate nudges the model to expand.
    """

    id: str = ""
    title: str
    kind: TaskKind = "primitive"
    #: For a primitive task this is authoritative. For a compound task it is IGNORED on input
    #: and overwritten by :meth:`TaskNetwork.normalize` with the status derived from children.
    status: TaskStatus = "pending"
    #: The step's done-condition — a one-line, checkable acceptance criterion ("prints DONE",
    #: "all tests pass", "GET /health returns 200"). This IS the verification half of the plan:
    #: it is what tells the author (and a reader) when the step is objectively finished. For a
    #: ``blocked`` step, say WHY it is stuck here instead. Optional, but recommended on every
    #: primitive step; rendered into the re-injected plan so it stays in view.
    note: str = ""
    #: Ids of earlier steps that must reach a terminal status before THIS step is actionable
    #: (a dependency edge, e.g. ``["1", "2"]``). Validated by :meth:`TaskNetwork.normalize`:
    #: unknown ids, self-references, and cycles are dropped (fail-open), so a malformed graph
    #: can never freeze the agent. Empty (the default) = no dependencies, i.e. document order.
    blocked_by: list[str] = Field(default_factory=list)
    children: list[Task] = Field(default_factory=list)
    #: ADR-0110 — who authored the step: the model (default) or the harness (a skill section,
    #: an investigation step, the request anchor). Rendered nowhere; read by the plan gate.
    origin: Literal["model", "harness"] = "model"
    #: ADR-0110 — the harness's REQUEST ANCHOR: a step standing for the user's ask itself,
    #: seeded when deep work began to change the workspace with no plan. The one step the plan
    #: gate may close on the model's behalf at a conclusion; the model's own steps never are.
    anchor: bool = False
    #: ADR-0110 — what the step PRODUCED, one line, recorded when it closes ("the flake is a
    #: stale cache", "route added in app/users.py"). Set by the model (``update_plan``
    #: ``outcome``) or filled by the harness from the last evidence line when the model leaves
    #: it blank, so a closed step never reads as "done, no record".
    outcome: str = ""
    #: ADR-0110 — the tool calls that ran while this step was current, one compact line each
    #: ("read_file app/users.py ✓", "bash pytest -q ✗"), oldest first, bounded to
    #: :data:`MAX_EVIDENCE_PER_STEP` (older lines drop). Attached by the harness at the tool
    #: execution seam; the model never writes it. This is the step's own record of what it did.
    evidence: list[str] = Field(default_factory=list)

    def is_compound(self) -> bool:
        """True when this node has been decomposed into children (an actual sub-network)."""
        return self.kind == "compound" and bool(self.children)


class TaskNetwork(BaseModel):
    """A small forest of :class:`Task` trees — the live plan for the current goal.

    The network is the single source of truth for "the plan": persisted on the session,
    re-injected into context each iteration, and consulted by the plan gate. All structural
    invariants (derived parent status, dotted ids, a single focused ``in_progress`` primitive)
    are re-established by :meth:`normalize`, which the authoring tool calls on every edit, so
    no consumer ever sees a half-consistent tree.
    """

    tasks: list[Task] = Field(default_factory=list)
    #: ADR-0110 — what the plan is for (the request anchor). Survives a full-replace; reset by
    #: the loop when a new plan starts on an empty network.
    context: PlanContext = Field(default_factory=PlanContext)
    #: ADR-0110 — the plan's history, oldest first, bounded to :data:`MAX_LOG_EVENTS`; the count
    #: of events folded off the front is kept so the record says how much it forgot. Outlives the
    #: steps: a turn-start reset clears ``tasks`` and keeps the log.
    log: list[PlanEvent] = Field(default_factory=list)
    log_folded: int = 0

    # ── authoring / normalization ───────────────────────────────────────────────

    def normalize(self) -> list[str]:
        """Re-establish every structural invariant; return human-readable advisories.

        Idempotent and total — safe to call on any (even model-authored, possibly malformed)
        tree. It (1) assigns dotted ids by position, (2) sanitizes dependency edges
        (``blocked_by``): drops self-references, unknown ids, and cycles so the graph is always a
        DAG, (3) enforces a single focused ``in_progress`` primitive (the first in document order
        wins; later ones are demoted to ``pending``), then (4) derives every compound's status
        from its children bottom-up. Focus is enforced BEFORE derivation so a demoted child can
        never leave its parent with a stale ``in_progress`` status. Returned strings are surfaced
        to the model by the tool so a demotion, a dropped edge, or an under-decomposed node is
        visible, never silent.
        """
        advisories: list[str] = []
        self._assign_ids(self.tasks, prefix="")
        self._sanitize_dependencies(advisories)
        self._flag_duplicate_siblings(advisories)
        self._enforce_single_focus(advisories)
        for task in self.tasks:
            self._derive_status(task, advisories)
        return advisories

    def insert_before(self, anchor: Task | None, steps: list[Task], *, reason: str = "") -> None:
        """Splice harness-authored ``steps`` in as siblings directly AHEAD of ``anchor``.

        The decompose-on-stuck recovery (ADR-0057) uses this to put investigative steps in
        front of the step the model is stuck on: ``anchor`` is demoted to ``pending`` so the
        first new step becomes the current work, and the stuck step follows in document order
        — investigate, decide, then retry. ``anchor=None`` (no plan, or a finished one) appends
        at the top level: the investigation IS the plan. Normalizes afterwards, so ids, focus,
        and derived statuses are consistent for the next render. Harness-authored, so every
        step is stamped ``origin="harness"`` and the splice is logged (ADR-0110) — ``reason``
        names why, else the titles do.
        """
        for step in steps:
            self._stamp_harness(step)
        siblings = self._siblings_of(anchor) if anchor is not None else None
        if siblings is None:
            self.tasks.extend(steps)
        else:
            if anchor is not None and anchor.status == "in_progress":
                anchor.status = "pending"
            index = next(i for i, task in enumerate(siblings) if task is anchor)
            siblings[index:index] = steps
        self.normalize()
        if steps:
            self.record(
                "seeded",
                step=steps[0] if len(steps) == 1 else None,
                detail=reason or "; ".join(clip(s.title, 40) for s in steps[:6]),
            )

    @staticmethod
    def _stamp_harness(step: Task) -> None:
        step.origin = "harness"
        for child in step.children:
            TaskNetwork._stamp_harness(child)

    # ── memory: history, evidence, carry-over (ADR-0110) ─────────────────────────

    def record(self, kind: str, *, step: Task | None = None, detail: str = "") -> PlanEvent:
        """Append one :class:`PlanEvent`; the oldest fold off past :data:`MAX_LOG_EVENTS`."""
        event = PlanEvent(
            seq=self.log_folded + len(self.log) + 1,
            at=_now(),
            kind=kind,
            step=step.id if step is not None else "",
            title=clip(step.title, MAX_LINE_CHARS) if step is not None else "",
            detail=clip(detail, MAX_LINE_CHARS),
        )
        self.log.append(event)
        overflow = len(self.log) - MAX_LOG_EVENTS
        if overflow > 0:
            del self.log[:overflow]
            self.log_folded += overflow
        return event

    def attach_evidence(self, task: Task, line: str) -> None:
        """Append one evidence line to ``task`` — the step that was current when a tool ran."""
        if not line.strip():
            return
        task.evidence.append(clip(line, MAX_LINE_CHARS))
        overflow = len(task.evidence) - MAX_EVIDENCE_PER_STEP
        if overflow > 0:
            del task.evidence[:overflow]

    @staticmethod
    def _title_key(title: str) -> str:
        return " ".join(title.lower().split())

    def replace_from_author(self, tasks: list[Task]) -> list[str]:
        """Install a model-authored tree (full-replace), keeping what the model cannot resend.

        ``update_plan`` hands a fresh tree by position, so without this every step's memory
        would vanish on each edit. A new node inherits from its predecessor of the same title
        (the one thing a model preserves across an edit; ids shift when steps are inserted):
        ``evidence``, ``origin`` and ``anchor``, and — when the model left it blank — the prior
        ``outcome``. A leaf that just reached a terminal status with no outcome gets one from
        its last evidence line, so a closed step never reads as "done, no record". Every leaf's
        start (``-> in_progress``) and close, and the (re)authoring itself when the set of
        titles changed, are logged. Returns :meth:`normalize`'s advisories.
        """
        prior_by_title: dict[str, Task] = {}
        for task in self._iter():
            prior_by_title.setdefault(self._title_key(task.title), task)
        prior_titles = [self._title_key(t.title) for t in self._iter()]
        self.tasks = tasks
        advisories = self.normalize()
        for task in self._iter():
            prior = prior_by_title.get(self._title_key(task.title))
            if prior is None:
                if not task.children and task.status in _TERMINAL:
                    tail = f" — {task.outcome}" if task.outcome else ""
                    self.record("step", step=task, detail=f"new -> {task.status}{tail}")
                continue
            task.evidence = list(prior.evidence)
            task.origin = prior.origin
            task.anchor = prior.anchor
            if not task.outcome:
                task.outcome = prior.outcome
            if task.children or task.status == prior.status:
                continue
            if task.status in _TERMINAL and not task.outcome and task.evidence:
                task.outcome = "last action: " + task.evidence[-1]
            if task.status in _TERMINAL or task.status == "in_progress":
                tail = f" — {task.outcome}" if task.status in _TERMINAL and task.outcome else ""
                self.record("step", step=task, detail=f"{prior.status} -> {task.status}{tail}")
        if [self._title_key(t.title) for t in self._iter()] != prior_titles:
            leaves = self.leaves()
            titles = "; ".join(clip(t.title, 40) for t in leaves[:6])
            self.record("authored", detail=f"{len(leaves)} step(s): {titles}")
        return advisories

    def recent_closed(self, limit: int = 3) -> list[Task]:
        """The most recently CLOSED leaves (done/cancelled), newest first (ADR-0110).

        Read from the log, which is chronological where document order is not; a closed step
        the model has since re-titled or dropped is skipped. When the log is silent about
        closures (a plan restored by an older build), document order stands in.
        """
        by_title = {self._title_key(t.title): t for t in self.leaves()}
        out: list[Task] = []
        for event in reversed(self.log):
            closing = "-> done" in event.detail or "-> cancelled" in event.detail
            if event.kind != "step" or not closing:
                continue
            task = by_title.get(self._title_key(event.title))
            if task is None or task.status not in _TERMINAL or any(t is task for t in out):
                continue
            out.append(task)
            if len(out) >= limit:
                return out
        if not out:
            out = [t for t in reversed(self.leaves()) if t.status in _TERMINAL][:limit]
        return out

    def last_closed(self) -> Task | None:
        """The most recently closed leaf, or ``None``."""
        closed = self.recent_closed(1)
        return closed[0] if closed else None

    def contains(self, task: Task) -> bool:
        """True while this exact ``task`` object is still part of the network.

        Identity, not equality: a harness that spliced a step in (:meth:`insert_before`) asks
        whether the model has since replaced the plan around it, and a same-titled step the
        model re-authored is a different object.
        """
        return any(node is task for node in self._iter())

    def _siblings_of(self, task: Task) -> list[Task] | None:
        """The sibling list holding ``task`` (by identity), or ``None`` when it is not here."""

        def visit(nodes: list[Task]) -> list[Task] | None:
            for node in nodes:
                if node is task:
                    return nodes
                found = visit(node.children)
                if found is not None:
                    return found
            return None

        return visit(self.tasks)

    def _flag_duplicate_siblings(self, advisories: list[str]) -> None:
        """Advise on same-titled sibling steps (ADR-0050 — the duplicate-subtask check).

        The ayoai-processor's ``HTNPlanner.check_subtasks`` DROPPED a duplicate task key at
        one decomposition level as a loop indicator (``moveTo → moveTo → moveTo``); the
        open-domain analog is an identical title among siblings. Dropping a model-authored
        step would violate fail-open authoring, so it is advised, never dropped.
        """

        def visit(siblings: list[Task]) -> None:
            seen: dict[str, str] = {}
            for task in siblings:
                key = " ".join(task.title.lower().split())
                if key and key in seen:
                    advisories.append(
                        f"steps {seen[key]} and {task.id} have the same title ({task.title!r}) "
                        "— possible duplicated or looping step; merge or differentiate them."
                    )
                else:
                    seen[key] = task.id
                visit(task.children)

        visit(self.tasks)

    @staticmethod
    def _assign_ids(tasks: list[Task], *, prefix: str) -> None:
        for index, task in enumerate(tasks, start=1):
            task.id = f"{prefix}{index}"
            TaskNetwork._assign_ids(task.children, prefix=f"{task.id}.")

    def _sanitize_dependencies(self, advisories: list[str]) -> None:
        """Make ``blocked_by`` a valid DAG: drop self/unknown references, then break cycles.

        Fail-open by design — a malformed dependency graph must never freeze the agent, so every
        invalid edge is dropped (with an advisory) rather than raised. Runs after ids are assigned
        so references resolve against the current positions.
        """
        by_id = {t.id: t for t in self._iter()}
        for task in self._iter():
            kept: list[str] = []
            for dep in task.blocked_by:
                if dep == task.id:
                    advisories.append(f"task {task.id} cannot depend on itself; dropped.")
                elif dep not in by_id:
                    advisories.append(f"task {task.id} depends on unknown step {dep!r}; dropped.")
                elif dep not in kept:  # de-duplicate, preserve order
                    kept.append(dep)
            task.blocked_by = kept
        self._break_dependency_cycles(by_id, advisories)

    @staticmethod
    def _break_dependency_cycles(by_id: dict[str, Task], advisories: list[str]) -> None:
        """Drop the back-edges that would make ``blocked_by`` cyclic (depth-first, fail-open)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(by_id, WHITE)

        def visit(tid: str) -> None:
            color[tid] = GRAY
            task = by_id[tid]
            kept: list[str] = []
            for dep in task.blocked_by:
                if color[dep] == GRAY:  # back-edge into the current stack -> would cycle
                    advisories.append(
                        f"dropped dependency {tid} -> {dep} (it would create a cycle)."
                    )
                    continue
                kept.append(dep)
                if color[dep] == WHITE:
                    visit(dep)
            task.blocked_by = kept
            color[tid] = BLACK

        for tid in by_id:
            if color[tid] == WHITE:
                visit(tid)

    def _deps_satisfied(self, task: Task, by_id: dict[str, Task]) -> bool:
        """True when every dependency of ``task`` has reached a terminal status."""
        return all(by_id[d].status in _TERMINAL for d in task.blocked_by if d in by_id)

    @staticmethod
    def _derive_status(task: Task, advisories: list[str]) -> None:
        """Bottom-up: a compound's status is a pure function of its children's."""
        if not task.children:
            if task.kind == "compound":
                advisories.append(
                    f"task {task.id} ({task.title!r}) is compound but has no sub-tasks — "
                    "decompose it into primitive steps before working on it."
                )
            return
        for child in task.children:
            TaskNetwork._derive_status(child, advisories)
        statuses = [c.status for c in task.children]
        if all(s == "cancelled" for s in statuses):
            task.status = "cancelled"
        elif all(s in _TERMINAL for s in statuses):
            task.status = "done"
        elif any(s == "blocked" for s in statuses) and not any(
            s == "in_progress" for s in statuses
        ):
            task.status = "blocked"
        elif any(s == "in_progress" for s in statuses) or any(s in _TERMINAL for s in statuses):
            task.status = "in_progress"
        else:
            task.status = "pending"

    def _enforce_single_focus(self, advisories: list[str]) -> None:
        """Keep at most one primitive ``in_progress`` (focus discipline); demote the rest."""
        seen = False
        for task in self._iter():
            if task.is_compound() or task.status != "in_progress":
                continue
            if seen:
                task.status = "pending"
                advisories.append(
                    f"task {task.id} ({task.title!r}) was demoted to pending — keep exactly "
                    "one task in_progress at a time so the focus is unambiguous."
                )
            else:
                seen = True

    # ── traversal ────────────────────────────────────────────────────────────────

    def _iter(self) -> list[Task]:
        """All tasks in document (depth-first, left-to-right) order."""
        out: list[Task] = []

        def visit(nodes: list[Task]) -> None:
            for node in nodes:
                out.append(node)
                visit(node.children)

        visit(self.tasks)
        return out

    def leaves(self) -> list[Task]:
        """Leaf tasks in document order — the primitives (plus any under-decomposed compound).

        A childless node is a leaf regardless of declared kind; an under-decomposed compound
        therefore appears here so the frontier surfaces it as the thing blocking progress.
        """
        return [t for t in self._iter() if not t.children]

    # ── derived plan state (consumed by the loop's plan gate + re-injection) ──────

    def is_empty(self) -> bool:
        """True when no plan has been laid out yet."""
        return not self.tasks

    def current(self) -> Task | None:
        """The task the model should be working on now.

        Order of preference: the first ``in_progress`` leaf; else the first non-terminal,
        non-``blocked`` leaf in document order **whose dependencies are all satisfied**; else
        (fail-open, so a tangled dependency graph can never freeze the agent) the first
        non-terminal, non-``blocked`` leaf ignoring dependencies; else ``None`` (plan done).
        """
        leaves = self.leaves()
        by_id = {t.id: t for t in self._iter()}
        for leaf in leaves:
            if leaf.status == "in_progress":
                return leaf
        actionable = [
            leaf for leaf in leaves if leaf.status not in _TERMINAL and leaf.status != "blocked"
        ]
        for leaf in actionable:
            if self._deps_satisfied(leaf, by_id):
                return leaf
        return actionable[0] if actionable else None

    def actionable_remaining(self) -> list[Task]:
        """Leaves that still owe work — neither finished (``done``/``cancelled``) nor blocked.

        This is the set the completion gate guards: a turn must not end ``completed`` while it
        is non-empty (the model must either do the work or explicitly mark it done/cancelled).
        """
        return [t for t in self.leaves() if t.status not in _TERMINAL and t.status != "blocked"]

    def undecomposed(self) -> list[Task]:
        """Compound tasks with no children — declared as goals but not yet broken down."""
        return [t for t in self._iter() if t.kind == "compound" and not t.children]

    def is_complete(self) -> bool:
        """True when a plan exists and every top-level task reached a terminal status."""
        return bool(self.tasks) and all(t.status in _TERMINAL for t in self.tasks)

    def progress(self) -> tuple[int, int]:
        """``(finished, total)`` over primitive leaves — the headline progress fraction."""
        leaves = self.leaves()
        total = len(leaves)
        finished = sum(1 for leaf in leaves if leaf.status in _TERMINAL)
        return finished, total

    def has_step_in_flight(self) -> bool:
        """True while any task is ``in_progress`` — the model is mid-step, not between steps.

        The task-boundary say hold (ADR-0052) keys on this: a pending user message waits for
        the current step's seam rather than landing mid-focus.
        """
        return any(t.status == "in_progress" for t in self._iter())

    def progress_signature(self) -> str:
        """A stable ``(id, status, title)`` snapshot of every task in document order.

        Two networks with equal signatures are identical in structure (ids), advancement
        (statuses), and content (titles): any step transition, decomposition, or plan edit changes
        it, while an untouched plan reproduces it exactly. The loop's staleness guard (issue #32)
        compares this across turn-starts to detect an abandoned plan — one the model has neither
        advanced nor edited — and drop it rather than let it haunt. Including the title makes the
        guard conservative (any edit resets the idle counter, the fail-safe direction); ``repr`` of
        a tuple list avoids delimiter-collision ambiguity and is only ever compared for equality.
        """
        return repr([(t.id, t.status, t.title) for t in self._iter()])

    def structure_signature(self) -> str:
        """A stable ``(id, kind, title)`` snapshot — the plan's SHAPE, statuses excluded.

        Two networks with equal structure signatures decompose the goal identically: only
        adding, removing, retitling, or re-nesting steps changes it, while marking steps
        done / in_progress does not. The judged-plan critique (ADR-0050) compares this
        across an ``update_plan`` call so status ticks never re-trigger judgment.
        """
        return repr([(t.id, t.kind, t.title) for t in self._iter()])

    # ── structural quality (ADR-0050 — the ayoai-processor evaluate_candidate port) ─────

    def quality(self) -> tuple[float, list[str]]:
        """Deterministic structural quality of the plan in ``[0, 1]``, with named deficiencies.

        A port of the ayoai-processor's ``HTNPlanner.evaluate_candidate`` — the production
        HTN planner whose decomposition discipline this substrate mirrors — re-grounded for
        the open domain with the same three terms and weights (0.5 / 0.3 / 0.2):

        * **completeness** — there, ``1 - unresolved/total``; here the unresolved node is
          the compound with no children (a declared goal nobody broke down).
        * **verifiability** (the *feasibility* analog) — there, a primitive was feasible
          when the closed task VOCABULARY knew it; an open workspace has no vocabulary, so
          the evidence a step is executable-and-checkable is its ``note`` done-condition.
        * **granularity** (the *efficiency* analog) — a mild penalty past 10 primitives.
          The processor penalized from the first primitive, right for NPC micro-plans and
          wrong for code plans, so the curve is shifted rather than copied.

        Pure and free — no model call; the judged rubric (:mod:`zakcode.quality.plan`) is
        the semantic complement, scoring what structure alone cannot see. An empty plan
        scores 1.0 (nothing to fault). Deficiencies name the offending step ids so the
        surfaced line is actionable, not a bare number.
        """
        nodes = self._iter()
        if not nodes:
            return 1.0, []
        deficiencies: list[str] = []
        undecomposed = [t for t in nodes if t.kind == "compound" and not t.children]
        completeness = 1.0 - (len(undecomposed) / len(nodes))
        if undecomposed:
            ids = ", ".join(t.id for t in undecomposed[:3])
            deficiencies.append(f"{len(undecomposed)} step(s) not yet decomposed ({ids})")
        primitives = [t for t in self.leaves() if t.kind == "primitive"]
        noted = sum(1 for t in primitives if t.note)
        verifiability = (noted / len(primitives)) if primitives else 0.0
        missing = [t for t in primitives if not t.note]
        if missing:
            ids = ", ".join(t.id for t in missing[:3])
            deficiencies.append(
                f"{len(missing)} step(s) lack a done-condition note ({ids}) — say how each "
                "step is verified"
            )
        granularity = 1.0 / (1.0 + 0.1 * max(0, len(primitives) - 10))
        if len(primitives) > 10:
            deficiencies.append(
                f"{len(primitives)} primitive steps — merge trivial ones so each is one focused run"
            )
        score = 0.5 * completeness + 0.3 * verifiability + 0.2 * granularity
        return round(score, 3), deficiencies

    # ── rendering (the live plan folded back into context) ────────────────────────

    def render(self) -> str:
        """A compact, indented checklist of the live plan, marking the current task.

        This is the text re-injected near the end of context each iteration to counter
        instruction fade-out — the model's standing view of "the plan, and where I am in it".
        Empty plan renders as ``""`` so the caller injects nothing on un-planned turns.
        """
        if self.is_empty():
            return ""
        current = self.current()
        current_id = current.id if current is not None else None
        finished, total = self.progress()
        lines = [f"Current plan ({finished}/{total} steps done):"]

        def render_nodes(nodes: list[Task], depth: int) -> None:
            for node in nodes:
                glyph = _GLYPH.get(node.status, " ")
                indent = "  " * (depth + 1)
                marker = "  <- current" if node.id == current_id else ""
                # A closed step shows what it PRODUCED (ADR-0110); an open one its done-condition.
                shown = node.outcome if node.status in _TERMINAL and node.outcome else node.note
                detail = f" — {shown}" if shown else ""
                deps = f" (after {', '.join(node.blocked_by)})" if node.blocked_by else ""
                lines.append(f"{indent}[{glyph}] {node.id} {node.title}{detail}{deps}{marker}")
                render_nodes(node.children, depth + 1)

        render_nodes(self.tasks, 0)
        return "\n".join(lines)


# ── skill skeletons: a skill body's numbered sections as plan steps (ADR-0062) ─────────
#: Section headings that read as ORDERED WORK — the checklist a skill author already wrote
#: ("## Phase 1: …", "### Step 2.3 …", "## Lane 4 …", "## 3. …"). Anything else ("## Syntax",
#: "## Return Protocol", "## Chaining") is documentation and never becomes a step. Measured
#: on a Mind deployment's 130 skills: 78 carry such sections, 52 carry none.
_STEP_HEADING_RE = re.compile(
    r"^(#{2,3})\s+(?:\*\*)?(?:phase|step|lane|stage|part|task|\d+[.)])(?![a-z-])", re.I
)
#: The same ordered-work markers written as a bold lead-in instead of a heading — the shape
#: a Mind's control skills use ("**Step 0.7: Recovery Branch** — …", "**Step 1 — Find the
#: right session:**", "- **Step 2**: …"): at line start (a list bullet allowed), the word, a
#: number, then a separator or the closing bold. "**Phase 6 spark is NOT wrapped …**" is
#: prose (a word follows the number) and never matches. Measured (ADR-0064): /start, /stop
#: and /aspirations carry no step-like heading at all and 6 / 4 / 2 of these; a first
#: `/start` on a small model stopped after "Following Step 0.7 cleanup sequence." with an
#: empty plan and nothing to hold it.
_STEP_BOLD_RE = re.compile(
    r"^\s*(?:[-*+]\s+)?\*\*(?:phase|step|lane|stage|part|task)\s+\d[\w.]*\s*(?:[:.)\-—–]|\*\*)",
    re.I,
)
#: A skeleton never exceeds this many top-level steps; the rest fold into one closing step.
_MAX_SKELETON_STEPS = 60
#: …nor this many sub-steps under one section (the overflow is counted in the section's note).
_MAX_SKELETON_CHILDREN = 12
#: Step titles are the heading text, trimmed to this many characters.
_MAX_SKELETON_TITLE = 100


def _heading_title(line: str) -> str:
    text = " ".join(line.lstrip("#").replace("**", "").replace("`", "").split())
    if len(text) > _MAX_SKELETON_TITLE:
        text = text[: _MAX_SKELETON_TITLE - 1].rstrip() + "…"
    return text


_BARE_STEP_MARKER_RE = re.compile(
    r"^(?:phase|step|lane|stage|part|task)\s+(?:[a-z]{1,3}-?)?\d[\w.-]*\s*[:.)\-—–]*\s*$", re.I
)


def _bold_title(line: str) -> str:
    """The title of a ``**Step N …**`` lead-in: the bold span without its trailing separator
    — or, when the span is the bare marker (``**Step 2**: resume the agent.``), the marker
    plus the first sentence that follows it."""
    _, span, *tail = line.split("**")
    title = _heading_title(span).rstrip(" :—–-")
    if _BARE_STEP_MARKER_RE.match(span.replace("`", "")):
        clause = _heading_title("".join(tail)).lstrip(" :—–-")
        clause = re.split(r"\.(?:\s|$)", clause, maxsplit=1)[0].strip()
        if clause:
            title = f"{title}: {clause}"
    return _heading_title(title)


def skill_skeleton(body: str, *, skill: str) -> list[Task]:
    """Plan steps from a skill body's sections (ADR-0062) — ``[]`` when it has none.

    One step per page of :func:`skill_pages` (the same outline — page ``k`` is always step
    ``k``): a step-like ``##`` heading; a step-like ``###`` heading, a bold ``**Step N …**``
    lead-in (ADR-0064) or a fenced ``# Phase N`` comment (ADR-0084) that sits under no
    step section; and the parts a section over the page budget was cut into. Markers
    inside a page that was not cut become its sub-steps (the section becomes
    ``compound``). Every note opens with the ``from /<skill>`` marker — how the loop
    recognises a skeleton it already seeded. Pure: no model, no I/O.
    """
    outline = _outline(body, skill=skill)
    if len(outline.pages) < 2 and not any(unit.marker for unit in outline.pages):
        return []  # no numbered section anywhere: the plan is the model's own (ADR-0062)
    note = f"from /{skill}; done when this section has been carried out"
    top: list[Task] = []
    for unit in outline.pages:
        if unit.folded:
            top.append(
                Task(
                    title=unit.title,
                    note=f"from /{skill}; add them to the plan with update_plan as you reach them",
                )
            )
            continue
        task = Task(title=unit.title, note=note)
        for child in unit.children[:_MAX_SKELETON_CHILDREN]:
            task.children.append(Task(title=child, note=note))
            task.kind = "compound"
        extra = len(unit.children) - _MAX_SKELETON_CHILDREN
        if extra > 0:
            task.note += f" (+{extra} more sub-sections not listed)"
        top.append(task)
    return top


# ── skill pages: a sectioned skill delivered one section at a time (ADR-0067) ──────────
#: The most skill text one page puts in context — ~3–4k tokens on the tokenizers measured
#: (ADR-0084). A section over this is cut at its deeper ordered-work markers, then at any
#: heading, then at paragraph breaks; front matter over it is paged the same way. Measured
#: 2026-08-29 on a Mind deployment: nine of its largest skills (51–116 KB) had no step-like
#: ``##`` heading and were delivered whole — /worker-loop, 84 KB, on every worker unit.
PAGE_BUDGET_CHARS = 12_000
#: An ordered-work marker written as a pseudocode comment at column 0 inside a fenced
#: block — the shape a Mind's loop skills use ("# Phase -0.5 — LIGHT PRIME …", "# Phase 1 —
#: SELECT …"): /worker-loop is one fence carrying 23 of these and no heading at all.
_STEP_FENCED_RE = re.compile(
    r"^#\s*(?:phase|step|lane|stage|part|task)\s+-?\d[\w.]*(?![\w-])", re.I
)
#: Any heading outside a fence — the cut of last resort before paragraph breaks.
_ANY_HEADING_RE = re.compile(r"^#{2,4}\s+\S")
#: The leading ordered-work token of a section title — ``Step 0.7``, ``Phase -3``, ``Lane 1``,
#: ``3.`` — the part a model keeps when it rewrites the step ("Step 0.5 + 0.6: …"), so a page
#: can still find its step after the plan has been reshaped.
#: A step id may carry a short letter prefix (``Step B2.5``, ``Phase GS-1``, ``Phase S4.6``):
#: measured 2026-08-29 over coach's 39 paged skills, 60 of 210 pages had no marker, and
#: the lettered ids were a third of them (ADR-0095).
_SECTION_MARKER_RE = re.compile(
    r"^(?:(phase|step|lane|stage|part|task)\s+(-?(?:[a-z]{1,3}-?)?\d[\w.-]*)|(\d+)[.)])", re.I
)
#: The header every delivered page opens with — and the marker the loop reads back from the
#: transcript to learn how far a skill was paged before a restart.
PAGE_HEADER_RE = re.compile(r"\[/(?P<skill>[^\s\]]+) — page (?P<page>\d+) of (?P<count>\d+):")
#: The cut levels inside a section, shallowest first: step-like ``###`` headings and bold
#: lead-ins, fenced ``# Phase N`` comments, any heading. Paragraph breaks come after all.
_CUT_LEVELS = ("sub", "fenced", "heading")


def _section_marker(heading: str) -> str:
    text = _heading_title(heading).lower()
    match = _SECTION_MARKER_RE.match(text)
    if match is None:
        return ""
    if match.group(1):
        return f"{match.group(1)} {match.group(2).rstrip('.:')}"
    return match.group(3)


def _fenced_title(line: str) -> str:
    """The title of a fenced ``# Phase N …`` comment: its first sentence."""
    text = line.lstrip("#").strip()
    return _heading_title(re.split(r"\.(?:\s|$)", text, maxsplit=1)[0])


@dataclass
class _Unit:
    """One page of a skill: its title, marker token, delivered text, and the sub-markers
    inside it (the skeleton's sub-steps). ``folded`` marks the closing catch-all page."""

    title: str
    marker: str
    text: str
    children: list[str] = field(default_factory=list)
    folded: bool = False
    #: The sections a packed page carries (ADR-0088), ``(title, marker)`` each; empty for
    #: a page that is one section.
    sections: list[tuple[str, str]] = field(default_factory=list)


def _pack(pages: list[_Unit]) -> list[_Unit]:
    """Consecutive small sections share a page up to the budget (ADR-0088).

    A page costs a model turn to deliver — on a slow pod, minutes — and a Mind's skills
    are mostly short sections: measured 2026-08-29 over 131 skills, 976 pages became 321
    deliveries at the budget (/aspirations-precheck 55 → 18, /boot 26 → 4, /respond
    35 → 7). A section over the budget stays alone (it was already cut to fit); the
    folded closing page is never packed. A packed page is titled by its first section and
    counts the rest; its sections become the skeleton step's sub-steps and any of them
    names the page when the model rewrites the plan.
    """
    out: list[_Unit] = []
    run: list[_Unit] = []

    def flush() -> None:
        if not run:
            return
        if len(run) == 1:
            out.append(run[0])
        else:
            out.append(
                _Unit(
                    f"{run[0].title} (+{len(run) - 1} more)",
                    run[0].marker,
                    "\n\n".join(unit.text for unit in run),
                    [unit.title for unit in run],
                    sections=[(unit.title, unit.marker) for unit in run],
                )
            )
        run.clear()

    for unit in pages:
        if unit.folded:
            flush()
            out.append(unit)
            continue
        if run and sum(len(held.text) + 2 for held in run) + len(unit.text) > PAGE_BUDGET_CHARS:
            flush()
        run.append(unit)
    flush()
    return out


@dataclass
class _Outline:
    """The shared reading of a skill body: what travels up front, and the pages in order.
    ``paged`` is False when the whole body packs into one page (ADR-0088): it is then
    delivered whole, and ``pages`` are its sections — the plan's steps, not pages."""

    front: str
    pages: list[_Unit]
    paged: bool = True


class _Body:
    """A skill body as lines, with each line's fence state — the walk both projections use."""

    def __init__(self, body: str) -> None:
        self.lines = body.splitlines()
        self.state: list[str] = []  # out | open | in | close
        self.opener: list[str] = []  # the opening fence line of a fenced line's block
        inside, current = False, ""
        for line in self.lines:
            if line.lstrip().startswith(("```", "~~~")):
                if inside:
                    self.state.append("close")
                    self.opener.append(current)
                    inside, current = False, ""
                else:
                    inside, current = True, line
                    self.state.append("open")
                    self.opener.append(current)
            else:
                self.state.append("in" if inside else "out")
                self.opener.append(current if inside else "")

    def size(self, start: int, end: int) -> int:
        return sum(len(line) + 1 for line in self.lines[start:end])

    def blank(self, start: int, end: int) -> bool:
        return all(not line.strip() for line in self.lines[start:end])

    def text(self, start: int, end: int) -> str:
        """``lines[start:end]`` as a page: a cut made inside a fence is re-fenced on both
        sides, so every page is well-formed markdown on its own."""
        chunk = list(self.lines[start:end])
        if not chunk:
            return ""
        if self.state[start] in ("in", "close"):
            chunk.insert(0, self.opener[start] or "```")
        if self.state[end - 1] in ("in", "open"):
            chunk.append("```")
        return "\n".join(chunk).rstrip()

    def markers(self, start: int, end: int, level: str) -> list[tuple[int, str, str]]:
        """``(line, title, marker)`` for every boundary of ``level`` in ``lines[start:end]``."""
        out: list[tuple[int, str, str]] = []
        for i in range(start, end):
            line = self.lines[i]
            state = self.state[i]
            if level == "fenced":
                if state == "in" and _STEP_FENCED_RE.match(line):
                    title = _fenced_title(line)
                    out.append((i, title, _section_marker(title)))
                continue
            if state != "out":
                continue
            if level == "top":
                match = _STEP_HEADING_RE.match(line)
                if match is not None and len(match.group(1)) == 2:
                    out.append((i, _heading_title(line), _section_marker(line)))
            elif level == "sub":
                match = _STEP_HEADING_RE.match(line)
                if match is not None and len(match.group(1)) == 3:
                    out.append((i, _heading_title(line), _section_marker(line)))
                elif _STEP_BOLD_RE.match(line):
                    title = _bold_title(line)
                    out.append((i, title, _section_marker(title)))
            elif level == "heading" and _ANY_HEADING_RE.match(line):
                out.append((i, _heading_title(line), _section_marker(line)))
        return out

    def unit(self, title: str, marker: str, start: int, end: int) -> _Unit:
        children = self.markers(start + 1, end, "sub") or self.markers(start + 1, end, "fenced")
        return _Unit(title, marker, self.text(start, end), [t for _, t, _ in children])

    def bounded(
        self, title: str, marker: str, start: int, end: int, levels: tuple[str, ...]
    ) -> list[_Unit]:
        """``lines[start:end]`` as pages under the budget: one page when it fits; else cut at
        the first level in ``levels`` with a marker inside (the span's own first line is
        never a cut), each piece bounded again by the levels after it; paragraphs last."""
        if self.size(start, end) <= PAGE_BUDGET_CHARS:
            return [self.unit(title, marker, start, end)]
        for k, level in enumerate(levels):
            found = self.markers(start + 1, end, level)
            if not found:
                continue
            cuts = [start, *(i for i, _, _ in found), end]
            names = [(title, marker), *((t, m) for _, t, m in found)]
            units: list[_Unit] = []
            for j, (name, mark) in enumerate(names):
                piece_start, piece_end = cuts[j], cuts[j + 1]
                if self.blank(piece_start, piece_end) or (
                    j == 0 and self.blank(piece_start + 1, piece_end)
                ):
                    continue  # a heading with nothing under it before the first cut
                units.extend(self.bounded(name, mark, piece_start, piece_end, levels[k + 1 :]))
            return units
        return self.paragraphs(title, marker, start, end)

    def paragraphs(self, title: str, marker: str, start: int, end: int) -> list[_Unit]:
        """Cut ``lines[start:end]`` at blank lines and pack the paragraphs up to the budget;
        a lone paragraph over it stays whole. Parts are titled ``<title> (k/n)``."""
        breaks = [start]
        for i in range(start, end - 1):
            if not self.lines[i].strip() and self.lines[i + 1].strip():
                breaks.append(i + 1)
        breaks.append(end)
        # The span's own title line is not a paragraph: it stays with the text under it.
        if len(breaks) > 2 and sum(bool(self.lines[i].strip()) for i in range(*breaks[:2])) == 1:
            del breaks[1]
        chunks: list[tuple[int, int]] = []
        chunk_start, size = start, 0
        for j in range(len(breaks) - 1):
            para_start, para_end = breaks[j], breaks[j + 1]
            para = self.size(para_start, para_end)
            if size and size + para > PAGE_BUDGET_CHARS:
                chunks.append((chunk_start, para_start))
                chunk_start, size = para_start, 0
            size += para
        chunks.append((chunk_start, end))
        if len(chunks) == 1:
            return [self.unit(title, marker, start, end)]
        return [
            _Unit(f"{title} ({k}/{len(chunks)})", marker, self.text(s, e))
            for k, (s, e) in enumerate(chunks, 1)
        ]


def _outline(body: str, *, skill: str) -> _Outline:
    """Read a skill body once, for both the skeleton and the pages (ADR-0084).

    The body is partitioned at every ``##`` heading outside a fence. A step-like section
    is pages: one when it fits the budget, else cut at its deeper markers. Anything else
    — the preamble and each documentation section — travels up front when it fits and
    holds no ordered-work marker; a documentation section that HOLDS markers (a "## The
    loop" fence of ``# Phase`` comments, a "## Mode 1" of ``### Step`` headings) is a
    container: its intro goes up front and each marker opens a page; one over the budget
    with no markers is paged at its headings, then paragraphs. Consecutive small sections
    then share a page up to the budget (ADR-0088, :func:`_pack`); a skill that packs into
    ONE page is not paged at all — it is delivered whole (``paged`` False) and its
    sections stay the plan's steps. Past :data:`_MAX_SKELETON_STEPS` pages the rest fold
    into one closing page.
    """
    text = _Body(body)
    heads = [
        i for i, line in enumerate(text.lines) if text.state[i] == "out" and line.startswith("## ")
    ]
    bounds = [0, *heads, len(text.lines)]
    front: list[str] = []
    pages: list[_Unit] = []

    def intro(title: str, start: int, end: int) -> None:
        if text.blank(start, end):
            return
        if text.size(start, end) <= PAGE_BUDGET_CHARS:
            front.append(text.text(start, end))
        else:
            pages.extend(text.bounded(title, "", start, end, ("heading",)))

    for j in range(len(bounds) - 1):
        start, end = bounds[j], bounds[j + 1]
        if start == end:
            continue
        heading = text.lines[start] if j > 0 else ""
        match = _STEP_HEADING_RE.match(heading) if heading else None
        if match is not None and len(match.group(1)) == 2:
            pages.extend(
                text.bounded(
                    _heading_title(heading), _section_marker(heading), start, end, _CUT_LEVELS
                )
            )
            continue
        title = _heading_title(heading) if heading else f"/{skill}"
        body_start = start + 1 if heading else start
        for k, level in enumerate(("sub", "fenced")):
            inner = text.markers(body_start, end, level)
            if inner:
                intro(title, start, inner[0][0])
                cuts = [*(i for i, _, _ in inner), end]
                for n, (_, name, mark) in enumerate(inner):
                    pages.extend(
                        text.bounded(name, mark, cuts[n], cuts[n + 1], _CUT_LEVELS[k + 1 :])
                    )
                break
        else:
            intro(title, start, end)

    packed = _pack(pages)
    paged = len(packed) >= 2
    if paged:
        pages = packed
    if len(pages) > _MAX_SKELETON_STEPS:
        keep = _MAX_SKELETON_STEPS - 1
        rest = pages[keep:]
        pages = [
            *pages[:keep],
            _Unit(
                f"Remaining sections of /{skill} ({len(rest)} more)",
                "",
                "\n\n".join(unit.text for unit in rest),
                folded=True,
            ),
        ]
    return _Outline(front="\n\n".join(front), pages=pages, paged=paged)


#: The seeded marker where a rewrite tends to put it (ADR-0092). A seeded step's NOTE opens
#: with ``from /<skill>``; the plan renders a step as ``title — note``, and a model that
#: copies the render into its next ``update_plan`` folds ``— from /<skill>`` into the TITLE
#: and writes its own note — the very thing the rails ask for ("mark it cancelled with a
#: note saying why"). Measured 2026-08-29 (coach-w2, then every worker on the packed build):
#: after one rewrite no step carried the note marker, so the seeded structure was gone,
#: every page fell to title matching, /start's cancelled branches (no marker token) were
#: delivered as "dropped", and a token two sections share reopened the wrong page.
_NOTE_MARKER_RE = re.compile(r"^from\s+/([a-z0-9][a-z0-9_-]*)(?![a-z0-9_-])", re.I)
_TITLE_MARKER_RE = re.compile(
    r"\s+[—–-]+\s*from\s+/([a-z0-9][a-z0-9_-]*)(?![a-z0-9_-]).*$", re.I | re.S
)


def step_skill(title: str, note: str) -> str | None:
    """The skill a plan step was seeded from (lower-case): the marker its note opens with,
    else the one a rewrite carried into its title — ``None`` for a step of the model's own."""
    found = _NOTE_MARKER_RE.match(note) or _TITLE_MARKER_RE.search(title)
    return found.group(1).lower() if found else None


def bare_title(title: str) -> str:
    """``title`` without a marker a rewrite folded into it (see :func:`step_skill`)."""
    return _TITLE_MARKER_RE.sub("", title)


#: Words almost any title carries — they cannot tell one page from another. The marker
#: words are here because a marker is matched as a marker (:meth:`SkillPage.matches`).
_TITLE_STOP_WORDS = frozenset(
    {
        # fmt: off
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "only",
        "when",
        "then",
        "else",
        "not",
        "any",
        "all",
        "its",
        "are",
        "was",
        "more",
        "via",
        "per",
        "each",
        "one",
        "two",
        "how",
        "what",
        "which",
        "does",
        "done",
        "step",
        "steps",
        "phase",
        "phases",
        "lane",
        "stage",
        "part",
        "task",
        "section",
        # fmt: on
    }
)


def _title_tokens(title: str) -> frozenset[str]:
    """The words of a title that can tell it from another: lower-cased, three characters or
    more (``1/3`` and ``0.5`` count — they tell a split page's parts apart), minus the words
    almost any title carries."""
    return frozenset(
        token
        for token in re.findall(r"[\w./-]+", title.lower())
        if len(token) >= 3 and token not in _TITLE_STOP_WORDS
    )


@dataclass(frozen=True)
class SkillPage:
    """One page of a skill: the text the loop hands over when the plan reaches its step."""

    index: int
    title: str
    marker: str
    text: str
    #: The sections a packed page carries (ADR-0088), ``(title, marker)`` each; a step
    #: titled or marked like any of them is this page's.
    sections: tuple[tuple[str, str], ...] = ()

    def matches(self, step_title: str, *, exact: bool = False) -> bool:
        """Whether a plan step (possibly rewritten by the model) is this page's step: the
        seeded title verbatim — a marker the rewrite folded into it stripped first — or,
        unless ``exact``, the page's marker token as a whole word in the title; for the
        page itself or any section packed into it."""
        norm = " ".join(bare_title(step_title).lower().split())
        for title, marker in ((self.title, self.marker), *self.sections):
            if norm == " ".join(title.lower().split()):
                return True
            if not exact and marker and re.search(rf"(?<![\w.]){re.escape(marker)}(?![\w.])", norm):
                return True
        return False

    def overlap(self, step_title: str) -> int:
        """How many telling words a plan step's title shares with this page's title — or
        with a section packed into it, the best one counting — when that is at least two
        words, or every telling word the step has; else ``0``. A page whose heading is a
        branch name ("RUNNING + requested mode is autonomous", "IDLE (…) (2/3)") carries no
        marker, so a paraphrase of it ("RUNNING + autonomous mode", "IDLE (2/3)") can be
        found no other way (ADR-0095)."""
        step = _title_tokens(bare_title(step_title))
        if not step:
            return 0
        best = max(
            len(step & _title_tokens(title))
            for title, _ in ((self.title, self.marker), *self.sections)
        )
        return best if best >= 2 or best == len(step) else 0


@dataclass(frozen=True)
class SkillPages:
    """A skill split for paging (ADR-0067): ``front`` (the preamble and every documentation
    section that fits — the definitions and rules the pages rely on) travels with page 1;
    every page is a plan step, in order (ADR-0084: one per section, or per part of a
    section cut to the page budget)."""

    skill: str
    front: str
    pages: tuple[SkillPage, ...]

    @property
    def count(self) -> int:
        return len(self.pages)

    def header(self, index: int) -> str:
        page = self.pages[index - 1]
        return f"[/{self.skill} — page {index} of {self.count}: {page.title}]"

    def render(self, index: int) -> str:
        """Page ``index`` as delivered: header, the section, and what to do when it is done."""
        page = self.pages[index - 1]
        after = (
            "This is the last section."
            if index == self.count
            else (
                f"When this section is done, mark its step done with update_plan (send the "
                f"whole plan); section {index + 1} of {self.count} arrives in the reply to that "
                "call — nothing arrives on its own."
            )
        )
        return f"{self.header(index)}\n{page.text.rstrip()}\n\n{after}"

    def first(self) -> str:
        """What a load delivers: the front matter and page 1, with the paging contract."""
        parts = [self.front.rstrip(), self.render(1)]
        return "\n\n".join(p for p in parts if p)


def skill_pages(body: str, *, skill: str) -> SkillPages | None:
    """Split a skill body into pages (ADR-0067/0084), or ``None`` when it has fewer than
    two — such a body is delivered whole, as before. Page ``k`` is always skeleton step
    ``k``: both come from :func:`_outline`. Pure: no model, no I/O.
    """
    outline = _outline(body, skill=skill)
    if not outline.paged or len(outline.pages) < 2:
        return None
    pages = tuple(
        SkillPage(
            index=i + 1,
            title=unit.title,
            marker=unit.marker,
            text=unit.text,
            sections=tuple(unit.sections),
        )
        for i, unit in enumerate(outline.pages)
    )
    return SkillPages(skill=skill, front=outline.front, pages=pages)


__all__ = [
    "PAGE_HEADER_RE",
    "SkillPage",
    "SkillPages",
    "Task",
    "TaskNetwork",
    "TaskStatus",
    "TaskKind",
    "skill_pages",
    "skill_skeleton",
]
