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
from dataclasses import dataclass
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

    def insert_before(self, anchor: Task | None, steps: list[Task]) -> None:
        """Splice harness-authored ``steps`` in as siblings directly AHEAD of ``anchor``.

        The decompose-on-stuck recovery (ADR-0057) uses this to put investigative steps in
        front of the step the model is stuck on: ``anchor`` is demoted to ``pending`` so the
        first new step becomes the current work, and the stuck step follows in document order
        — investigate, decide, then retry. ``anchor=None`` (no plan, or a finished one) appends
        at the top level: the investigation IS the plan. Normalizes afterwards, so ids, focus,
        and derived statuses are consistent for the next render.
        """
        siblings = self._siblings_of(anchor) if anchor is not None else None
        if siblings is None:
            self.tasks.extend(steps)
        else:
            if anchor is not None and anchor.status == "in_progress":
                anchor.status = "pending"
            index = next(i for i, task in enumerate(siblings) if task is anchor)
            siblings[index:index] = steps
        self.normalize()

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
                detail = f" — {node.note}" if node.note else ""
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
_MAX_SKELETON_STEPS = 40
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
    r"^(?:phase|step|lane|stage|part|task)\s+\d[\w.]*\s*[:.)\-—–]*\s*$", re.I
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
    """Plan steps from a skill body's numbered sections (ADR-0062) — ``[]`` when it has none.

    Walks the markdown outside fenced code: a step-like ``##`` heading becomes a top-level
    step; a step-like ``###`` heading, or a bold ``**Step N …**`` lead-in at line start
    (ADR-0064), a sub-step of the section it sits in (which becomes ``compound``) — or a
    step of its own when no section is open; a non-step ``##`` heading closes the current
    section. Every note opens with the ``from /<skill>`` marker — how the loop recognises a
    skeleton it already seeded. Pure: no model, no I/O.
    """
    note = f"from /{skill}; done when this section has been carried out"
    top: list[Task] = []
    parent: Task | None = None
    overflow: dict[int, int] = {}
    fenced = False
    for line in body.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = _STEP_HEADING_RE.match(line)
        if match is not None:
            task = Task(title=_heading_title(line), note=note)
            top_level = len(match.group(1)) == 2
        elif _STEP_BOLD_RE.match(line):
            task = Task(title=_bold_title(line), note=note)
            top_level = False
        else:
            if line.startswith("## "):
                parent = None
            continue
        if top_level:
            top.append(task)
            parent = task
        elif parent is None:
            top.append(task)
        elif len(parent.children) < _MAX_SKELETON_CHILDREN:
            parent.children.append(task)
            parent.kind = "compound"
        else:
            overflow[id(parent)] = overflow.get(id(parent), 0) + 1
    for section in top:
        extra = overflow.get(id(section))
        if extra:
            section.note += f" (+{extra} more sub-sections not listed)"
    if len(top) > _MAX_SKELETON_STEPS:
        rest = len(top) - (_MAX_SKELETON_STEPS - 1)
        top = top[: _MAX_SKELETON_STEPS - 1] + [
            Task(
                title=f"Remaining sections of /{skill} ({rest} more)",
                note=f"from /{skill}; add them to the plan with update_plan as you reach them",
            )
        ]
    return top


# ── skill pages: a sectioned skill delivered one section at a time (ADR-0067) ──────────
#: The leading ordered-work token of a section title — ``Step 0.7``, ``Phase -3``, ``Lane 1``,
#: ``3.`` — the part a model keeps when it rewrites the step ("Step 0.5 + 0.6: …"), so a page
#: can still find its step after the plan has been reshaped.
_SECTION_MARKER_RE = re.compile(
    r"^(?:(phase|step|lane|stage|part|task)\s+(-?\d[\w.]*)|(\d+)[.)])", re.I
)
#: The header every delivered page opens with — and the marker the loop reads back from the
#: transcript to learn how far a skill was paged before a restart.
PAGE_HEADER_RE = re.compile(r"\[/(?P<skill>[^\s\]]+) — page (?P<page>\d+) of (?P<count>\d+):")


def _section_marker(heading: str) -> str:
    text = _heading_title(heading).lower()
    match = _SECTION_MARKER_RE.match(text)
    if match is None:
        return ""
    if match.group(1):
        return f"{match.group(1)} {match.group(2).rstrip('.:')}"
    return match.group(3)


@dataclass(frozen=True)
class SkillPage:
    """One top-level section of a skill: the page the loop hands over when the plan reaches it."""

    index: int
    title: str
    marker: str
    text: str

    def matches(self, step_title: str) -> bool:
        """Whether a plan step (possibly rewritten by the model) is this section's step: the
        seeded title verbatim, or the section's marker token as a whole word in the title."""
        norm = " ".join(step_title.lower().split())
        if norm == " ".join(self.title.lower().split()):
            return True
        if not self.marker:
            return False
        return re.search(rf"(?<![\w.]){re.escape(self.marker)}(?![\w.])", norm) is not None


@dataclass(frozen=True)
class SkillPages:
    """A sectioned skill split for paging (ADR-0067): ``front`` (the preamble and every
    non-step ``##`` section — the definitions and rules a section relies on) travels with
    page 1; every top-level step section is a page, in order, one per skeleton step."""

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
                f"whole plan) and section {index + 1} of {self.count} arrives in the next message."
            )
        )
        return f"{self.header(index)}\n{page.text.rstrip()}\n\n{after}"

    def first(self) -> str:
        """What a load delivers: the front matter and page 1, with the paging contract."""
        parts = [self.front.rstrip(), self.render(1)]
        return "\n\n".join(p for p in parts if p)


def skill_pages(body: str, *, skill: str) -> SkillPages | None:
    """Split a skill body into pages by its top-level step sections (ADR-0067), or ``None``
    when it has fewer than two — such a body is delivered whole, as before.

    The same walk as :func:`skill_skeleton` (outside fenced code; a step-like ``##`` heading
    opens a section; a non-step ``##`` heading is documentation and goes to ``front``), folded
    past :data:`_MAX_SKELETON_STEPS` exactly as the skeleton folds, so page ``k`` is always
    skeleton step ``k``. Pure: no model, no I/O.
    """
    lines = body.splitlines()
    boundaries: list[tuple[int, bool]] = []  # (line index, is a step section)
    fenced = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or not line.startswith("## "):
            continue
        match = _STEP_HEADING_RE.match(line)
        boundaries.append((i, match is not None and len(match.group(1)) == 2))
    if sum(1 for _, is_step in boundaries if is_step) < 2:
        return None
    front: list[str] = []
    sections: list[tuple[str, str, str]] = []
    preamble = "\n".join(lines[: boundaries[0][0]]).strip()
    if preamble:
        front.append(preamble)
    for j, (start, is_step) in enumerate(boundaries):
        end = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        chunk = "\n".join(lines[start:end]).rstrip()
        if is_step:
            sections.append((_heading_title(lines[start]), _section_marker(lines[start]), chunk))
        else:
            front.append(chunk)
    if len(sections) > _MAX_SKELETON_STEPS:
        keep = _MAX_SKELETON_STEPS - 1
        rest = sections[keep:]
        sections = sections[:keep] + [
            (
                f"Remaining sections of /{skill} ({len(rest)} more)",
                "",
                "\n\n".join(text for _, _, text in rest),
            )
        ]
    pages = tuple(
        SkillPage(index=i + 1, title=title, marker=marker, text=text)
        for i, (title, marker, text) in enumerate(sections)
    )
    return SkillPages(skill=skill, front="\n\n".join(front), pages=pages)


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
