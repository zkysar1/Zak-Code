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
    #: Optional one-line detail or acceptance criterion ("prints DONE", "all tests pass").
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
        self._enforce_single_focus(advisories)
        for task in self.tasks:
            self._derive_status(task, advisories)
        return advisories

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


__all__ = ["Task", "TaskNetwork", "TaskStatus", "TaskKind"]
