"""Tests for the hierarchical task-network substrate (``zakcode.agent.tasks``)."""

from __future__ import annotations

from zakcode.tasks import Task, TaskNetwork


def _net(*tasks: Task) -> TaskNetwork:
    net = TaskNetwork(tasks=list(tasks))
    net.normalize()
    return net


def test_empty_network_renders_nothing_and_is_not_complete() -> None:
    net = TaskNetwork()
    assert net.normalize() == []
    assert net.is_empty()
    assert net.render() == ""
    assert not net.is_complete()
    assert net.current() is None
    assert net.progress() == (0, 0)


def test_ids_are_assigned_as_dotted_paths_by_position() -> None:
    net = _net(
        Task(title="first"),
        Task(
            title="second",
            kind="compound",
            children=[Task(title="2a"), Task(title="2b")],
        ),
    )
    ids = [t.id for t in net._iter()]
    assert ids == ["1", "2", "2.1", "2.2"]


def test_compound_status_is_derived_from_children_not_trusted() -> None:
    # Parent is *declared* done, but a child is still pending -> derived back to in_progress.
    net = _net(
        Task(
            title="parent",
            kind="compound",
            status="done",
            children=[
                Task(title="a", status="done"),
                Task(title="b", status="pending"),
            ],
        )
    )
    parent = net.tasks[0]
    assert parent.status == "in_progress"

    # All children terminal -> parent done.
    net2 = _net(
        Task(
            title="parent",
            kind="compound",
            children=[Task(title="a", status="done"), Task(title="b", status="cancelled")],
        )
    )
    assert net2.tasks[0].status == "done"
    assert net2.is_complete()


def test_single_in_progress_focus_is_enforced() -> None:
    net = _net(
        Task(title="a", status="in_progress"),
        Task(title="b", status="in_progress"),
    )
    statuses = [t.status for t in net.tasks]
    assert statuses == ["in_progress", "pending"]  # second demoted


def test_current_prefers_in_progress_then_first_unfinished() -> None:
    net = _net(
        Task(title="a", status="done"),
        Task(title="b", status="pending"),
        Task(title="c", status="in_progress"),
    )
    assert net.current().title == "c"

    net2 = _net(
        Task(title="a", status="done"),
        Task(title="b", status="pending"),
        Task(title="c", status="pending"),
    )
    assert net2.current().title == "b"


def test_current_descends_into_compound_children() -> None:
    net = _net(
        Task(title="done-leaf", status="done"),
        Task(
            title="parent",
            kind="compound",
            children=[Task(title="sub-a", status="done"), Task(title="sub-b", status="pending")],
        ),
    )
    cur = net.current()
    assert cur is not None and cur.title == "sub-b" and cur.id == "2.2"


def test_blocked_leaf_is_not_actionable_and_skipped_by_current() -> None:
    net = _net(
        Task(title="a", status="blocked", note="waiting on api key"),
        Task(title="b", status="pending"),
    )
    assert net.current().title == "b"
    assert [t.title for t in net.actionable_remaining()] == ["b"]


def test_under_decomposed_compound_is_flagged_and_surfaced() -> None:
    net = TaskNetwork(tasks=[Task(title="big goal", kind="compound")])
    advisories = net.normalize()
    assert any("decompose" in a for a in advisories)
    assert [t.title for t in net.undecomposed()] == ["big goal"]
    # A childless compound is still a leaf, so it shows up as actionable (blocking progress).
    assert net.current().title == "big goal"


def test_progress_counts_only_primitive_leaves() -> None:
    net = _net(
        Task(
            title="parent",
            kind="compound",
            children=[Task(title="a", status="done"), Task(title="b", status="pending")],
        ),
        Task(title="standalone", status="done"),
    )
    # leaves: a, b, standalone -> 2 of 3 finished (parent is not a leaf).
    assert net.progress() == (2, 3)


def test_actionable_remaining_drives_completion_gate() -> None:
    net = _net(Task(title="a", status="done"), Task(title="b", status="pending"))
    assert [t.title for t in net.actionable_remaining()] == ["b"]
    assert not net.is_complete()

    done = _net(Task(title="a", status="done"), Task(title="b", status="cancelled"))
    assert done.actionable_remaining() == []
    assert done.is_complete()


def test_render_marks_current_and_shows_progress_and_glyphs() -> None:
    net = _net(
        Task(title="set up", status="done"),
        Task(
            title="build",
            kind="compound",
            children=[
                Task(title="route", status="in_progress"),
                Task(title="handler", status="pending", note="returns 200"),
            ],
        ),
        Task(title="test", status="pending"),
    )
    out = net.render()
    # Leaves: set up, route, handler, test (4 primitives); "build" is compound, not counted.
    assert "Current plan (1/4 steps done):" in out
    assert "[x] 1 set up" in out
    assert "[~] 2.1 route  <- current" in out
    assert "[ ] 2.2 handler — returns 200" in out
    assert "[ ] 3 test" in out


def test_focus_enforced_before_derivation_leaves_no_stale_parent() -> None:
    # Two compounds each with an in_progress child. Single-focus keeps only the first child
    # in_progress; the second parent MUST derive to pending, not a stale in_progress. (This is
    # the bug that the old derive-then-enforce ordering produced.)
    net = _net(
        Task(title="g1", kind="compound", children=[Task(title="a", status="in_progress")]),
        Task(title="g2", kind="compound", children=[Task(title="b", status="in_progress")]),
    )
    assert net.tasks[0].children[0].status == "in_progress"
    assert net.tasks[0].status == "in_progress"
    assert net.tasks[1].children[0].status == "pending"  # demoted by single-focus
    assert net.tasks[1].status == "pending"  # derived AFTER demotion — not stale in_progress


def test_dependencies_reorder_current_until_satisfied() -> None:
    # Step 1 depends on step 2, so the frontier is step 2 first.
    net = _net(Task(title="write tests", blocked_by=["2"]), Task(title="implement"))
    assert net.current().title == "implement"
    # Once the dependency is done, the dependent step becomes current.
    net.tasks[1].status = "done"
    net.normalize()
    assert net.current().title == "write tests"
    assert "(after 2)" in net.render()


def test_dependencies_drop_self_unknown_and_cycles_failopen() -> None:
    # 1 depends on {2, 9 (unknown), 1 (self)}; 2 depends on 1 -> a 1<->2 cycle.
    net = _net(Task(title="a", blocked_by=["2", "9", "1"]), Task(title="b", blocked_by=["1"]))
    # self + unknown dropped; the cycle is broken so the graph is a DAG.
    assert net.tasks[0].blocked_by == ["2"]
    assert net.tasks[1].blocked_by == []  # the back-edge 2->1 was dropped
    # Crucially, the agent is never frozen: current() still returns an actionable leaf.
    assert net.current() is not None


def test_blocked_dependency_does_not_remove_work_from_the_gate() -> None:
    # A step waiting on an unfinished dependency still "owes work" (drives the completion gate).
    net = _net(Task(title="a"), Task(title="b", blocked_by=["1"]))
    assert {t.title for t in net.actionable_remaining()} == {"a", "b"}
    assert not net.is_complete()


def test_normalize_is_idempotent() -> None:
    net = _net(
        Task(
            title="parent",
            kind="compound",
            children=[Task(title="a", status="done"), Task(title="b", status="in_progress")],
        )
    )
    first = net.render()
    net.normalize()
    assert net.render() == first


def test_progress_signature_equal_when_untouched_changes_on_any_edit() -> None:
    # The loop's staleness guard (issue #32) detects an abandoned plan by an UNCHANGED signature
    # across turn-starts, and protects an active plan because ANY edit changes the signature.
    net = _net(Task(title="A"), Task(title="B"))
    sig0 = net.progress_signature()
    # An identical plan reproduces the signature exactly ("the model did not touch it").
    assert _net(Task(title="A"), Task(title="B")).progress_signature() == sig0
    # A status advance changes it.
    assert _net(Task(title="A", status="done"), Task(title="B")).progress_signature() != sig0
    # A structural change (decomposition adds child ids) changes it.
    assert (
        _net(
            Task(title="A", kind="compound", children=[Task(title="A1")]),
            Task(title="B"),
        ).progress_signature()
        != sig0
    )
    # New content (a retitled step) changes it too — conservative, the fail-safe direction.
    assert _net(Task(title="A"), Task(title="reworded")).progress_signature() != sig0
    # An empty network has a stable, distinct signature.
    assert TaskNetwork().progress_signature() == repr([])


# ── structural quality (ADR-0050 — the evaluate_candidate port) ───────────────


def test_quality_perfect_plan_scores_full() -> None:
    net = _net(Task(title="A", note="tests pass"), Task(title="B", note="file exists"))
    net.normalize()
    quality, deficiencies = net.quality()
    assert quality == 1.0
    assert deficiencies == []


def test_quality_flags_steps_without_done_conditions() -> None:
    net = _net(Task(title="A"), Task(title="B", note="ok"))
    net.normalize()
    quality, deficiencies = net.quality()
    assert quality < 1.0
    assert any("done-condition" in d and "1" in d for d in deficiencies)


def test_quality_penalizes_undecomposed_compounds_hardest() -> None:
    net = _net(Task(title="Goal", kind="compound"))
    net.normalize()
    quality, deficiencies = net.quality()
    # completeness 0, verifiability 0 (no primitives), granularity 1.0 -> 0.2
    assert quality == 0.2
    assert any("not yet decomposed" in d for d in deficiencies)


def test_quality_granularity_penalty_past_ten_primitives() -> None:
    net = _net(*[Task(title=f"S{i}", note="ok") for i in range(14)])
    net.normalize()
    quality, deficiencies = net.quality()
    assert quality < 1.0
    assert any("primitive steps" in d for d in deficiencies)


def test_quality_truncated_id_list_says_so() -> None:
    """Four noteless steps used to read '4 step(s) ... (1, 2, 3)' — the count and the list
    disagreed with nothing marking the cut (ADR-0114)."""
    net = _net(*[Task(title=f"S{i}") for i in range(4)])
    net.normalize()
    _, deficiencies = net.quality()
    line = next(d for d in deficiencies if "done-condition" in d)
    assert "4 step(s)" in line and "(1, 2, 3, …)" in line
    net3 = _net(*[Task(title=f"S{i}") for i in range(3)])
    net3.normalize()
    _, deficiencies3 = net3.quality()
    assert any("(1, 2, 3)" in d and "…" not in d for d in deficiencies3)


def test_quality_empty_plan_is_silent() -> None:
    assert TaskNetwork().quality() == (1.0, [])


def test_duplicate_sibling_titles_are_advised_never_dropped() -> None:
    net = _net(Task(title="grep the log"), Task(title="Grep the log"))
    advisories = net.normalize()
    assert len(net.tasks) == 2  # fail-open: advised, not dropped (the processor DROPPED)
    assert any("same title" in a for a in advisories)


def test_structure_signature_ignores_status_ticks_but_not_shape_edits() -> None:
    net = _net(Task(title="A"), Task(title="B"))
    net.normalize()
    signature = net.structure_signature()
    net.tasks[0].status = "done"
    net.normalize()
    assert net.structure_signature() == signature  # a tick is not a new decomposition
    net.tasks.append(Task(title="C"))
    net.normalize()
    assert net.structure_signature() != signature  # a shape edit is
