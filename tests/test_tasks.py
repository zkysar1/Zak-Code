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
