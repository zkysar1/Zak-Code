"""PEARL knowledge-base reading (§10.3/§10.4) — the pre-projected bundle and the
lean-agent raw-note fallbacks.

WHY THIS IS CORE AND NOT ``server/``. Everything here is business logic over a
workspace directory: it opens files, coerces shapes, and returns plain dicts. It holds
no FastAPI, no request, no response and no transport concern, so by the repo's
core/interface separation rule it belongs in the importable engine and the HTTP layer
should be a thin caller. It lived in ``server/app.py`` only because that is where its
first (and so far only) caller happens to be.

WHAT THE REDACTION GUARANTEE IS, AND WHY IT SURVIVES THE MOVE. ``KnowledgeProjection``
runs in-process on the box in the MIND (where the stores live) and writes an
already-filtered + redacted bundle to ``<workspace>/.knowledge-bundle.json`` — filter
at the source. Nothing in this module projects, filters or redacts; it only READS what
was already filtered, plus raw notes that a lean agent wrote under its own
``<workspace>/knowledge/``. Every path this module touches is workspace-relative, so it
can never reach a framework ``world/`` store outside the workspace. Keep it that way:
the moment something here resolves a path from outside ``workspace_root``, the
"the daemon holds no projection logic" property is gone.

FAIL-OPEN THROUGHOUT, DELIBERATELY. A missing, unreadable, malformed or wrong-typed
file yields an empty section rather than raising. A corrupt store must degrade a browse
to "nothing here yet", never 500 it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KNOWLEDGE_BUNDLE_FILE = ".knowledge-bundle.json"

_KNOWLEDGE_SECTIONS = ("tree", "hypotheses", "guardrails", "lessons")


def _empty_bundle() -> dict[str, Any]:
    return {"counts": {}, "tree": [], "hypotheses": [], "guardrails": [], "lessons": []}


#: Caps for the lean-agent raw-note fallback (g-335-191): a short sampler ``summary``
#: for the wiki map vs the fuller clickable ``body``. Mirrors the Mind projection's
#: summary/body split so the viewer shows a sampler in the tree and the full note on
#: click. The raw path is workspace-isolated (a research agent writes only its own
#: domain notes under ``knowledge/tree/``) — unlike the kid-facing PEARL path, which
#: serves the Mind's already-redacted projected bundle, this fallback carries the note
#: as written and never reads a framework ``system/`` path.
_RAW_NODE_SUMMARY_CAP = 500
_RAW_NODE_BODY_CAP = 32_000


def _raw_summary(text: str) -> str:
    """A short sampler for a raw markdown note: the first non-heading paragraph, capped
    at ``_RAW_NODE_SUMMARY_CAP``. The full note is carried separately as ``body`` so the
    viewer renders a sampler in the map and the whole article on click (g-335-191)."""
    for para in text.split("\n\n"):
        stripped = "\n".join(
            ln for ln in para.splitlines() if not ln.strip().startswith("#")
        ).strip()
        if stripped:
            return stripped[:_RAW_NODE_SUMMARY_CAP]
    return text[:_RAW_NODE_SUMMARY_CAP]


def _read_raw_tree(workspace_root: Path) -> list[dict[str, Any]]:
    """Fallback for lean research agents that write raw markdown notes to
    ``<workspace>/knowledge/tree/*.md`` instead of running the Mind's
    ``KnowledgeProjection``. Each ``<key>.md`` becomes a node: the first ``# ``
    heading is the title, a short first-paragraph sampler is the ``summary``, and the
    full note (capped) is the ``body`` — so the map stays light and clicking a node
    shows the whole article (g-335-191). Flat — raw notes carry no parent/child edges.
    Returns ``[]`` when the directory is absent, so a
    full Mind (whose tree lives elsewhere and is surfaced via the projected bundle)
    is never affected: there is simply nothing to read at this path.
    """
    tree_dir = workspace_root / "knowledge" / "tree"
    try:
        files = sorted(tree_dir.glob("*.md"))
    except OSError:
        return []
    nodes: list[dict[str, Any]] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title = f.stem
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
        nodes.append(
            {
                "key": f.stem,
                "title": title,
                "summary": _raw_summary(text),
                "body": text[:_RAW_NODE_BODY_CAP],
                "parent": "",
                "children": [],
            }
        )
    return nodes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a newline-delimited JSON file into a list of dicts, skipping blank /
    malformed lines. Returns ``[]`` when the file is absent or unreadable. Never
    raises — a corrupt store must not 500 a browse."""
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _read_raw_hypotheses(workspace_root: Path) -> list[dict[str, Any]]:
    """Fallback for agents that record hypotheses as raw JSONL at
    ``<workspace>/knowledge/hypotheses.jsonl`` instead of running the Mind's
    ``KnowledgeProjection``. Each line becomes a hypothesis card, mapping the
    common Mind pipeline field names onto the viewer shape (statement / horizon /
    status / outcome). Field lookups are defensive (multiple aliases) so a lean
    agent and a full pipeline record both surface. Returns ``[]`` when the file
    is absent, so a full Mind (which surfaces hypotheses via the projected
    bundle) is never affected.

    Reads two lean store names, BOTH under ``<workspace>/knowledge/`` — never the
    framework ``world/`` tree outside the workspace:
      * ``knowledge/hypotheses.jsonl`` — a lean agent's raw export, and
      * ``knowledge/pipeline.jsonl`` — the REAL framework pipeline store, when a
        sidecar Mind runs the mind_api daemon with ``AYOAI_WORLD`` pointed at
        ``<workspace>/knowledge`` (the PEARL sidecar layout). The daemon names its
        pipeline store ``pipeline.jsonl``, so reading only ``hypotheses.jsonl``
        would miss every hypothesis a real ``pipeline-add.sh`` call produced.
    Both live UNDER ``knowledge/`` and hold only the research agent's own domain
    records (it never writes the framework ``system/`` subtree), so the redaction
    guarantee is preserved: this fallback still never reads a framework ``world/``
    path outside the workspace, and only fires when the projected bundle's section
    is empty.
    """
    items: list[dict[str, Any]] = []
    for name in ("hypotheses.jsonl", "pipeline.jsonl"):
        for row in _read_jsonl(workspace_root / "knowledge" / name):
            statement = (
                row.get("statement")
                or row.get("hypothesis")
                or row.get("prediction")
                or row.get("text")
                or row.get("title")
                or ""
            )
            if not str(statement).strip():
                continue
            items.append(
                {
                    "statement": str(statement),
                    "horizon": str(row.get("horizon") or ""),
                    "status": str(row.get("status") or row.get("stage") or ""),
                    "outcome": str(row.get("outcome") or row.get("resolution") or ""),
                }
            )
    return items


def _read_raw_guardrails(workspace_root: Path) -> list[dict[str, Any]]:
    """Fallback for agents that record guardrails as raw JSONL at
    ``<workspace>/knowledge/guardrails.jsonl``. Each line becomes a guardrail card
    (viewer shape is just ``rule``). Defensive field aliases; returns ``[]`` when
    absent. Reads ONLY the lean export path, never the framework
    ``world/guardrails.jsonl`` — same redaction-firewall guarantee as
    ``_read_raw_hypotheses``.
    """
    items: list[dict[str, Any]] = []
    for row in _read_jsonl(workspace_root / "knowledge" / "guardrails.jsonl"):
        rule = row.get("rule") or row.get("text") or row.get("statement") or ""
        if not str(rule).strip():
            continue
        items.append({"rule": str(rule)})
    return items


def _read_knowledge_bundle(workspace_root: Path) -> dict[str, Any]:
    """The pre-projected knowledge bundle the Mind's periodic export wrote.

    ``KnowledgeProjection`` runs in-process on the box in the MIND (where the stores
    live) and writes an already-filtered + redacted bundle to
    ``<workspace>/.knowledge-bundle.json`` (PEARL §10.3 — filter at the source). The
    daemon serves that artifact read-only and holds NO projection logic, so it can
    never see raw framework internals. Fail-open: a missing / unreadable / malformed /
    non-dict file yields the empty bundle (the loop simply hasn't exported yet).

    Lean-agent fallback: when no projected ``tree`` is present, surface the raw
    ``knowledge/tree/*.md`` notes directly (see ``_read_raw_tree``). This makes the
    wiki work for research agents that write raw markdown and never project, while
    leaving a full Mind's redacted-bundle path unchanged — its ``tree`` is non-empty
    whenever a bundle exists, and it keeps no raw notes under this path.
    """
    out = _empty_bundle()
    try:
        data: Any = json.loads((workspace_root / KNOWLEDGE_BUNDLE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict):
        # Coerce each expected section to a list so a corrupt field cannot 500 a browse.
        out["counts"] = data.get("counts") if isinstance(data.get("counts"), dict) else {}
        for section in _KNOWLEDGE_SECTIONS:
            val = data.get(section)
            out[section] = val if isinstance(val, list) else []
    if not out["tree"]:
        out["tree"] = _read_raw_tree(workspace_root)
    if not out["hypotheses"]:
        out["hypotheses"] = _read_raw_hypotheses(workspace_root)
    if not out["guardrails"]:
        out["guardrails"] = _read_raw_guardrails(workspace_root)
    return out
