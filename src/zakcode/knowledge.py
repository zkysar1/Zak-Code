"""Knowledge-base reading and export shaping — the SDK side of PEARL §10.4/§10.5.

Everything here is pure workspace/bundle logic with no HTTP in it: reading the
Mind's pre-projected ``.knowledge-bundle.json`` (filter-at-the-source), the
lean-agent raw-note fallback under ``knowledge/``, and the OKF transfer-bundle
export shaping. It lived inline in ``server/app.py`` until the interface-purity
pass (guard-4547: business logic belongs in the SDK, interfaces stay thin); the
server routes now call :func:`read_knowledge_bundle` / :func:`okf_bundle` and do
nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["KNOWLEDGE_BUNDLE_FILE", "read_knowledge_bundle", "okf_bundle"]

KNOWLEDGE_BUNDLE_FILE = ".knowledge-bundle.json"

_KNOWLEDGE_SECTIONS = ("tree", "hypotheses", "guardrails", "lessons")


def _empty_bundle() -> dict[str, Any]:
    # `self` and `program` are OBJECTS, not lists, and are deliberately absent
    # from _KNOWLEDGE_SECTIONS above — see the coercion in read_knowledge_bundle.
    return {
        "counts": {},
        "tree": [],
        "hypotheses": [],
        "guardrails": [],
        "lessons": [],
        "self": {},
        "program": {},
    }


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


def read_knowledge_bundle(workspace_root: Path) -> dict[str, Any]:
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
        # `self` is the agent's projected identity — an OBJECT ({} or
        # {purpose, created, last_updated}), NOT a list, because emptiness is the
        # signal: "no identity published" vs "published and blank" (guard-5493).
        # THIS ASSIGNMENT MUST STAY BELOW THE LOOP. The loop coerces every
        # _KNOWLEDGE_SECTIONS member to a list, so running last is what makes this
        # authoritative — registering `self` up there is then harmless, but hoisting
        # this above the loop while registering it flattens the object to [] and the
        # route serves empty forever. Both orders were mutation-tested; see the
        # three-state table in tests/test_server_knowledge_self.py.
        self_val = data.get("self")
        out["self"] = self_val if isinstance(self_val, dict) else {}
        # `program` is the same OBJECT-not-list shape as `self` above, and for the
        # same reason: emptiness is the signal ("nothing published" vs "published
        # and blank"). It is likewise absent from _KNOWLEDGE_SECTIONS and MUST stay
        # BELOW the loop — registering it up there and hoisting this would flatten
        # the object to [] and the route would serve empty forever.
        program_val = data.get("program")
        out["program"] = program_val if isinstance(program_val, dict) else {}
    if not out["tree"]:
        out["tree"] = _read_raw_tree(workspace_root)
    if not out["hypotheses"]:
        out["hypotheses"] = _read_raw_hypotheses(workspace_root)
    if not out["guardrails"]:
        out["guardrails"] = _read_raw_guardrails(workspace_root)
    return out


# ── OKF transfer-bundle export (PEARL §10.5 / g-335-45) ──────────────────────
# `/knowledge/export` returns a PORTABLE, HUMAN-READABLE WIKI — "Markdown nodes
# plus a manifest, not a database dump" (§10.5). The shape is the framework's
# own OKF-aligned contract (core/config/conventions/transfer-bundle-export-shape.md):
#
#   1. bundle = the unit of distribution — a self-contained directory tree,
#      carried here as a path -> file-content map so the whole chain stays JSON
#      (the sidecar proxy and the gateway both forward JSON; a binary archive
#      would break both). Write `files` to disk verbatim and you have the
#      git-shippable bundle.
#   2. concept = ONE .md with YAML frontmatter.
#   3. exactly one REQUIRED frontmatter key: the `type` discriminator.
#   4. unknown keys are PRESERVED — any field on a source record that this
#      producer does not model is carried into frontmatter rather than dropped,
#      so the boundary never silently loses a producer's data.
#   6. links are bundle-relative and MAY dangle — a child link to a node that
#      was filtered out by KnowledgeProjection is a frontier marker, not an
#      error, so no link is validated or pruned here.
#   7. index.md is the optional progressive-disclosure index.
#
# NOTE this producer adds NO dependency. Frontmatter scalars/lists are emitted
# via json.dumps, which is valid YAML by construction (YAML 1.2 is a JSON
# superset) and escapes arbitrary content correctly — strictly safer than a
# hand-rolled quoter, and it does not put pyyaml on a customer box's runtime.
OKF_BUNDLE_FORMAT = "okf-transfer-bundle"
OKF_BUNDLE_VERSION = 1

# Fields this producer renders into the BODY rather than the frontmatter; every
# other field on a record falls through to frontmatter under invariant 4.
_OKF_BODY_FIELDS = {
    # "node", not "concept": the Mind's own OKF writer (knowledge-export.py
    # writeokf_bundle) already ships `type: node` for tree records, and both
    # producers emit into `nodes/`. The convention deliberately does not
    # enumerate type values (invariant 5 — consumers tolerate unknown ones), so
    # neither spelling is non-conforming; but two producers of ONE declared
    # format disagreeing on the required discriminator is precisely the
    # misroute invariant 3 exists to prevent. Align on the shipped spelling.
    "node": ("title", "summary", "body"),
    "hypothesis": ("statement",),
    "guardrail": ("rule",),
    # "lesson" leads the tuple because it is the ONLY prose key a projected
    # lesson actually carries: the Mind's KnowledgeProjection builds each record
    # as exactly {title, lesson} (knowledge_projection.py, bundle.lessons), and
    # .knowledge-bundle.json is the only thing this path ever reads. Omitting it
    # cost both halves at once — the prose fell through to FRONTMATTER as an
    # unmodelled field while the body expression below resolved to "", so every
    # lesson rendered as a blank page under its heading. Keep it FIRST: once the
    # key is modelled it no longer reaches frontmatter, so a record carrying a
    # stray content/text/summary alongside it would otherwise drop the lesson
    # prose from BOTH places. content/text/summary stay for hand-authored and
    # legacy records.
    "lesson": ("lesson", "title", "content", "text", "summary"),
}


def _okf_slug(raw: str, fallback: str) -> str:
    """A safe, stable bundle-relative filename stem.

    Restrictive on purpose: the stem lands in a path a consumer will write to
    disk, so anything outside [a-z0-9._-] is collapsed, leading dots are
    dropped (no hidden files, no traversal), and an empty result falls back to
    the caller's positional name.
    """
    out = []
    for ch in str(raw).strip().lower():
        out.append(ch if (ch.isalnum() and ch.isascii()) or ch in "-_." else "-")
    slug = "".join(out).strip("-.")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:120] or fallback


def _okf_frontmatter(fields: dict[str, Any]) -> str:
    """Emit a YAML frontmatter block. `type` is written first (invariant 3)."""
    lines = ["---"]
    ordered = ["type"] + sorted(k for k in fields if k != "type")
    for k in ordered:
        if k not in fields:
            continue
        v = fields[k]
        if v is None or v == "" or v == []:
            continue
        key = _okf_slug(k, "field").replace(".", "_")
        if isinstance(v, (str, int, float, bool)):
            lines.append(f"{key}: {json.dumps(v, ensure_ascii=False)}")
        elif isinstance(v, list):
            lines.append(f"{key}: {json.dumps([str(x) for x in v], ensure_ascii=False)}")
        else:
            # Unmodelled composite — preserve it losslessly as a JSON scalar
            # rather than dropping it (invariant 4). Encoded twice on purpose:
            # once to JSON, then again so the result is a single-line scalar.
            inner = json.dumps(v, ensure_ascii=False)
            lines.append(f"{key}: {json.dumps(inner, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def _okf_doc(kind: str, record: dict[str, Any], heading: str, body_parts: list[str]) -> str:
    """One concept document: frontmatter + markdown body."""
    modelled = set(_OKF_BODY_FIELDS.get(kind, ())) | {"key", "children"}
    fm: dict[str, Any] = {"type": kind}
    for k, v in record.items():
        if k in modelled:
            continue
        fm[k] = v
    if record.get("key"):
        fm["key"] = str(record["key"])
    parts = [_okf_frontmatter(fm), "", f"# {heading}".rstrip(), ""]
    parts.extend(p for p in body_parts if p)
    return "\n".join(parts).rstrip() + "\n"


def okf_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """Project the read bundle into the OKF transfer-bundle export shape."""
    files: dict[str, str] = {}
    used: set[str] = set()

    def _path(folder: str, stem: str) -> str:
        # De-collide: two records that slug identically must not overwrite each
        # other (silent data loss at the export boundary).
        candidate, n = f"{folder}/{stem}.md", 1
        while candidate in used:
            n += 1
            candidate = f"{folder}/{stem}-{n}.md"
        used.add(candidate)
        return candidate

    listing: dict[str, list[tuple[str, str]]] = {}

    for i, n in enumerate(bundle.get("tree") or []):
        if not isinstance(n, dict):
            continue
        key = str(n.get("key") or "")
        title = str(n.get("title") or key or f"node-{i + 1}")
        body = str(n.get("body") or n.get("summary") or "")
        # Bundle-relative child links; a dangling one is a frontier marker.
        kids = [str(c) for c in (n.get("children") or []) if c]
        links = (
            ["", "## Explore from here", ""]
            + [f"- [{c}](./{_okf_slug(c, 'node')}.md)" for c in kids]
            if kids
            else []
        )
        p = _path("nodes", _okf_slug(key or title, f"node-{i + 1}"))
        files[p] = _okf_doc("node", n, title, [body, *links])
        listing.setdefault("nodes", []).append((title, p))

    for i, h in enumerate(bundle.get("hypotheses") or []):
        if not isinstance(h, dict):
            continue
        statement = str(h.get("statement") or "")
        p = _path("hypotheses", _okf_slug(statement[:60], f"hypothesis-{i + 1}"))
        files[p] = _okf_doc("hypothesis", h, statement or f"Hypothesis {i + 1}", [])
        listing.setdefault("hypotheses", []).append((statement or p, p))

    for i, g in enumerate(bundle.get("guardrails") or []):
        if not isinstance(g, dict):
            continue
        rule = str(g.get("rule") or "")
        p = _path("guardrails", _okf_slug(rule[:60], f"guardrail-{i + 1}"))
        files[p] = _okf_doc("guardrail", g, rule or f"Guardrail {i + 1}", [])
        listing.setdefault("guardrails", []).append((rule or p, p))

    for i, lesson in enumerate(bundle.get("lessons") or []):
        if not isinstance(lesson, dict):
            continue
        heading = str(lesson.get("title") or lesson.get("text") or f"Lesson {i + 1}")
        # "lesson" first — see the _OKF_BODY_FIELDS["lesson"] comment. A
        # projected record carries only {title, lesson}, so the other three
        # resolve to "" on every real lesson.
        body = str(
            lesson.get("lesson")
            or lesson.get("content")
            or lesson.get("text")
            or lesson.get("summary")
            or ""
        )
        p = _path("lessons", _okf_slug(heading[:60], f"lesson-{i + 1}"))
        files[p] = _okf_doc("lesson", lesson, heading, [body])
        listing.setdefault("lessons", []).append((heading, p))

    index = ["---", 'type: "index"', "---", "", "# Knowledge base", ""]
    if not listing:
        index += ["Nothing here yet — the agent has not recorded anything so far."]
    for folder in ("nodes", "hypotheses", "guardrails", "lessons"):
        rows = listing.get(folder) or []
        if not rows:
            continue
        index += [f"## {folder} ({len(rows)})", ""]
        index += [f"- [{t}](./{p})" for t, p in rows]
        index += [""]
    files["index.md"] = "\n".join(index).rstrip() + "\n"

    return {
        "bundle": {
            "format": OKF_BUNDLE_FORMAT,
            "version": OKF_BUNDLE_VERSION,
            "counts": {k: len(v) for k, v in listing.items()},
            "file_count": len(files),
        },
        "files": files,
    }
