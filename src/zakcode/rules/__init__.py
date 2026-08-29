"""Rules — always-on, operator-authored guidance folded into the system prompt.

A *rule* is a Markdown file of standing guidance ("always use tabs", "this repo's
HTTP layer lives in ``api/``", a team convention). Unlike a **skill** (M7) — which
is model-*invokable* and whose body loads lazily on demand — a rule is **always
applied**: its text is rendered into the *stable, cacheable* tier of the system
prompt every turn. This is the Cursor-rules model (flat ``.zakcode/rules`` /
``.claude/rules`` ``*.md`` files).

**Not to be confused with a root ``CLAUDE.md`` / ``AGENTS.md``:** those are *agent
guides*, discovered by :func:`zakcode.agent.prompt.discover_context` and folded into
the *dynamic* context tier (not here, not the cached tier). Rules are the always-on
``rules/`` directory; the guides are the project's root-level instruction files.
This is the static-guidance seam a self-learning framework (see ``docs/INTEGRATIONS``)
drops durable conventions into.

Discovery mirrors skills, with one shape difference: rules are **flat ``*.md``
files** in a rules directory (not one subdirectory per rule). Roots, in increasing
precedence (later overrides earlier by rule name):

* bundled (``zakcode/rules/bundled``),
* user (``~/.config/zakcode/rules``),
* project (``<workspace>/.zakcode/rules`` then ``<workspace>/.claude/rules`` — the
  latter for Claude-Code / Claude-Mind compatibility).

A rule file may carry an optional ``---`` frontmatter block (``name`` / ``description``
are recognized; anything else is ignored); without one, the file *stem* is the rule
name and the whole file is the body. Parsing is defensive — an unreadable file is
recorded and skipped, never raised.

Because the rendered block sits in the cached prefix, it must be bounded: each rule
is capped at :data:`MAX_RULE_FILE_CHARS` and the combined render at
:data:`MAX_RULES_TOTAL_CHARS`, so a large ``.claude/rules`` directory can never blow
the context window (a learning framework that keeps many rules for *on-demand* use
should let its skills read them via the file tools rather than relying on always-on
injection).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("zakcode.rules")

#: Largest slice of any single rule file folded into the prompt (~8 KB).
MAX_RULE_FILE_CHARS = 8_192

#: Largest combined size of all rendered rules (~32 KB).
MAX_RULES_TOTAL_CHARS = 32_768

#: Per-rule summary cap in the compact :meth:`RuleRegistry.render_index` (lean) output.
_INDEX_SUMMARY_CHARS = 140


class RuleError(Exception):
    """Raised when a rule file cannot be read or is empty."""


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split an optional ``---`` frontmatter block off the front of ``text``.

    Returns ``(meta, body)``. With no leading ``---`` fence (or an unterminated
    one), ``meta`` is empty and ``body`` is the whole text. Only simple ``key:
    value`` lines are read; ``name`` and ``description`` are the recognized keys.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta: dict[str, str] = {}
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in ("name", "description", "alwaysapply"):
            meta[key] = value.strip().strip("\"'")
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _truthy(value: str | None) -> bool:
    """A frontmatter flag: ``true`` / ``yes`` / ``1`` (any case) — anything else is off."""
    return (value or "").strip().lower() in ("true", "yes", "1")


class Rule:
    """One discovered always-on rule: a ``name`` and a Markdown ``content`` body.

    ``always_apply`` is the Cursor ``alwaysApply:`` frontmatter flag (ADR-0105): the
    rule's FULL body rides in the prompt even under the lean index, and it is folded
    first under the full render, so the budget can never drop it behind a sibling.
    """

    def __init__(
        self,
        name: str,
        content: str,
        path: Path,
        *,
        description: str = "",
        always_apply: bool = False,
    ) -> None:
        self.name = name
        self.content = content
        self.path = path
        self.description = description
        self.always_apply = always_apply


class RuleRegistry:
    """A name-keyed collection of rules with a bounded, cache-stable render."""

    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}

    def add(self, rule: Rule, *, replace: bool = False) -> bool:
        """Register a rule. Returns ``False`` (without replacing) on a name clash."""
        if rule.name in self._rules and not replace:
            return False
        self._rules[rule.name] = rule
        return True

    def get(self, name: str) -> Rule | None:
        return self._rules.get(name)

    def names(self) -> list[str]:
        return list(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def _ordered(self) -> list[Rule]:
        """Discovery order, with ``always_apply`` rules first (ADR-0105).

        Both renders fold rules front-to-back until the budget is reached, so order IS
        priority: a rule the operator pinned must be considered before any sibling that
        merely sorts earlier by name.
        """
        pinned = [r for r in self._rules.values() if r.always_apply]
        rest = [r for r in self._rules.values() if not r.always_apply]
        return pinned + rest

    @staticmethod
    def _body_block(rule: Rule) -> str:
        """``## name`` plus the body capped at :data:`MAX_RULE_FILE_CHARS`; ``""`` if blank."""
        content = rule.content.strip()
        if not content:
            return ""
        if len(content) > MAX_RULE_FILE_CHARS:
            content = content[:MAX_RULE_FILE_CHARS]
        return f"## {rule.name}\n{content}"

    def render(self) -> str:
        """Render all rules as one prompt block for the stable tier (``""`` if none).

        Bounded by :data:`MAX_RULE_FILE_CHARS` per rule and
        :data:`MAX_RULES_TOTAL_CHARS` overall, so a large rules directory cannot
        blow the context window; rules beyond the budget are dropped (a note records
        how many). ``always_apply`` rules are folded first, so the budget drops them last.
        """
        if not self._rules:
            return ""
        header = "Project rules (operator-authored standing guidance — always apply these):"
        blocks: list[str] = []
        # Budget the FULL rendered size (header + the "\n\n" boundary + per-block "\n\n"
        # separators), with slack reserved for the omission note, so the returned string
        # is genuinely bounded by MAX_RULES_TOTAL_CHARS (the docstring's guarantee).
        note_slack = 80
        total = len(header) + 2  # header + the "\n\n" before the body
        dropped = 0
        for rule in self._ordered():
            block = self._body_block(rule)
            if not block:
                continue
            sep = 2 if blocks else 0  # the "\n\n" join separator before this block
            if total + sep + len(block) > MAX_RULES_TOTAL_CHARS - note_slack:
                dropped += 1
                continue
            blocks.append(block)
            total += sep + len(block)
        if not blocks:
            return ""
        body = "\n\n".join(blocks)
        if dropped:
            body += f"\n\n[{dropped} further rule(s) omitted: rules budget reached]"
        return f"{header}\n\n{body}"

    def render_index(self) -> str:
        """Render a COMPACT one-line-per-rule INDEX for the stable tier (``""`` if none).

        The lean alternative to :meth:`render`. Where :meth:`render` folds every rule's
        full body into the cached prefix on *every* turn — fine for a handful of rules, a
        per-turn cost multiplier for a rules-heavy "mind" carrying dozens — this emits one
        line per rule (``- name: summary [path]``) and tells the model to READ the full
        rule file on demand (via its file-read tool) when a rule looks relevant to the
        current step. This is exactly the on-demand model this module's docstring
        recommends for a learning framework that keeps many rules.

        The summary is the rule's frontmatter ``description`` when present, else the first
        non-empty body line (with any leading Markdown heading marker stripped), capped at
        :data:`_INDEX_SUMMARY_CHARS`. The on-disk ``path`` lets the model fetch the body
        with its file-read tool (directly usable for workspace-local rules such as
        ``.claude/rules/*.md`` — the dominant "mind" case). Bounded overall by
        :data:`MAX_RULES_TOTAL_CHARS` for parity with :meth:`render` (the index is far
        smaller, but the cap is enforced for safety); rules beyond the budget are dropped
        with a recorded note. Returns ``""`` when there are no rules.
        """
        if not self._rules:
            return ""
        header = (
            "Project rules (operator-authored standing guidance). Each rule is listed by "
            "name with a one-line summary; when a rule looks relevant to the current step, "
            "call read_rule with its name to get the full text, then apply it. (The file "
            "path is shown too, so your file-read tool also works if read_rule is "
            "unavailable.):"
        )
        lines: list[str] = []
        note_slack = 80
        total = len(header) + 2  # header + the "\n\n" before the body
        dropped = 0
        # ADR-0105: a rule pinned with ``alwaysApply: true`` keeps its FULL body under the
        # lean index — the index tells the model to fetch a body on demand, and a small
        # model measured over an hour of fleet turns never did (0 read_rule calls), so the
        # rules an operator cannot afford to lose ride in full, ahead of the index, within
        # the same budget. Every other rule stays one line.
        pinned_header = "Pinned rules (full text — always apply these):"
        pinned_blocks: list[str] = []
        if any(rule.always_apply for rule in self._rules.values()):
            total += len(pinned_header) + 2 + 2  # its heading + "\n\n", and the join to the index
        for rule in self._ordered():
            if not rule.always_apply:
                continue
            block = self._body_block(rule)
            if not block:
                continue
            sep = 2 if pinned_blocks else 0
            if total + sep + len(block) > MAX_RULES_TOTAL_CHARS - note_slack:
                dropped += 1
                continue
            pinned_blocks.append(block)
            total += sep + len(block)
        for rule in self._ordered():
            if rule.always_apply:
                continue
            summary = (rule.description or "").strip()
            if not summary:
                first = next((ln.strip() for ln in rule.content.splitlines() if ln.strip()), "")
                summary = first.lstrip("#").strip()  # drop a leading Markdown heading marker
            if len(summary) > _INDEX_SUMMARY_CHARS:
                summary = summary[: _INDEX_SUMMARY_CHARS - 3].rstrip() + "..."
            line = (
                f"- {rule.name}: {summary} [{rule.path}]"
                if summary
                else f"- {rule.name} [{rule.path}]"
            )
            sep = 1 if lines else 0  # the single "\n" join separator before this line
            if total + sep + len(line) > MAX_RULES_TOTAL_CHARS - note_slack:
                dropped += 1
                continue
            lines.append(line)
            total += sep + len(line)
        if not lines and not pinned_blocks:
            return ""
        body = "\n".join(lines)
        if dropped:
            body += f"\n[{dropped} further rule(s) omitted: rules index budget reached]"
        index = f"{header}\n\n{body}" if lines else body
        if pinned_blocks:
            pinned = f"{pinned_header}\n\n" + "\n\n".join(pinned_blocks)
            return f"{pinned}\n\n{index}" if index else pinned
        return index


def discover_rule_dir(rules_dir: str | Path) -> tuple[list[Rule], dict[str, str]]:
    """Discover flat ``*.md`` rule files under ``rules_dir``.

    Returns ``(rules, errors)``; an unreadable/empty file is recorded in ``errors``
    (by filename) and skipped. A missing dir yields empties. Files are returned
    sorted by name for a deterministic, cache-stable order.
    """
    rules: list[Rule] = []
    errors: dict[str, str] = {}
    root = Path(rules_dir)
    if not root.is_dir():
        return rules, errors
    for entry in sorted(root.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".md":
            continue
        try:
            text = entry.read_text(encoding="utf-8")
            meta, body = _split_frontmatter(text)
            content = body.strip()
            if not content:
                raise RuleError("empty rule body")
        except Exception as exc:  # noqa: BLE001 — a bad rule is data, not a crash
            errors[entry.name] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping rule file %s: %s", entry, exc)
            continue
        name = meta.get("name") or entry.stem
        rules.append(
            Rule(
                name,
                content,
                entry,
                description=meta.get("description", ""),
                always_apply=_truthy(meta.get("alwaysapply")),
            )
        )
    return rules, errors


def default_rule_dirs(workspace_root: str | Path) -> list[Path]:
    """Candidate rule roots: bundled, user, then project (project wins on clash).

    ``.claude/rules`` is included after ``.zakcode/rules`` for Claude-Code /
    Claude-Mind compatibility.
    """
    bundled = Path(__file__).parent / "bundled"
    ws = Path(workspace_root)
    return [
        bundled,
        Path.home() / ".config" / "zakcode" / "rules",
        ws / ".zakcode" / "rules",
        ws / ".claude" / "rules",
    ]


def discover_rules(workspace_root: str | Path) -> tuple[RuleRegistry, dict[str, str]]:
    """Discover all rules (bundled → user → project) into a :class:`RuleRegistry`.

    Later sources override earlier ones by name (so a project rule shadows a bundled
    one of the same name). Returns ``(registry, errors)``.
    """
    registry = RuleRegistry()
    all_errors: dict[str, str] = {}
    for d in default_rule_dirs(workspace_root):
        rules, errors = discover_rule_dir(d)
        all_errors.update(errors)
        for rule in rules:
            registry.add(rule, replace=True)
    return registry, all_errors


__all__ = [
    "MAX_RULE_FILE_CHARS",
    "MAX_RULES_TOTAL_CHARS",
    "RuleError",
    "Rule",
    "RuleRegistry",
    "_split_frontmatter",  # shared with zakcode.identity (self.md frontmatter parsing)
    "discover_rule_dir",
    "default_rule_dirs",
    "discover_rules",
]
