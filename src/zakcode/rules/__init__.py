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
import re
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


def _within(path: Path, root: Path) -> bool:
    """Whether ``path``'s real path (junctions/symlinks/``..`` collapsed) is ``root`` itself
    or lives under it — the same containment :mod:`zakcode.skills` enforces for skill writes.
    """
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def project_rules_dir(workspace_root: str | Path) -> Path:
    """The project-level rules root (``<workspace>/.zakcode/rules``).

    Where runtime-authored rules (see :func:`save_rule`) are written so they travel with
    the repository and are discovered next session. It is the zakcode-native project dir,
    matching ``<workspace>/.zakcode/skills``. Note :func:`default_rule_dirs` scans
    ``.claude/rules`` *after* this one, so a same-named rule in that compatibility dir
    still shadows a rule written here — deliberate (a checked-in project rule outranks a
    runtime-authored one), but worth knowing when a saved rule seems not to take effect.
    """
    return Path(workspace_root) / ".zakcode" / "rules"


def save_rule(
    name: str,
    description: str,
    body: str,
    *,
    rules_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Author a rule file under ``rules_dir`` and return its path.

    The write counterpart to :class:`~zakcode.tools.builtins.read_rule.ReadRuleTool`.
    ``read_rule`` let a turn READ project rules; nothing let it WRITE one, so a rule the
    agent learned by experience could only enter the store by an out-of-band human edit.
    This is the storage primitive that closes that asymmetry; it makes no decision about
    *when* a rule is worth authoring.

    Writes ``<rules_dir>/<name>.md`` with a frontmatter block :func:`_split_frontmatter`
    round-trips, followed by ``body``. Deliberately mirrors :func:`zakcode.skills.save_skill`:
    ``name`` must be a safe kebab-case identifier (``[a-z0-9][a-z0-9-]{0,63}``) so it has no
    path separators or ``..``, and the target's REAL path is verified to live under
    ``rules_dir`` (so a pre-planted junction/symlink cannot redirect the write out of tree).

    Three constraints that are NOT arbitrary:

    * The frontmatter fence is the FIRST byte of the file. :func:`_split_frontmatter` tests
      ``lines[0].strip() == "---"``, so anything above it — a blank line, a comment — makes
      the parser return *no metadata* and treat the whole file as the body. The rule would
      still load, silently, under its filename stem with its description lost.
    * ``name`` and ``description`` must be single-line. The frontmatter parser is line-based,
      so an embedded newline either invents a bogus key or (if the line is ``---``) closes
      the block early.
    * ``body`` is capped at :data:`MAX_RULE_FILE_CHARS`, because that is the bound
      :meth:`RuleRegistry._body_block` truncates at when rendering. Refusing here is better
      than authoring a rule that is silently cut in half every turn it is read.

    There is no ``always_apply`` parameter, and its absence is the point: ``alwaysApply``
    puts a rule's full body in every turn's prompt, so a self-authored always-on rule would
    let a turn permanently spend the shared context budget. An author who genuinely wants
    that sets the frontmatter flag by hand, as a human review step.

    Raises :class:`RuleError` on a bad name, a multi-line name/description, an out-of-tree
    target, an empty or over-budget body, or an existing rule when ``overwrite`` is false.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise RuleError(
            "rule name must be kebab-case ([a-z0-9-], <=64 chars) and contain no path separators"
        )
    if not body or not body.strip():
        raise RuleError("rule body must be non-empty")
    if len(body) > MAX_RULE_FILE_CHARS:
        raise RuleError(
            f"rule body is {len(body)} chars, over the {MAX_RULE_FILE_CHARS}-char per-rule "
            "budget; the render would truncate it silently. Split it or shorten it."
        )
    if "\n" in description or "\r" in description:
        raise RuleError("rule description must be a single line (the frontmatter is line-based)")
    root = Path(rules_dir)
    path = root / f"{name}.md"
    if not _within(path, root):
        raise RuleError(f"rule {name!r} resolves outside the rules directory {str(root)!r}")
    if path.exists() and not overwrite:
        raise RuleError(f"rule {name!r} already exists at {path} (pass overwrite=True to replace)")
    root.mkdir(parents=True, exist_ok=True)
    # The fence is byte 0 — see the docstring. The description is written UNQUOTED because
    # quoting buys nothing: the parser partitions on the FIRST ':' and takes the remainder
    # verbatim, so an internal colon already round-trips without a wrapper.
    # The NORMALISATION below is the load-bearing half. The parser ends with .strip("\"'"),
    # so a description that starts or ends with a quote char cannot survive a round trip in
    # this format at all — that is the reader's behaviour and no writer can change it.
    # Stripping the same way here means what is written is exactly what is read back, so
    # editing a rule repeatedly (read description -> save it again) is idempotent rather
    # than shedding one character per pass. (Measured while mutation-proving this writer:
    # with this line present, quoted and unquoted are equivalent — the quoted form is
    # redundant, not harmful, and an earlier comment here claiming otherwise was wrong.)
    desc = description.strip().strip("\"'")
    frontmatter = f"---\nname: {name}\ndescription: {desc}\n---"
    path.write_text(f"{frontmatter}\n{body.strip()}\n", encoding="utf-8")
    return path


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
    "project_rules_dir",
    "save_rule",
]
