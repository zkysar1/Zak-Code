"""Skills (M7) — progressive-disclosure, model-invokable capabilities as markdown.

A *skill* is a ``SKILL.md`` file with a small YAML-ish frontmatter block followed by
a markdown body:

    ---
    name: commit-helper
    description: Write a conventional-commit message from a diff.
    allowed-tools: [read_file, bash]
    ---
    <the body — instructions the model follows when the skill is invoked>

**Three-level disclosure** keeps context cheap:

* **L0** — ``name`` + ``description`` only. Always cheap to surface (the catalog the
  model sees so it knows a skill exists).
* **L1** — the markdown **body**, loaded *on demand* (only when the skill is invoked,
  never at discovery/startup).
* **L2** — referenced sibling files/resources, pulled lazily by the body's own tool
  calls (no special machinery here; they're just files in the skill dir).

Discovery mirrors plugins: scan a bundled dir + user (`~/.config/zakcode/skills`) +
project (`.zakcode/skills`) roots, one subdirectory per skill. Parsing is defensive
— a malformed ``SKILL.md`` is recorded and skipped, never raised. The frontmatter
parser is a hand-rolled minimal subset (``key: value`` lines + simple lists), so the
core takes on no YAML dependency.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger("zakcode.skills")

SKILL_FILENAME = "SKILL.md"


class SkillError(Exception):
    """Raised when a ``SKILL.md`` is malformed (missing frontmatter / required field)."""


class SkillFrontmatter(BaseModel):
    """The L0 metadata parsed from a skill's frontmatter."""

    name: str
    description: str = ""
    #: Optional tool allow-list (advisory; surfaced to the operator/model).
    allowed_tools: list[str] = Field(default_factory=list)
    version: str = "0.0.0"
    #: Every frontmatter key this parser does not type explicitly, PRESERVED verbatim
    #: (audit P1-2): Mind skills carry cognitive metadata — ``minimum_mode``,
    #: ``companion_scripts``, ``user_invocable``, ``triggers``, … — that must survive
    #: parsing and stay queryable by the host. Bracketed values arrive as lists,
    #: everything else as the (de-quoted) string. Keys are normalized ``-`` → ``_``.
    extras: dict[str, str | list[str]] = Field(default_factory=dict)


def _coerce_list(value: str) -> list[str]:
    """Parse a frontmatter list value: ``[a, b]`` or ``a, b`` -> ``["a", "b"]``."""
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [item.strip().strip("\"'") for item in v.split(",") if item.strip()]


def parse_frontmatter(text: str) -> tuple[SkillFrontmatter, str]:
    """Split a ``SKILL.md`` into ``(frontmatter, body)``.

    The frontmatter is the block between the leading ``---`` fence and the next
    ``---`` line; the body is everything after. ``name``/``description``/``version``/
    ``allowed-tools`` (alias ``allowed_tools``) get typed fields; every OTHER key is
    preserved verbatim in :attr:`SkillFrontmatter.extras` (audit P1-2 — Mind skills'
    cognitive metadata must survive). Lists parse in BOTH YAML spellings: inline
    (``triggers: ["/start"]``) and block sequence (``triggers:`` followed by
    ``- "/start"`` lines) — the block form is what real Claude-Mind skills
    overwhelmingly use (measured 2026-08-20: 60 of 78 skills in a live Mind), and
    before this it silently parsed as an empty string, so trigger routing and the
    extras-preservation promise both quietly degraded. Raises :class:`SkillError`
    if the fence or ``name`` is missing.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillError("missing leading '---' frontmatter fence")
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        raise SkillError("unterminated frontmatter (no closing '---')")

    fields: dict[str, object] = {}
    extras: dict[str, str | list[str]] = {}
    fm = lines[1:end]
    idx = 0
    while idx < len(fm):
        line = fm[idx].strip()
        idx += 1
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().replace("-", "_")
        value = value.strip()
        block_items: list[str] | None = None
        if not value:
            # YAML block-sequence lookahead: a bare `key:` followed by `- item` lines is
            # the list. Only dash lines are consumed (comments/blank/other lines end the
            # block), so a bare `key:` with no items keeps its empty-string behavior, and
            # a `- name: x` item survives as the string "name: x" rather than vanishing.
            items: list[str] = []
            while idx < len(fm):
                cand = fm[idx].strip()
                if not cand.startswith("- "):
                    break
                items.append(cand[2:].strip().strip("\"'"))
                idx += 1
            if items:
                block_items = items
        if key in ("allowed_tools",):
            fields[key] = block_items if block_items is not None else _coerce_list(value)
        elif key in ("name", "description", "version"):
            fields[key] = value.strip("\"'")
        elif key:
            if block_items is not None:
                extras[key] = block_items
            else:
                extras[key] = _coerce_list(value) if value.startswith("[") else value.strip("\"'")

    if "name" not in fields or not fields["name"]:
        raise SkillError("frontmatter is missing a 'name'")
    body = "\n".join(lines[end + 1 :]).strip()
    return SkillFrontmatter(**fields, extras=extras), body  # type: ignore[arg-type]


class Skill:
    """One discovered skill — L0 metadata eagerly, L1 body lazily."""

    def __init__(self, frontmatter: SkillFrontmatter, path: Path) -> None:
        self.frontmatter = frontmatter
        self.path = path  # the SKILL.md file
        self._body: str | None = None

    @property
    def name(self) -> str:
        return self.frontmatter.name

    @property
    def description(self) -> str:
        return self.frontmatter.description

    @property
    def directory(self) -> Path:
        """The skill's directory (where its L2 resource files live)."""
        return self.path.parent

    def body(self) -> str:
        """Load and cache the L1 markdown body (read on first call, not at discovery)."""
        if self._body is None:
            text = self.path.read_text(encoding="utf-8")
            _, self._body = parse_frontmatter(text)
        return self._body

    @property
    def body_loaded(self) -> bool:
        """Whether the body has been read yet (for asserting lazy disclosure)."""
        return self._body is not None


class SkillRegistry:
    """A name-keyed collection of discovered skills (L0 catalog + lazy bodies)."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def add(self, skill: Skill, *, replace: bool = False) -> bool:
        """Register a skill. Returns ``False`` (without replacing) on a name clash."""
        if skill.name in self._skills and not replace:
            return False
        self._skills[skill.name] = skill
        return True

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def resolve(self, token: str) -> Skill | None:
        """Resolve a slash token to a skill: by ``name`` first, else by any skill whose
        ``triggers:`` frontmatter lists it — Claude Code's skill->slash mechanism (a skill
        ``looper`` with ``triggers: ["/start"]`` is reachable as ``/start``). A leading ``/`` is
        optional on both sides; matching is CASE-INSENSITIVE (CC matches names/triggers
        case-insensitively, and the CLI lower-cases a typed ``/Command``); a name match wins.
        """
        needle = token.lstrip("/")
        direct = self._skills.get(needle) or self._skills.get(token)
        if direct is not None:
            return direct
        lowered = needle.lower()
        for skill in self._skills.values():  # name match wins over any trigger match
            if skill.name.lower() == lowered:
                return skill
        for skill in self._skills.values():
            triggers = skill.frontmatter.extras.get("triggers")
            if isinstance(triggers, list) and any(
                lowered == str(t).lstrip("/").lower() for t in triggers
            ):
                return skill
        return None

    def names(self) -> list[str]:
        return list(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def catalog(self) -> list[tuple[str, str]]:
        """L0: ``(name, description)`` for every skill, in registration order."""
        return [(s.name, s.description) for s in self._skills.values()]

    def render_catalog(self) -> str:
        """Render the L0 catalog as a compact prompt block (empty string if none).

        This is static per session (discovery runs once), so it is cache-safe to
        place in the stable system-prompt tier.
        """
        if not self._skills:
            return ""
        lines = [
            "Available skills (these are NOT tools — never emit a skill name as a tool "
            "call). To run a skill, call the `use_skill` tool with the skill's name; that "
            "loads its full instructions, which you then follow. A skill's steps may tell "
            "you to use another skill (they chain). Each entry shows the exact call to make:",
        ]
        for name, desc in self.catalog():
            call = f'use_skill(name="{name}")'
            lines.append(f"- {call} — {desc}" if desc else f"- {call}")
        # Invocation provenance (the other half of user-invocable enforcement): the runtime
        # composes a <command-name> frame ONLY for a human-typed slash, so this contract line
        # is what lets a skill that forbids model self-invocation run when the USER asks.
        # Without it, a rule-following model refuses the operator's own keystroke (live
        # 2026-08-19: a Mind's /start — "user-invocable only" — was declined as self-invocation).
        lines.append(
            "When a user message BEGINS with a <command-name> block, the human operator "
            "typed that slash command in their terminal: the skill was invoked BY THE USER, "
            "not by you. Any rule limiting a skill to user/human invocation is satisfied in "
            "that case — do not refuse or defer; carry out the instructions in that message "
            "as the current turn's task."
        )
        return "\n".join(lines)


def _within(path: Path, root: Path) -> bool:
    """Whether ``path``'s real path (junctions/symlinks/``..`` collapsed) is ``root`` itself
    or lives under it — the same containment the file-tool resolver enforces
    (:mod:`zakcode.tools.builtins._safety`). (audit3 #2)
    """
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def discover_skill_dir(skills_dir: str | Path) -> tuple[list[Skill], dict[str, str]]:
    """Discover skills under ``skills_dir`` (one subdir per skill, each with SKILL.md).

    Returns ``(skills, errors)``; a malformed/unreadable skill is recorded in
    ``errors`` (by directory name) and skipped. Only the frontmatter is parsed here
    — bodies stay unloaded (lazy disclosure). A missing dir yields empties. A subdir
    (or its SKILL.md) whose real path escapes ``skills_dir`` — e.g. a planted junction
    or symlink — is skipped and recorded, so discovery never pulls out-of-tree content
    into the prompt (the same containment the file tools enforce).
    """
    skills: list[Skill] = []
    errors: dict[str, str] = {}
    root = Path(skills_dir)
    if not root.is_dir():
        return skills, errors
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        md = entry / SKILL_FILENAME
        try:
            # Containment first: a junction/symlink entry (or SKILL.md) whose realpath
            # escapes the skills root must not be read into the catalog/prompt. (audit3 #2)
            if not _within(md, root):
                raise SkillError("resolves outside the skills directory")
            if not md.is_file():
                raise SkillError(f"missing {SKILL_FILENAME}")
            frontmatter, _body = parse_frontmatter(md.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — a bad skill is data, not a crash
            errors[entry.name] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping skill dir %s: %s", entry, exc)
            continue
        skills.append(Skill(frontmatter, md))
    return skills, errors


def user_skills_dir() -> Path:
    """The user-level skills root (``~/.config/zakcode/skills``)."""
    return Path.home() / ".config" / "zakcode" / "skills"


def project_skills_dir(workspace_root: str | Path) -> Path:
    """The project-level skills root (``<workspace>/.zakcode/skills``).

    This is where runtime-authored skills (see :func:`save_skill`) are written so
    they travel with the repository and are discovered next session.
    """
    return Path(workspace_root) / ".zakcode" / "skills"


def default_skill_dirs(workspace_root: str | Path) -> list[Path]:
    """Candidate skill roots, in increasing precedence (later wins on a name clash).

    Bundled → user → project ``.zakcode/skills`` → project ``.claude/skills`` (the
    last for Claude-Code / Claude-Mind compatibility, mirroring rule discovery).
    """
    bundled = Path(__file__).parent / "bundled"
    return [
        bundled,
        user_skills_dir(),
        project_skills_dir(workspace_root),
        Path(workspace_root) / ".claude" / "skills",
    ]


def _serialize_frontmatter(
    name: str,
    description: str,
    allowed_tools: list[str] | None,
    version: str,
    extras: dict[str, str | list[str]] | None = None,
) -> str:
    """Render a ``SKILL.md`` frontmatter block the project parser round-trips."""
    lines = ["---", f"name: {name}"]
    if description:
        # Keep the description a single physical line (the parser is line-based and
        # splits on CR, LF, AND CRLF), collapsing any newline run to one space.
        single_line = re.sub(r"\s*[\r\n]+\s*", " ", description.strip())
        lines.append(f"description: {single_line}")
    if allowed_tools:
        lines.append("allowed-tools: [" + ", ".join(allowed_tools) + "]")
    lines.append(f"version: {version}")
    # Extras (audit P1-2) round-trip after the typed fields, in insertion order. A
    # list renders in the bracketed form _coerce_list parses back; values are kept
    # to one physical line for the same line-based-parser reason as description.
    for key, value in (extras or {}).items():
        if isinstance(value, list):
            lines.append(f"{key}: [" + ", ".join(str(v) for v in value) + "]")
        else:
            single = re.sub(r"\s*[\r\n]+\s*", " ", str(value).strip())
            lines.append(f"{key}: {single}")
    lines.append("---")
    return "\n".join(lines)


def save_skill(
    name: str,
    description: str,
    body: str,
    *,
    skills_dir: str | Path,
    allowed_tools: list[str] | None = None,
    version: str = "0.0.0",
    extras: dict[str, str | list[str]] | None = None,
    overwrite: bool = False,
) -> Path:
    """Author a ``SKILL.md`` under ``skills_dir`` and return its path.

    Writes ``<skills_dir>/<name>/SKILL.md`` with a frontmatter block this package's
    own parser round-trips, followed by ``body``. ``name`` must be a safe, kebab-case
    identifier (``[a-z0-9][a-z0-9-]{0,63}``) so it has no path separators or ``..``, and
    the target dir's REAL path is verified to live under ``skills_dir`` (so a pre-planted
    junction/symlink at ``<skills_dir>/<name>`` cannot redirect the write out of tree).
    Raises :class:`SkillError` on a bad name, an out-of-tree target, an empty body, or an
    existing skill when ``overwrite`` is false.

    This is the storage primitive a self-learning framework (or the ``save_skill``
    tool) builds skill-extraction on; it makes no decision about *when* to author.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise SkillError(
            "skill name must be kebab-case ([a-z0-9-], <=64 chars) and contain no path separators"
        )
    if not body or not body.strip():
        raise SkillError("skill body must be non-empty")
    root = Path(skills_dir)
    skill_dir = root / name
    # Containment: refuse if <skills_dir>/<name> resolves out of the skills tree (a planted
    # junction/symlink) — the kebab-case name check alone never resolves reparse points so
    # cannot catch this on its own. Checked before any mkdir/write. (audit3 #2)
    if not _within(skill_dir, root):
        raise SkillError(f"skill {name!r} resolves outside the skills directory {str(root)!r}")
    md = skill_dir / SKILL_FILENAME
    if md.exists() and not overwrite:
        raise SkillError(f"skill {name!r} already exists at {md} (pass overwrite=True to replace)")
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = _serialize_frontmatter(name, description, allowed_tools, version, extras)
    md.write_text(f"{frontmatter}\n{body.strip()}\n", encoding="utf-8")
    return md


def discover_skills(
    workspace_root: str | Path,
    *,
    extra_skill_dirs: Sequence[str | Path] | None = None,
) -> tuple[SkillRegistry, dict[str, str]]:
    """Discover all skills (bundled -> user -> project [-> extra]) into a :class:`SkillRegistry`.

    Later sources override earlier ones by name (so a project skill shadows a bundled
    one of the same name). ``extra_skill_dirs``, when provided, are scanned *after* the
    defaults — so an external skill directory (e.g. a claude-mind ``skills/`` tree)
    shadows same-named project skills. Returns ``(registry, errors)``.
    """
    registry = SkillRegistry()
    all_errors: dict[str, str] = {}
    dirs = default_skill_dirs(workspace_root)
    if extra_skill_dirs:
        dirs.extend(Path(d) for d in extra_skill_dirs)
    for d in dirs:
        skills, errors = discover_skill_dir(d)
        all_errors.update(errors)
        for skill in skills:
            registry.add(skill, replace=True)
    return registry, all_errors


__all__ = [
    "SKILL_FILENAME",
    "SkillError",
    "SkillFrontmatter",
    "Skill",
    "SkillRegistry",
    "parse_frontmatter",
    "discover_skill_dir",
    "discover_skills",
    "default_skill_dirs",
    "user_skills_dir",
    "project_skills_dir",
    "save_skill",
]
