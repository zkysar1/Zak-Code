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


def _coerce_list(value: str) -> list[str]:
    """Parse a frontmatter list value: ``[a, b]`` or ``a, b`` -> ``["a", "b"]``."""
    v = value.strip()
    if v.startswith("[") and v.endswith("]"):
        v = v[1:-1]
    return [item.strip().strip("\"'") for item in v.split(",") if item.strip()]


def parse_frontmatter(text: str) -> tuple[SkillFrontmatter, str]:
    """Split a ``SKILL.md`` into ``(frontmatter, body)``.

    The frontmatter is the block between the leading ``---`` fence and the next
    ``---`` line; the body is everything after. Only ``name``/``description``/
    ``version``/``allowed-tools`` (alias ``allowed_tools``) are recognized; unknown
    keys are ignored. Raises :class:`SkillError` if the fence or ``name`` is missing.
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
    for raw in lines[1:end]:
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().replace("-", "_")
        value = value.strip()
        if key in ("allowed_tools",):
            fields[key] = _coerce_list(value)
        elif key in ("name", "description", "version"):
            fields[key] = value.strip("\"'")

    if "name" not in fields or not fields["name"]:
        raise SkillError("frontmatter is missing a 'name'")
    body = "\n".join(lines[end + 1 :]).strip()
    return SkillFrontmatter(**fields), body  # type: ignore[arg-type]


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
            "Available skills (invoke a skill by name to load its full instructions):",
        ]
        for name, desc in self.catalog():
            lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        return "\n".join(lines)


def discover_skill_dir(skills_dir: str | Path) -> tuple[list[Skill], dict[str, str]]:
    """Discover skills under ``skills_dir`` (one subdir per skill, each with SKILL.md).

    Returns ``(skills, errors)``; a malformed/unreadable skill is recorded in
    ``errors`` (by directory name) and skipped. Only the frontmatter is parsed here
    — bodies stay unloaded (lazy disclosure). A missing dir yields empties.
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
    name: str, description: str, allowed_tools: list[str] | None, version: str
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
    overwrite: bool = False,
) -> Path:
    """Author a ``SKILL.md`` under ``skills_dir`` and return its path.

    Writes ``<skills_dir>/<name>/SKILL.md`` with a frontmatter block this package's
    own parser round-trips, followed by ``body``. ``name`` must be a safe, kebab-case
    identifier (``[a-z0-9][a-z0-9-]{0,63}``) so it can never escape ``skills_dir`` via
    path separators or ``..``. Raises :class:`SkillError` on a bad name, an empty
    body, or an existing skill when ``overwrite`` is false.

    This is the storage primitive a self-learning framework (or the ``save_skill``
    tool) builds skill-extraction on; it makes no decision about *when* to author.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", name):
        raise SkillError(
            "skill name must be kebab-case ([a-z0-9-], <=64 chars) and contain no path separators"
        )
    if not body or not body.strip():
        raise SkillError("skill body must be non-empty")
    skill_dir = Path(skills_dir) / name
    md = skill_dir / SKILL_FILENAME
    if md.exists() and not overwrite:
        raise SkillError(f"skill {name!r} already exists at {md} (pass overwrite=True to replace)")
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = _serialize_frontmatter(name, description, allowed_tools, version)
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
