"""Shared ignore filter for the search tools (``grep`` / ``glob`` / ``list_dir``).

Large repos bury real code under build/vendor/cache output; searching it wastes context and
adds noise (worse for a small model). This module decides which paths the search tools skip:

* ``.git`` is ALWAYS pruned (VCS internals — never source, huge), even with ``include_ignored``.
* A small built-in set of UNAMBIGUOUS junk directories (``node_modules`` / ``__pycache__`` /
  ``.venv`` / caches …) is pruned by default.
* ``.gitignore`` and an optional ``.zakcodeignore`` at the workspace root add project-specific
  ignores (a conservative gitignore subset: names, ``*``/``?``/``[]``/``**`` globs, anchored
  paths, directory-only ``dir/``, and ``!`` negation).

The matcher is deliberately **FAIL-OPEN**: a pattern it cannot confidently compile is skipped,
so the worst case is leaving some noise in — never HIDING a file the agent needs. The
``soft=False`` mode (the tools' ``include_ignored`` flag) turns the whole soft layer off
(``.git`` still pruned) for the rare time the model must search ignored/generated files.

Only the workspace-root ignore files are read (nested per-directory ``.gitignore`` is not yet
supported); patterns are matched against POSIX paths relative to the root.
"""

from __future__ import annotations

import re
from pathlib import Path

#: VCS internals: always skipped, even with ``include_ignored`` (huge; never source).
_ALWAYS_PRUNE = frozenset({".git"})

#: Unambiguous build/vendor/cache directories skipped by default (basename, any depth).
#: Deliberately conservative — only dirs that are ~never source. Project-specific build output
#: (``build/`` / ``dist/`` / ``target/`` …) is left to ``.gitignore`` so we never hardcode-hide
#: a directory that might be real source in some project.
DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".venv",
        "venv",
        "node_modules",
        "bower_components",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".cache",
    }
)

_IGNORE_FILES = (".gitignore", ".zakcodeignore")


class _Rule:
    """One compiled ignore pattern."""

    __slots__ = ("regex", "exact", "negate", "dir_only")

    def __init__(
        self, regex: re.Pattern[str], exact: re.Pattern[str], negate: bool, dir_only: bool
    ) -> None:
        self.regex = regex  # matches the path OR anything under it (dir contents)
        self.exact = exact  # matches the path exactly (used for the dir-only/file distinction)
        self.negate = negate
        self.dir_only = dir_only


def _translate(pattern: str) -> str | None:
    """Translate a gitignore glob to a regex body that respects ``/`` boundaries.

    ``pattern`` has had any leading ``!``, anchoring ``/``, and trailing ``/`` removed already.
    Returns ``None`` for a construct we can't confidently handle (fail-open: the rule is dropped,
    so we never over-ignore on a pattern we don't understand).
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 3] == "**/":
                out.append("(?:.*/)?")  # zero or more leading directories
                i += 3
                continue
            if pattern[i : i + 2] == "**":
                out.append(".*")  # matches across directory separators
                i += 2
                continue
            out.append("[^/]*")  # single * does not cross /
            i += 1
            continue
        if c == "?":
            out.append("[^/]")
            i += 1
            continue
        if c == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                return None  # unterminated character class -> fail-open
            cls = pattern[i + 1 : j]
            cls = ("^" + cls[1:]) if cls.startswith("!") else cls
            out.append("[" + cls + "]")
            i = j + 1
            continue
        out.append(re.escape(c))
        i += 1
    return "".join(out)


def _compile_line(line: str) -> _Rule | None:
    """Compile one ignore-file line into a :class:`_Rule` (``None`` to skip it)."""
    s = line.rstrip("\n").rstrip()
    if not s or s.startswith("#"):
        return None
    negate = s.startswith("!")
    if negate:
        s = s[1:]
    if s[:2] in ("\\#", "\\!"):  # an escaped leading # or ! is a literal
        s = s[1:]
    dir_only = s.endswith("/")
    if dir_only:
        s = s[:-1]
    # gitignore: a pattern containing a '/' (anywhere but a trailing one, already stripped) is
    # anchored to the ignore file's directory (the root here); otherwise it matches a basename
    # at any depth.
    anchored = s.startswith("/") or "/" in s
    s = s.lstrip("/")
    if not s:
        return None
    body = _translate(s)
    if body is None:
        return None
    prefix = "" if anchored else "(?:.*/)?"
    try:
        regex = re.compile("^" + prefix + body + "(?:/.*)?$")
        exact = re.compile("^" + prefix + body + "$")
    except re.error:
        return None  # fail-open
    return _Rule(regex, exact, negate, dir_only)


class IgnoreSpec:
    """A compiled set of ignore rules for one workspace root."""

    def __init__(self, rules: list[_Rule]) -> None:
        self._rules = rules

    def match(
        self, rel_posix: str, *, is_dir: bool, soft: bool = True, apply_rules: bool = True
    ) -> bool:
        """True if the path (POSIX, relative to the root) should be skipped.

        ``soft=False`` disables the default-dir + gitignore layers (the ``include_ignored``
        escape hatch); the ``.git`` hard floor still applies. ``apply_rules=False`` keeps the
        ``.git`` + default-dir floor but skips the user ``.gitignore``/``.zakcodeignore`` rules —
        used for the basename-only fallback on a path that is NOT under this root (an extra
        workspace root), where this root's project ignores must not govern another tree's files.
        """
        rel = rel_posix.strip("/")
        if not rel:
            return False
        parts = rel.split("/")
        if any(p in _ALWAYS_PRUNE for p in parts):
            return True  # .git: hard floor
        if not soft:
            return False
        if any(p in DEFAULT_IGNORE_DIRS for p in parts):
            return True
        if not apply_rules:
            return False
        ignored = False
        for rule in self._rules:  # gitignore semantics: last matching rule wins
            if not rule.regex.match(rel):
                continue
            if rule.dir_only and not is_dir and rule.exact.match(rel):
                continue  # a dir-only pattern must not match a plain file of the same name
            ignored = not rule.negate
        return ignored

    def is_ignored_path(self, path: Path, root: Path, *, is_dir: bool, soft: bool = True) -> bool:
        """Convenience: relativize ``path`` to ``root`` and :meth:`match`.

        A path not under ``root`` (e.g. an extra workspace root from the M-3 multi-root sandbox)
        is matched on its basename against the ``.git`` / default-dir floor ONLY — the user
        ``.gitignore`` rules are NOT applied, because they belong to *this* root and a basename
        like ``config.py`` would otherwise silently hide a same-named real file in another root,
        breaking the module's FAIL-OPEN guarantee.
        """
        try:
            rel = path.relative_to(root).as_posix()
            under_root = True
        except ValueError:
            rel = path.name
            under_root = False
        return self.match(rel, is_dir=is_dir, soft=soft, apply_rules=under_root)


def load_ignore(root: Path) -> IgnoreSpec:
    """Build an :class:`IgnoreSpec` from the workspace root's ignore files (+ built-in dirs)."""
    rules: list[_Rule] = []
    for fname in _IGNORE_FILES:
        try:
            text = (root / fname).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            rule = _compile_line(line)
            if rule is not None:
                rules.append(rule)
    return IgnoreSpec(rules)


__all__ = ["IgnoreSpec", "load_ignore", "DEFAULT_IGNORE_DIRS"]
