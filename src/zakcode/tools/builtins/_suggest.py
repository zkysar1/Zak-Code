"""Closest-path suggestions for a file tool's "not found" answer (ADR-0040).

Field incident 2026-08-27: asked to review "the python script to list the google drive
files", the model looked for ``google-drive-list`` by path, got ``File not found``, marked
the step blocked and asked the user for the path — twice. The user typed "you can't grep
it?"; a single content search returned seven hits, including the skill directory that
carried the name. A not-found answer is a fact about ONE path, not about the workspace;
the tool that reports it is the right place to say what the workspace DOES have.

:func:`suggest` walks the workspace once (ignore-aware, symlink-safe, bounded by file
count and wall clock) and returns two small lists: paths whose NAME contains the missing
name, and text files whose CONTENT mentions it. The tools append them to their error so a
small model gets the answer inside the failure instead of a hint to go find it.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Sequence
from pathlib import Path

from zakcode.tools.builtins._ignore import load_ignore

#: Hard ceilings so a suggestion never costs more than the model's own next search would.
_MAX_FILES = 40_000
_TIME_BUDGET_S = 2.0
_MAX_NAME_HITS = 6
_MAX_CONTENT_HITS = 6
_MAX_CONTENT_FILES = 4_000
_MAX_CONTENT_BYTES = 512 * 1024
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GENERIC_TOKENS = frozenset({"the", "and", "file", "files", "script", "src", "lib", "txt"})


def _tokens(stem: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(stem.lower()) if len(t) >= 3 and t not in _GENERIC_TOKENS]


def _display(path: Path, workspace_root: Path) -> str:
    """Workspace-relative when possible, always with forward slashes — the same text on
    every platform (Windows CI rendered ``.zakcode\\skills\\…``; a model-facing path with
    ``/`` works on Windows Python too, and a stable form is what tests and readers key on)."""
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


def suggest(
    missing: str,
    workspace_root: Path,
    extra_roots: Sequence[Path] = (),
    *,
    soft: bool = True,
) -> tuple[list[str], list[str]]:
    """Return ``(by_name, by_content)`` for a path that does not exist.

    ``by_name``: files/dirs whose basename contains the missing basename (exact, then
    without extension, then every ≥3-char token). ``by_content``: ``path:line`` of text
    files mentioning the missing stem. Both are short, deterministic, and never raise —
    a suggestion that fails is an empty list, never a second error on top of the first.
    """
    try:
        return _suggest(missing, Path(workspace_root), [Path(r) for r in extra_roots], soft=soft)
    except Exception:  # noqa: BLE001 - a helper on the error path must never raise
        return [], []


def _suggest(
    missing: str, workspace_root: Path, extra_roots: list[Path], *, soft: bool
) -> tuple[list[str], list[str]]:
    needle = Path(missing.rstrip("/\\")).name.lower()
    if not needle or needle in {".", ".."}:
        return [], []
    stem = needle.rsplit(".", 1)[0] if "." in needle[1:] else needle
    tokens = _tokens(stem)
    if len(stem) < 3 and not tokens:
        return [], []

    ignore = load_ignore(workspace_root)
    ignore_root = workspace_root.resolve()
    deadline = time.monotonic() + _TIME_BUDGET_S
    scored: list[tuple[int, int, str, Path]] = []
    text_candidates: list[Path] = []
    seen = 0
    roots = [workspace_root, *extra_roots]
    for root in roots:
        if not root.is_dir():
            continue
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            current_path = Path(current)
            dirnames[:] = [
                d
                for d in dirnames
                if not ignore.is_ignored_path(current_path / d, ignore_root, is_dir=True, soft=soft)
            ]
            entries = [(d, True) for d in dirnames] + [(f, False) for f in sorted(filenames)]
            for name, is_dir in entries:
                seen += 1
                if seen > _MAX_FILES or time.monotonic() > deadline:
                    break
                leaf = current_path / name
                if leaf.is_symlink():
                    continue
                if not is_dir and ignore.is_ignored_path(
                    leaf, ignore_root, is_dir=False, soft=soft
                ):
                    continue
                lower = name.lower()
                score = 0
                if needle in lower:
                    score = 3
                elif stem in lower:
                    score = 2
                elif tokens and all(t in lower for t in tokens):
                    score = 1
                if score:
                    rel = _display(leaf, workspace_root)
                    scored.append((-score, len(rel), rel, leaf))
                if not is_dir:
                    text_candidates.append(leaf)
            if seen > _MAX_FILES or time.monotonic() > deadline:
                break

    scored.sort()
    by_name = [rel for _, _, rel, _ in scored[:_MAX_NAME_HITS]]

    by_content: list[str] = []
    if len(stem) >= 4:
        probe = stem.encode("utf-8")
        for leaf in text_candidates[:_MAX_CONTENT_FILES]:
            if len(by_content) >= _MAX_CONTENT_HITS or time.monotonic() > deadline:
                break
            try:
                if leaf.stat().st_size > _MAX_CONTENT_BYTES:
                    continue
                raw = leaf.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096] or probe not in raw.lower():
                continue
            for line_no, line in enumerate(raw.splitlines(), start=1):
                if probe in line.lower():
                    by_content.append(f"{_display(leaf, workspace_root)}:{line_no}")
                    break
    return by_name, by_content


def render(missing: str, by_name: list[str], by_content: list[str]) -> str:
    """The lines a tool appends under its not-found message ('' when nothing to say)."""
    lines: list[str] = []
    if by_name:
        lines.append("Closest paths by name:")
        lines.extend(f"  {p}" for p in by_name)
    if by_content:
        stem = Path(missing.rstrip("/\\")).name
        lines.append(f"Files whose content mentions {stem!r}:")
        lines.extend(f"  {p}" for p in by_content)
    return "\n".join(lines)


def not_found_fix(missing: str, found_any: bool) -> str:
    """The ``fix`` rail for a not-found error: what to do next, in one line."""
    stem = Path(missing.rstrip("/\\")).name
    if found_any:
        return (
            "read or list one of the paths above; a not-found answer is about ONE path, "
            "not the workspace. Do not ask the user for a path the suggestions already name."
        )
    return (
        f'search before concluding it is missing: grep(pattern="{re.escape(stem)}") from the '
        "workspace root searches every file by content, and list_dir shows what a directory "
        "actually holds. Only when both come back empty is the file genuinely absent."
    )


def literal_stem(pattern: str) -> str:
    """The longest glob-free path segment of ``pattern`` — the name the caller was after.

    ``**/google-drive-list*`` → ``google-drive-list``; a bare extension glob (``*.py``) has
    no name in it and yields ``''``, so an empty listing stays an empty listing.
    """
    pieces = [Path(p.strip("/\\")).name.strip(".") for p in re.split(r"[*?\[\]{}]+", pattern)]
    pieces = [p for p in pieces if len(p) >= 3]
    return max(pieces, key=len) if pieces else ""
