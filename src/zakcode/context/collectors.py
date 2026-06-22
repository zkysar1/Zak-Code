"""Deterministic candidate collectors for the context gatherer (step 1).

Each is a pure function of the turn payload -- no model; the only I/O is reading
files under the workspace. They surface candidates with a cheap relevance prior;
the gatherer's classifier does the ranking. Everything is bounded (file count,
file size, walk size) so a huge repo can never make the gather slow or
memory-hungry -- enable-and-forget.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

from zakcode.hooks import LLMContextPayload

from .gatherer import Candidate

_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "target",
        ".idea",
        ".vscode",
        ".tox",
    }
)
_TEXT_EXT = frozenset(
    {
        ".py",
        ".md",
        ".rst",
        ".txt",
        ".json",
        ".toml",
        ".yaml",
        ".yml",
        ".cfg",
        ".ini",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".mjs",
        ".rs",
        ".go",
        ".java",
        ".rb",
        ".php",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".sh",
        ".bash",
        ".sql",
        ".html",
        ".css",
        ".scss",
    }
)
_MAX_FILE_BYTES = 12_000  # head of each file we are willing to inject
_MAX_WALK_FILES = 4000  # hard cap on how many files we will even stat


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9_]+", text.lower()) if len(t) > 2}


def _overlap(a: set[str], b: set[str]) -> float:
    """Fraction of the task tokens ``b`` that appear in ``a`` (0 when no task tokens)."""
    return len(a & b) / len(b) if b else 0.0


def _walk(root: Path) -> Iterator[Path]:
    seen = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if Path(name).suffix.lower() not in _TEXT_EXT:
                continue
            seen += 1
            if seen > _MAX_WALK_FILES:
                return
            yield Path(dirpath) / name


def _read_head(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(_MAX_FILE_BYTES)
    except OSError:
        return ""


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def recent_files_collector(
    payload: LLMContextPayload, *, limit: int = 12, max_age_days: float = 14.0
) -> list[Candidate]:
    """Recently-modified workspace files, scored by recency + name-overlap with the task."""
    root = Path(payload.cwd or ".")
    if not root.is_dir():
        return []
    task_tokens = _tokens(payload.user_text)
    now = time.time()
    scored: list[tuple[float, Path]] = []
    for path in _walk(root):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        age_days = (now - mtime) / 86400.0
        if age_days > max_age_days:
            continue
        recency = max(0.0, 1.0 - age_days / max_age_days)
        name_overlap = _overlap(_tokens(path.name), task_tokens)
        scored.append((0.6 * recency + 0.4 * name_overlap, path))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[Candidate] = []
    for score, path in scored[:limit]:
        body = _read_head(path)
        if body:
            out.append(
                Candidate(
                    source="file", ref=_relpath(path, root), text=body, cheap_score=round(score, 4)
                )
            )
    return out


def mentioned_files_collector(payload: LLMContextPayload) -> list[Candidate]:
    """Files named verbatim in the task text -- an explicit mention is the strongest prior."""
    root = Path(payload.cwd or ".")
    if not root.is_dir() or not payload.user_text:
        return []
    out: list[Candidate] = []
    seen: set[str] = set()
    for tok in re.findall(r"[\w./\\-]+", payload.user_text):
        if "." not in tok and "/" not in tok and "\\" not in tok:
            continue
        rel = tok.replace("\\", "/").lstrip("/")
        if rel in seen:
            continue
        candidate_path = root / rel
        if candidate_path.is_file():
            body = _read_head(candidate_path)
            if body:
                out.append(Candidate(source="file", ref=rel, text=body, cheap_score=1.0))
                seen.add(rel)
    return out
