"""Which BUILD is running — the question ``__version__`` alone cannot answer.

``version`` in ``pyproject.toml`` is hand-maintained, so every install between two bumps
reports the same string. For a project installed straight from git that is every install:
two checkouts months apart both say ``0.0.1``, and "is this current?" becomes unanswerable
from inside the process.

Measured 2026-08-21: a container running a build from six weeks earlier reported the same
``0.0.1`` as HEAD, and the older build silently lacked three settings the newer one had.
Nothing in the CLI could distinguish them.

The commit is recovered from PEP 610 ``direct_url.json``, which pip/uv write at install
time for a VCS install. No build-time stamping, no new dependency, and it degrades quietly
to ``None`` for a PyPI or plain-path install, where there is no commit to report.
"""

from __future__ import annotations

import json
from functools import lru_cache

__all__ = ["build_commit", "build_dir", "build_source", "build_url", "version_line"]


@lru_cache(maxsize=1)
def _direct_url() -> dict[str, object]:
    """PEP 610 install metadata, or ``{}`` when absent/unreadable."""
    try:
        from importlib.metadata import distribution

        raw = distribution("zakcode").read_text("direct_url.json")
    except Exception:  # pragma: no cover - defensive: metadata layouts vary
        return {}
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:  # pragma: no cover - defensive
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_commit(short: bool = True) -> str | None:
    """The commit this build was installed from, or ``None`` if not a VCS install."""
    vcs = _direct_url().get("vcs_info")
    if not isinstance(vcs, dict):
        return None
    commit = vcs.get("commit_id")
    if not isinstance(commit, str) or not commit.strip():
        return None
    commit = commit.strip()
    return commit[:12] if short else commit


def build_url() -> str | None:
    """The VCS URL this build was installed from, or ``None`` for a non-VCS install.

    This is what lets ``zakcode update`` self-locate its source: PEP 610 records the
    exact URL pip installed from, so the update needs no configuration.
    """
    if not isinstance(_direct_url().get("vcs_info"), dict):
        return None
    url = _direct_url().get("url")
    return url if isinstance(url, str) and url.strip() else None


def build_dir() -> str | None:
    """The local directory this build was installed from, or ``None`` (not a path install).

    A ``file://`` URL plus ``dir_info`` is what pip/uv record for an install from a
    local checkout (``uv tool install 'zakcode @ file:///path/to/clone'``) — the shape
    ``zakcode update`` refreshes with a git pull before reinstalling.
    """
    info = _direct_url()
    if not isinstance(info.get("dir_info"), dict):
        return None
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    path = url2pathname(urlparse(url).path)
    return path if path.strip() else None


def build_source() -> str | None:
    """Where this build came from: ``git``, ``local path``, or ``None`` (a normal install)."""
    info = _direct_url()
    if isinstance(info.get("vcs_info"), dict):
        return "git"
    if isinstance(info.get("dir_info"), dict):
        return "local path"
    return None


def version_line(version: str) -> str:
    """``version`` plus build identity when there is any to add.

    ``0.0.1`` -> ``0.0.1 (git ab2c313e0482)`` for a VCS install; unchanged otherwise, so a
    released install keeps a clean version string.
    """
    commit = build_commit()
    if commit:
        return f"{version} (git {commit})"
    source = build_source()
    if source == "git":
        # A VCS install whose commit_id is missing/blank. Say so rather than printing a
        # bare "(git)": the useful signal is that the version string is NOT pinned to a
        # readable commit, which is exactly when you must not trust it.
        return f"{version} (git, commit unknown)"
    if source:
        return f"{version} ({source})"
    return version
