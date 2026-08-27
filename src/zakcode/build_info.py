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
import subprocess
from functools import lru_cache
from pathlib import Path

__all__ = [
    "build_commit",
    "build_dir",
    "build_source",
    "build_url",
    "install_changed",
    "install_identity",
    "running_build",
    "version_line",
]


def _read_direct_url() -> dict[str, object]:
    """PEP 610 install metadata read FRESH from disk, or ``{}`` when absent/unreadable."""
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


@lru_cache(maxsize=1)
def _direct_url() -> dict[str, object]:
    """PEP 610 install metadata as it was when this process first asked (cached)."""
    return _read_direct_url()


def _commit_of(info: dict[str, object], short: bool = True) -> str | None:
    vcs = info.get("vcs_info")
    if not isinstance(vcs, dict):
        return None
    commit = vcs.get("commit_id")
    if not isinstance(commit, str) or not commit.strip():
        return None
    commit = commit.strip()
    return commit[:12] if short else commit


def _dir_of(info: dict[str, object]) -> str | None:
    if not isinstance(info.get("dir_info"), dict):
        return None
    url = info.get("url")
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    path = url2pathname(urlparse(url).path)
    return path if path.strip() else None


def build_commit(short: bool = True) -> str | None:
    """The commit this build was installed from, or ``None`` if not a VCS install."""
    return _commit_of(_direct_url(), short)


def _install_marker() -> float | None:
    """Mtime of the install's own ``direct_url.json`` — rewritten by every (re)install.

    The one signal that says "the package on disk is not the one this process loaded"
    for EVERY install shape: a git-URL install (whose commit may change), a local-checkout
    install (whose metadata carries no commit at all), and a no-op reinstall of the same
    commit. ``None`` when there is no installed distribution (running from a bare source
    tree), so nothing downstream ever fires there.
    """
    try:
        from importlib.metadata import distribution

        dist = distribution("zakcode")
        for entry in dist.files or []:
            if entry.name == "direct_url.json" and ".dist-info" in str(entry):
                return Path(entry.locate()).stat().st_mtime
        path = getattr(dist, "_path", None)  # PathDistribution: the dist-info directory
        if path is not None:
            return (path / "direct_url.json").stat().st_mtime
    except Exception:  # noqa: BLE001 — a probe, never a failure
        return None
    return None


def _checkout_head(directory: str) -> str | None:
    """Short HEAD sha of the checkout a local-path install came from, or ``None``."""
    try:
        proc = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:  # noqa: BLE001 — no git, no checkout, or a hung probe: unlabelled
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


def install_identity() -> tuple[str, float | None]:
    """``(label, marker)`` of the zakcode install on disk RIGHT NOW (ADR-0034).

    ``label`` is the human-facing build identity — the recorded commit for a git-URL
    install, the checkout's HEAD for a local-path install, ``""`` when there is none — and
    ``marker`` is :func:`_install_marker`. Read fresh every call; :func:`running_build` is
    the same reading frozen at import, which is what makes :func:`install_changed` a
    comparison between "what I loaded" and "what is on disk".
    """
    info = _read_direct_url()
    label = _commit_of(info)
    if label is None:
        directory = _dir_of(info)
        label = _checkout_head(directory) if directory is not None else None
    return (label or "", _install_marker())


#: The install this PROCESS loaded, frozen at import — before any update can move the disk.
_RUNNING_IDENTITY: tuple[str, float | None] = install_identity()


def running_build() -> str:
    """The build identity of the code this process is running (``""`` when unknown)."""
    return _RUNNING_IDENTITY[0]


def install_changed() -> tuple[str, str] | None:
    """``(running_label, installed_label)`` when the install on disk is no longer the one
    this process loaded — i.e. a ``zakcode update`` (or any reinstall) landed while the
    process was running — else ``None``. Keyed on the install marker, not the label, so a
    dev checkout whose HEAD moves without a reinstall never trips it.
    """
    running_label, running_marker = _RUNNING_IDENTITY
    if running_marker is None:
        return None  # no installed distribution to compare against
    installed_label, installed_marker = install_identity()
    if installed_marker is None or installed_marker == running_marker:
        return None
    return (running_label, installed_label)


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
    return _dir_of(_direct_url())


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
