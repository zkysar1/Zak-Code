"""Tooling contracts (audit P2-2 / P2-5): version sync + config-docs completeness."""

from __future__ import annotations

import tomllib
from pathlib import Path

from zakcode.config import Settings
from zakcode.version import __version__

_ROOT = Path(__file__).resolve().parents[1]


def test_version_sync() -> None:
    """``zakcode.version.__version__`` and pyproject's ``[project].version`` are kept
    in sync by hand — this test is the automation (audit P2-2): a bump to one without
    the other fails CI instead of shipping a mismatched package."""
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__


def test_every_settings_field_is_documented() -> None:
    """docs/CONFIG.md must document every Settings field (audit P2-5): adding a field
    without documenting it fails here, so the reference can never silently rot."""
    doc = (_ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8")
    missing = [name for name in Settings.model_fields if f"`{name}`" not in doc]
    assert not missing, f"Settings fields missing from docs/CONFIG.md: {missing}"


def test_every_documented_env_var_matches_a_field() -> None:
    """The inverse direction: every ``ZAKCODE_*`` env var named in the doc's tables
    corresponds to a real field, so the doc can't advertise a knob that doesn't exist."""
    import re

    doc = (_ROOT / "docs" / "CONFIG.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`ZAKCODE_([A-Z0-9_]+)`", doc))
    real = {name.upper() for name in Settings.model_fields}
    ghosts = sorted(v for v in documented if v not in real)
    assert not ghosts, f"docs/CONFIG.md documents nonexistent settings: {ghosts}"
