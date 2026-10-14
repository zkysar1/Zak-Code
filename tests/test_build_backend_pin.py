"""The build backend is pinned to one exact release (ADR-0300).

``zakcode update`` builds the checkout it has just pulled, and the build installs whatever
``[build-system] requires`` names into its own throwaway environment. ``uv.lock`` and the
constraints file exported from it hold the project's dependencies and nothing of the build
backend (the export has no ``hatchling`` line), so ``requires = ["hatchling"]`` made every update
build with the newest hatchling of the day, outside the lock. This pins the declaration; the
update end to end (``tests/test_update_e2e.py``) shows the pin reaching a real build.

Hermetic: it reads ``pyproject.toml`` and nothing else.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def loose(requires: list[str]) -> list[str]:
    """The entries of ``requires`` that are not one exact release: a bare name, a range, a
    wildcard (``==1.*``), arbitrary equality (``===``), or more than one clause."""
    found: list[str] = []
    for text in requires:
        clauses = list(Requirement(text).specifier)
        if not (
            len(clauses) == 1 and clauses[0].operator == "==" and "*" not in clauses[0].version
        ):
            found.append(text)
    return found


def test_the_predicate_accepts_one_exact_release() -> None:
    """The control: without it the cases below could all be failing for another reason."""
    assert loose(["hatchling==1.32.4"]) == []
    assert loose(["hatchling==1.32.4", "pathspec==0.12.1"]) == []


@pytest.mark.parametrize(
    "entry",
    [
        "hatchling",
        "hatchling>=1.32",
        "hatchling~=1.32.4",
        "hatchling<2",
        "hatchling==1.*",
        "hatchling===1.32.4",
        "hatchling>=1.32,<1.33",
    ],
)
def test_the_predicate_flags_anything_that_is_not_one_exact_release(entry: str) -> None:
    assert loose([entry]) == [entry]


def test_a_second_requirement_that_floats_is_flagged_beside_a_pinned_one() -> None:
    assert loose(["hatchling==1.32.4", "pathspec"]) == ["pathspec"]


def test_every_build_requirement_of_the_project_is_one_exact_release() -> None:
    build = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["build-system"]

    assert build["requires"], "no build requirement is declared at all"
    assert loose(build["requires"]) == []


def test_the_backend_that_builds_the_project_is_one_of_the_pinned_requirements() -> None:
    """A pin on a package the build does not use would leave the real backend floating."""
    build = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["build-system"]

    assert build["build-backend"] == "hatchling.build"
    pinned = {canonicalize_name(Requirement(text).name) for text in build["requires"]}
    assert "hatchling" in pinned
