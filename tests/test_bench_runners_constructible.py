"""The bench runners must not be able to die silently again (ADR-0143).

``bench/`` is excluded from every package gate on purpose — ``ruff`` via
``extend-exclude``, ``pytest`` via ``testpaths = ["tests"]``, ``mypy`` via
``packages = ["zakcode"]``. That is defensible for experiment code, but it means a
runner can break and nothing says so until someone runs it by hand. Six of the seven
did break, identically, and the quality-engine arm of the harness — the evidence
ADR-0011 cites for shipping best-of-N — was dead for as long as nobody looked.

The bug: ``Settings.model_copy(update={"workspace_root": str(ws)})``. ``model_copy``
does NOT re-validate, so the ``str`` defeats the ``workspace_root: Path`` annotation
and survives to ``load_settings_permissions``, which does ``workspace_root / ".claude"``
and raises ``TypeError: unsupported operand type(s) for /: 'str' and 'str'`` — before a
single model call. ``run_task.py`` documents the trap in a nine-line comment and is the
one runner that avoided it.

This test READS the bench sources; it never imports or executes them, so the harness
stays outside the runtime gates while the regression class stays closed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"


def _bench_modules() -> list[Path]:
    return sorted(BENCH.glob("*.py"))


def test_bench_dir_is_present() -> None:
    """A positive control: an empty glob would make every test below vacuously pass."""
    mods = _bench_modules()
    assert mods, f"no bench/*.py found under {BENCH} — this suite would pass vacuously"
    assert (BENCH / "run_task.py") in mods


@pytest.mark.parametrize("path", _bench_modules(), ids=lambda p: p.name)
def test_workspace_root_is_never_stringified(path: Path) -> None:
    """No runner may hand ``workspace_root`` a ``str(...)`` — it must stay a ``Path``.

    Checked on the AST rather than by substring so reformatting cannot hide it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if not (isinstance(key, ast.Constant) and key.value == "workspace_root"):
                continue
            assert not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "str"
            ), (
                f"{path.name}:{key.lineno} passes workspace_root as str(...). "
                "model_copy does not re-validate, so this dies in "
                "load_settings_permissions before any model call. Pass the Path."
            )
