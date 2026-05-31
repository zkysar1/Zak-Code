"""Plugin discovery (M6) — turn directories and installed packages into Plugins.

A directory plugin lives at ``<plugins_dir>/<name>/`` with:

* ``plugin.json`` — the :class:`~zakcode.plugins.PluginManifest` fields, and
* a module (``plugin.py`` by default) exposing ``def register(ctx): ...``.

Installed packages may also contribute via the ``zakcode.plugins`` entry-point
group (each entry point resolves to a ``register`` callable).

Discovery is **defensive**: a directory with bad JSON, a missing module, or no
``register`` function is recorded in the returned errors map and skipped — it never
raises, so one broken plugin can't stop the others (or startup). Discovery does
**not** run plugin code beyond importing the module; ``register`` is invoked later,
trust-gated, by :meth:`~zakcode.plugins.PluginManager.load_into`.

Trust: a plugin participates only if its manifest is ``trusted``. Rather than make
operators edit each manifest, pass ``trusted_names`` (an allowlist) — any plugin
named in it is marked trusted at discovery time.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from zakcode.plugins import Plugin, PluginManifest

logger = logging.getLogger("zakcode.plugins.discovery")

#: Entry-point group installed packages use to contribute plugins.
ENTRY_POINT_GROUP = "zakcode.plugins"


def default_plugin_dirs(workspace_root: str | Path) -> list[Path]:
    """Candidate plugin roots: project ``.zakcode/plugins`` then user config dir."""
    return [
        Path(workspace_root) / ".zakcode" / "plugins",
        Path.home() / ".config" / "zakcode" / "plugins",
    ]


def _apply_trust(manifest: PluginManifest, trusted_names: set[str] | None) -> PluginManifest:
    if trusted_names and manifest.name in trusted_names:
        return manifest.model_copy(update={"trusted": True})
    return manifest


def _load_register_from_file(module_path: Path, plugin_name: str) -> Any:
    """Import ``module_path`` in isolation and return its ``register`` callable.

    The synthetic module name is made unique per absolute path (a short hash) so two
    plugins that share a directory name — or repeated loads across one process — never
    collide in ``sys.modules``. The module is registered in ``sys.modules`` before
    execution so dataclasses/enums it defines resolve their own module correctly.
    """
    resolved = module_path.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    mod_name = f"zakcode_plugin_{plugin_name}_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    register = getattr(module, "register", None)
    if not callable(register):
        raise AttributeError(f"{module_path} has no callable 'register(ctx)'")
    return register


def discover_dir_plugins(
    plugins_dir: str | Path, *, trusted_names: set[str] | None = None
) -> tuple[list[Plugin], dict[str, str]]:
    """Discover plugins under ``plugins_dir`` (one subdirectory per plugin).

    Returns ``(plugins, errors)`` — ``errors`` maps a plugin directory name to the
    reason it could not be loaded. A missing ``plugins_dir`` yields empties.
    """
    plugins: list[Plugin] = []
    errors: dict[str, str] = {}
    root = Path(plugins_dir)
    if not root.is_dir():
        return plugins, errors
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "plugin.json"
        try:
            if not manifest_path.is_file():
                raise FileNotFoundError("missing plugin.json")
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = PluginManifest(**data)
            module_path = entry / f"{manifest.module}.py"
            if not module_path.is_file():
                raise FileNotFoundError(f"missing module {module_path.name}")
            register = _load_register_from_file(module_path, manifest.name)
        except Exception as exc:  # noqa: BLE001 — a bad plugin is data, not a crash
            errors[entry.name] = f"{type(exc).__name__}: {exc}"
            logger.warning("skipping plugin dir %s: %s", entry, exc)
            continue
        plugins.append(
            Plugin(
                manifest=_apply_trust(manifest, trusted_names),
                register=register,
                origin=str(entry),
            )
        )
    return plugins, errors


def discover_entrypoint_plugins(
    *, trusted_names: set[str] | None = None
) -> tuple[list[Plugin], dict[str, str]]:
    """Discover plugins from the ``zakcode.plugins`` entry-point group.

    Each entry point resolves to a ``register`` callable; the entry-point name is the
    plugin name. Entry-point plugins start *untrusted* unless named in
    ``trusted_names`` (installing a package is not by itself consent to run it).
    """
    plugins: list[Plugin] = []
    errors: dict[str, str] = {}
    try:
        from importlib.metadata import entry_points

        eps: Iterable[Any] = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 — never let metadata problems crash discovery
        logger.warning("entry-point discovery failed: %s", exc)
        return plugins, errors
    for ep in eps:
        try:
            register = ep.load()
            if not callable(register):
                raise TypeError(f"entry point {ep.name!r} is not callable")
            manifest = _apply_trust(
                PluginManifest(name=ep.name, module=getattr(ep, "value", "")), trusted_names
            )
        except Exception as exc:  # noqa: BLE001
            errors[ep.name] = f"{type(exc).__name__}: {exc}"
            continue
        plugins.append(
            Plugin(manifest=manifest, register=register, origin=f"entry-point:{ep.name}")
        )
    return plugins, errors


def discover_plugins(
    workspace_root: str | Path,
    *,
    trusted_names: set[str] | None = None,
    include_entry_points: bool = True,
) -> tuple[list[Plugin], dict[str, str]]:
    """Discover all plugins (project + user dirs, then entry points).

    Returns ``(plugins, errors)``. Later sources do not override earlier ones by
    name — the first plugin discovered for a given name wins, and a duplicate is
    recorded in ``errors``.
    """
    all_plugins: list[Plugin] = []
    all_errors: dict[str, str] = {}
    seen: set[str] = set()

    sources: list[tuple[list[Plugin], dict[str, str]]] = [
        discover_dir_plugins(d, trusted_names=trusted_names)
        for d in default_plugin_dirs(workspace_root)
    ]
    if include_entry_points:
        sources.append(discover_entrypoint_plugins(trusted_names=trusted_names))

    for plugins, errors in sources:
        all_errors.update(errors)
        for plugin in plugins:
            if plugin.manifest.name in seen:
                all_errors[plugin.manifest.name] = "duplicate plugin name (first one wins)"
                continue
            seen.add(plugin.manifest.name)
            all_plugins.append(plugin)
    return all_plugins, all_errors


__all__ = [
    "ENTRY_POINT_GROUP",
    "default_plugin_dirs",
    "discover_dir_plugins",
    "discover_entrypoint_plugins",
    "discover_plugins",
]
