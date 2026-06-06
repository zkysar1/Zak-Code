"""Tests for filesystem plugin discovery (M6-3).

Discovery is defensive: bad manifests / missing modules / no register() are
recorded as errors and skipped, never raised. These write real plugin dirs under
tmp_path and assert the (plugins, errors) split, trust gating, and dedup.
"""

from __future__ import annotations

import json
from pathlib import Path

from zakcode.plugins.discovery import (
    default_plugin_dirs,
    discover_dir_plugins,
    discover_plugins,
)


def _write_plugin(
    root: Path,
    name: str,
    *,
    manifest: dict | None = None,
    module_name: str = "plugin",
    module_body: str = "def register(ctx):\n    pass\n",
    skip_module: bool = False,
) -> None:
    pdir = root / name
    pdir.mkdir(parents=True)
    m = {"name": name, **(manifest or {})}
    (pdir / "plugin.json").write_text(json.dumps(m), encoding="utf-8")
    if not skip_module:
        (pdir / f"{module_name}.py").write_text(module_body, encoding="utf-8")


def test_discover_missing_dir_is_empty(tmp_path: Path) -> None:
    plugins, errors = discover_dir_plugins(tmp_path / "nope")
    assert plugins == []
    assert errors == {}


def test_discover_valid_plugin(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "demo", module_body="def register(ctx):\n    ctx.log('hi')\n")
    plugins, errors = discover_dir_plugins(tmp_path)
    assert errors == {}
    assert len(plugins) == 1
    assert plugins[0].manifest.name == "demo"
    # B1 security: the module is NOT imported at discovery, so `.register` is None and
    # `.loader` is set; resolve() imports on demand and returns the register callable.
    assert plugins[0].register is None
    assert callable(plugins[0].resolve())
    assert plugins[0].manifest.trusted is False  # untrusted by default


def test_trusted_names_marks_trusted(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "demo")
    plugins, _ = discover_dir_plugins(tmp_path, trusted_names={"demo"})
    assert plugins[0].manifest.trusted is True


def test_bad_json_is_recorded_not_raised(tmp_path: Path) -> None:
    pdir = tmp_path / "broken"
    pdir.mkdir()
    (pdir / "plugin.json").write_text("{not json", encoding="utf-8")
    plugins, errors = discover_dir_plugins(tmp_path)
    assert plugins == []
    assert "broken" in errors


def test_missing_module_is_recorded(tmp_path: Path) -> None:
    _write_plugin(tmp_path, "nomod", skip_module=True)
    plugins, errors = discover_dir_plugins(tmp_path)
    assert plugins == []
    assert "nomod" in errors


def test_module_without_register_discovers_but_fails_to_resolve(tmp_path: Path) -> None:
    # The module is NOT imported at discovery (security), so "no register()" is not a
    # discovery error — the plugin is discovered, and resolve() fails at load time.
    _write_plugin(tmp_path, "noreg", module_body="x = 1\n")
    plugins, errors = discover_dir_plugins(tmp_path)
    assert [p.manifest.name for p in plugins] == ["noreg"]
    assert errors == {}
    import pytest

    with pytest.raises(AttributeError):
        plugins[0].resolve()


def test_custom_module_name(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "custom",
        manifest={"module": "entry"},
        module_name="entry",
        module_body="def register(ctx):\n    pass\n",
    )
    plugins, errors = discover_dir_plugins(tmp_path)
    assert errors == {}
    assert [p.manifest.name for p in plugins] == ["custom"]


def test_discovered_register_is_callable(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "real",
        module_body=(
            "def register(ctx):\n"
            "    from zakcode.commands import CommandResult\n"
            "    ctx.register_command('rc', lambda a: CommandResult.ok('ran'))\n"
        ),
    )
    plugins, _ = discover_dir_plugins(tmp_path)
    # The resolved register is the real module function (we can't run it without a
    # context, but we can confirm it loaded as a callable named 'register').
    assert plugins[0].resolve().__name__ == "register"


def test_default_plugin_dirs_shape(tmp_path: Path) -> None:
    dirs = default_plugin_dirs(tmp_path)
    assert dirs[0] == tmp_path / ".zakcode" / "plugins"  # project first
    assert dirs[-1].name == "plugins"


def test_discover_plugins_dedups_by_name(tmp_path: Path) -> None:
    # Same plugin name in the project dir → only one wins; no entry points.
    proj = tmp_path / ".zakcode" / "plugins"
    _write_plugin(proj, "dup")
    plugins, errors = discover_plugins(tmp_path, include_entry_points=False)
    assert [p.manifest.name for p in plugins] == ["dup"], errors


# ── audit3 #4: workspace plugins are origin-blind no more ─────────────────────


def test_trusted_workspace_plugin_logs_security_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    # The documented name-allowlist flow still trusts a workspace plugin, but because the
    # grant is name-based (a same name in any opened repo would match), trusting an
    # opened-repo plugin is surfaced with a prominent security warning.
    import logging

    import zakcode.plugins.discovery as disc

    ws = tmp_path / "ws" / ".zakcode" / "plugins"
    user = tmp_path / "user"
    _write_plugin(ws, "blessed")
    monkeypatch.setattr(disc, "default_plugin_dirs", lambda root: [ws, user])
    with caplog.at_level(logging.WARNING, logger="zakcode.plugins.discovery"):
        plugins, _ = discover_plugins(
            tmp_path, trusted_names={"blessed"}, include_entry_points=False
        )
    blessed = next(p for p in plugins if p.manifest.name == "blessed")
    assert blessed.manifest.trusted is True  # documented allowlist flow preserved
    assert any(
        "TRUSTED workspace plugin" in r.message and "blessed" in r.message for r in caplog.records
    )


def test_installed_plugin_wins_over_workspace_shadow(tmp_path: Path, monkeypatch) -> None:
    # An installed/user plugin of the same name resolves FIRST (wins) and IS trusted; the
    # workspace shadow is dropped as a duplicate.
    import zakcode.plugins.discovery as disc

    ws = tmp_path / "ws" / ".zakcode" / "plugins"
    user = tmp_path / "user"
    _write_plugin(user, "blessed")
    _write_plugin(ws, "blessed")
    monkeypatch.setattr(disc, "default_plugin_dirs", lambda root: [ws, user])
    plugins, errors = discover_plugins(
        tmp_path, trusted_names={"blessed"}, include_entry_points=False
    )
    blessed = [p for p in plugins if p.manifest.name == "blessed"]
    assert len(blessed) == 1
    assert blessed[0].origin == str(user / "blessed")  # the installed one won
    assert blessed[0].manifest.trusted is True
    assert errors.get("blessed") == "duplicate plugin name (first one wins)"


def test_workspace_plugin_explicit_manifest_trust_is_honored(tmp_path: Path, monkeypatch) -> None:
    # The per-project consent path still works: a workspace plugin that declares
    # "trusted": true in its own plugin.json remains trusted.
    import zakcode.plugins.discovery as disc

    ws = tmp_path / "ws" / ".zakcode" / "plugins"
    user = tmp_path / "user"
    _write_plugin(ws, "local", manifest={"trusted": True})
    monkeypatch.setattr(disc, "default_plugin_dirs", lambda root: [ws, user])
    plugins, _ = discover_plugins(tmp_path, include_entry_points=False)
    local = next(p for p in plugins if p.manifest.name == "local")
    assert local.manifest.trusted is True
