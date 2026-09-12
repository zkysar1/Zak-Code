#!/usr/bin/env python
"""Held-out oracle for 06-plugin-conventions. Run with cwd = the agent's workspace.

WHAT THIS TASK IS FOR. ADR-0151 measured the suite saturated for BOTH agents -- Claude Code scored
5/5 -- so no task in it carries information about which agent is better. ADR-0154 measured why the
suite is easy: bench and production exercise nearly disjoint slices, and the slice the bench misses
is the one real work lives in -- inferring a codebase's conventions from the codebase. Every
existing task states its full contract in the prompt. This one does not.

The prompt asks only for "a renderer that outputs YAML". Everything else is DISCOVERABLE and
nothing is invented: CONTRIBUTING.md states four hard rules in plain language at the workspace root,
and both sibling renderers demonstrate each one in code. The load-bearing rule is stdlib-only --
`import yaml` is the obvious implementation, pyyaml IS installed so it would run fine, and it is
forbidden by the project. That is a convention the interpreter does not enforce, which is precisely
the kind real repositories are made of.

Exits 0 on success, 1 on the first failure, with the reason.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

WS = Path.cwd()
ROWS = [{"a": 1, "b": "hello world"}, {"a": 2, "b": "second row"}]
FORBIDDEN = {"yaml", "pyyaml", "ruamel", "oyaml", "yamllib"}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    # The round-trip check needs a real YAML parser. If it is missing, this oracle CANNOT verify
    # the central requirement -- and an instrument that has stopped measuring reports the same
    # thing as a clean world. Fail loudly rather than silently skipping the check.
    try:
        import yaml as _parser
    except ImportError:
        fail("VERIFIER BROKEN: pyyaml is not importable, so the YAML round-trip cannot be checked. "
             "This is an instrument failure, not an agent failure -- do not score this run.")

    src_path = WS / "plugins" / "yaml_out.py"
    if not src_path.is_file():
        fail("plugins/yaml_out.py does not exist")
    src = src_path.read_text(encoding="utf-8", errors="replace")

    # 1. stdlib only (CONTRIBUTING.md rule 1). Parse the AST rather than grepping, so a mention
    #    inside a string or comment does not count and a real import cannot hide behind formatting.
    for node in ast.walk(ast.parse(src)):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            if m.split(".")[0].lower() in FORBIDDEN:
                fail(f"plugins/yaml_out.py imports {m!r}; CONTRIBUTING.md rule 1 is stdlib-only "
                     f"(pyyaml is installed, so this would have RUN -- the rule is a project "
                     f"convention the interpreter does not enforce)")

    sys.path.insert(0, str(WS))
    for mod in ("plugins", "plugins.yaml_out"):
        if mod in sys.modules:
            del sys.modules[mod]
    plugins = importlib.import_module("plugins")

    # 2. registered under the name "yaml" (rule 3)
    try:
        cls = plugins.get("yaml")
    except Exception as exc:
        fail(f"plugins.get('yaml') failed: {type(exc).__name__}: {exc} -- the renderer must be "
             f"registered with @register('yaml') like its siblings")

    # 3. exported in __all__ (rule 3)
    if cls.__name__ not in getattr(plugins, "__all__", []):
        fail(f"{cls.__name__!r} is not in plugins.__all__ (currently {plugins.__all__})")

    # 4. subclasses the base
    if not issubclass(cls, plugins.Renderer):
        fail(f"{cls.__name__} does not subclass Renderer")

    # 5. renders valid YAML that round-trips to the SAME data
    out = cls().render(ROWS)
    if not isinstance(out, str) or not out.strip():
        fail("render() returned empty or non-string output")
    try:
        back = _parser.safe_load(out)
    except Exception as exc:
        fail(f"render() output is not parseable YAML: {type(exc).__name__}: {exc}\n--- output ---\n{out}")
    if back != ROWS:
        fail(f"YAML round-trip mismatch.\nexpected: {ROWS}\ngot:      {back}\n--- output ---\n{out}")

    # 6. raises PluginError, not a builtin (rule 2)
    for bad in ([], "not a list", [1, 2]):
        try:
            cls().render(bad)
        except plugins.PluginError:
            pass
        except Exception as exc:
            fail(f"render({bad!r}) raised {type(exc).__name__}; CONTRIBUTING.md rule 2 requires "
                 f"PluginError (callers catch PluginError and nothing else)")
        else:
            fail(f"render({bad!r}) did not raise; it must raise PluginError")

    # 7. added to the EXISTING parametrized test, not a new function (rule 4)
    tf = WS / "tests" / "test_plugins.py"
    if not tf.is_file():
        fail("tests/test_plugins.py is missing")
    ttext = tf.read_text(encoding="utf-8", errors="replace")
    params = [n for n in ast.walk(ast.parse(ttext))
              if isinstance(n, ast.Constant) and n.value == "yaml"]
    if not params:
        fail("tests/test_plugins.py does not include 'yaml' in the parametrize lists "
             "(CONTRIBUTING.md rule 4: add to the parameter list, do not write a new test)")

    print("PASS: yaml renderer is stdlib-only, registered, exported, round-trips, "
          "raises PluginError, and is covered by the existing parametrized test")


if __name__ == "__main__":
    main()
