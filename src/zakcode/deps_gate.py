"""Declared-vs-undeclared dependency gate — the first concrete step toward safe
self-remediation (see ``docs/SELF-REMEDIATION.md``).

A package-install command (``pip install X``, ``uv add X``, ``npm add X`` …) is only safe to
run autonomously when the package is one the project already vouches for — i.e. it already
appears in a lockfile/manifest. An *undeclared* install is the typosquatting / malicious-
dependency / prompt-injection-escalation vector, so it must escalate (ASK) — and, with no
real execution sandbox yet, hard-DENY in ``autonomous`` mode. (Download-and-execute,
``curl | bash``, is already a hard floor in :data:`zakcode.permissions.DANGEROUS_PATTERNS`.)

This module is **pure** (no permission logic, no prompting): it parses a shell command into
the package names it would explicitly install (:func:`installed_specs`) and reads the
project's declared package set from its manifests (:func:`read_declared_packages`). The
permission policy combines the two — see :meth:`zakcode.permissions.PermissionPolicy.decide`.

Design choices, deliberately conservative:

* **Manifest reads are permissive.** ``uv.lock`` carries the *fully resolved* dependency set
  (direct + transitive), so an install of any resolved dependency is recognised as declared.
  A package missing from EVERY manifest is what gets gated — the rare false "undeclared" is a
  prompt, not a silent block (and in autonomous a clear, recoverable tool error).
* **Local and lock-file installs are not gated.** ``pip install .``/``-e .`` (the project),
  ``pip install -r req.txt`` / ``-c`` (declared files), ``uv sync``, ``npm install`` /
  ``npm ci`` with no named package — none introduce an unvetted package, so they pass through.
* **URL / VCS installs are always treated as undeclared** (``git+…``, ``https://…``,
  ``*.whl``) — they fetch code that no manifest pinned, the same risk class as the named-but-
  undeclared case.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

try:  # py311+ ships tomllib; never hard-fail the import if absent.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

#: npm-family install verbs (``yarn add`` / ``npm install`` / ``npm i`` / ``pnpm add`` …).
#: ``npm ci`` / ``pnpm install --frozen`` install only what the lockfile pins, so ``ci`` is
#: deliberately absent — a no-named-package install is never gated.
_NPM_FAMILY = {"npm", "pnpm", "yarn"}
_NPM_VERBS = {"install", "i", "add"}
_PIP_NAMES = {"pip", "pip3"}

#: Ecosystem tags. A declared/installed identity carries its ecosystem (``pypi:requests`` /
#: ``npm:left-pad``) so a name only ever vouches for an install in the SAME ecosystem — an
#: npm-declared ``left-pad`` must NOT make ``pip install left-pad`` (a different, unvetted
#: PyPI package that happens to share the string) look declared, and vice versa.
_ECO_PYPI = "pypi"
_ECO_NPM = "npm"

#: Flags that consume the FOLLOWING token as a value (so ``-r requirements.txt`` doesn't leave
#: ``requirements.txt`` looking like a package). Union across pip/uv/npm; harmless if a flag
#: doesn't apply to a given tool.
_VALUE_FLAGS = {
    "-r",
    "--requirement",
    "-c",
    "--constraint",
    "-i",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "-t",
    "--target",
    "--prefix",
    "-p",
    "--python",
    "--registry",
    "--index",
    "-d",
    "--save-exact",
    "--prefer-dist",
}

_PEP503 = re.compile(r"[-_.]+")
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")


def normalize(name: str) -> str:
    """PEP 503 normalization: lowercase, collapse runs of ``- _ .`` to a single ``-``.

    Applied to both manifest names and parsed install names so ``Foo_Bar`` and ``foo-bar``
    compare equal. Harmless for npm names (already lowercase, rarely contain ``_``/``.``).
    """
    return _PEP503.sub("-", name.strip().lower()).strip("-")


def _tag(eco: str, name: str | None) -> str | None:
    """Ecosystem-tag a bare name (``requests`` → ``pypi:requests``), or ``None`` if falsy."""
    return f"{eco}:{name}" if name else None


def display_name(spec: str) -> str:
    """Strip the ecosystem tag for a human-facing message (``pypi:evil`` → ``evil``)."""
    return spec.split(":", 1)[1] if ":" in spec else spec


def _name_from_spec(spec: str, *, npm: bool) -> str | None:
    """Reduce one install token to a comparable, ecosystem-tagged identity, or ``None`` to skip.

    Returns ``<eco>:<normalized name>`` for a plain ``name``/``name==1.0``/``name[extra]``/
    ``name@ver``; ``<eco>:<lowercased spec>`` for a URL/VCS install (so it can never match a
    manifest name and is therefore always treated as undeclared); and ``None`` for a local-path /
    project install (``.``, ``./x``, ``/abs``, ``*.whl`` files, requirement files), not gated.
    The ``<eco>`` tag (``pypi``/``npm``) keeps the two ecosystems' namespaces from colliding.
    """
    eco = _ECO_NPM if npm else _ECO_PYPI
    s = spec.strip().strip("\"'")
    if not s:
        return None
    low = s.lower()
    # URL / VCS install → always undeclared (no manifest pins arbitrary remote code).
    if re.match(r"(git|hg|svn|bzr)\+", low) or "://" in low:
        return _tag(eco, low)
    # npm scoped package (``@scope/name`` / ``@scope/name@version``) — contains ``/`` but is a
    # registry name, not a path, so it must be recognised BEFORE the local-path guard below.
    if npm and s.startswith("@"):
        parts = s.split("@")  # ['', 'scope/name', 'version'?]
        name = "@" + parts[1] if len(parts) > 1 else s
        return _tag(eco, name.lower())
    # Local path / project / file install → not an unvetted package; skip.
    if (
        s in (".", "..")
        or s.startswith(("./", "../", "/", "~", ".\\", "..\\"))
        or "/" in s
        or "\\" in s
    ):
        return None
    if low.endswith((".txt", ".cfg", ".toml", ".whl", ".tar.gz", ".zip")):
        # A wheel/sdist file path is a local install; a .txt is a stray requirements file.
        return _tag(eco, low) if low.endswith((".whl", ".tar.gz", ".zip")) else None
    if npm:
        # npm non-scoped: ``name@version`` strips the trailing ``@version``.
        return _tag(eco, s.split("@", 1)[0].lower())
    # pip/uv/poetry: cut at the first version/extra/marker/direct-ref delimiter.
    name = re.split(r"[\[<>=!~;@(\s]", s, maxsplit=1)[0]
    return _tag(eco, normalize(name))


def _basename(token: str) -> str:
    """The lowercased command name from a path/exe token (``C:\\x\\pip.exe`` → ``pip``)."""
    base = re.split(r"[\\/]", token)[-1].lower()
    for suffix in (".exe", ".bat", ".cmd"):
        base = base.removesuffix(suffix)
    return base


def _is_python_launcher(token: str) -> bool:
    """True for ``python`` / ``python3`` / ``py`` / ``python3.11`` / a full path to one."""
    base = _basename(token)
    return base in ("python", "py") or re.fullmatch(r"python3(\.\d+)?", base) is not None


def _install_scan_start(toks_low: list[str]) -> tuple[bool, int] | None:
    """Resolve the package-manager invocation to ``(is_npm, index)`` or ``None``.

    ``index`` is where package tokens begin (just past the install verb). This sees through
    the launcher prefixes that the project's OWN self-fix hint emits — ``python -m pip
    install``, ``uv pip install``, ``"<full/path/python.exe>" -m pip install`` — not just a
    bare ``pip``/``uv``, so an undeclared install can't slip past the gate by spelling the
    invocation differently. Returns ``None`` for non-installs and for no-named-package
    installs (``uv sync``, ``npm ci``, ``pip download`` …).
    """
    n = len(toks_low)
    if n == 0:
        return None
    head = toks_low[0]

    # python -m pip install …  (any launcher spelling, incl. a full path to the interpreter)
    if _is_python_launcher(head):
        if (
            n >= 4
            and toks_low[1] == "-m"
            and toks_low[2] in _PIP_NAMES
            and toks_low[3] == "install"
        ):
            return (False, 4)
        return None

    base = _basename(head)

    # uv pip install … | uv run pip install … | uv add …  (NOT uv sync / uv lock / uv run <x>)
    if base == "uv":
        if n >= 3 and toks_low[1] == "pip" and toks_low[2] == "install":
            return (False, 3)
        if (
            n >= 4
            and toks_low[1] == "run"
            and toks_low[2] in _PIP_NAMES
            and toks_low[3] == "install"
        ):
            return (False, 4)
        if n >= 2 and toks_low[1] == "add":
            return (False, 2)
        return None

    # pip install … / pip3 install …  (tolerate flags before the verb: ``pip -q install foo``)
    if base in _PIP_NAMES or base == "pipx":
        verb = next((j for j in range(1, n) if toks_low[j] == "install"), None)
        if verb is not None and all(toks_low[j].startswith("-") for j in range(1, verb)):
            return (False, verb + 1)
        return None

    if base == "poetry":
        if n >= 2 and toks_low[1] == "add":
            return (False, 2)
        return None

    # npm / pnpm / yarn  <verb> …
    if base in _NPM_FAMILY:
        if n >= 2 and toks_low[1] in _NPM_VERBS:
            return (True, 2)
        return None

    return None


def installed_specs(command: str) -> list[str]:
    """The packages a shell command would EXPLICITLY install, as ecosystem-tagged identities.

    Pure and cheap (no I/O) so the permission policy can call it on every shell command and
    only read manifests when this returns something. Each entry is tagged with its ecosystem
    (``pypi:requests`` / ``npm:left-pad``) to match :func:`read_declared_packages`. Returns
    ``[]`` for a non-install, or a lockfile/manifest/local install that names no loose package.
    Handles ``&&`` / ``;`` / ``|`` chained commands by scanning each segment, and sees through a
    leading PowerShell call operator (``& "C:\\py.exe" -m pip install …``).
    """
    found: list[str] = []
    for segment in _SEGMENT_SPLIT.split(command):
        seg = segment.strip()
        if not seg:
            continue
        try:
            tokens = shlex.split(seg, posix=True)
        except ValueError:
            tokens = seg.split()
        # See through a leading PowerShell call operator so ``& <py> -m pip install`` is caught.
        if tokens and tokens[0] == "&":
            tokens = tokens[1:]
        if not tokens:
            continue
        toks_low = [t.lower() for t in tokens]
        start = _install_scan_start(toks_low)
        if start is None:
            continue
        npm, i = start
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("-"):
                if tok in _VALUE_FLAGS and "=" not in tok:
                    i += 1  # also consume this flag's value token
                i += 1
                continue
            name = _name_from_spec(tok, npm=npm)
            if name is not None:
                found.append(name)
            i += 1
    return found


def undeclared_install_specs(command: str, declared: set[str]) -> list[str]:
    """The install targets in ``command`` that are NOT in the project's ``declared`` set.

    Both sides are ecosystem-tagged (:func:`installed_specs` / :func:`read_declared_packages`),
    so the membership test is ecosystem-scoped: a ``pypi:`` install is only vouched for by a
    ``pypi:`` declaration. A URL/VCS spec is never in ``declared`` and so is always returned
    (flagged). Empty when the command installs nothing loose, or every named package is declared.
    Entries are returned tagged; use :func:`display_name` for a human-facing message.
    """
    return [spec for spec in installed_specs(command) if spec not in declared]


# ── manifest reading ──────────────────────────────────────────────────────────


def _names_from_pep508(reqs: object) -> set[str]:
    """Normalized names from a list of PEP 508 requirement strings (or a single string)."""
    out: set[str] = set()
    items = reqs if isinstance(reqs, list) else [reqs]
    for item in items:
        if isinstance(item, str):
            name = re.split(r"[\[<>=!~;@(\s]", item.strip(), maxsplit=1)[0]
            if name:
                out.add(normalize(name))
        elif isinstance(item, dict):
            # PEP 735 ``{include-group = "..."}`` entries carry no package name; skip.
            continue
    return out


def _read_pyproject(path: Path) -> set[str]:
    if tomllib is None or not path.is_file():
        return set()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out: set[str] = set()
    project = data.get("project", {})
    if isinstance(project, dict):
        out |= _names_from_pep508(project.get("dependencies", []))
        opt = project.get("optional-dependencies", {})
        if isinstance(opt, dict):
            for group in opt.values():
                out |= _names_from_pep508(group)
    groups = data.get("dependency-groups", {})  # PEP 735
    if isinstance(groups, dict):
        for group in groups.values():
            out |= _names_from_pep508(group)
    # Poetry tables key dependencies by name.
    tool = data.get("tool", {})
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    if isinstance(poetry, dict):
        for table_key in ("dependencies", "dev-dependencies"):
            table = poetry.get(table_key, {})
            if isinstance(table, dict):
                out |= {normalize(k) for k in table if k.lower() != "python"}
        # Poetry 1.2+ group dependencies: ``[tool.poetry.group.<name>.dependencies]``.
        group = poetry.get("group", {})
        if isinstance(group, dict):
            for grp in group.values():
                deps = grp.get("dependencies", {}) if isinstance(grp, dict) else {}
                if isinstance(deps, dict):
                    out |= {normalize(k) for k in deps if k.lower() != "python"}
    return out


def _read_uv_lock(path: Path) -> set[str]:
    """The FULL resolved set from ``uv.lock`` (each ``[[package]]`` has ``name``)."""
    if tomllib is None or not path.is_file():
        return set()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    packages = data.get("package", [])
    if not isinstance(packages, list):
        return set()
    # Guard the name type too: a hand-edited/corrupt lock with a non-string ``name`` must not
    # raise out of the "never raises" reader (normalize() would call .strip() on a non-str).
    return {
        normalize(p["name"])
        for p in packages
        if isinstance(p, dict) and isinstance(p.get("name"), str) and p["name"]
    }


_REQ_INCLUDE = re.compile(r"^(?:-r|--requirement)[=\s]+(.+)$")


def _read_requirements_file(path: Path, out: set[str], visited: set[Path]) -> None:
    """Read one requirements file into ``out``, following ``-r``/``--requirement`` includes.

    Includes resolve relative to the including file and are de-duplicated via ``visited``
    (resolved paths), so an include cycle terminates and a layered ``-r requirements/base.txt``
    chain is fully followed. Defensive: an unreadable file or unresolvable include path
    contributes nothing, never raises.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return
    if resolved in visited or len(visited) > 64:  # cycle / runaway guard
        return
    visited.add(resolved)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        include = _REQ_INCLUDE.match(line)
        if include:
            _read_requirements_file(path.parent / include.group(1).strip(), out, visited)
            continue
        if line.startswith("-"):  # other options (-e / -c / --hash / ...) carry no top-level name
            continue
        name = re.split(r"[\[<>=!~;@(\s]", line, maxsplit=1)[0]
        if name and "/" not in name and "\\" not in name:
            out.add(normalize(name))


def _collect_requirements(root: Path) -> set[str]:
    """Every package named across the project's requirements files.

    Broad-matches ``*requirements*.txt`` at the root (so ``dev-requirements.txt`` /
    ``requirements-dev.txt`` count, not only the ``requirements*.txt`` prefix) and
    ``requirements/*.txt`` (the requirements-dir split convention), and follows ``-r`` includes
    from each (:func:`_read_requirements_file`) so packages reachable only through an include
    are still recognised.
    """
    out: set[str] = set()
    visited: set[Path] = set()
    for pattern in ("*requirements*.txt", "requirements/*.txt"):
        try:
            files = list(root.glob(pattern))
        except OSError:
            continue
        for req in files:
            if req.is_file():
                _read_requirements_file(req, out, visited)
    return out


def _read_package_json(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    out: set[str] = set()
    if isinstance(data, dict):
        for field in (
            "dependencies",
            "devDependencies",
            "optionalDependencies",
            "peerDependencies",
        ):
            section = data.get(field, {})
            if isinstance(section, dict):
                out |= {k.lower() for k in section}
    return out


def read_declared_packages(workspace_root: str | Path) -> set[str]:
    """Union of every package the project declares, ecosystem-tagged, across its manifests.

    Reads ``pyproject.toml`` (project + optional + dependency-groups + poetry incl. 1.2+
    groups), ``uv.lock`` (the full resolved set), ``requirements*.txt`` / ``*requirements*.txt``
    / ``requirements/*.txt`` (following ``-r`` includes), and ``package.json``. Python names are
    tagged ``pypi:``, npm names ``npm:`` (matching :func:`installed_specs`), so a name only
    vouches for an install in its OWN ecosystem. Every read is defensive — a missing or
    malformed file contributes nothing, never raises — so the gate fails toward asking (an
    unreadable manifest just means fewer packages count as declared), never toward a crash.
    """
    root = Path(workspace_root)
    pypi: set[str] = set()
    pypi |= _read_pyproject(root / "pyproject.toml")
    pypi |= _read_uv_lock(root / "uv.lock")
    pypi |= _collect_requirements(root)
    npm = _read_package_json(root / "package.json")
    return {f"{_ECO_PYPI}:{n}" for n in pypi} | {f"{_ECO_NPM}:{n}" for n in npm}


__all__ = [
    "installed_specs",
    "undeclared_install_specs",
    "read_declared_packages",
    "normalize",
    "display_name",
]
