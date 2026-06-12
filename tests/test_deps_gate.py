"""The declared-vs-undeclared dependency gate (self-remediation Step 1).

Three layers, mirroring the module's own split:

* the **pure parser** — what packages does a shell command explicitly install
  (:func:`zakcode.deps_gate.installed_specs`)? Launcher-aware (``pip``, ``uv pip``,
  ``python -m pip``, ``npm``…), version/extra/URL aware, ecosystem-tagged, and a no-op for
  non-installs.
* the **manifest reader** — which packages does a project declare across its
  manifests/lockfiles (:func:`zakcode.deps_gate.read_declared_packages`)?
* the **permission integration** — :meth:`PermissionPolicy.decide`/``authorize`` escalates an
  undeclared install to ASK (hard DENY in ``autonomous``), passes declared/local/lockfile
  installs through, can't be waived by a blanket session grant, and is a complete no-op when
  the gate is off (no ``declared_packages`` provider).

Identities are ecosystem-tagged (``pypi:requests`` / ``npm:left-pad``) so a name only ever
vouches for an install in its OWN ecosystem.
"""

from __future__ import annotations

import asyncio

import pytest

from zakcode.config import PermissionTier
from zakcode.deps_gate import (
    display_name,
    installed_specs,
    normalize,
    read_declared_packages,
    undeclared_install_specs,
)
from zakcode.permissions import (
    PermissionDecision,
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionRequest,
)
from zakcode.tools.base import ConcurrencyClass, ToolSpec

# ── fixtures ────────────────────────────────────────────────────────────────────


def _spec(name: str, tier: PermissionTier) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} test tool",
        parameters={"type": "object", "properties": {}},
        required_permission=tier,
        concurrency=ConcurrencyClass.NEVER_PARALLEL,
    )


SHELL = _spec("bash", PermissionTier.DANGER_FULL_ACCESS)


class _ScriptedPrompter:
    """Returns scripted outcomes; records every confirmation request."""

    def __init__(self, outcomes: list[PermissionOutcome] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.calls: list[PermissionRequest] = []

    async def confirm(self, request: PermissionRequest) -> PermissionOutcome:
        self.calls.append(request)
        return self.outcomes.pop(0) if self.outcomes else PermissionOutcome.DENY_ONCE


def _auth(policy: PermissionPolicy, spec: ToolSpec, args: dict) -> tuple[bool, str]:
    return asyncio.run(policy.authorize(spec, args))


# ── 1. pure parser: installed_specs (ecosystem-tagged) ───────────────────────────


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # bare pip / pip3 → pypi-tagged
        ("pip install requests", ["pypi:requests"]),
        ("pip3 install Flask", ["pypi:flask"]),
        # PEP 503 normalization of the install target
        ("pip install Foo_Bar.Baz", ["pypi:foo-bar-baz"]),
        # version / extras / markers are stripped to the bare name
        ("pip install 'requests==2.31.0'", ["pypi:requests"]),
        ("pip install 'uvicorn[standard]>=0.20'", ["pypi:uvicorn"]),
        # multiple targets in one command
        ("pip install foo bar baz", ["pypi:foo", "pypi:bar", "pypi:baz"]),
        # uv add / uv pip install / uv run pip install (launcher see-through)
        ("uv add rich", ["pypi:rich"]),
        ("uv pip install rich", ["pypi:rich"]),
        ("uv run pip install rich", ["pypi:rich"]),
        # poetry
        ("poetry add black", ["pypi:black"]),
        # python -m pip install (the form the self-fix hint emits)
        ("python -m pip install httpx", ["pypi:httpx"]),
        ("python3 -m pip install httpx", ["pypi:httpx"]),
        ("py -m pip install httpx", ["pypi:httpx"]),
        # full path to the interpreter (sys.executable form), quoted
        ('"/usr/bin/python3.11" -m pip install httpx', ["pypi:httpx"]),
        ('"C:\\Py\\python.exe" -m pip install httpx', ["pypi:httpx"]),
        # the exact shape pip_install_hint emits: --python <path> consumed as a value flag
        ('uv pip install --python "/usr/bin/python3" ddgs httpx', ["pypi:ddgs", "pypi:httpx"]),
        # PowerShell call operator see-through
        ('& "/usr/bin/python3" -m pip install httpx', ["pypi:httpx"]),
        # npm family → npm-tagged
        ("npm install left-pad", ["npm:left-pad"]),
        ("npm i left-pad", ["npm:left-pad"]),
        ("npm add left-pad", ["npm:left-pad"]),
        ("pnpm add left-pad", ["npm:left-pad"]),
        ("yarn add left-pad", ["npm:left-pad"]),
        # npm scope kept; trailing @version stripped
        ("npm install @scope/pkg", ["npm:@scope/pkg"]),
        ("npm install pkg@1.2.3", ["npm:pkg"]),
        ("npm install @scope/pkg@1.2.3", ["npm:@scope/pkg"]),
        # chained commands (&&, ;, |) — every segment scanned
        ("pip install foo && echo done", ["pypi:foo"]),
        ("echo hi ; uv add bar", ["pypi:bar"]),
        ("pip install foo | tee log", ["pypi:foo"]),
        # flag before the verb tolerated
        ("pip -q install foo", ["pypi:foo"]),
        # save flags do not eat the package
        ("npm install --save-dev foo", ["npm:foo"]),
        ("npm install -D foo", ["npm:foo"]),
    ],
)
def test_installed_specs_detects_named_installs(command: str, expected: list[str]) -> None:
    assert installed_specs(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        # not an install at all
        "git status",
        "ls -la",
        "echo pip install foo",  # 'echo' head — not an installer
        # no-named-package installs: governed by mode, not the dependency gate
        "uv sync",
        "uv lock",
        "npm ci",
        "npm install",  # install everything in package.json's lockfile, names nothing new
        "pip install",  # nothing named
        # requirement / constraint files are declared inputs, not loose packages
        "pip install -r requirements.txt",
        "pip install --requirement reqs/dev.txt",
        "pip install -c constraints.txt -r requirements.txt",
        # local / editable / project installs introduce no unvetted PyPI package
        "pip install .",
        "pip install -e .",
        "pip install ./local/pkg",
        "pip install /abs/path/pkg",
        # index/registry flags consume their value, not a package
        "pip install -i https://my.index/simple",
    ],
)
def test_installed_specs_ignores_non_named_installs(command: str) -> None:
    assert installed_specs(command) == []


def test_installed_specs_flags_url_and_vcs_installs() -> None:
    # URL / VCS installs fetch code no manifest pinned → always surfaced (and never declared).
    assert installed_specs("pip install git+https://github.com/x/y.git") == [
        "pypi:git+https://github.com/x/y.git"
    ]
    assert installed_specs("pip install https://example.com/pkg-1.0.whl") == [
        "pypi:https://example.com/pkg-1.0.whl"
    ]


def test_normalize_is_pep503() -> None:
    assert normalize("Foo_Bar") == "foo-bar"
    assert normalize("Foo.Bar__Baz") == "foo-bar-baz"
    assert normalize("requests") == "requests"


def test_display_name_strips_the_ecosystem_tag() -> None:
    assert display_name("pypi:evil") == "evil"
    assert display_name("npm:@scope/pkg") == "@scope/pkg"
    assert display_name("pypi:git+https://x/y") == "git+https://x/y"  # only the first ':' splits
    assert display_name("untagged") == "untagged"


def test_undeclared_install_specs_filters_against_declared() -> None:
    declared = {"pypi:requests", "pypi:rich"}
    assert undeclared_install_specs("pip install requests rich", declared) == []
    assert undeclared_install_specs("pip install requests evil", declared) == ["pypi:evil"]
    # a URL install is never in the declared set
    assert undeclared_install_specs("pip install git+https://x/y", declared) == [
        "pypi:git+https://x/y"
    ]


def test_cross_ecosystem_name_does_not_vouch_across_ecosystems() -> None:
    # An npm-declared 'left-pad' must NOT make a PyPI 'pip install left-pad' look declared (F5).
    declared = {"npm:left-pad", "pypi:requests"}
    assert undeclared_install_specs("pip install left-pad", declared) == ["pypi:left-pad"]
    assert undeclared_install_specs("npm install left-pad", declared) == []
    assert undeclared_install_specs("pip install requests", declared) == []


# ── 2. manifest reader: read_declared_packages ───────────────────────────────────


def test_read_declared_packages_unions_every_manifest(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
        [project]
        name = "demo"
        dependencies = ["Requests>=2", "rich"]
        [project.optional-dependencies]
        web = ["httpx", "ddgs"]
        [dependency-groups]
        dev = ["pytest", {include-group = "lint"}]
        """,
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        """
        [[package]]
        name = "anyio"
        version = "4.0.0"

        [[package]]
        name = "sniffio"
        version = "1.3.0"
        """,
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text(
        "tomli==2.0.1\n# a comment\n-e .\nclick\n", encoding="utf-8"
    )
    (tmp_path / "package.json").write_text(
        '{"dependencies": {"left-pad": "^1.0.0"}, "devDependencies": {"jest": "^29"}}',
        encoding="utf-8",
    )

    declared = read_declared_packages(tmp_path)

    # pyproject: project deps (normalized), optional groups, PEP 735 groups (include-group skipped)
    assert {"pypi:requests", "pypi:rich", "pypi:httpx", "pypi:ddgs", "pypi:pytest"} <= declared
    # uv.lock: full resolved set
    assert {"pypi:anyio", "pypi:sniffio"} <= declared
    # requirements.txt: names only, skipping comments / -e lines
    assert {"pypi:tomli", "pypi:click"} <= declared
    # package.json: dependencies + devDependencies, npm-tagged
    assert {"npm:left-pad", "npm:jest"} <= declared
    # and the npm names are NOT also present pypi-tagged (ecosystems stay separate)
    assert "pypi:left-pad" not in declared


def test_poetry_group_dependencies_are_read(tmp_path) -> None:
    # Modern Poetry 1.2+ layout: [tool.poetry.group.<name>.dependencies] (F1).
    (tmp_path / "pyproject.toml").write_text(
        """
        [tool.poetry]
        name = "demo"
        [tool.poetry.dependencies]
        python = "^3.11"
        flask = "^3"
        [tool.poetry.group.test.dependencies]
        pytest = "^8"
        httpx = "^0.27"
        [tool.poetry.group.dev.dependencies]
        ruff = "^0.5"
        """,
        encoding="utf-8",
    )
    declared = read_declared_packages(tmp_path)
    assert {"pypi:flask", "pypi:pytest", "pypi:httpx", "pypi:ruff"} <= declared
    assert "pypi:python" not in declared  # the interpreter pin is not a package


def test_requirements_subdir_dev_prefix_and_includes_are_read(tmp_path) -> None:
    # F2: requirements/dev.txt (subdir) + dev-requirements.txt (non-prefix); F3: -r includes.
    (tmp_path / "requirements.txt").write_text("-r requirements/base.txt\nrequests\n", "utf-8")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "base.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "requirements" / "dev.txt").write_text("pytest\nblack\n", encoding="utf-8")
    (tmp_path / "dev-requirements.txt").write_text("mypy\n", encoding="utf-8")

    declared = read_declared_packages(tmp_path)
    assert {
        "pypi:requests",  # root requirements.txt
        "pypi:flask",  # reached only via the -r include
        "pypi:pytest",  # requirements/dev.txt (subdir)
        "pypi:black",
        "pypi:mypy",  # dev-requirements.txt (non-'requirements' prefix)
    } <= declared


def test_requirements_include_cycle_terminates(tmp_path) -> None:
    # A -r include cycle must not infinitely recurse.
    (tmp_path / "requirements.txt").write_text("-r a.txt\nroot-pkg\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("-r b.txt\na-pkg\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("-r a.txt\nb-pkg\n", encoding="utf-8")  # back-edge
    declared = read_declared_packages(tmp_path)
    assert {"pypi:root-pkg", "pypi:a-pkg", "pypi:b-pkg"} <= declared


def test_uv_lock_with_non_string_name_does_not_raise(tmp_path) -> None:
    # F4: a hand-edited/corrupt lock with a non-string name must not raise (contract: defensive).
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = 123\n\n[[package]]\nname = "anyio"\n', encoding="utf-8"
    )
    declared = read_declared_packages(tmp_path)  # must not raise
    assert "pypi:anyio" in declared


def test_read_declared_packages_is_defensive(tmp_path) -> None:
    # No manifests at all → empty set, never raises.
    assert read_declared_packages(tmp_path) == set()
    # Malformed files contribute nothing rather than crashing.
    (tmp_path / "pyproject.toml").write_text("this is { not valid toml", encoding="utf-8")
    (tmp_path / "package.json").write_text("{not json", encoding="utf-8")
    assert read_declared_packages(tmp_path) == set()


# ── 3. permission integration: PermissionPolicy.decide ───────────────────────────


def _policy(mode: PermissionMode, declared_pypi: set[str] | None = None) -> PermissionPolicy:
    # declared_pypi: bare PyPI names; tagged here to match real read_declared_packages output.
    if declared_pypi is None:
        provider = None
    else:
        tagged = {f"pypi:{n}" for n in declared_pypi}
        provider = lambda: tagged  # noqa: E731
    return PermissionPolicy(mode, declared_packages=provider)


def test_gate_off_by_default_never_escalates() -> None:
    # No declared_packages provider → the gate is a no-op; an install is judged on mode alone.
    policy = PermissionPolicy(PermissionMode.ALLOW)  # gate off
    decision, _ = policy.decide(SHELL, {"command": "pip install totally-unvetted"})
    assert decision is PermissionDecision.ALLOW


def test_undeclared_install_escalates_to_ask_in_allow_mode() -> None:
    # 'allow' would auto-allow the shell call; an undeclared install tightens it to ASK.
    policy = _policy(PermissionMode.ALLOW, {"requests"})
    decision, reason = policy.decide(SHELL, {"command": "pip install evil-typosquat"})
    assert decision is PermissionDecision.ASK
    assert "evil-typosquat" in reason
    assert "pypi:" not in reason  # the human message shows the bare name, not the tag


def test_declared_install_passes_through_in_allow_mode() -> None:
    policy = _policy(PermissionMode.ALLOW, {"requests", "rich"})
    decision, _ = policy.decide(SHELL, {"command": "pip install requests rich"})
    assert decision is PermissionDecision.ALLOW


def test_undeclared_install_hard_denies_in_autonomous() -> None:
    # No prompt exists headless and there is no execution sandbox yet → deterministic DENY.
    policy = _policy(PermissionMode.AUTONOMOUS, {"requests"})
    decision, reason = policy.decide(SHELL, {"command": "uv add evil"})
    assert decision is PermissionDecision.DENY
    assert "evil" in reason


def test_declared_install_allowed_in_autonomous() -> None:
    policy = _policy(PermissionMode.AUTONOMOUS, {"requests"})
    decision, _ = policy.decide(SHELL, {"command": "uv add requests"})
    assert decision is PermissionDecision.ALLOW


def test_lockfile_and_local_installs_are_not_gated() -> None:
    # `uv sync` / `npm ci` / `pip install .` name no loose package → never escalated, even
    # in autonomous with a strict declared set. The provider must not even be consulted.
    calls = {"n": 0}

    def provider() -> set[str]:
        calls["n"] += 1
        return set()

    policy = PermissionPolicy(PermissionMode.AUTONOMOUS, declared_packages=provider)
    for cmd in ("uv sync", "npm ci", "pip install -e .", "git status"):
        decision, _ = policy.decide(SHELL, {"command": cmd})
        assert decision is PermissionDecision.ALLOW, cmd
    assert calls["n"] == 0, "manifest provider read for a non-install command"


def test_dangerous_pattern_outranks_the_dependency_gate() -> None:
    # A command that is BOTH dangerous and an undeclared install reports the dangerous reason
    # (the catastrophic floor is checked first), and is denied in autonomous either way.
    policy = _policy(PermissionMode.AUTONOMOUS, {"requests"})
    decision, reason = policy.decide(
        SHELL, {"command": "pip install evil ; sudo rm -rf / --no-preserve-root"}
    )
    assert decision is PermissionDecision.DENY
    assert "dangerous" in reason.lower()


def test_manifest_read_failure_fails_toward_asking() -> None:
    # If the declared-set read raises, treat nothing as declared → escalate (never crash).
    def boom() -> set[str]:
        raise RuntimeError("manifest unreadable")

    policy = PermissionPolicy(PermissionMode.ALLOW, declared_packages=boom)
    decision, _ = policy.decide(SHELL, {"command": "pip install anything"})
    assert decision is PermissionDecision.ASK


def test_child_view_propagates_the_gate() -> None:
    policy = _policy(PermissionMode.AUTONOMOUS, {"requests"})
    child = policy.child_view()
    decision, _ = child.decide(SHELL, {"command": "uv add evil"})
    assert decision is PermissionDecision.DENY
    # and an off-gate parent yields an off-gate child
    off_child = PermissionPolicy(PermissionMode.ALLOW).child_view()
    decision, _ = off_child.decide(SHELL, {"command": "pip install evil"})
    assert decision is PermissionDecision.ALLOW


# ── 3b. the gate is not waivable by a blanket session grant (authorize path) ──────


def test_session_grant_does_not_waive_the_gate_interactive() -> None:
    # A blanket "allow bash for the session" must NOT silently authorize an undeclared install:
    # the gate re-escalates it to a prompt even under the grant (mirrors the dangerous floor).
    prompter = _ScriptedPrompter([PermissionOutcome.DENY_ONCE])
    policy = _policy(PermissionMode.ALLOW, {"requests"})
    policy.prompter = prompter
    policy._session_allow.add("bash")  # simulate an earlier ALLOW_SESSION / restored grant

    # declared install rides the grant with no prompt
    allowed, _ = _auth(policy, SHELL, {"command": "pip install requests"})
    assert allowed is True
    assert prompter.calls == []

    # undeclared install re-prompts despite the grant (here the operator denies)
    allowed, _ = _auth(policy, SHELL, {"command": "pip install evil"})
    assert allowed is False
    assert len(prompter.calls) == 1


def test_session_grant_does_not_waive_the_gate_autonomous() -> None:
    # Even with a restored grant, an undeclared install is a hard DENY in autonomous.
    policy = _policy(PermissionMode.AUTONOMOUS, {"requests"})
    policy._session_allow.add("bash")
    allowed, reason = _auth(policy, SHELL, {"command": "uv add evil"})
    assert allowed is False
    assert "undeclared" in reason.lower()
    # a declared install still rides the grant
    allowed, _ = _auth(policy, SHELL, {"command": "uv add requests"})
    assert allowed is True


# ── 4. facade wiring: the dependency_gate setting toggles the provider ────────────


def test_facade_wires_gate_on_by_default(tmp_path) -> None:
    import zakcode

    # tmp_path has no manifests → every package is undeclared. Default dependency_gate=True
    # + an empty workspace means an install in 'allow' mode escalates to ASK.
    agent = zakcode.Agent(workspace_root=tmp_path, permission_mode="allow")
    decision, _ = agent.permission_policy.decide(SHELL, {"command": "pip install evil"})
    assert decision is PermissionDecision.ASK


def test_facade_gate_can_be_disabled(tmp_path) -> None:
    import zakcode

    agent = zakcode.Agent(workspace_root=tmp_path, permission_mode="allow", dependency_gate=False)
    decision, _ = agent.permission_policy.decide(SHELL, {"command": "pip install evil"})
    assert decision is PermissionDecision.ALLOW
