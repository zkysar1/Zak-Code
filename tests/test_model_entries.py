"""Tests for the litellm entries a deployment ships (``zakcode.providers.model_entries``).

A process that cannot fetch litellm's remote price map loads the map bundled in the installed
litellm, which can lack a model added after the release (ADR-0281). These tests pin that such a
process still prices and describes the models the deployment recipe names, that an entry litellm
already holds is never touched, and that the snapshot is litellm's own entries, current for the
pinned litellm.

Offline and deterministic: the unit tests hand the loader a fake litellm module, and the subprocess
tests force litellm's bundled map with ``LITELLM_LOCAL_MODEL_COST_MAP=True`` (litellm then makes no
fetch). None of them reads the fetched map.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from zakcode.providers import model_entries
from zakcode.providers.model_entries import register_missing_entries

_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT_PATH = _ROOT / "src" / "zakcode" / "providers" / "litellm_model_entries.json"
_REFRESH_SCRIPT = _ROOT / "scripts" / "refresh_litellm_model_entries.py"


def _snapshot() -> dict[str, Any]:
    return json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))


class _FakeLitellm:
    """The two members of the litellm module the loader touches."""

    def __init__(self, model_cost: dict[str, Any]) -> None:
        self.model_cost = model_cost
        self.register_calls: list[dict[str, Any]] = []

    def register_model(self, entries: dict[str, Any]) -> None:
        self.register_calls.append(entries)
        self.model_cost.update(entries)


class _RaisingLitellm(_FakeLitellm):
    def register_model(self, entries: dict[str, Any]) -> None:
        raise RuntimeError("register_model failed")


class TestRegisterMissingEntries:
    def test_registers_every_shipped_id_the_map_lacks(self) -> None:
        shipped = _snapshot()["entries"]
        fake = _FakeLitellm({"gpt-4o": {"sentinel": 1}})

        registered = register_missing_entries(fake)

        assert sorted(registered) == sorted(shipped)
        for model_id, entry in shipped.items():
            assert fake.model_cost[model_id] == entry  # litellm's own entry, verbatim
        assert fake.model_cost["gpt-4o"] == {"sentinel": 1}

    def test_never_overrides_an_entry_litellm_holds(self) -> None:
        shipped = _snapshot()["entries"]
        held = {"input_cost_per_token": 999, "sentinel": True}
        fake = _FakeLitellm({"gpt-6-luna": dict(held)})

        registered = register_missing_entries(fake)

        assert fake.model_cost["gpt-6-luna"] == held
        assert sorted(registered) == sorted(set(shipped) - {"gpt-6-luna"})

    def test_registers_nothing_when_the_map_holds_every_id(self) -> None:
        # The fetched map: every shipped id is there already.
        shipped = _snapshot()["entries"]
        fake = _FakeLitellm({model_id: {"sentinel": model_id} for model_id in shipped})

        assert register_missing_entries(fake) == []
        assert fake.register_calls == []

    def test_a_second_call_registers_nothing(self) -> None:
        fake = _FakeLitellm({})

        assert register_missing_entries(fake)
        assert register_missing_entries(fake) == []

    def test_a_failing_register_model_never_raises(self) -> None:
        assert register_missing_entries(_RaisingLitellm({})) == []

    def test_a_missing_snapshot_never_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(model_entries, "_SNAPSHOT_PATH", tmp_path / "absent.json")
        fake = _FakeLitellm({})

        assert register_missing_entries(fake) == []
        assert fake.register_calls == []


def _run_python(script: str) -> Any:
    """Run ``script`` in a fresh interpreter on litellm's BUNDLED price map; return its JSON line.

    ``LITELLM_LOCAL_MODEL_COST_MAP=True`` is what a process with no route to the remote map falls
    back to. The script's stdout must be that one JSON line and nothing else, so a banner litellm
    prints while the entries are registered at import fails the assertion below.
    """
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(_ROOT / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        LITELLM_LOG="ERROR",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=100,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1, f"stdout is not one JSON line (import-time output?):\n{result.stdout}"
    return json.loads(lines[0])


_SEALED_PROBE = """
import copy, json
import litellm
before = copy.deepcopy(litellm.model_cost)
from zakcode.agent.loop import _answer_room
from zakcode.providers.litellm_provider import LiteLLMProvider
from zakcode.providers.registry import get_capabilities

out = {
    "added": sorted(set(litellm.model_cost) - set(before)),
    "removed": sorted(set(before) - set(litellm.model_cost)),
    "changed": sorted(k for k in before if litellm.model_cost[k] != before[k]),
}
for model in ("openai/gpt-6-luna", "gpt-6-luna", "openai/gpt-5-mini", "openai/gpt-5-nano"):
    caps = get_capabilities(model)
    out[model] = {
        "cost": LiteLLMProvider._litellm_token_cost(model, 100000, 10000, 90000, 0),
        "context_window": caps.context_window,
        "max_output": caps.max_output,
        "supports_tools": caps.supports_tools,
        "supports_vision": caps.supports_vision,
        "supports_caching": caps.supports_caching,
        "answer_room": _answer_room(caps),
    }
print(json.dumps(out))
"""


class TestSealedProcess:
    """A fresh interpreter on litellm's bundled price map."""

    @pytest.fixture(scope="class")
    def sealed(self) -> Any:
        return _run_python(_SEALED_PROBE)

    @pytest.mark.parametrize("model", ["openai/gpt-6-luna", "gpt-6-luna"])
    def test_the_recipe_model_is_priced_and_described(self, sealed: Any, model: str) -> None:
        record = sealed[model]

        assert record["cost"] == pytest.approx(0.0069)
        assert (record["context_window"], record["max_output"]) == (922_000, 128_000)
        assert record["supports_tools"] is True
        assert record["supports_vision"] is True
        assert record["supports_caching"] is True
        assert record["answer_room"] == 16_384

    @pytest.mark.parametrize("model", ["openai/gpt-5-mini", "openai/gpt-5-nano"])
    def test_the_other_recipe_models_are_priced(self, sealed: Any, model: str) -> None:
        assert sealed[model]["cost"] is not None
        assert sealed[model]["context_window"]

    def test_registration_only_adds_shipped_ids(self, sealed: Any) -> None:
        assert sealed["removed"] == []
        assert sealed["changed"] == []
        assert set(sealed["added"]) <= set(_snapshot()["entries"])

    def test_the_registry_alone_sees_the_entries(self) -> None:
        # The capability lookup is a second reader of the price map: importing only the registry
        # must still leave the shipped entries registered.
        probe = (
            "import json\n"
            "from zakcode.providers.registry import get_capabilities\n"
            "caps = get_capabilities('openai/gpt-6-luna')\n"
            "print(json.dumps([caps.context_window, caps.max_output, caps.supports_caching]))\n"
        )

        assert _run_python(probe) == [922_000, 128_000, True]


class TestSnapshot:
    def test_it_is_canonical_json(self) -> None:
        text = _SNAPSHOT_PATH.read_text(encoding="utf-8")

        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"

    def test_it_lists_the_ids_it_ships(self) -> None:
        snapshot = _snapshot()

        assert snapshot["_meta"]["models"] == sorted(snapshot["entries"])
        assert "gpt-6-luna" in snapshot["entries"]

    def test_each_entry_carries_what_the_resolvers_read(self) -> None:
        keys = (
            "input_cost_per_token",
            "output_cost_per_token",
            "max_input_tokens",
            "max_output_tokens",
            "supports_function_calling",
            "supports_vision",
            "supports_prompt_caching",
        )
        for model_id, entry in _snapshot()["entries"].items():
            for key in keys:
                assert key in entry, f"{model_id} has no {key}"

    def test_it_matches_the_installed_litellm(self) -> None:
        # Moving the litellm pin fails this until scripts/refresh_litellm_model_entries.py is run.
        assert _snapshot()["_meta"]["litellm_version"] == md.version("litellm")


class TestRefreshScriptCheck:
    @staticmethod
    def _check(snapshot_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(_REFRESH_SCRIPT), "--check", "--output", str(snapshot_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_it_passes_on_the_shipped_snapshot(self) -> None:
        assert self._check(_SNAPSHOT_PATH).returncode == 0

    def test_it_fails_on_a_snapshot_for_another_litellm(self, tmp_path: Path) -> None:
        stale = _snapshot()
        stale["_meta"]["litellm_version"] = "0.0.0"
        path = tmp_path / "stale.json"
        path.write_text(json.dumps(stale), encoding="utf-8")

        result = self._check(path)

        assert result.returncode == 1
        assert "0.0.0" in result.stderr
