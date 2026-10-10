#!/usr/bin/env python3
"""Refresh the bundled litellm model-entries snapshot.

Imports litellm WITHOUT ``LITELLM_LOCAL_MODEL_COST_MAP`` so the remote price map
is fetched, reads the entries for the requested model ids, and writes the snapshot
deterministically (sorted keys, indent 2) to
``src/zakcode/providers/litellm_model_entries.json``.

It refuses (exit 1, nothing written) when the loaded map lacks a requested id: a process that
cannot reach the remote map silently loads litellm's bundled one, and a snapshot written from it
would drop the very models this file exists to ship (ADR-0281).

Usage::

    # default models (the recipe's mix):
    python scripts/refresh_litellm_model_entries.py

    # explicit list:
    python scripts/refresh_litellm_model_entries.py --models gpt-6-luna gpt-5-nano gpt-5-mini

    # offline check: exits non-zero when the snapshot's litellm_version differs
    # from the installed litellm (CI-safe, no network):
    python scripts/refresh_litellm_model_entries.py --check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Default recipe models (bare keys as they appear in litellm.model_cost).
_DEFAULT_MODELS = ["gpt-5-mini", "gpt-5-nano", "gpt-6-luna"]

_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "zakcode"
    / "providers"
    / "litellm_model_entries.json"
)


def _check(snapshot_path: Path) -> int:
    """Offline check: exit 0 if snapshot version matches installed litellm, else 1."""
    import importlib.metadata as md

    installed = md.version("litellm")
    with open(snapshot_path, encoding="utf-8") as f:
        snapshot = json.load(f)
    snapshot_version = snapshot.get("_meta", {}).get("litellm_version")
    if snapshot_version == installed:
        print(f"OK: snapshot litellm_version {snapshot_version} == installed {installed}")
        return 0
    print(
        f"MISMATCH: snapshot litellm_version {snapshot_version} != installed {installed}",
        file=sys.stderr,
    )
    return 1


def _refresh(models: list[str], snapshot_path: Path) -> int:
    """Fetch the remote map and write the snapshot."""
    # Ensure the remote map is fetched, not the bundled one.
    os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)

    import importlib.metadata as md

    import litellm

    version = md.version("litellm")
    entries: dict[str, dict[str, object]] = {}
    missing: list[str] = []

    for model_id in sorted(set(models)):
        if model_id in litellm.model_cost:
            entries[model_id] = litellm.model_cost[model_id]
        else:
            missing.append(model_id)

    if missing:
        print(
            f"ERROR: not in the price map litellm loaded: {missing}. Is the fetch blocked (litellm "
            "then loads its bundled map)? Nothing written.",
            file=sys.stderr,
        )
        return 1

    snapshot = {
        "_meta": {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "litellm_version": version,
            "models": sorted(set(models)),
            "source": "litellm.model_cost (fetched map, env LITELLM_LOCAL_MODEL_COST_MAP unset)",
        },
        "entries": entries,
    }

    with open(snapshot_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(entries)} entries to {snapshot_path} (litellm {version})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh bundled litellm model entries")
    parser.add_argument(
        "--models",
        nargs="+",
        default=_DEFAULT_MODELS,
        help="Model ids to include (bare keys, as in litellm.model_cost)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Offline: exit non-zero if snapshot version != installed litellm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_SNAPSHOT_PATH,
        help="Snapshot output path",
    )
    args = parser.parse_args()

    if args.check:
        return _check(args.output)
    return _refresh(args.models, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
