"""Litellm's own price and capability entries, for the models a deployment is configured with.

A process that cannot fetch litellm's remote price map (a network-sealed sandbox) loads the map
bundled in the installed litellm, and that map can lack a model added after the release. Every
call to such a model then reads as unpriced, and the model is described as one nothing is known
about: no context window, no output cap, no vision, no prompt caching (ADR-0281).

``litellm_model_entries.json`` beside this module ships litellm's OWN entries, verbatim, for the
models the deployment recipe names. :func:`register_missing_entries` registers an entry with
``litellm.register_model`` only when the loaded ``litellm.model_cost`` lacks that id: it never
overrides what litellm already holds and guesses no rate, because a copy of litellm's entry is
litellm's price. ``scripts/refresh_litellm_model_entries.py`` rewrites the snapshot from the
fetched map whenever the litellm pin moves.

The litellm module is a parameter, not an import: ``tests/test_contracts.py`` lets only
``litellm_provider.py`` and ``registry.py`` import litellm.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("zakcode.providers")

_SNAPSHOT_PATH = Path(__file__).with_name("litellm_model_entries.json")


def register_missing_entries(litellm_module: Any) -> list[str]:
    """Register the shipped entries that ``litellm_module.model_cost`` lacks; return their ids.

    Idempotent: a registered id is present on the next call, so a second call registers nothing.
    Offline, and best effort: an unreadable snapshot or a failing ``register_model`` is logged at
    debug and never raised into a turn (the ids registered before a failure are still returned).
    """
    registered: list[str] = []
    try:
        snapshot = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        entries: dict[str, dict[str, Any]] = snapshot.get("entries", {})
        for model_id, entry in entries.items():
            if model_id not in litellm_module.model_cost:
                litellm_module.register_model({model_id: entry})
                registered.append(model_id)
    except Exception:
        logger.debug("model_entries: could not register the shipped entries", exc_info=True)
    return registered
