"""Anthropic betas litellm must forward although its own list may not name them (ADR-0341).

litellm passes every ``anthropic-beta`` value through a per-provider allowlist and drops, without
a word, a value the list does not name. It reads the list from litellm's GitHub ``main`` the first
time a request needs it and, when that fetch fails (a network-sealed sandbox, a 403 from GitHub),
from the copy bundled in the installed release. A beta added after the release is absent from the
bundled copy, so a process that cannot fetch the list sends the field a beta permits and not the
beta: Anthropic answers 400 "Extra inputs are not permitted", and the session fails over to its
fallback model.

:func:`allow_provider_betas` makes every list litellm loads name the betas this runtime sends. It
adds a name only where the loaded list lacks that key, so a list that already names the beta, or
names it as unsupported, is left exactly as it is; it removes and remaps nothing. The names ride
litellm's own loader, so nothing is fetched at import and a reload keeps them.

The litellm manager module is a parameter, not an import: ``tests/test_contracts.py`` lets only
``litellm_provider.py`` and ``registry.py`` import litellm.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger("zakcode.providers")

#: Set on the wrapper this module puts on litellm's loader: the (provider, betas) it names, so a
#: call that repeats the one before it does not wrap the wrapper.
_NAMED = "_zakcode_named_betas"


def _name_betas(config: Any, provider: str, betas: Sequence[str]) -> None:
    """Add each of ``betas`` to ``config[provider]`` where that mapping lacks the key."""
    mapping = config.get(provider) if isinstance(config, dict) else None
    if isinstance(mapping, dict):
        for beta in betas:
            mapping.setdefault(beta, beta)


def allow_provider_betas(manager: Any, provider: str, betas: Sequence[str]) -> bool:
    """Make ``manager``'s beta list name ``betas`` for ``provider``, now and on every later load.

    ``manager`` is litellm's ``anthropic_beta_headers_manager``. A list it has already loaded gets
    the names at once. A list it loads later, at first use or on ``reload_beta_headers_config``,
    gets them through a wrapper on ``get_beta_headers_config``, which the fetched list and the
    bundled one both pass through. A bare string is one beta. True when the wrapper is installed (a
    loaded list of another shape is left as it is); a call that repeats the one before it wraps
    nothing more, and calls with other names stack one wrapper each. Best effort: a manager that
    lacks the loader, or will not take the wrapper, is logged at debug and False comes back (a
    list it had already loaded has been named by then, which is harmless), because what that
    leaves is the failure this removes and not a new one.
    """
    wanted = (betas,) if isinstance(betas, str) else tuple(betas)
    try:
        _name_betas(getattr(manager, "_BETA_HEADERS_CONFIG", None), provider, wanted)
        loader: Callable[..., Any] = manager.get_beta_headers_config
        if getattr(loader, _NAMED, None) == (provider, wanted):
            return True

        @functools.wraps(loader)
        def get_beta_headers_config(*args: Any, **kwargs: Any) -> Any:
            config = loader(*args, **kwargs)
            _name_betas(config, provider, wanted)
            return config

        setattr(get_beta_headers_config, _NAMED, (provider, wanted))
        manager.get_beta_headers_config = get_beta_headers_config
    except Exception:
        logger.debug("beta_headers: could not name the betas in litellm's list", exc_info=True)
        return False
    return True
