"""Shared pytest configuration.

The FastAPI server is an **optional** extra (``zakcode[server]``). Its test modules
import ``fastapi`` / ``httpx`` at module scope, so when the extra is not installed
(a plain ``uv sync && pytest``, or a ``uv run pytest`` that did not carry
``--extra server``) they would fail at *collection* with ``ModuleNotFoundError``
rather than skipping. Ignore those modules when the extra is absent — they run
normally whenever it is present (CI installs it; see ``.github/workflows/ci.yml``).
"""

from __future__ import annotations

import importlib.util

#: Server test modules that import the optional ``server`` extra at module scope.
_SERVER_TEST_MODULES = [
    "test_server_app.py",
    "test_server_client.py",
    "test_server_webclient.py",
    "test_server_ws.py",
]

collect_ignore: list[str] = []
if importlib.util.find_spec("fastapi") is None:
    collect_ignore += _SERVER_TEST_MODULES
