"""``load_settings`` must export the local ``.env`` into the process environment.

litellm reads provider keys (``OPENAI_API_KEY``, ``GROQ_API_KEY``, …) from
``os.environ`` — never from our settings surface. pydantic-settings alone extracts
only ``ZAKCODE_*`` entries from the env file, so without the ``load_dotenv`` step in
:func:`zakcode.config.load_settings` a provider key placed in ``.env`` was silently
ignored (the program would raise ``AuthError`` despite a correct-looking file).
"""

from __future__ import annotations

import os
from pathlib import Path

from zakcode.config import load_settings

#: Deliberately fake name — never a real provider variable, so a failure to clean up
#: can't leak state into other tests or shadow a real credential.
_SENTINEL = "ZAKCODE_TEST_FAKE_PROVIDER_KEY"


def test_dotenv_provider_key_reaches_process_env(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text(f"{_SENTINEL}=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(_SENTINEL, raising=False)
    try:
        load_settings()
        assert os.environ.get(_SENTINEL) == "from-dotenv"
    finally:
        os.environ.pop(_SENTINEL, None)


def test_real_environment_wins_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    """``override=False``: a variable already set in the environment is never clobbered."""
    (tmp_path / ".env").write_text(f"{_SENTINEL}=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(_SENTINEL, "from-real-env")
    load_settings()
    assert os.environ[_SENTINEL] == "from-real-env"


def test_missing_dotenv_is_a_noop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()  # must not raise
    assert settings is not None
