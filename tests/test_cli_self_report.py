"""``zakcode info`` / ``version`` must answer "what is actually running here?".

Both were incomplete in ways that hid real misconfiguration:

* ``info`` rendered neither ``local_only`` nor ``extra_body`` / ``extra_headers`` — the
  settings that fail SILENTLY. An older build has no ``local_only`` field at all and a
  shadowed one never applies; in both cases every visible row still read correct.
* ``version`` printed the hand-maintained ``0.0.1`` for every build, so two checkouts
  weeks apart were indistinguishable from inside the process.
"""

from __future__ import annotations

import json

import pytest

from zakcode.build_info import version_line
from zakcode.cli import build_info_lines
from zakcode.config import Settings

POD = "http://10.0.0.250:9090/v1"


def _rows(settings: Settings) -> dict[str, str]:
    return dict(build_info_lines(settings))


class TestCostRowsAreVisible:
    def test_local_only_on_is_reported(self) -> None:
        rows = _rows(Settings(local_only=True))
        assert rows["Local only"].startswith("on")

    def test_local_only_off_is_reported(self) -> None:
        assert _rows(Settings(local_only=False))["Local only"].startswith("off")

    def test_empty_allowlist_names_the_env_var_to_set(self) -> None:
        # The actionable case: local_only on with no allowlist trusts ANY api_base,
        # including a gateway that forwards to metered providers.
        assert "ZAKCODE_LOCAL_API_BASES" in _rows(Settings(local_only=True))["Local api_bases"]

    def test_populated_allowlist_is_listed(self) -> None:
        rows = _rows(Settings(local_only=True, local_api_bases=[POD]))
        assert rows["Local api_bases"] == POD

    def test_allowlist_row_absent_when_local_only_off(self) -> None:
        assert "Local api_bases" not in _rows(Settings(local_only=False))


class TestRequestShapingRows:
    def test_extra_body_is_rendered(self) -> None:
        body = {"chat_template_kwargs": {"enable_thinking": False}}
        assert json.loads(_rows(Settings(extra_body=body))["Extra body"]) == body

    def test_extra_headers_show_names_only_never_values(self) -> None:
        # CLAUDE.md rule 4: info may report presence, never a secret value. Headers are
        # the usual carrier for an auth token.
        secret = "super-secret-token"
        rows = _rows(Settings(extra_headers={"Authorization": secret, "X-Trace": "abc"}))
        rendered = rows["Extra headers"]
        assert "Authorization" in rendered and "X-Trace" in rendered
        assert secret not in rendered
        assert "abc" not in rendered

    def test_absent_when_unset(self) -> None:
        rows = _rows(Settings())
        assert "Extra body" not in rows
        assert "Extra headers" not in rows


@pytest.fixture()
def _clean_provenance() -> object:
    """Reset zakcode's PROCESS-GLOBAL provenance cache between tests.

    ``config.env_source`` is documented as best-effort within one process: dotenv exports
    stick in ``os.environ`` and their source is remembered in a module-level dict. That is
    fine for a CLI process (one load), but across tests a name exported by an earlier case
    is still reported as ".env"-sourced in a later one.
    """
    import zakcode.config as cfg

    cfg._DOTENV_EXPORTED.clear()
    cfg._ENV_SOURCES.clear()
    yield
    cfg._DOTENV_EXPORTED.clear()
    cfg._ENV_SOURCES.clear()


@pytest.mark.usefixtures("_clean_provenance")
class TestProvenanceAnnotation:
    def test_user_env_value_is_annotated(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        home = tmp_path / ".zakcode"
        home.mkdir()
        (home / ".env").write_text("ZAKCODE_PERMISSION_MODE=allow\n", encoding="utf-8")
        monkeypatch.setenv("ZAKCODE_HOME", str(home))
        monkeypatch.delenv("ZAKCODE_PERMISSION_MODE", raising=False)
        from zakcode.config import load_settings

        settings = load_settings()
        assert "user .env" in _rows(settings)["Permission mode"]

    def test_annotation_uses_no_rich_markup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Square brackets are rich STYLE MARKUP and are silently swallowed on render.

        Regression: the annotation was first written as ``[user .env]``. It was present in
        build_info_lines() and invisible in the panel — the exact class of silent failure
        this feature exists to surface.
        """
        home = tmp_path / ".zakcode"
        home.mkdir()
        (home / ".env").write_text("ZAKCODE_PERMISSION_MODE=allow\n", encoding="utf-8")
        monkeypatch.setenv("ZAKCODE_HOME", str(home))
        monkeypatch.delenv("ZAKCODE_PERMISSION_MODE", raising=False)
        from zakcode.config import load_settings

        value = _rows(load_settings())["Permission mode"]
        assert "[" not in value and "]" not in value

    def test_real_env_var_is_not_annotated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ZAKCODE_PERMISSION_MODE", "allow")
        from zakcode.config import load_settings

        assert _rows(load_settings())["Permission mode"] == "allow"


class TestVersionLine:
    def test_plain_version_when_no_install_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import zakcode.build_info as bi

        bi._direct_url.cache_clear()
        monkeypatch.setattr(bi, "_direct_url", lambda: {})
        assert version_line("1.2.3") == "1.2.3"

    def test_git_install_reports_the_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import zakcode.build_info as bi

        bi._direct_url.cache_clear()
        monkeypatch.setattr(
            bi, "_direct_url", lambda: {"vcs_info": {"vcs": "git", "commit_id": "a" * 40}}
        )
        assert version_line("0.0.1") == f"0.0.1 (git {'a' * 12})"

    def test_local_path_install_is_labelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import zakcode.build_info as bi

        bi._direct_url.cache_clear()
        monkeypatch.setattr(bi, "_direct_url", lambda: {"dir_info": {}})
        assert version_line("0.0.1") == "0.0.1 (local path)"

    def test_vcs_install_with_unreadable_commit_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import zakcode.build_info as bi

        bi._direct_url.cache_clear()
        monkeypatch.setattr(bi, "_direct_url", lambda: {"vcs_info": {"commit_id": ""}})
        assert version_line("0.0.1") == "0.0.1 (git, commit unknown)"
