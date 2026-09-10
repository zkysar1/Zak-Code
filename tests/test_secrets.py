"""Tests for the secret-redaction guard (zakcode.secrets)."""

from __future__ import annotations

from zakcode.secrets import redact_secrets, strip_url_credentials


def test_strip_url_credentials_masks_userinfo() -> None:
    # audit3 #7: an api_base with embedded user:pass@ must be masked for display/serialization,
    # preserving the host and path.
    masked = strip_url_credentials("https://user:s3cret-token@gateway.example.com:8443/v1")
    assert masked is not None
    assert "s3cret-token" not in masked and "user" not in masked
    assert "gateway.example.com:8443/v1" in masked
    # URLs without userinfo (and None) pass through unchanged.
    assert strip_url_credentials("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"
    assert strip_url_credentials(None) is None


def test_redacts_url_userinfo_in_freeform_text() -> None:
    # SRV-07: a provider error echoing a credentialed api_base must not leak the password.
    # redact_secrets is the chokepoint litellm_provider._map_error scrubs through.
    out, n = redact_secrets("RequestFailed: could not reach https://svc:Hunter2Pass@gw.example/v1")
    assert n >= 1
    assert "Hunter2Pass" not in out
    assert "://***@gw.example/v1" in out


def test_redacts_url_userinfo_with_only_a_token() -> None:
    # SRV-07 hardening (Phase 4): a USER-ONLY credential (no password) -- e.g. an API key carried as
    # the userinfo username -- must be masked, not leaked verbatim.
    out, n = redact_secrets("connect https://s3cr3t-token@gateway.example.com/v1 failed")
    assert n >= 1
    assert "s3cr3t-token" not in out
    assert "://***@gateway.example.com/v1" in out


def test_redacts_url_userinfo_with_at_in_password() -> None:
    # SRV-07 hardening (Phase 4): a password containing '@' must be FULLY masked (to the last @).
    out, n = redact_secrets("redis://default:p@ss@w0rd@cache.internal:6379 timed out")
    assert n >= 1
    assert "p@ss@w0rd" not in out and "ss@w0rd" not in out  # no password fragment survives
    assert "://***@cache.internal:6379" in out


def test_does_not_redact_url_without_userinfo() -> None:
    # SRV-07: a plain URL (no userinfo @) and a path '@' are NOT redacted (no false positive).
    out, n = redact_secrets("GET https://api.example.com/v1/users?at=@here returned 500")
    assert n == 0
    assert "https://api.example.com/v1/users" in out


def test_redacts_openai_style_key() -> None:
    out, n = redact_secrets("here it is sk-ABCDEFGHIJKLMNOP1234567890 ok")
    assert n == 1
    assert "sk-ABCDEFGHIJKLMNOP" not in out
    assert "[REDACTED]" in out


def test_redacts_aws_and_github_and_slack() -> None:
    out, n = redact_secrets(
        "AKIAIOSFODNN7EXAMPLE and ghp_0123456789012345678901234567890123 and "
        "xoxb-1234567890-abcdEFGH"
    )
    assert n == 3
    assert "AKIA" not in out
    assert "ghp_" not in out
    assert "xoxb-" not in out


def test_redacts_assignment_keeps_key_name() -> None:
    out, n = redact_secrets("password = hunter2hunter2")
    assert n == 1
    assert out.startswith("password")  # key name preserved
    assert "hunter2hunter2" not in out
    assert "[REDACTED]" in out


def test_redacts_pem_block() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIByyz...\n-----END RSA PRIVATE KEY-----"
    out, n = redact_secrets(f"key:\n{pem}\nend")
    assert n == 1
    assert "PRIVATE KEY" in out  # marker mentions it
    assert "MIIByyz" not in out


def test_ordinary_prose_is_untouched() -> None:
    text = "The API key lives in the environment, never in code. Run the tests."
    out, n = redact_secrets(text)
    assert n == 0  # no key=value, no token shape
    assert out == text


def test_empty_is_safe() -> None:
    assert redact_secrets("") == ("", 0)


def test_redacts_google_oauth_api_key_and_jwt_shapes() -> None:
    # ADR-0116: a `gcloud auth print-access-token` value (ya29.…) was echoed into a transcript
    # twice; the shape was not in the guard. Google API keys and JWTs ride along.
    ya29 = "ya29.c.c0AZ4bNp" + "x" * 60 + ".Q1w2E3r4T5y6U7i8O9p0"
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV"
    api_key = "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q"  # AIza + 35, the real shape
    out, n = redact_secrets(f"token: {ya29}\nkey {api_key}\n{jwt}")
    assert n >= 3
    assert "ya29." not in out and "AIzaSy" not in out and "eyJhbGci" not in out


def test_redact_credential_tokens_scrubs_shapes_but_never_code_assignments() -> None:
    from zakcode.secrets import redact_credential_tokens

    ya29 = "ya29.a0AfB_byC" + "k" * 50
    code = "api_key = settings.api_key\nOPENAI_API_KEY=sk-live-" + "a" * 24 + "\n"
    out, n = redact_credential_tokens(code + ya29 + "\n")
    assert n == 2  # the sk- token and the ya29 token
    assert "api_key = settings.api_key" in out  # the key=value layer is NOT applied here
    assert "ya29." not in out and "sk-live-" not in out
    assert redact_credential_tokens("plain prose, nothing to hide") == (
        "plain prose, nothing to hide",
        0,
    )


_OAUTH = "wDIrZA" + "kQ9x" * 56  # an opaque 230-char OAuth token: no prefix to recognise
_HEX32 = "e114adce" * 4


def test_credential_values_after_a_secret_key_are_scrubbed_but_identifiers_never() -> None:
    """ADR-0125. Measured 2026-09-10: ``cat .yahoo_token.json`` passed the seam untouched —
    the token layer knows provider PREFIXES, and an OAuth token has none. The value layer
    tests the SHAPE of the value, so a credential file is scrubbed and source code is not."""
    from zakcode.secrets import redact_credential_tokens

    # A credential file read verbatim: both secrets gone, the non-secret fields intact.
    yahoo = (
        '{"access_token": "' + _OAUTH + '", "refresh_token": "AEFMlG7pZq7pZq7pZq7pZq", '
        '"token_type": "bearer", "expires_at": 1756570800}'
    )
    out, n = redact_credential_tokens(yahoo)
    assert n == 2 and _OAUTH not in out and "AEFMlG" not in out
    assert '"access_token": "[REDACTED]"' in out  # key and quotes kept, value gone
    assert '"token_type": "bearer"' in out and "1756570800" in out

    # A .env file: env-style names carry the key as a SUFFIX; hex and dashed shapes both.
    env = (
        f"YAHOO_CLIENT_SECRET={_HEX32}\nTAVILY_API_KEY=tvly-Ab3Ab3Ab3Ab3Ab3\n"
        "ZAKCODE_MODEL=gpt-4o-mini\n"
    )
    out, n = redact_credential_tokens(env)
    assert n == 2 and _HEX32 not in out and "tvly-" not in out
    assert "YAHOO_CLIENT_SECRET=[REDACTED]" in out and "ZAKCODE_MODEL=gpt-4o-mini" in out

    # YAML: an unquoted value that is a credential, beside ones that are words.
    yaml = (
        f"password: hunter2\naccess_token: {_OAUTH}\nname: alpha-agent-v2\n"
        "model: qwen/qwen3.6-27b\n"
    )
    out, n = redact_credential_tokens(yaml)
    assert n == 1 and _OAUTH not in out
    assert "password: hunter2" in out and "alpha-agent-v2" in out and "qwen/qwen3.6-27b" in out

    # THE ADR-0116 PROPERTY, KEPT: source code that names a secret is never rewritten.
    code = (
        "api_key = settings.api_key\n"
        'token = os.environ["TOKEN"]\n'
        "self.access_token = access_token\n"
        "password = get_password()\n"
        'token: str = Field(default="")\n'
        "secret_key = config.secret_key_v2\n"
        'API_KEY_ENV_NAME = "OPENAI_API_KEY"\n'
    )
    assert redact_credential_tokens(code) == (code, 0)

    # Placeholders and dummies are not secrets.
    ph = (
        'api_key = "{{secret:OPENAI}}"\ntoken = "[REDACTED]"\npassword = "changeme"\n'
        'api_key = "your_api_key_here"\nauthorization: Bearer ${TOKEN}\n'
    )
    assert redact_credential_tokens(ph) == (ph, 0)


def test_the_blanket_layer_no_longer_misses_quoted_or_env_style_keys() -> None:
    """The web_search screen uses ``redact_secrets`` to REFUSE a query carrying a credential.
    A JSON-quoted key (``"access_token": "…"``) and an env-style name (``FOO_API_KEY=``)
    both slipped its word boundaries, so a pasted credential file was not refused."""
    out, n = redact_secrets('"access_token": "' + _OAUTH + '"')
    assert n >= 1 and _OAUTH not in out
    out, n = redact_secrets(f"YAHOO_CLIENT_SECRET={_HEX32}")
    assert n >= 1 and _HEX32 not in out
