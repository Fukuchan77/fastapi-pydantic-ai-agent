"""Unit tests for the `trust_proxy_headers` startup warning (Req 11.4).

`SecurityHeadersMiddleware` (L4.5) emits `Strict-Transport-Security` only when
`request.url.scheme == "https"`, resolved entirely at the ASGI-server layer -
the middleware never reads a forwarded header. Outside development, that
resolution requires the server to already trust the TLS-terminating proxy's
forwarded scheme (`--forwarded-allow-ips`/`FORWARDED_ALLOW_IPS`, L4.6), so
this warning is the only signal that the deployment and application halves
have not silently disagreed about it.
"""

import logging

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from app.config import Settings
from app.main import create_app


# staging/production both reject the default wildcard host allow-list
# (Req 1.x), so name a concrete host to keep these tests focused on the
# trust_proxy_headers warning.
NON_WILDCARD_HOSTS = ["api.example.com"]


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ."""
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "session_signing_key": "test-session-signing-key-1234567890",
        "llm_model": "openai:gpt-4",
        "llm_api_key": "test-llm-key-12345",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _has_trust_proxy_warning(records: list[logging.LogRecord]) -> bool:
    """Return True if any warning record names `trust_proxy_headers`."""
    return any(
        "trust_proxy_headers" in record.message
        for record in records
        if record.levelno == logging.WARNING
    )


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_warns_when_non_development_and_proxy_headers_untrusted(
    app_env: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-development `app_env` with the default `trust_proxy_headers=False` warns.

    Args:
        app_env: A non-development permitted `app_env` value.
        caplog: Pytest fixture capturing log records.
    """
    app = create_app(
        settings=_build_settings(app_env=app_env, allowed_hosts=NON_WILDCARD_HOSTS),
        model=TestModel(),
    )

    # TestClient triggers lifespan startup on context manager entry.
    with caplog.at_level(logging.WARNING), TestClient(app):
        pass

    assert _has_trust_proxy_warning(caplog.records), (
        f"Expected trust_proxy_headers warning for app_env={app_env!r}. "
        f"Found: {[r.message for r in caplog.records]}"
    )


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_no_warning_when_non_development_and_proxy_headers_trusted(
    app_env: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-development `app_env` with `trust_proxy_headers=True` stays silent.

    Args:
        app_env: A non-development permitted `app_env` value.
        caplog: Pytest fixture capturing log records.
    """
    app = create_app(
        settings=_build_settings(
            app_env=app_env,
            allowed_hosts=NON_WILDCARD_HOSTS,
            trust_proxy_headers=True,
        ),
        model=TestModel(),
    )

    with caplog.at_level(logging.WARNING), TestClient(app):
        pass

    assert not _has_trust_proxy_warning(caplog.records), (
        f"Expected NO trust_proxy_headers warning for app_env={app_env!r} "
        f"when trust_proxy_headers=True. Found: {[r.message for r in caplog.records]}"
    )


def test_no_warning_in_development_regardless_of_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`development` never warns, even with the default `trust_proxy_headers=False`.

    Args:
        caplog: Pytest fixture capturing log records.
    """
    app = create_app(settings=_build_settings(app_env="development"), model=TestModel())

    with caplog.at_level(logging.WARNING), TestClient(app):
        pass

    assert not _has_trust_proxy_warning(caplog.records), (
        f"Expected NO trust_proxy_headers warning in development. "
        f"Found: {[r.message for r in caplog.records]}"
    )


def test_warning_reads_flag_from_settings_not_environ(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning check reads `settings.trust_proxy_headers`, never `os.environ` directly.

    `Settings(trust_proxy_headers=False, ...)` explicitly pins the field, so
    the constructed instance's `trust_proxy_headers` is `False` regardless of
    `os.environ` (`build_test_settings`'s documented isolation contract: an
    explicit constructor argument bypasses environment lookups for that
    field). `TRUST_PROXY_HEADERS=true` is set in `os.environ` to the opposite
    value specifically so that an implementation calling
    `os.environ.get("TRUST_PROXY_HEADERS")` instead of reading
    `settings.trust_proxy_headers` would wrongly suppress the warning - only
    reading the `Settings` field produces the correct (warning-present)
    outcome here.

    Args:
        monkeypatch: Pytest fixture for environment isolation.
        caplog: Pytest fixture capturing log records.
    """
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")

    app = create_app(
        settings=_build_settings(
            app_env="production",
            allowed_hosts=NON_WILDCARD_HOSTS,
            trust_proxy_headers=False,
        ),
        model=TestModel(),
    )

    with caplog.at_level(logging.WARNING), TestClient(app):
        pass

    assert _has_trust_proxy_warning(caplog.records), (
        "Expected trust_proxy_headers warning to fire from the constructed "
        f"Settings value, ignoring os.environ. Found: {[r.message for r in caplog.records]}"
    )
