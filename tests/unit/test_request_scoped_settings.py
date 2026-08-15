"""Tests that request-path settings resolve to the app's own `Settings`.

`create_app(settings=...)` injects an explicit instance and the lifespan
publishes it as `app.state.settings`. Code that reads process-global
`get_settings()` instead reads the *environment*, so a single decision could be
assembled from two different objects - which is exactly what happened in the
rate limiter: `get_client_identifier` read `trusted_proxies` from the
environment while `enforce_llm_rate_limit`, in the same module, read the limit
from the injected instance. An injected `TRUSTED_PROXIES` was silently ignored.

These tests pin that `app.state.settings` wins whenever it exists, and that the
`get_settings()` fallback survives only for an app whose lifespan never
populated `app.state`.
"""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.deps.settings import get_request_settings
from app.middleware.rate_limit import get_client_identifier


REAL_CLIENT_IP = "203.0.113.9"


def _settings(trusted_proxies: list[str], api_key: str = "state-api-key-1234567890") -> Settings:
    """Build an isolated `Settings` with explicit field values."""
    return Settings(
        api_key=SecretStr(api_key),
        session_signing_key=SecretStr("test-signing-key-1234567890"),
        llm_model="openai:gpt-4o",
        llm_api_key=SecretStr("test-llm-key-1234567890"),
        trusted_proxies=trusted_proxies,
    )


class TestRateLimitIdentitySettingsSource:
    """`get_client_identifier` must key on the app's own `trusted_proxies`."""

    def test_app_state_settings_win_over_process_global(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected allow-list is honoured even when the environment disagrees.

        The environment says nothing is trusted; the app says `10.0.0.0/8` is.
        Reading the environment would ignore the header and key on the proxy,
        collapsing every client behind it into one bucket.
        """
        monkeypatch.setattr(
            "app.middleware.rate_limit.get_settings",
            lambda: _settings([]),
        )

        app = FastAPI()
        app.state.settings = _settings(["10.0.0.0/8"])

        @app.get("/test")
        async def route(request: Request) -> dict[str, str]:
            return {"client": get_client_identifier(request)}

        with TestClient(app, client=("10.0.0.5", 12345)) as client:
            response = client.get("/test", headers={"X-Forwarded-For": REAL_CLIENT_IP})

        assert response.json()["client"] == REAL_CLIENT_IP

    def test_app_state_settings_win_when_they_are_the_stricter_side(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The precedence holds in both directions, not just the permissive one."""
        monkeypatch.setattr(
            "app.middleware.rate_limit.get_settings",
            lambda: _settings(["10.0.0.0/8"]),
        )

        app = FastAPI()
        app.state.settings = _settings([])

        @app.get("/test")
        async def route(request: Request) -> dict[str, str]:
            return {"client": get_client_identifier(request)}

        with TestClient(app, client=("10.0.0.5", 12345)) as client:
            response = client.get("/test", headers={"X-Forwarded-For": REAL_CLIENT_IP})

        assert response.json()["client"] == "10.0.0.5"

    def test_falls_back_to_process_global_without_app_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bare harness with no lifespan still resolves settings."""
        monkeypatch.setattr(
            "app.middleware.rate_limit.get_settings",
            lambda: _settings(["10.0.0.0/8"]),
        )

        app = FastAPI()

        @app.get("/test")
        async def route(request: Request) -> dict[str, str]:
            return {"client": get_client_identifier(request)}

        with TestClient(app, client=("10.0.0.5", 12345)) as client:
            response = client.get("/test", headers={"X-Forwarded-For": REAL_CLIENT_IP})

        assert response.json()["client"] == REAL_CLIENT_IP


class TestGetRequestSettingsDependency:
    """The `Depends` counterpart used by `verify_api_key`."""

    async def test_returns_app_state_settings_when_present(self) -> None:
        """The injected instance is returned identically, not a rebuilt copy."""
        app = FastAPI()
        injected = _settings([])
        app.state.settings = injected

        request = Request({"type": "http", "app": app, "headers": []})

        assert await get_request_settings(request) is injected

    async def test_falls_back_when_app_state_is_unpopulated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No `app.state.settings` (lifespan never ran) falls back, never fails."""
        fallback = _settings([])
        monkeypatch.setattr("app.deps.settings.get_settings", lambda: fallback)

        app = FastAPI()
        request = Request({"type": "http", "app": app, "headers": []})

        assert await get_request_settings(request) is fallback

    async def test_non_settings_state_attribute_is_not_trusted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A harness leaving a mock on `app.state` must not become the settings.

        Unit harnesses build `app.state` out of `MagicMock`s, whose attribute
        access invents any name asked for. Type-checking the attribute keeps
        such a stand-in from silently becoming the configuration a security
        check reads.
        """
        fallback = _settings([])
        monkeypatch.setattr("app.deps.settings.get_settings", lambda: fallback)

        app = FastAPI()
        app.state.settings = object()
        request = Request({"type": "http", "app": app, "headers": []})

        assert await get_request_settings(request) is fallback
