"""Unit tests for trusted proxy validation in rate limiter.

Verify that X-Forwarded-For header is only trusted when the
immediate client is in the trusted proxy list, preventing header spoofing attacks.
"""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.middleware.rate_limit import get_client_identifier


# Simulates a request arriving from a trusted proxy. `trusted_proxies` accepts
# only IP addresses and CIDR networks, so TestClient must be given a real source
# address rather than its default "testclient" host.
TRUSTED_PROXY_CLIENT = ("10.0.0.1", 12345)


@pytest.fixture
def mock_settings_with_trusted_proxies(monkeypatch):
    """Create settings with trusted proxies configured (but NOT testclient).

    This simulates requests from real trusted proxies.
    TestClient uses "testclient" as client.host, which is NOT in this list,
    so X-Forwarded-For will be ignored (testing untrusted proxy behavior).
    """

    def mock_get_settings():
        return Settings(
            api_key=SecretStr("test-api-key-12345678"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
            trusted_proxies=["10.0.0.1", "10.0.0.2", "192.168.1.1"],
        )

    monkeypatch.setattr("app.middleware.rate_limit.get_settings", mock_get_settings)
    return mock_get_settings()


@pytest.fixture
def mock_settings_testclient_is_trusted(monkeypatch):
    """Create settings where the simulated proxy address is trusted.

    Tests using this fixture construct TestClient with
    `client=TRUSTED_PROXY_CLIENT` so `request.client.host` is a real IP inside
    the trusted list. `trusted_proxies` only accepts IP addresses and CIDR
    networks, so the previous "testclient" placeholder is no longer valid.
    """

    def mock_get_settings():
        return Settings(
            api_key=SecretStr("test-api-key-12345678"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
            trusted_proxies=["10.0.0.1", "10.0.0.2", "192.168.1.1"],
        )

    monkeypatch.setattr("app.middleware.rate_limit.get_settings", mock_get_settings)
    return mock_get_settings()


@pytest.fixture
def mock_settings_no_trusted_proxies(monkeypatch):
    """Create settings with empty trusted proxies list."""

    def mock_get_settings():
        return Settings(
            api_key=SecretStr("test-api-key-12345678"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
            trusted_proxies=[],
        )

    monkeypatch.setattr("app.middleware.rate_limit.get_settings", mock_get_settings)
    return mock_get_settings()


class TestTrustedProxyValidation:
    """Test trusted proxy validation in get_client_identifier."""

    def test_trusts_forwarded_for_from_trusted_proxy(self, mock_settings_testclient_is_trusted):
        """Should use X-Forwarded-For when request comes from trusted proxy."""
        # Arrange: Create mock request from trusted proxy with X-Forwarded-For
        app = FastAPI()

        @app.get("/test")
        async def test_route(request: Request):
            return {"client": get_client_identifier(request)}

        # Act: Request from trusted proxy (10.0.0.1) with X-Forwarded-For header
        # Simulate: Real client 203.0.113.5 -> Trusted proxy 10.0.0.1 -> Our app
        with TestClient(
            app, base_url="http://testserver", client=TRUSTED_PROXY_CLIENT
        ) as test_client:
            response = test_client.get(
                "/test",
                headers={"X-Forwarded-For": "203.0.113.5"},
            )

        # Assert: Should use the forwarded IP
        assert response.json()["client"] == "203.0.113.5"

    def test_ignores_forwarded_for_from_untrusted_client(self, mock_settings_with_trusted_proxies):
        """Should ignore X-Forwarded-For when request comes from untrusted client."""
        # Arrange: Create mock request from untrusted client with spoofed X-Forwarded-For
        app = FastAPI()

        @app.get("/test")
        async def test_route(request: Request):
            return {"client": get_client_identifier(request)}

        # Act: Request from untrusted client (203.0.113.99) trying to spoof X-Forwarded-For
        # Simulate: Attacker at 203.0.113.99 -> Our app (no proxy)
        with TestClient(app, base_url="http://testserver") as test_client:
            response = test_client.get(
                "/test",
                headers={"X-Forwarded-For": "1.2.3.4"},  # Spoofed header
            )

        # Assert: Should use direct client IP, ignoring the spoofed header
        # Note: TestClient always uses "testclient" as client.host,
        # so we check it's not the spoofed IP
        result = response.json()["client"]
        assert result != "1.2.3.4", "Should not trust X-Forwarded-For from untrusted client"

    def test_uses_direct_ip_when_no_forwarded_for_header(self, mock_settings_with_trusted_proxies):
        """Should use direct client IP when no X-Forwarded-For header is present."""
        # Arrange
        app = FastAPI()

        @app.get("/test")
        async def test_route(request: Request):
            return {"client": get_client_identifier(request)}

        # Act: Request without X-Forwarded-For header
        with TestClient(app, base_url="http://testserver") as test_client:
            response = test_client.get("/test")

        # Assert: Should use direct client IP
        result = response.json()["client"]
        assert result is not None
        assert result != ""

    def test_ignores_all_forwarded_for_when_no_trusted_proxies(
        self, mock_settings_no_trusted_proxies
    ):
        """Should never trust X-Forwarded-For when trusted_proxies list is empty."""
        # Arrange
        app = FastAPI()

        @app.get("/test")
        async def test_route(request: Request):
            return {"client": get_client_identifier(request)}

        # Act: Request with X-Forwarded-For but empty trusted proxies list
        with TestClient(app, base_url="http://testserver") as test_client:
            response = test_client.get(
                "/test",
                headers={"X-Forwarded-For": "1.2.3.4"},
            )

        # Assert: Should ignore X-Forwarded-For
        result = response.json()["client"]
        assert result != "1.2.3.4", "Should not trust X-Forwarded-For when no trusted proxies"

    def test_handles_multiple_ips_in_forwarded_for(self, mock_settings_testclient_is_trusted):
        """Should extract the last untrusted IP from an X-Forwarded-For chain.

        The chain is walked right-to-left, skipping hops that are themselves
        trusted proxies. Here only `10.0.0.1` is trusted, so `198.51.100.1` -
        the address that trusted hop actually observed - is the identity.

        `203.0.113.5` is deliberately *not* the answer even though it is the
        leftmost element: it was relayed by `198.51.100.1`, which is not in the
        trusted list, so it is an unverified claim. Taking it would let any
        caller behind a documented appending proxy (Nginx, ALB, Cloudflare) pick
        its own rate-limit bucket - see
        `tests/unit/test_middleware_rate_limit_forwarded_chain.py`.
        """
        # Arrange
        app = FastAPI()

        @app.get("/test")
        async def test_route(request: Request):
            return {"client": get_client_identifier(request)}

        # Act: X-Forwarded-For with multiple IPs (claim, observed, trusted hop)
        # Simulate: 203.0.113.5 claimed -> 198.51.100.1 observed -> trusted 10.0.0.1 -> app
        with TestClient(
            app, base_url="http://testserver", client=TRUSTED_PROXY_CLIENT
        ) as test_client:
            response = test_client.get(
                "/test",
                headers={"X-Forwarded-For": "203.0.113.5, 198.51.100.1, 10.0.0.1"},
            )

        # Assert: the last element the trusted infrastructure actually observed
        assert response.json()["client"] == "198.51.100.1"

    def test_handles_missing_client_attribute(self, mock_settings_testclient_is_trusted):
        """Should handle gracefully when request.client is None."""
        # Arrange: Mock request without client attribute
        app = FastAPI()

        @app.get("/test")
        async def test_route(request: Request):
            # Temporarily set client to None to simulate edge case
            original_client = request.client
            request.scope["client"] = None
            result = get_client_identifier(request)
            request.scope["client"] = original_client  # Restore
            return {"client": result}

        # Act
        with TestClient(app, base_url="http://testserver") as test_client:
            response = test_client.get("/test")

        # Assert: Should return "unknown" when client is None
        assert response.json()["client"] == "unknown"


class TestTrustedProxiesConfigValidation:
    """Test Settings validation for trusted_proxies field."""

    def test_trusted_proxies_accepts_list_of_ips(self):
        """Should accept list of IP addresses for trusted_proxies."""
        # Act & Assert: Should not raise validation error
        settings = Settings(
            api_key=SecretStr("test-api-key-12345678"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
            trusted_proxies=["10.0.0.1", "192.168.1.1", "172.16.0.1"],
        )
        assert settings.trusted_proxies == ["10.0.0.1", "192.168.1.1", "172.16.0.1"]

    def test_trusted_proxies_accepts_empty_list(self):
        """Should accept empty list for trusted_proxies (no proxies trusted)."""
        # Act & Assert
        settings = Settings(
            api_key=SecretStr("test-api-key-12345678"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
            trusted_proxies=[],
        )
        assert settings.trusted_proxies == []

    def test_trusted_proxies_defaults_to_empty_list(self):
        """Should default to empty list when not provided."""
        # Act & Assert
        settings = Settings(
            api_key=SecretStr("test-api-key-12345678"),
            llm_model="openai:gpt-4o",
            llm_api_key=SecretStr("test-llm-key-1234567890"),
        )
        assert settings.trusted_proxies == []
        assert isinstance(settings.trusted_proxies, list)
