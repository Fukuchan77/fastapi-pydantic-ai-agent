"""Regression tests for CIDR support in trusted proxy validation.

`trusted_proxies` was compared with an exact string match while every
deployment target in `docs/production_deployment.md` is documented as a CIDR
range (Nginx `10.0.0.0/8`, an ALB's VPC CIDR, Cloudflare's 15 published
ranges). Configured as documented, the membership test never matched,
`X-Forwarded-For` was always ignored, and every client behind the proxy shared
a single rate-limit bucket keyed on the proxy's own address.

That takes out two controls at once: the global 1000/minute default and the
stricter `llm_rate_limit` (Req 11.3) that bounds LLM spend on
chat/stream/query. Both fail open and silently - the service keeps serving.
"""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings
from app.middleware.rate_limit import get_client_identifier


REAL_CLIENT_IP = "203.0.113.5"


def _settings(trusted_proxies: list[str]) -> Settings:
    """Build settings with the given trusted proxy entries."""
    return Settings(
        api_key=SecretStr("test-api-key-12345678"),
        session_signing_key=SecretStr("test-signing-key-1234567890"),
        llm_model="openai:gpt-4o",
        llm_api_key=SecretStr("test-llm-key-1234567890"),
        trusted_proxies=trusted_proxies,
    )


def _identify(monkeypatch: pytest.MonkeyPatch, trusted: list[str], client_ip: str) -> str:
    """Resolve the rate-limit identity for a request originating from `client_ip`."""
    monkeypatch.setattr(
        "app.middleware.rate_limit.get_settings",
        lambda: _settings(trusted),
    )

    app = FastAPI()

    @app.get("/test")
    async def route(request: Request) -> dict[str, str]:
        return {"client": get_client_identifier(request)}

    with TestClient(app, client=(client_ip, 12345)) as client:
        response = client.get("/test", headers={"X-Forwarded-For": REAL_CLIENT_IP})

    return response.json()["client"]


class TestCIDRMatching:
    """Trusted proxies expressed as CIDR networks must match."""

    @pytest.mark.parametrize(
        ("network", "proxy_ip"),
        [
            ("10.0.0.0/8", "10.1.2.3"),
            ("10.0.0.0/8", "10.255.255.254"),
            ("192.168.1.0/24", "192.168.1.77"),
            ("172.16.0.0/12", "172.20.5.5"),
            # A Cloudflare range straight out of docs/production_deployment.md
            ("173.245.48.0/20", "173.245.55.1"),
        ],
    )
    def test_forwarded_for_trusted_within_cidr(
        self, monkeypatch: pytest.MonkeyPatch, network: str, proxy_ip: str
    ) -> None:
        """A proxy inside the configured network is trusted."""
        assert _identify(monkeypatch, [network], proxy_ip) == REAL_CLIENT_IP

    @pytest.mark.parametrize(
        ("network", "proxy_ip"),
        [
            ("10.0.0.0/8", "192.0.2.1"),
            ("192.168.1.0/24", "192.168.2.1"),
            ("173.245.48.0/20", "8.8.8.8"),
        ],
    )
    def test_forwarded_for_ignored_outside_cidr(
        self, monkeypatch: pytest.MonkeyPatch, network: str, proxy_ip: str
    ) -> None:
        """A client outside the configured network cannot spoof the header."""
        assert _identify(monkeypatch, [network], proxy_ip) == proxy_ip

    def test_bare_ip_entry_still_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Existing single-address configuration keeps working."""
        assert _identify(monkeypatch, ["10.0.0.1"], "10.0.0.1") == REAL_CLIENT_IP
        assert _identify(monkeypatch, ["10.0.0.1"], "10.0.0.2") == "10.0.0.2"

    def test_empty_list_never_trusts_forwarded_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty allow-list means the header is always ignored."""
        assert _identify(monkeypatch, [], "10.0.0.1") == "10.0.0.1"

    def test_ipv6_network_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """IPv6 proxies are matched the same way."""
        assert _identify(monkeypatch, ["2001:db8::/32"], "2001:db8::1") == REAL_CLIENT_IP


class TestForwardedValueValidation:
    """The forwarded value is validated before becoming a bucket key."""

    def test_malformed_forwarded_value_falls_back_to_direct_ip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trusted proxy relaying garbage must not create arbitrary buckets."""
        monkeypatch.setattr(
            "app.middleware.rate_limit.get_settings",
            lambda: _settings(["10.0.0.0/8"]),
        )

        app = FastAPI()

        @app.get("/test")
        async def route(request: Request) -> dict[str, str]:
            return {"client": get_client_identifier(request)}

        with TestClient(app, client=("10.0.0.1", 12345)) as client:
            response = client.get("/test", headers={"X-Forwarded-For": "not-an-ip-address"})

        assert response.json()["client"] == "10.0.0.1"


class TestTrustedProxiesConfigValidation:
    """Unparseable entries are rejected at startup, not silently ignored."""

    @pytest.mark.parametrize(
        "invalid",
        ["testclient", "not-an-ip", "10.0.0.0/33", "999.999.999.999", "10.0.0.1-10.0.0.9"],
    )
    def test_invalid_entry_rejected(self, invalid: str) -> None:
        """An entry that could never match must fail configuration validation."""
        with pytest.raises(ValidationError, match=r"(?i)not a valid IP address or CIDR"):
            _settings([invalid])

    def test_valid_mixed_entries_accepted(self) -> None:
        """Bare addresses and CIDR networks can be mixed."""
        settings = _settings(["127.0.0.1", "10.0.0.0/8", "2001:db8::/32"])
        assert settings.trusted_proxies == ["127.0.0.1", "10.0.0.0/8", "2001:db8::/32"]
