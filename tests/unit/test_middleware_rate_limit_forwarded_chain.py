"""Regression tests for which element of `X-Forwarded-For` becomes the rate-limit key.

`get_client_identifier` took `forwarded.split(",")[0]` - the *leftmost* element.
Every proxy `docs/production_deployment.md` documents except Apache appends to
the header rather than replacing it:

- Nginx (L64, L106): `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`
  expands to `$http_x_forwarded_for, $remote_addr`
- AWS ALB (L277) and Cloudflare (L318) both append the observed client address

So under every documented deployment the leftmost element is whatever the client
itself sent. Trusting it let any caller choose its own rate-limit bucket and
rotate through unlimited distinct ones, defeating the global limit and the
stricter `llm_rate_limit` (Req 11.3) that bounds LLM spend.

This was latent until CIDR matching was fixed: before that, a CIDR-configured
deployment never trusted the header at all, so the parsing bug was unreachable
(fail-closed). Restoring trust in the header made it reachable - and fail-open.

The fix walks the chain right-to-left and returns the first element that is not
itself a trusted proxy.
"""

import pytest
from fastapi import FastAPI
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.config import Settings
from app.middleware.rate_limit import get_client_identifier


# The address the trusted infrastructure actually observed - what every case
# below must key on, regardless of what the client claimed.
REAL_CLIENT_IP = "203.0.113.7"
EDGE_PROXY_IP = "10.0.0.5"


def _settings(trusted_proxies: list[str]) -> Settings:
    """Build settings with the given trusted proxy entries."""
    return Settings(
        api_key=SecretStr("test-api-key-12345678"),
        session_signing_key=SecretStr("test-signing-key-1234567890"),
        llm_model="openai:gpt-4o",
        llm_api_key=SecretStr("test-llm-key-1234567890"),
        trusted_proxies=trusted_proxies,
    )


def _identify(
    monkeypatch: pytest.MonkeyPatch,
    trusted: list[str],
    peer_ip: str,
    forwarded: str,
) -> str:
    """Resolve the rate-limit identity for a request arriving with `forwarded`."""
    monkeypatch.setattr(
        "app.middleware.rate_limit.get_settings",
        lambda: _settings(trusted),
    )

    app = FastAPI()

    @app.get("/test")
    async def route(request: Request) -> dict[str, str]:
        return {"client": get_client_identifier(request)}

    with TestClient(app, client=(peer_ip, 12345)) as client:
        response = client.get("/test", headers={"X-Forwarded-For": forwarded})

    return response.json()["client"]


class TestClientSuppliedPrefixIsIgnored:
    """A client-supplied leftmost element must never become the bucket key."""

    @pytest.mark.parametrize("spoofed", ["1.1.1.1", "2.2.2.2", "8.8.8.8", "2001:db8::dead"])
    def test_nginx_appended_chain_keys_on_the_observed_address(
        self, monkeypatch: pytest.MonkeyPatch, spoofed: str
    ) -> None:
        """`$proxy_add_x_forwarded_for` yields `<client claim>, <observed>`."""
        identity = _identify(
            monkeypatch,
            ["10.0.0.0/8"],
            EDGE_PROXY_IP,
            f"{spoofed}, {REAL_CLIENT_IP}",
        )
        assert identity == REAL_CLIENT_IP

    def test_rotating_the_claimed_prefix_does_not_create_new_buckets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One real client must map to exactly one bucket however it rotates the header.

        This is the property the whole fix exists for: a caller that varies the
        leftmost element per request would otherwise get a fresh rate-limit
        budget every time.
        """
        buckets = {
            _identify(
                monkeypatch,
                ["10.0.0.0/8"],
                EDGE_PROXY_IP,
                f"{claim}, {REAL_CLIENT_IP}",
            )
            for claim in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4")
        }
        assert buckets == {REAL_CLIENT_IP}

    def test_multi_hop_chain_skips_every_trusted_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With two trusted hops appended, the last untrusted element still wins."""
        identity = _identify(
            monkeypatch,
            ["10.0.0.0/8", "172.16.0.0/12"],
            EDGE_PROXY_IP,
            f"9.9.9.9, {REAL_CLIENT_IP}, 172.16.4.4, 10.0.0.9",
        )
        assert identity == REAL_CLIENT_IP


class TestSingleElementChainUnchanged:
    """The replacing-proxy case (Apache `RequestHeader set`) keeps working."""

    def test_single_element_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A one-element header from a trusted proxy is the client address."""
        assert _identify(monkeypatch, ["10.0.0.0/8"], EDGE_PROXY_IP, REAL_CLIENT_IP) == (
            REAL_CLIENT_IP
        )

    def test_untrusted_peer_still_ignores_the_header_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No trust in the peer means no trust in any element of the chain."""
        identity = _identify(
            monkeypatch,
            ["10.0.0.0/8"],
            "192.0.2.50",
            f"1.1.1.1, {REAL_CLIENT_IP}",
        )
        assert identity == "192.0.2.50"


class TestDegenerateChains:
    """Chains with nothing identifiable fall back to the peer, never to a claim."""

    def test_all_elements_trusted_falls_back_to_peer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When every hop is trusted infrastructure, no client address is knowable."""
        identity = _identify(
            monkeypatch,
            ["10.0.0.0/8"],
            EDGE_PROXY_IP,
            "10.0.0.7, 10.0.0.8",
        )
        assert identity == EDGE_PROXY_IP

    def test_malformed_rightmost_element_stops_the_walk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed hop makes everything to its left unverifiable.

        Skipping past it would resume trusting client-supplied elements, so the
        walk stops and the peer address is used instead.
        """
        identity = _identify(
            monkeypatch,
            ["10.0.0.0/8"],
            EDGE_PROXY_IP,
            f"1.1.1.1, {REAL_CLIENT_IP}, garbage",
        )
        assert identity == EDGE_PROXY_IP

    def test_wholly_malformed_header_falls_back_to_peer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unchanged behaviour for a header carrying no parseable address at all."""
        assert _identify(monkeypatch, ["10.0.0.0/8"], EDGE_PROXY_IP, "not-an-ip") == EDGE_PROXY_IP

    def test_whitespace_around_elements_is_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proxies vary in spacing; elements are stripped before parsing."""
        identity = _identify(
            monkeypatch,
            ["10.0.0.0/8"],
            EDGE_PROXY_IP,
            f"1.1.1.1 ,  {REAL_CLIENT_IP} ,10.0.0.9",
        )
        assert identity == REAL_CLIENT_IP
