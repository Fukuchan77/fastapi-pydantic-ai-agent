"""Unit tests for Vary header preservation in CORSMiddleware.

Covers Requirement 11.7: when a response already carries a Vary header
(set by an upstream handler), the CORS middleware must append "Origin" to
the existing value rather than replacing it, and must not duplicate
"Origin" when it is already present.
"""

from fastapi import FastAPI
from fastapi import Response
from fastapi.testclient import TestClient

from app.middleware.cors import CORSMiddleware


def _build_client(upstream_vary: str) -> TestClient:
    """Build a test app whose endpoint sets Vary before CORS runs."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://example.com"],
        allow_credentials=False,
    )

    @app.get("/test")
    async def test_endpoint(response: Response) -> dict[str, str]:
        response.headers["Vary"] = upstream_vary
        return {"status": "ok"}

    return TestClient(app)


def test_vary_header_survives_and_gains_origin() -> None:
    """An existing Vary value is preserved, with Origin appended to it."""
    client = _build_client("Accept-Encoding")

    response = client.get("/test", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    vary_values = [v.strip() for v in response.headers.get("Vary", "").split(",")]
    assert "Accept-Encoding" in vary_values
    assert "Origin" in vary_values


def test_vary_origin_not_duplicated_when_already_present() -> None:
    """Origin already listed in Vary is not appended a second time."""
    client = _build_client("Origin")

    response = client.get("/test", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    vary_values = [v.strip() for v in response.headers.get("Vary", "").split(",")]
    assert vary_values.count("Origin") == 1


def test_vary_origin_not_duplicated_when_present_with_different_case() -> None:
    """A differently-cased "origin" token still counts as present (RFC 9110 §12.5.5)."""
    client = _build_client("Accept-Encoding, origin")

    response = client.get("/test", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    vary_values = [v.strip().casefold() for v in response.headers.get("Vary", "").split(",")]
    assert vary_values.count("origin") == 1
