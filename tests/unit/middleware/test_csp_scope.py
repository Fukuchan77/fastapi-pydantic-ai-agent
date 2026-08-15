"""Unit tests for Content-Security-Policy path scoping (Req 11.5, 11.6, 11.8).

Documentation routes need `'unsafe-inline'` and the Swagger/ReDoc CDN hosts;
every other path -- including the schema document and all `/v1/*` API
routes -- must stay strict. Built against the real `create_app()` output
rather than a bare `FastAPI()` app, so the docs paths under test
(`app.docs_url`/`app.redoc_url`/`app.swagger_ui_oauth2_redirect_url`) are the
ones the live application actually serves.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import build_test_settings


@pytest.fixture
def client() -> TestClient:
    """Build the real app (no lifespan) so docs/openapi/v1 routes all exist."""
    app = create_app(settings=build_test_settings())
    return TestClient(app)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
def test_documentation_paths_get_relaxed_csp(client: TestClient, path: str) -> None:
    """Docs paths -- including the OAuth2 redirect sub-path -- allow inline scripts and the CDN."""
    response = client.get(path)

    csp = response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" in csp
    assert "https://cdn.jsdelivr.net" in csp
    assert "font-src https://fonts.gstatic.com" in csp


@pytest.mark.parametrize(
    "path",
    [
        "/openapi.json",
        "/v1/agent/chat",
        "/v1/rag/query",
        "/v1/does-not-exist",
    ],
)
def test_schema_and_api_paths_get_strict_csp(client: TestClient, path: str) -> None:
    """The schema document and every `/v1/*` route -- including unknown ones -- stay strict."""
    response = client.get(path)

    csp = response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in csp
    assert "cdn.jsdelivr.net" not in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "frame-ancestors 'none'" in csp


@pytest.mark.parametrize(
    "path",
    ["/docs2", "/redocs", "/documentation", "/v1/docs", "/docsx"],
)
def test_paths_resembling_documentation_stay_strict(client: TestClient, path: str) -> None:
    """Paths that merely resemble a docs route must not match it (exact-set, not a prefix).

    Guards `_is_documentation_path`'s exact-set check against regressing into a
    prefix or substring match, which would leak the relaxed policy to routes
    that only happen to start with "/docs" or "/redoc" (Req 11.6/11.8).
    """
    response = client.get(path)

    csp = response.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in csp
    assert "cdn.jsdelivr.net" not in csp


def test_docs_and_api_csp_both_terminate_every_directive(client: TestClient) -> None:
    """Both the relaxed and strict policies contain no trailing/duplicated whitespace (Req 11.5)."""
    relaxed_csp = client.get("/docs").headers["Content-Security-Policy"]
    strict_csp = client.get("/openapi.json").headers["Content-Security-Policy"]

    for csp in (relaxed_csp, strict_csp):
        assert csp == csp.strip()
        assert "  " not in csp
        assert csp.endswith(";")
