"""Unit tests for the stricter, per-route LLM rate limit dependency (Req 11.3)."""

from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.rate_limit import add_rate_limiting
from app.middleware.rate_limit import enforce_llm_rate_limit
from tests.conftest import build_test_settings


def _build_app(llm_rate_limit: str) -> FastAPI:
    app = FastAPI()
    add_rate_limiting(app, default_limits=["1000/minute"])
    app.state.settings = build_test_settings(llm_rate_limit=llm_rate_limit)

    @app.get("/llm-endpoint", dependencies=[Depends(enforce_llm_rate_limit)])
    async def llm_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    return app


def test_enforce_llm_rate_limit_allows_requests_within_budget() -> None:
    """Requests within the configured llm_rate_limit succeed."""
    app = _build_app("3/minute")
    client = TestClient(app)

    for _ in range(3):
        response = client.get("/llm-endpoint")
        assert response.status_code == 200


def test_enforce_llm_rate_limit_blocks_requests_exceeding_budget() -> None:
    """A request past the configured llm_rate_limit is rejected with 429."""
    app = _build_app("3/minute")
    client = TestClient(app)

    for _ in range(3):
        response = client.get("/llm-endpoint")
        assert response.status_code == 200

    response = client.get("/llm-endpoint")
    assert response.status_code == 429
    assert response.json()["code"] == "RATE_LIMIT_EXCEEDED"


def test_enforce_llm_rate_limit_uses_configured_settings_value() -> None:
    """A stricter configured limit (1/minute) blocks the second request."""
    app = _build_app("1/minute")
    client = TestClient(app)

    first = client.get("/llm-endpoint")
    assert first.status_code == 200

    second = client.get("/llm-endpoint")
    assert second.status_code == 429
