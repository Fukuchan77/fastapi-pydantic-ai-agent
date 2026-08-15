"""Unit tests for the stricter, per-route LLM rate limit dependency (Req 11.3)."""

import limits
from fastapi import Depends
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from slowapi.wrappers import Limit as SlowApiLimit

from app.middleware.rate_limit import add_rate_limiting
from app.middleware.rate_limit import enforce_llm_rate_limit
from app.middleware.rate_limit import get_client_identifier
from tests.conftest import build_test_settings


def _build_app(llm_rate_limit: str) -> FastAPI:
    app = FastAPI()
    add_rate_limiting(app, default_limits=["1000/minute"])
    app.state.settings = build_test_settings(llm_rate_limit=llm_rate_limit)

    @app.get("/llm-endpoint", dependencies=[Depends(enforce_llm_rate_limit)])
    async def llm_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    return app


def _raise_rate_limit_exceeded_without_recording_state() -> None:
    """Raise `RateLimitExceeded` the way a future site could, forgetting Req 1.4's step.

    Never sets `request.state.view_rate_limit`, unlike `enforce_llm_rate_limit`
    itself - this is the only way to exercise the handler's defensive
    `getattr` guard (Req 1.5) once 3.4 makes every real raise site record it.
    """
    item = limits.parse("1/minute")
    raise RateLimitExceeded(
        SlowApiLimit(item, get_client_identifier, None, False, None, None, None, 1, False)
    )


def _build_app_with_unrecorded_raise() -> FastAPI:
    app = FastAPI()
    add_rate_limiting(app, default_limits=["1000/minute"])

    @app.get(
        "/unrecorded",
        dependencies=[Depends(_raise_rate_limit_exceeded_without_recording_state)],
    )
    async def unrecorded_endpoint(request: Request) -> JSONResponse:
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
    body = response.json()
    assert set(body) == {"message", "code"}
    assert body["code"] == "RATE_LIMIT_EXCEEDED"
    assert "error" not in body
    assert "Retry-After" in response.headers
    assert "X-RateLimit-Limit" in response.headers


def test_enforce_llm_rate_limit_uses_configured_settings_value() -> None:
    """A stricter configured limit (1/minute) blocks the second request."""
    app = _build_app("1/minute")
    client = TestClient(app)

    first = client.get("/llm-endpoint")
    assert first.status_code == 200

    second = client.get("/llm-endpoint")
    assert second.status_code == 429


def test_unrecorded_view_rate_limit_still_yields_429_not_500() -> None:
    """A raise site that never sets `view_rate_limit` still gets a flat 429 (Req 1.5).

    Without the handler's `getattr(request.state, "view_rate_limit", None)`
    guard, this would raise `AttributeError` inside the handler itself and
    surface as an internal 500 instead.
    """
    app = _build_app_with_unrecorded_raise()
    client = TestClient(app)

    response = client.get("/unrecorded")

    assert response.status_code == 429
    body = response.json()
    assert set(body) == {"message", "code"}
    assert body["code"] == "RATE_LIMIT_EXCEEDED"
