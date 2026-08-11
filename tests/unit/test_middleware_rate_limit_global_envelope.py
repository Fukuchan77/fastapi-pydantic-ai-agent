"""Regression test for the global rate limit's 429 envelope (Req 1.1, 1.2, 1.3, 1.6).

Every other rate-limit unit test decorates its route with `@limiter.limit(...)`,
which slowapi's `_should_exempt` treats as "the decorator will handle it" and
skips in `SlowAPIMiddleware` - so none of them exercise the middleware's own
check. This is the path that shipped broken: `SlowAPIMiddleware` reaches the
registered handler through `sync_check_limits`, which uses
`inspect.iscoroutinefunction` to detect an `async def` handler and silently
swaps it for slowapi's own default handler (`{"error": ...}`), breaking this
project's flat `{message, code}` envelope contract.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi.middleware import SlowAPIMiddleware

from app.middleware.rate_limit import add_rate_limiting


def _client() -> TestClient:
    """Build a client whose only route is limited exclusively by the middleware.

    No route carries a `@limiter.limit` decorator, so `_should_exempt` does
    not exempt it and the request is checked by `SlowAPIMiddleware` itself.

    Returns:
        TestClient: A second request within the window yields 429.
    """
    app = FastAPI()
    add_rate_limiting(app, default_limits=["1/minute"])
    app.add_middleware(SlowAPIMiddleware)  # type: ignore[arg-type]

    @app.get("/undecorated")
    async def _undecorated() -> JSONResponse:
        return JSONResponse(content={"ok": True})

    return TestClient(app)


def test_global_rate_limit_429_is_flat_with_no_error_key() -> None:
    """The global limit's 429 body holds exactly `message` and `code` (Req 1.1, 1.2)."""
    client = _client()
    client.get("/undecorated")
    response = client.get("/undecorated")

    assert response.status_code == 429
    body = response.json()
    assert set(body) == {"message", "code"}
    assert body["code"] == "RATE_LIMIT_EXCEEDED"
    assert "error" not in body


def test_global_rate_limit_429_carries_rate_limit_headers() -> None:
    """The global limit's 429 carries `X-RateLimit-Limit` and `Retry-After` (Req 1.3)."""
    client = _client()
    client.get("/undecorated")
    response = client.get("/undecorated")

    assert response.status_code == 429
    assert "X-RateLimit-Limit" in response.headers
    assert "Retry-After" in response.headers
