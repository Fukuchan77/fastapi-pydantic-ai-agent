"""Tests for the unified flat error envelope (Req 8.1-8.5, 8.7).

Scoped to task 2.1's boundary (`app/api/errors.py`, `app/main.py`): the two
new handlers and their single registration entry point. Task 2.2 adds one
test for the 413 `code` it gives the size-limit middleware. Task 2.3 adds one
test for the session-authorization rejection path's flat shape. Task 2.4
widens the module to every reachable status - 401, 403, 413, 422, 429, 500,
502, 504 - and adds one aggregate test asserting none of them carries the
legacy nested `{"detail": ...}` shape (Req 8.7).

Validation remediation adds 404 and 405, which are raised by starlette's
*router* rather than by any route in this codebase. 8.7 does not enumerate
them, but 8.1 ("regardless of status code") and 8.4 ("any route or status
code") both reach them, and they were flat-envelope holes: the handler was
registered against `fastapi.HTTPException` while the router raises its
`starlette.exceptions.HTTPException` superclass, which starlette's
MRO-upward handler lookup never matches to a subclass registration.
"""

from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.testclient import TestClient
from httpx import Response

from app.api.errors import register_error_handlers
from app.main import create_app
from app.middleware.rate_limit import add_rate_limiting
from app.middleware.request_size import RequestSizeLimitMiddleware
from app.models.errors import ErrorResponse
from app.security.principal import Principal
from app.services.session_service import authorize_session
from tests.conftest import build_test_settings


def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app with only the error handlers registered.

    Returns:
        FastAPI: An app exposing routes that raise each `detail` shape this
            feature's handler must re-render, a route that fails request
            validation, and a route that exercises the real session-ownership
            rejection path (Req 8.2).
    """
    app = FastAPI()
    register_error_handlers(app)
    settings = build_test_settings()

    @app.get("/string-detail")
    def _string_detail() -> None:
        raise HTTPException(status_code=504, detail="Workflow timed out")

    @app.get("/mapping-detail")
    def _mapping_detail() -> None:
        raise HTTPException(
            status_code=401,
            detail={"message": "Unauthorized", "code": "UNAUTHORIZED"},
        )

    @app.get("/other-detail")
    def _other_detail() -> None:
        raise HTTPException(status_code=502, detail={"unexpected": "shape"})

    @app.post("/validated")
    def _validated(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    @app.get("/forbidden-session")
    async def _forbidden_session() -> None:
        attacker = Principal(id="attacker0000000")
        await authorize_session(attacker, "not-a-signed-session-id", settings)

    return app


def _client() -> TestClient:
    """Build a `TestClient` over the minimal test app.

    Returns:
        TestClient: A client for the app built by `_build_test_app()`.
    """
    return TestClient(_build_test_app(), raise_server_exceptions=False)


def _size_limited_client(max_size: int = 10) -> TestClient:
    """Build a client whose one route is guarded by the request-size middleware.

    Args:
        max_size: The middleware's configured byte ceiling.

    Returns:
        TestClient: Posting a body larger than `max_size` yields 413.
    """
    app = FastAPI()
    app.add_middleware(RequestSizeLimitMiddleware, max_size=max_size)

    @app.post("/echo")
    def _echo() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def _rate_limited_client() -> TestClient:
    """Build a client whose one route allows a single request per window.

    Returns:
        TestClient: A second call to `/limited` yields 429.
    """
    app = FastAPI()
    limiter = add_rate_limiting(app, default_limits=["1/minute"])

    @app.get("/limited")
    @limiter.limit("1/minute")
    async def _limited(request: Request) -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def _full_app_client() -> TestClient:
    """Build the real composition-root app, plus one route that raises unhandled.

    Only the global `Exception` handler (Req 8) - registered in
    `create_app()` itself, not in `app/api/errors.py` - can be exercised this
    way, since it is not part of `register_error_handlers()`'s public surface.

    Returns:
        TestClient: `GET /boom` yields 500 via that handler.
    """
    app = create_app(settings=build_test_settings())

    @app.get("/boom")
    def _boom() -> None:
        raise RuntimeError("Simulated internal error")

    return TestClient(app, raise_server_exceptions=False)


def test_string_detail_renders_flat_body_with_no_nested_detail_key() -> None:
    """A string `detail` (e.g. today's 502/504 raise sites) becomes a flat body."""
    response = _client().get("/string-detail")

    assert response.status_code == 504
    body = response.json()
    assert "detail" not in body
    assert body["message"] == "Workflow timed out"
    assert body["code"] == "WORKFLOW_TIMEOUT"


def test_mapping_detail_renders_flat_body_with_no_nested_detail_key() -> None:
    """A mapping `detail` (e.g. today's 401/403 raise sites) is flattened, not nested."""
    response = _client().get("/mapping-detail")

    assert response.status_code == 401
    body = response.json()
    assert "detail" not in body
    assert body["message"] == "Unauthorized"
    assert body["code"] == "UNAUTHORIZED"


def test_other_detail_shape_still_renders_a_flat_body() -> None:
    """A `detail` that is neither a string nor a message/code mapping still flattens."""
    response = _client().get("/other-detail")

    assert response.status_code == 502
    body = response.json()
    assert "detail" not in body
    assert "message" in body
    assert body["code"] == "UPSTREAM_GROUNDING_FAILED"


def test_validation_error_returns_generic_message_with_no_field_echo() -> None:
    """A request-validation failure returns a generic message, not per-field detail."""
    response = _client().post("/validated", content=b"not json")

    assert response.status_code == 422
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "VALIDATION_ERROR"
    assert "not json" not in body["message"]
    assert set(body) == {"message", "code"}


def test_session_forbidden_renders_flat_body_with_no_nested_detail_key() -> None:
    """The session-authorization rejection path (Req 8.2) is flat, not nested.

    Exercises `app.services.session_service.authorize_session`'s real
    malformed-id rejection path (not a synthetic mapping like the tests
    above), so the assertion pins the actual raise site, not just the
    handler's generic mapping-flattening behaviour.
    """
    response = _client().get("/forbidden-session")

    assert response.status_code == 403
    body = response.json()
    assert "detail" not in body
    assert body["message"] == "Session does not belong to this principal"
    assert body["code"] == "FORBIDDEN"


def test_request_too_large_response_carries_a_code() -> None:
    """The 413 body gains the `code` it previously omitted (8.3).

    Emitted directly by the size-limit middleware, never reaching
    `HTTPException`.
    """
    response = _size_limited_client().post(
        "/echo", content=b"x" * 20, headers={"content-length": "20"}
    )

    assert response.status_code == 413
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "REQUEST_TOO_LARGE"


def test_rate_limit_exceeded_renders_flat_body_with_no_nested_detail_key() -> None:
    """A 429 (rate limit exceeded, `app/middleware/rate_limit.py`) is flat, not nested."""
    client = _rate_limited_client()
    client.get("/limited")  # consumes the one allowed request in the window
    response = client.get("/limited")

    assert response.status_code == 429
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "RATE_LIMIT_EXCEEDED"
    ErrorResponse.model_validate(body)


def test_unhandled_exception_renders_flat_body_with_no_nested_detail_key() -> None:
    """A 500 (unhandled exception, `create_app()`'s global handler) is flat, not nested."""
    response = _full_app_client().get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "INTERNAL_ERROR"
    ErrorResponse.model_validate(body)


def test_router_raised_not_found_renders_flat_body_with_no_nested_detail_key() -> None:
    """A 404 raised by starlette's router - not by any route - is flat, not nested.

    No route in this codebase raises 404; starlette's `Router.not_found`
    raises `starlette.exceptions.HTTPException` directly, so this pins the
    superclass registration rather than the mapping/string rendering the
    tests above cover (Req 8.1, 8.4).
    """
    response = _client().get("/no-such-route")

    assert response.status_code == 404
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "NOT_FOUND"
    ErrorResponse.model_validate(body)


def test_real_composition_root_also_flattens_router_raised_not_found() -> None:
    """The 404 flattening holds through the real middleware stack, not just a bare app.

    `_build_test_app()` registers only these handlers; `create_app()` stacks
    security headers, CORS, rate limiting and `TrustedHostMiddleware` around
    them. This pins that none of those re-wraps or bypasses the envelope on
    the unauthenticated 404 path.
    """
    response = _full_app_client().get("/v1/no-such-route")

    assert response.status_code == 404
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "NOT_FOUND"


def test_router_raised_method_not_allowed_renders_flat_body_and_keeps_allow_header() -> None:
    """A 405 raised by starlette's router is flat, and its `Allow` header survives.

    `Route.handle` raises with a populated `headers` mapping; the handler
    forwards `exc.headers`, so flattening the body must not drop the
    method-negotiation header a client needs.
    """
    response = _client().post("/string-detail")  # a GET-only route

    assert response.status_code == 405
    body = response.json()
    assert "detail" not in body
    assert body["code"] == "METHOD_NOT_ALLOWED"
    assert "allow" in response.headers
    ErrorResponse.model_validate(body)


def test_every_reachable_status_validates_against_the_flat_model_with_no_legacy_nesting() -> None:
    """8.7: every reachable status validates against `ErrorResponse`, never nested.

    Re-drives 8.7's eight statuses - 401, 403, 413, 422, 429, 500, 502, 504 -
    plus the two router-raised statuses 8.7 omits but 8.1/8.4 still reach
    (404, 405), one status each, and asserts in one place that none of them
    carries the legacy nested `{"detail": ...}` shape. The per-status tests
    above pin each status's own message/code; this test pins the
    cross-cutting invariant: no route, at any status, ever nests.
    """
    minimal_client = _client()
    rate_limited_client = _rate_limited_client()
    rate_limited_client.get("/limited")

    responses: dict[int, Response] = {
        401: minimal_client.get("/mapping-detail"),
        403: minimal_client.get("/forbidden-session"),
        404: minimal_client.get("/no-such-route"),
        405: minimal_client.post("/string-detail"),
        413: _size_limited_client().post(
            "/echo", content=b"x" * 20, headers={"content-length": "20"}
        ),
        422: minimal_client.post("/validated", content=b"not json"),
        429: rate_limited_client.get("/limited"),
        500: _full_app_client().get("/boom"),
        502: minimal_client.get("/other-detail"),
        504: minimal_client.get("/string-detail"),
    }

    for expected_status, response in responses.items():
        assert response.status_code == expected_status
        body = response.json()
        assert "detail" not in body, f"status {expected_status} leaked a nested 'detail' key"
        ErrorResponse.model_validate(body)
