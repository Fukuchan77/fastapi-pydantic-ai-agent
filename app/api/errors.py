"""Unified HTTP error envelope handlers (Req 8).

Converges every error response - regardless of status code or which layer
raised it - onto one flat `ErrorResponse` body, so a client needs one parser
rather than branching on body shape. Does not own the global `Exception`
handler (stays in `app/main.py`, since it also owns background-task logging)
or the 413/429 responses (emitted directly by their own middleware/handler,
never reaching `HTTPException`).
"""

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.models.errors import ErrorResponse


# Default `code` per status, used when a raise site's `detail` carries none
# of its own (a bare string or an unrecognised object). A `code` already
# present on a mapping `detail` (e.g. the existing 401/403 raise sites) is
# read back unchanged and takes precedence over this table. This is the
# vocabulary task 2.2 documents in `app/models/errors.py`.
_DEFAULT_CODE_BY_STATUS: dict[int, str] = {
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMIT_EXCEEDED",
    500: "INTERNAL_ERROR",
    502: "UPSTREAM_GROUNDING_FAILED",
    504: "WORKFLOW_TIMEOUT",
}


def _render_detail(detail: Any, status_code: int) -> ErrorResponse:  # noqa: ANN401
    """Re-render an `HTTPException.detail` of any shape into a flat `ErrorResponse`.

    Args:
        detail: The raised exception's `detail` - a string, a mapping (e.g.
            `ErrorResponse.model_dump()`), or any other object.
        status_code: The exception's HTTP status code, used to look up a
            default `code` when `detail` is not a mapping carrying its own.

    Returns:
        ErrorResponse: The flat body to serialize.
    """
    if isinstance(detail, Mapping):
        message = str(detail.get("message", detail))
        code = detail.get("code") or _DEFAULT_CODE_BY_STATUS.get(status_code)
        return ErrorResponse(message=message, code=code)

    message = detail if isinstance(detail, str) else str(detail)
    return ErrorResponse(message=message, code=_DEFAULT_CODE_BY_STATUS.get(status_code))


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render any raised `HTTPException` as the single flat error body.

    Covers every status this project raises `HTTPException` for today -
    401/403 (mapping `detail`) and 502/504 (string `detail`) - the 404/405
    starlette's *router* raises with no route involved, and any future raise
    site regardless of what shape it passes as `detail` (Req 8.1, 8.2, 8.4).

    `exc` is typed `Exception`, not `HTTPException`, to match
    `Starlette.add_exception_handler`'s actual signature - it dispatches by
    the registered exception class below, but the handler type itself is not
    generic over it (same idiom as `rate_limit.py`'s
    `rate_limit_exceeded_handler`). Registration guarantees only
    `StarletteHTTPException` instances ever reach this handler.

    Args:
        request: The incoming request. Unused; required by FastAPI's
            exception-handler signature.
        exc: The raised `HTTPException`, typed `Exception` per the note above.

    Returns:
        JSONResponse: A flat `ErrorResponse` body at `exc.status_code`,
            forwarding any headers the exception carries (the 405 `Allow`
            header depends on this).
    """
    del request
    # Narrowed to the starlette base class, not `fastapi.HTTPException`: the
    # router raises the base directly, so narrowing to the subclass would
    # make this handler unusable for exactly the 404/405 it must now cover.
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registration-guaranteed, narrows for the type checker
    error_response = _render_detail(exc.detail, exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
        headers=exc.headers,
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a request-validation failure as a generic flat error body.

    Deliberately drops FastAPI's default per-field `detail` list so the body
    never echoes request content back to the client (Req 8.5). `exc` is typed
    `Exception` for the same registration-signature reason documented on
    `http_exception_handler`.

    Args:
        request: The incoming request. Unused; required by FastAPI's
            exception-handler signature.
        exc: The validation error. Unused beyond its registered type - no
            per-field detail is surfaced.

    Returns:
        JSONResponse: A flat `ErrorResponse` body at 422 with a generic
            message and the `VALIDATION_ERROR` code.
    """
    del request, exc
    error_response = ErrorResponse(message="Request validation failed", code="VALIDATION_ERROR")
    return JSONResponse(status_code=422, content=error_response.model_dump())


def register_error_handlers(app: FastAPI) -> None:
    """Register the flat-envelope handlers on `app` (single entry point).

    Called once from `create_app()` (composition root).

    Registers against `starlette.exceptions.HTTPException`, the superclass of
    `fastapi.HTTPException`, deliberately. Starlette resolves a handler by
    walking the raised class's MRO *upward*, so a registration on the FastAPI
    subclass is never matched by the base class its router raises for 404 and
    405 - those would fall through to FastAPI's default handler and emit the
    legacy `{"detail": ...}` body (Req 8.1, 8.4). Registering on the base
    covers both, since every `fastapi.HTTPException` is also one of these.

    Args:
        app: The FastAPI application to register handlers on.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
