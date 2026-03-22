"""Request size limit middleware to prevent DoS attacks."""

from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from app.models.errors import ErrorResponse


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce maximum request size limit.

    Prevents denial-of-service attacks via oversized request bodies
    by rejecting requests that exceed the configured size limit.

    Default limit: 10MB
    """

    def __init__(self, app: ASGIApp, max_size: int = 10 * 1024 * 1024) -> None:
        """Initialize the middleware with a maximum request size.

        Args:
            app: ASGI application
            max_size: Maximum request size in bytes (default: 10MB)
        """
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and check content-length header.

        IMPORTANT: This middleware only checks the Content-Length header.
        Requests using Transfer-Encoding: chunked or omitting Content-Length
        will bypass this check. For production deployments, configure your
        reverse proxy (Nginx, ALB, etc.) to enforce actual body size limits:

        - Nginx: client_max_body_size directive
        - AWS ALB: Maximum content length (10MB default)
        - Cloudflare: Maximum upload size in dashboard

        This middleware provides an additional layer of validation and
        friendly error messages for clients that include Content-Length.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler

        Returns:
            HTTP response (413 if too large, otherwise normal response)
        """
        # Check Content-Length header if present
        # NOTE: This does not protect against chunked transfer encoding
        # or requests without Content-Length header
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_size:
                error_response = ErrorResponse(
                    message=f"Request entity too large (max {self.max_size // (1024 * 1024)}MB)"
                )
                return JSONResponse(
                    status_code=413,
                    content=error_response.model_dump(),
                )

        # Process request normally
        return await call_next(request)
