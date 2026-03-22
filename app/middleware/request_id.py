"""Request ID middleware for distributed tracing."""

import uuid
from collections.abc import Awaitable
from collections.abc import Callable
from contextvars import ContextVar

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp


# Context variable to store request ID for the current request
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and track request IDs for distributed tracing.

    This middleware:
    1. Generates a unique request ID for each incoming request
       (or uses X-Request-ID header if provided)
    2. Stores the request ID in a context variable accessible throughout the request lifecycle
    3. Adds the request ID to the response headers (X-Request-ID)

    The request ID is useful for:
    - Correlating logs across different services
    - Debugging distributed systems
    - Tracing requests through multiple components
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process the request and inject request ID.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler

        Returns:
            HTTP response with X-Request-ID header
        """
        # Use existing X-Request-ID header if provided, otherwise generate new UUID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store request ID in context variable for access during request processing
        request_id_var.set(request_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers for client correlation
        response.headers["X-Request-ID"] = request_id

        return response
