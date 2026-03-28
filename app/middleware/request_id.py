"""Request ID middleware for distributed tracing."""

import re
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

# Regex pattern for valid request ID format (alphanumeric, hyphens, underscores, 1-64 chars)
# Prevents CRLF injection (CWE-113) by rejecting any characters that could be used for
# HTTP header injection attacks (newlines, carriage returns, etc.)
REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


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
        # Use existing X-Request-ID header if valid, otherwise generate new UUID
        # Validation prevents HTTP header injection attacks (CWE-113)
        client_request_id = request.headers.get("X-Request-ID", "")
        if client_request_id and REQUEST_ID_PATTERN.match(client_request_id):
            request_id = client_request_id
        else:
            request_id = str(uuid.uuid4())

        # Store request ID in context variable for access during request processing
        request_id_var.set(request_id)

        # Process request
        response = await call_next(request)

        # Add request ID to response headers for client correlation
        response.headers["X-Request-ID"] = request_id

        return response
