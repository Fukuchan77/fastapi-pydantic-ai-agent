"""Security headers middleware for adding security-related HTTP headers."""

from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses.

    This middleware adds common security headers to protect against:
    - Clickjacking (X-Frame-Options)
    - MIME sniffing (X-Content-Type-Options)
    - Man-in-the-middle attacks (Strict-Transport-Security)
    - Various injection attacks (Content-Security-Policy)
    - Information leakage (Referrer-Policy)
    - Unwanted feature access (Permissions-Policy)

    Headers can be customized via the custom_headers parameter.
    """

    def __init__(
        self,
        app: ASGIApp,
        custom_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize security headers middleware.

        Args:
            app: ASGI application
            custom_headers: Optional dict of custom headers to add or override defaults
        """
        super().__init__(app)

        # Default security headers
        self.default_headers: dict[str, str] = {
            # Prevent MIME sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Force HTTPS (31536000 seconds = 1 year)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            # Content Security Policy - only allow resources from same origin
            "Content-Security-Policy": "default-src 'self'",
            # Control referrer information
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Restrict access to sensitive features
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

        # Merge custom headers (custom headers override defaults)
        if custom_headers:
            self.default_headers.update(custom_headers)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request and add security headers to response.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler

        Returns:
            HTTP response with security headers added
        """
        # Process request
        response = await call_next(request)

        # Add security headers to response
        for header_name, header_value in self.default_headers.items():
            response.headers[header_name] = header_value

        return response
