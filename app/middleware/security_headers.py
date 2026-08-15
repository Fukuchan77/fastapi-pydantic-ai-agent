"""Security headers middleware for adding security-related HTTP headers."""

from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from app.config import Settings


def _join_directives(directives: list[str]) -> str:
    """Render CSP directives with no trailing/duplicated whitespace, each terminated (Req 11.5).

    Args:
        directives: Directive strings (e.g. "default-src 'self'"), no semicolons.

    Returns:
        A single policy string: directives joined by "; ", ending in ";".
    """
    return "; ".join(directives) + ";"


# 'unsafe-inline' and the CDN/font hosts below are scoped to the interactive
# documentation UI only (Req 11.6) - see _is_documentation_path.
_STRICT_CSP = _join_directives(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
    ]
)
_DOCS_CSP = _join_directives(
    [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "img-src 'self' data: https://fastapi.tiangolo.com",
        "font-src https://fonts.gstatic.com",
    ]
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses.

    This middleware adds common security headers to protect against:
    - Clickjacking (X-Frame-Options)
    - MIME sniffing (X-Content-Type-Options)
    - Man-in-the-middle attacks (Strict-Transport-Security)
    - Various injection attacks (Content-Security-Policy)
    - Information leakage (Referrer-Policy)
    - Unwanted feature access (Permissions-Policy)

    `Strict-Transport-Security` and `Content-Security-Policy` are computed
    per request (Req 11.3/11.4/11.6) rather than fixed at construction time;
    the remaining headers are static. Headers can be customized via the
    custom_headers parameter, which always takes precedence.
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: Settings,
        custom_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize security headers middleware.

        Args:
            app: ASGI application.
            settings: Application settings; reads `hsts_max_age` and
                `hsts_include_subdomains` (Req 11.4). Scheme trust itself is
                resolved at the ASGI-server layer, not read from settings or
                forwarded headers here (ADR-5).
            custom_headers: Optional dict of custom headers to add or override
                any default or computed header, including HSTS and CSP.
        """
        super().__init__(app)
        self._settings = settings
        self._custom_headers = dict(custom_headers) if custom_headers else {}

        # Headers that never vary by request.
        self._static_headers: dict[str, str] = {
            # Prevent MIME sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Control referrer information
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Restrict access to sensitive features
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

    def _hsts_value(self) -> str:
        """Build the Strict-Transport-Security value from settings (Req 11.4).

        Returns:
            The header value, e.g. "max-age=31536000; includeSubDomains".
        """
        value = f"max-age={self._settings.hsts_max_age}"
        if self._settings.hsts_include_subdomains:
            value += "; includeSubDomains"
        return value

    def _is_documentation_path(self, request: Request) -> bool:
        """Check whether the request path is part of the interactive docs UI (Req 11.6).

        Reads the docs paths off the live `FastAPI` app instance
        (`docs_url`/`redoc_url`/`swagger_ui_oauth2_redirect_url`) rather than
        hard-coding them, and deliberately includes the OAuth2 redirect
        sub-path - it is a real page that needs the relaxed policy, and the
        single reason an exact `path == "/docs"` check would be insufficient.

        Args:
            request: Incoming HTTP request.

        Returns:
            True if the request path is a documentation route.
        """
        app = request.app
        docs_paths = {app.docs_url, app.redoc_url, app.swagger_ui_oauth2_redirect_url}
        return request.url.path in docs_paths

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

        for header_name, header_value in self._static_headers.items():
            response.headers[header_name] = header_value

        # HSTS asserts "this host is always HTTPS" - never true of a
        # plaintext response, so it is omitted entirely rather than sent
        # unconditionally (Req 11.3/11.4). Scheme comes from the ASGI scope
        # as resolved by the server layer, never a forwarded header (ADR-5).
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = self._hsts_value()

        response.headers["Content-Security-Policy"] = (
            _DOCS_CSP if self._is_documentation_path(request) else _STRICT_CSP
        )

        for header_name, header_value in self._custom_headers.items():
            response.headers[header_name] = header_value

        return response
