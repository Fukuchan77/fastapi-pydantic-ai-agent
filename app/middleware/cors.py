"""CORS (Cross-Origin Resource Sharing) middleware."""

from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp


class CORSMiddleware(BaseHTTPMiddleware):
    """CORS middleware for handling cross-origin requests.

    This middleware handles CORS preflight (OPTIONS) requests and adds
    appropriate CORS headers to responses based on the origin whitelist.

    Features:
    - Origin whitelist validation
    - Wildcard origin support (*)
    - Preflight request handling
    - Configurable allowed methods and headers
    - Credentials support
    """

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list[str] | None = None,
        allow_credentials: bool = False,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
        max_age: int = 600,
    ) -> None:
        """Initialize CORS middleware.

        Args:
            app: ASGI application
            allow_origins: List of allowed origins (use ["*"] for all)
            allow_credentials: Whether to allow credentials
            allow_methods: List of allowed HTTP methods
            allow_headers: List of allowed headers
            max_age: Max age for preflight cache in seconds (default: 10 minutes)
        """
        super().__init__(app)
        self.allow_origins = allow_origins or []
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]
        self.max_age = max_age

        # Check for wildcard origin
        self.allow_all_origins = "*" in self.allow_origins

        # Credentials and wildcard are incompatible per CORS spec
        if self.allow_all_origins and self.allow_credentials:
            raise ValueError(
                "allow_credentials=True is incompatible with allow_origins=['*']. "
                "Use specific origins when credentials are enabled."
            )

    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if the origin is in the allowed list.

        Args:
            origin: Origin header value

        Returns:
            bool: True if origin is allowed
        """
        if self.allow_all_origins:
            return True
        return origin in self.allow_origins

    def _add_cors_headers(self, response: Response, origin: str) -> Response:
        """Add CORS headers to response.

        Args:
            response: Response object
            origin: Origin header value

        Returns:
            Response: Response with CORS headers added
        """
        if self.allow_all_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"
        else:
            response.headers["Access-Control-Allow-Origin"] = origin
            # Task 20.4: Add Vary: Origin header when returning specific origin
            # to prevent proxy caches from returning wrong CORS headers to different clients
            response.headers["Vary"] = "Origin"

        if self.allow_credentials and not self.allow_all_origins:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request and add CORS headers.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler

        Returns:
            HTTP response with CORS headers if applicable
        """
        origin = request.headers.get("Origin")

        # No Origin header - not a CORS request
        if not origin:
            return await call_next(request)

        # Check if origin is allowed
        if not self._is_allowed_origin(origin):
            # Origin not allowed - process request normally without CORS headers
            return await call_next(request)

        # Handle preflight OPTIONS request
        if request.method == "OPTIONS":
            # Check if this is a CORS preflight request
            request_method = request.headers.get("Access-Control-Request-Method")
            if request_method:
                # This is a preflight request - return CORS headers
                response = Response(status_code=200)
                response = self._add_cors_headers(response, origin)
                response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
                response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
                response.headers["Access-Control-Max-Age"] = str(self.max_age)
                return response

        # Handle actual request - add CORS headers to response
        response = await call_next(request)
        return self._add_cors_headers(response, origin)
