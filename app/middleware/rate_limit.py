"""Rate limiting middleware using slowapi."""

from collections.abc import Callable
from collections.abc import Sequence

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from app.models.errors import ErrorResponse


def get_client_identifier(request: Request) -> str:
    """Get client identifier considering proxy headers.

    When behind a proxy or load balancer, the X-Forwarded-For header contains
    the real client IP. This function extracts the first IP from the header,
    which is the actual client IP.

    Args:
        request: FastAPI request object

    Returns:
        str: Client identifier (IP address)
    """
    # Check for X-Forwarded-For header (set by proxies/load balancers)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
        # The first IP is the actual client
        return forwarded.split(",")[0].strip()

    # Fall back to direct connection IP
    return request.client.host if request.client else "unknown"


def add_rate_limiting(
    app: FastAPI,
    default_limits: Sequence[str] | None = None,
    key_func: Callable[[Request], str] | None = None,
) -> Limiter:
    """Add rate limiting to FastAPI application using slowapi.

    This function configures slowapi rate limiting on the FastAPI app with:
    - Configurable default rate limits
    - Automatic rate limit headers in responses
    - Custom error handler that returns structured ErrorResponse
    - Key function for identifying clients (default: IP address)

    Args:
        app: FastAPI application instance
        default_limits: List of default rate limit strings (e.g., ["5/minute", "100/hour"])
        key_func: Function to extract client identifier from request (default: get_remote_address)

    Returns:
        Limiter: Configured slowapi Limiter instance for use as decorator

    Example:
        ```python
        app = FastAPI()
        limiter = add_rate_limiting(app, default_limits=["5/minute"])

        @app.get("/api/endpoint")
        @limiter.limit("10/minute")
        async def my_endpoint():
            return {"status": "ok"}
        ```
    """
    # Use default key function if not provided
    if key_func is None:
        key_func = get_client_identifier

    # Use default limits if not provided
    if default_limits is None:
        default_limits = ["60/minute"]

    # Create limiter instance
    # Note: Convert Sequence to list for slowapi compatibility
    limiter = Limiter(
        key_func=key_func,
        default_limits=list(default_limits),
        headers_enabled=True,  # Add rate limit headers to responses
    )

    # Store limiter in app state for access by routes
    app.state.limiter = limiter

    # Custom exception handler for rate limit exceeded
    async def rate_limit_exceeded_handler(
        request: Request,
        exc: RateLimitExceeded,
    ) -> JSONResponse:
        """Handle rate limit exceeded exception with structured error response.

        Args:
            request: The request that exceeded rate limit
            exc: The rate limit exceeded exception

        Returns:
            JSONResponse: 429 response with ErrorResponse body and rate limit headers
        """
        error_response = ErrorResponse(
            message="Rate limit exceeded. Please try again later.",
            code="RATE_LIMIT_EXCEEDED",
        )

        # Get rate limit headers from exception
        headers: dict[str, str] = {}
        if hasattr(exc, "headers") and exc.headers:
            headers = dict(exc.headers)

        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(),
            headers=headers,
        )

    # Register exception handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    return limiter
