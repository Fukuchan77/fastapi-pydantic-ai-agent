"""Rate limiting middleware using slowapi."""

from collections.abc import Callable
from collections.abc import Sequence

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.models.errors import ErrorResponse


def get_client_identifier(request: Request) -> str:
    """Get client identifier considering proxy headers with trusted proxy validation.

    Task 20.1: Only trust X-Forwarded-For header when the immediate client
    (request.client.host) is in the trusted_proxies list. This prevents
    header spoofing attacks where untrusted clients set fake X-Forwarded-For values.

    When behind a trusted proxy or load balancer, the X-Forwarded-For header contains
    the real client IP. This function extracts the first IP from the header,
    which is the actual client IP.

    Security:
        - Only trusts X-Forwarded-For when request comes from a trusted proxy
        - Prevents attackers from bypassing rate limiting by spoofing the header
        - Empty trusted_proxies list means X-Forwarded-For is never trusted

    Args:
        request: FastAPI request object

    Returns:
        str: Client identifier (IP address)
    """
    # Get trusted proxy configuration
    settings = get_settings()
    trusted_proxies = settings.trusted_proxies

    # Get the immediate client IP (the actual TCP connection source)
    direct_client_ip = request.client.host if request.client else "unknown"

    # Check for X-Forwarded-For header (set by proxies/load balancers)
    forwarded = request.headers.get("X-Forwarded-For")

    # Task 20.1: Only trust X-Forwarded-For if the immediate client is in trusted_proxies
    if forwarded and direct_client_ip in trusted_proxies:
        # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
        # The first IP is the actual client
        return forwarded.split(",")[0].strip()

    # Fall back to direct connection IP (ignore X-Forwarded-For from untrusted sources)
    return direct_client_ip


def add_rate_limiting(
    app: FastAPI,
    default_limits: Sequence[str] | None = None,
    key_func: Callable[[Request], str] | None = None,
) -> Limiter:
    """Add rate limiting to FastAPI application using slowapi.

    Task 20.2: Creates limiter instance and registers custom exception handler.
    The limiter is stored in app.state for access via dependencies.

    Args:
        app: FastAPI application instance
        default_limits: List of default rate limit strings
            (e.g., ["5/minute", "100/hour"])
        key_func: Function to extract client identifier from request
            (default: get_client_identifier)

    Returns:
        Limiter: Configured slowapi Limiter instance

    Example:
        ```python
        app = FastAPI()
        limiter = add_rate_limiting(app, default_limits=["60/minute"])
        ```
    """
    # Use default key function if not provided
    if key_func is None:
        key_func = get_client_identifier

    # Use default limits if not provided
    if default_limits is None:
        default_limits = ["60/minute"]

    # Create limiter instance
    limiter = Limiter(
        key_func=key_func,
        default_limits=list(default_limits),
        headers_enabled=True,
    )

    # Store limiter in app state for access via dependencies
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


# Task 20.2: FastAPI dependency function for rate limiting
# This is used as Depends() in route handlers to enforce rate limiting
async def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency that enforces rate limiting on the route.

    This dependency uses the limiter stored in app.state to check and enforce
    rate limits. It should be added to protected routes via Depends().

    Args:
        request: FastAPI request object

    Raises:
        RateLimitExceeded: If the rate limit is exceeded

    Example:
        ```python
        @router.post("/protected")
        async def protected_route(_: None = Depends(rate_limit_dependency)):
            return {"status": "ok"}
        ```
    """
    limiter: Limiter = request.app.state.limiter
    # Call the limiter's hit method to check and record this request
    # This will raise RateLimitExceeded if limit is exceeded
    await limiter.hit(limiter.default_limits[0], request)
