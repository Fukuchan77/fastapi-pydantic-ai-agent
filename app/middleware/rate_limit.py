"""Rate limiting middleware using slowapi."""

import time
from collections.abc import Callable
from collections.abc import Sequence

import limits
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.wrappers import Limit as SlowApiLimit

from app.config import Settings
from app.config import get_settings
from app.models.errors import ErrorResponse


def get_client_identifier(request: Request) -> str:
    """Get client identifier considering proxy headers with trusted proxy validation.

    Only trust X-Forwarded-For header when the immediate client
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

    # Only trust X-Forwarded-For if the immediate client is in trusted_proxies
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
    storage_uri: str | None = None,
) -> Limiter:
    """Add rate limiting to FastAPI application using slowapi.

    Creates limiter instance and registers custom exception handler.
    The limiter is stored in app.state for access via dependencies.

    Args:
        app: FastAPI application instance
        default_limits: List of default rate limit strings
            (e.g., ["5/minute", "100/hour"])
        key_func: Function to extract client identifier from request
            (default: get_client_identifier)
        storage_uri: Storage backend URI (e.g. a Redis URL) so limits are
            shared across processes (Req 11.4). `None` uses in-memory
            storage, suitable for single-instance/development deployments.

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

    # Create limiter instance. in_memory_fallback_enabled=True means a
    # configured storage_uri that becomes unreachable degrades to in-memory
    # storage (with a warning) instead of failing every request.
    limiter = Limiter(
        key_func=key_func,
        default_limits=list(default_limits),
        headers_enabled=True,
        storage_uri=storage_uri,
        in_memory_fallback_enabled=True,
    )

    # Store limiter in app state for access via dependencies
    app.state.limiter = limiter

    # Custom exception handler for rate limit exceeded
    async def rate_limit_exceeded_handler(
        request: Request,
        exc: RateLimitExceeded,
    ) -> JSONResponse:
        """Handle rate limit exceeded exception with structured error response.

        Adds Retry-After header to improve client UX and reduce retry storms.

        Args:
            request: The request that exceeded rate limit
            exc: The rate limit exceeded exception

        Returns:
            JSONResponse: 429 response with ErrorResponse body, rate limit headers,
                         and Retry-After header indicating seconds until reset
        """
        error_response = ErrorResponse(
            message="Rate limit exceeded. Please try again later.",
            code="RATE_LIMIT_EXCEEDED",
        )

        # Get rate limit headers from exception
        headers: dict[str, str] = {}
        if hasattr(exc, "headers") and exc.headers:
            headers = dict(exc.headers)

        # Add Retry-After header (RFC 6585, RFC 7231)
        # Calculate seconds until rate limit resets based on X-RateLimit-Reset header
        if "X-RateLimit-Reset" in headers:
            try:
                reset_timestamp = int(headers["X-RateLimit-Reset"])
                current_timestamp = int(time.time())
                retry_after_seconds = max(1, reset_timestamp - current_timestamp)
                headers["Retry-After"] = str(retry_after_seconds)
            except (ValueError, TypeError):
                # If parsing fails, default to 60 seconds (reasonable for most rate limits)
                headers["Retry-After"] = "60"
        else:
            # Fallback if X-RateLimit-Reset is not available
            headers["Retry-After"] = "60"

        return JSONResponse(
            status_code=429,
            content=error_response.model_dump(),
            headers=headers,
        )

    # Register exception handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)  # type: ignore[arg-type]

    return limiter


async def enforce_llm_rate_limit(request: Request) -> None:
    """Apply the stricter, configurable per-route limit to LLM-invoking endpoints (Req 11.3).

    Used as a route `dependencies=[Depends(enforce_llm_rate_limit)]` entry on
    `chat`/`stream_agent` (app/api/v1/agent.py) and `query` (app/api/v1/rag.py).

    Deliberately reuses `request.app.state.limiter` - the same per-app
    `Limiter`/storage `add_rate_limiting()` already wires to Redis when
    configured (Req 11.4) - rather than a second, independently-configured
    limiter, so this stricter check and the global default check share one
    consistent, correctly per-app-scoped storage backend.

    Args:
        request: FastAPI request object.

    Raises:
        RateLimitExceeded: If `settings.llm_rate_limit` is exceeded; handled
            by the same exception handler `add_rate_limiting()` registers.
    """
    limiter: Limiter = request.app.state.limiter
    settings: Settings = request.app.state.settings
    item = limits.parse(settings.llm_rate_limit)
    identifier = get_client_identifier(request)

    if not limiter.limiter.hit(item, identifier):
        raise RateLimitExceeded(
            SlowApiLimit(item, get_client_identifier, None, False, None, None, None, 1, False)
        )
