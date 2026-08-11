"""Rate limiting middleware using slowapi."""

from collections.abc import Callable
from collections.abc import Sequence

import limits
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from limits import RateLimitItem
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

    # Custom exception handler for rate limit exceeded.
    #
    # `exc` is typed `Exception`, not `RateLimitExceeded`, to match
    # `Starlette.add_exception_handler`'s actual signature - it dispatches by
    # the registered exception class below, but the handler type itself is
    # not generic over it.
    #
    # Deliberately synchronous (`def`, not `async def`): `SlowAPIMiddleware`
    # reaches this handler through `sync_check_limits`, which uses
    # `inspect.iscoroutinefunction` to detect an `async def` handler and
    # silently swaps it for slowapi's own default handler (which returns
    # `{"error": ...}`) instead of calling this one - that swap is what let
    # the global rate limit's 429 escape this project's flat envelope.
    # Starlette runs a synchronous handler via `run_in_threadpool`, so the
    # `enforce_llm_rate_limit` dependency's raise site is unaffected. If this
    # ever needs to become `async def` again, `SlowAPIMiddleware` must be
    # replaced too, or the same regression recurs silently.
    def rate_limit_exceeded_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle rate limit exceeded exception with structured error response.

        Args:
            request: The request that exceeded rate limit
            exc: The rate limit exceeded exception (typed `Exception`; see
                the handler's own note on `Starlette.add_exception_handler`)

        Returns:
            JSONResponse: 429 response with the flat `ErrorResponse` body.
                Header construction (`X-RateLimit-*` and a delay-seconds
                `Retry-After`) is delegated to `Limiter._inject_headers`,
                which knows the actual rate-limit window; this handler no
                longer computes any header itself.
        """
        error_response = ErrorResponse(
            message="Rate limit exceeded. Please try again later.",
            code="RATE_LIMIT_EXCEEDED",
        )
        response = JSONResponse(status_code=429, content=error_response.model_dump())

        # `view_rate_limit` is set by slowapi itself for both the middleware
        # and the `@limiter.limit`-decorated path. `enforce_llm_rate_limit`
        # sets it explicitly too, since it checks the limit directly rather
        # than through slowapi's own request-limit machinery. The `getattr`
        # default guards any future raise site that forgets to set it -
        # `_inject_headers` itself no-ops on a `None` current_limit, so this
        # keeps such a case a 429 rather than an `AttributeError` -> 500.
        view_rate_limit: tuple[RateLimitItem, list[str]] | None = getattr(
            request.state, "view_rate_limit", None
        )
        # `_inject_headers`'s own type hint omits the `None` it explicitly
        # handles at runtime, and it returns the same `JSONResponse` object
        # it was given (typed as the wider `Response` base class).
        return limiter._inject_headers(response, view_rate_limit)  # type: ignore

    # Register exception handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

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
        # Record the window this check hit, the same way slowapi's own
        # `_check_request_limit` does for the middleware and decorator
        # paths - this call checks the limit directly instead, so nothing
        # else sets it. Without this, `rate_limit_exceeded_handler` finds no
        # `view_rate_limit` and this 429 carries no `Retry-After`/
        # `X-RateLimit-*` headers at all (Req 1.4, 1.10).
        request.state.view_rate_limit = (item, [identifier])
        raise RateLimitExceeded(
            SlowApiLimit(item, get_client_identifier, None, False, None, None, None, 1, False)
        )
