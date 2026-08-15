"""Rate limiting middleware using slowapi."""

from collections.abc import Callable
from collections.abc import Sequence
from functools import cache
from ipaddress import IPv4Network
from ipaddress import IPv6Network
from ipaddress import ip_address
from ipaddress import ip_network

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


@cache
def _parse_trusted_proxies(entries: tuple[str, ...]) -> tuple[IPv4Network | IPv6Network, ...]:
    """Parse trusted proxy entries into network objects.

    Entries are validated at settings load time
    (`SecuritySettingsMixin.validate_trusted_proxies`), so parsing cannot fail
    here. Cached because this runs on the hot path for every request.

    Args:
        entries: Tuple of IP address or CIDR network strings.

    Returns:
        tuple: Parsed network objects for containment checks.
    """
    return tuple(ip_network(entry, strict=False) for entry in entries)


def _is_trusted_proxy(client_ip: str, trusted_proxies: list[str]) -> bool:
    """Check whether `client_ip` falls inside any configured trusted proxy network.

    Accepts both bare addresses ("10.0.0.1") and CIDR networks ("10.0.0.0/8").
    CIDR support is load-bearing: `docs/production_deployment.md` configures
    whole ranges (an ALB's VPC CIDR, Cloudflare's 15 published ranges), and an
    exact string comparison never matches any of them - which would silently
    disable real-client-IP extraction for exactly the deployments the guide
    describes.

    Args:
        client_ip: The immediate client IP address (TCP connection source).
        trusted_proxies: Configured trusted proxy addresses or networks.

    Returns:
        bool: True if `client_ip` is inside any trusted network.
    """
    if not trusted_proxies:
        return False

    try:
        address = ip_address(client_ip)
    except ValueError:
        # Not a parseable address (e.g. the "unknown" placeholder below, or a
        # Unix-socket transport reporting a path) - never trusted.
        return False

    return any(address in network for network in _parse_trusted_proxies(tuple(trusted_proxies)))


def _resolve_settings(request: Request) -> Settings:
    """Resolve the `Settings` this request's application was built with.

    `app.state.settings` is authoritative: `create_app(settings=...)` injects an
    explicit instance, and reading process-global `get_settings()` instead would
    silently apply an environment-derived `trusted_proxies` to an application
    configured with a different one - while `enforce_llm_rate_limit`, in this
    same module, reads the injected instance for the limit itself. One
    rate-limit decision must not be assembled from two different `Settings`.

    The `get_settings()` fallback covers only an application whose lifespan has
    not populated `app.state` (bare `FastAPI()` harnesses in unit tests). It can
    never override an injected value, because it is reached only when there is
    none.

    Args:
        request: The incoming request, whose `app.state` is consulted first.

    Returns:
        Settings: The application's settings.
    """
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def get_client_identifier(request: Request) -> str:
    """Get client identifier considering proxy headers with trusted proxy validation.

    Only trust X-Forwarded-For header when the immediate client
    (request.client.host) falls inside the trusted_proxies allow-list. This
    prevents header spoofing attacks where untrusted clients set fake
    X-Forwarded-For values.

    When behind a trusted proxy or load balancer, `X-Forwarded-For` is a chain:
    `client, proxy1, proxy2`. Every proxy this repository documents *appends* to
    it (Nginx's `$proxy_add_x_forwarded_for`, an ALB, and Cloudflare all do), so
    the leftmost element is whatever the client itself sent - taking it would let
    any caller choose its own rate-limit bucket and rotate through unlimited
    ones. This function therefore walks the chain from the right, skipping hops
    that are themselves trusted proxies, and takes the first address that is not.
    That address is the closest hop the trusted infrastructure actually observed
    and is the last one an untrusted party could not have fabricated.

    Security:
        - Only trusts X-Forwarded-For when request comes from a trusted proxy
        - Accepts CIDR networks, so a documented VPC/CDN range actually matches
        - Walks the chain right-to-left, so a client-supplied leftmost entry
          cannot become the bucket key and rate limiting cannot be bypassed by
          spoofing the header
        - Empty trusted_proxies list means X-Forwarded-For is never trusted
        - Validates each element is an IP address before returning it as a
          bucket key, so a trusted proxy relaying a malformed header cannot
          create unbounded distinct rate-limit buckets

    Args:
        request: FastAPI request object

    Returns:
        str: Client identifier (IP address)
    """
    # Get trusted proxy configuration from the settings this app was built with
    settings = _resolve_settings(request)
    trusted_proxies = settings.trusted_proxies

    # Get the immediate client IP (the actual TCP connection source)
    direct_client_ip = request.client.host if request.client else "unknown"

    # Check for X-Forwarded-For header (set by proxies/load balancers)
    forwarded = request.headers.get("X-Forwarded-For")

    # Only trust X-Forwarded-For if the immediate client is a trusted proxy
    if forwarded and _is_trusted_proxy(direct_client_ip, trusted_proxies):
        for candidate in reversed([element.strip() for element in forwarded.split(",")]):
            try:
                ip_address(candidate)
            except ValueError:
                # A malformed element means the chain can no longer be walked
                # reliably - everything to its left is unverifiable. Stop here
                # rather than skipping past it, and fall back to the direct
                # connection IP below.
                break
            if not _is_trusted_proxy(candidate, trusted_proxies):
                return candidate
        # Every element was a trusted proxy (or the chain was unwalkable):
        # no untrusted client address is identifiable, so key on the peer.
        return direct_client_ip

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
    settings: Settings = _resolve_settings(request)
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
