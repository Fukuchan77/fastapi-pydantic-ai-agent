"""FastAPI lifespan: startup wiring and shutdown teardown.

Split out of `app/main.py::create_app` (Req 4.2, 4.3) so `settings` and the
test-injected model override are explicit parameters instead of variables the
nested `lifespan` closure captured. `create_app` still owns middleware
registration, routing, and the store/model/observability factories
themselves - this module only owns startup/shutdown *ordering*.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic_ai.models import Model

from app.agents.chat_agent import build_chat_agent
from app.api.health import ReadinessProbeCache
from app.config import Settings
from app.http_client import build_http_client
from app.llm.factory import build_fallback_model
from app.logging_config import configure_logging
from app.observability import configure_logfire
from app.stores.factory import build_session_store
from app.stores.factory import build_vector_store
from app.stores.factory import dry_run_stores


logger = logging.getLogger(__name__)

# Minimum cleanup interval to avoid wasting CPU on frequent cleanups
# Even if session_ttl is very short (e.g., 60 seconds in tests), the cleanup
# interval should not be less than this value.
CLEANUP_INTERVAL_MIN: int = 300  # seconds (5 minutes)


async def _close_quietly(name: str, close: Callable[[], Awaitable[None]]) -> None:
    """Await `close()`, logging (not raising) on failure.

    Used for each independent shutdown step in `_shutdown` so a failure
    closing one resource (e.g. a Redis connection error) can never skip
    closing the ones after it - shutdown should always attempt every step.
    Logged at `error`, not `warning`: a close failure here means a leaked
    connection/pool, which should be visible to alerting.

    Args:
        name: Human-readable resource name, for the log message.
        close: Zero-argument callable returning the awaitable to close it.
    """
    try:
        await close()
        logger.info("Closed %s", name)
    except Exception:
        logger.error("Error closing %s during shutdown", name, exc_info=True)


async def _shutdown(app: FastAPI) -> None:
    """Cancel the cleanup task and close every store/client, isolating failures.

    Extracted from the post-`yield` teardown sequence so the same steps run
    on the normal shutdown path and, once the startup half is wrapped for
    failure cleanup, on the startup-failure path too - a resource opened
    partway through startup is released the same way either way.

    Args:
        app: FastAPI application instance whose `state` holds the resources
            to release. Each attribute is checked with `hasattr` since a
            startup failure may leave any of them unset.
    """
    if hasattr(app.state, "cleanup_task"):
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            logger.info("Session cleanup task successfully cancelled")

    # Each store/client is closed independently via _close_quietly(), so
    # a failure in one (e.g. a Redis connection error) can never skip
    # closing the ones after it.
    if hasattr(app.state, "vector_store"):
        await _close_quietly("vector store", app.state.vector_store.close)

    if hasattr(app.state, "session_store"):
        await _close_quietly("session store", app.state.session_store.close)

    if hasattr(app.state, "http_client"):
        await _close_quietly("HTTP client", app.state.http_client.aclose)


async def _startup(app: FastAPI, settings: Settings, model: Model | str | None) -> None:
    """Build every startup resource and store it on `app.state`.

    Handles, in order: Python logging, the HTTP client, the vector and
    session stores (plus their connectivity dry-run), the background
    session-cleanup task, the LLM model chain, the chat agent, and Logfire
    observability.

    Each resource is assigned to `app.state` as soon as it is built, so a
    failure partway through leaves every already-built resource discoverable
    there for the caller to release (Req 4.1) - the caller (`lifespan`) wraps
    this whole function in a `try`/`except` that calls `_shutdown(app)` on
    any exception, then re-raises unchanged.

    Args:
        app: FastAPI application instance whose `state` receives each
            resource as it is built.
        settings: Application settings.
        model: LLM model for the chat agent. Forwarded to `build_chat_agent()`
            as-is; when `None`, an eager `FallbackModel` chain is built from
            `settings` instead (Req 10.1).
    """
    app.state.settings = settings

    # Configure Python logging at startup
    # This must be done early, before any logging occurs
    configure_logging(settings)
    logger.info("Configured Python logging")
    logger.info("Initialized application settings")

    # Initialize HTTP client for agent tool usage (retrying transport,
    # timeout, and connection-pool limits built from settings).
    app.state.http_client = build_http_client(settings)
    logger.info(
        "Initialized HTTP client with %ss timeout (%ss connect), "
        "max_connections=%d, max_keepalive=%d, "
        "retry_max_attempts=%d, retry_base_delay=%.1fs",
        settings.http_timeout,
        settings.http_connect_timeout,
        settings.http_max_connections,
        settings.http_max_keepalive_connections,
        settings.http_retry_max_attempts,
        settings.http_retry_base_delay,
    )

    # Select and construct the vector/session stores from settings (store
    # factory), then probe connectivity so a misconfigured external store
    # fails startup instead of the first request.
    app.state.vector_store = build_vector_store(settings)
    logger.info(
        "Initialized vector store (backend=%s)",
        settings.vector_store_backend,
    )

    app.state.session_store = build_session_store(settings)
    logger.info(
        "Initialized session store (redis_enabled=%s)",
        settings.redis_session_store_enabled,
    )

    await dry_run_stores(app.state.vector_store, app.state.session_store)
    logger.info("Store connectivity dry-run passed")

    # Start background cleanup task for expired sessions
    async def cleanup_loop() -> None:
        """Background task that periodically cleans up expired sessions.

        Added comprehensive error handling to prevent cleanup
        task from stopping on transient errors, which would cause memory leaks.
        """
        session_store = app.state.session_store
        # Ensure cleanup interval has a minimum bound to avoid wasting CPU
        cleanup_interval = max(CLEANUP_INTERVAL_MIN, session_store.session_ttl // 2)
        logger.info("Starting session cleanup task (interval: %d seconds)", cleanup_interval)

        try:
            while True:
                await asyncio.sleep(cleanup_interval)
                try:
                    # cleanup_expired_sessions is now public
                    removed_count = await session_store.cleanup_expired_sessions()
                    if removed_count > 0:
                        logger.info("Cleaned up %d expired sessions", removed_count)
                except Exception as e:
                    # Catch all non-CancelledError exceptions
                    # Log the error but continue the cleanup loop to prevent memory leaks
                    logger.error(
                        "Error during session cleanup (will retry): %s",
                        e,
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled during shutdown")
            raise

    # Create and store the cleanup task
    app.state.cleanup_task = asyncio.create_task(cleanup_loop())

    # Build the FallbackModel chain eagerly so a misconfigured provider
    # chain fails startup instead of the first request (Req 10.1). A
    # test-injected `model` override bypasses this entirely, matching the
    # existing store/session-store test-isolation contract.
    resolved_model = model if model is not None else build_fallback_model(settings)

    # Publish the resolved model - the test-injected override when one was
    # supplied, otherwise the eagerly built chain - so chat and RAG share
    # exactly one model instance (Req 3.1, 3.2). Placed between the resolve
    # step and the chat-agent build so a later chat-agent build failure still
    # leaves it discoverable to shutdown.
    app.state.llm_model = resolved_model

    # Initialize chat agent, forwarding the resolved model
    app.state.chat_agent = build_chat_agent(model=resolved_model, settings=settings)
    logger.info("Initialized chat agent")

    # Short-TTL cache in front of /health/ready's dependency probes. That route
    # is unauthenticated and its LLM check is a real provider request, so the
    # cache is what stops inbound request volume from becoming outbound provider
    # volume (see ReadinessProbeCache). Created per application, never shared.
    app.state.readiness_cache = ReadinessProbeCache(settings.readiness_probe_cache_ttl)
    logger.info(
        "Initialized readiness probe cache (ttl=%ds)",
        settings.readiness_probe_cache_ttl,
    )

    # Configure Logfire observability
    configure_logfire(settings)
    logger.info("Configured Logfire observability")

    # Log warning if CORS_ORIGINS contains wildcard "*"
    # Check after logging is configured so warning is properly logged
    if "*" in settings.cors_origins:
        logger.warning(
            "CORS wildcard '*' detected in CORS_ORIGINS configuration. "
            "This allows requests from ANY origin and may pose a security risk in "
            "production. Consider restricting to specific origins for production "
            "deployments."
        )

    # Warn when the deployment and application halves could silently disagree
    # about the client-facing scheme (Req 11.4, ADR-5). SecurityHeadersMiddleware
    # (L4.5) emits Strict-Transport-Security only when `request.url.scheme ==
    # "https"`, which is resolved by the ASGI server, not read from a forwarded
    # header here. Outside development that resolution requires the server to
    # already trust the TLS-terminating proxy's forwarded scheme
    # (--forwarded-allow-ips/FORWARDED_ALLOW_IPS, L4.6); `trust_proxy_headers`
    # is the operator's confirmation that it does. Read from `settings`, never
    # `os.environ` directly, to stay Principle-4 compliant.
    if settings.app_env != "development" and not settings.trust_proxy_headers:
        logger.warning(
            "trust_proxy_headers is False while app_env is %r. Strict-Transport-"
            "Security will never be emitted behind a TLS-terminating proxy unless "
            "the ASGI server is configured to trust it "
            "(--forwarded-allow-ips/FORWARDED_ALLOW_IPS) and trust_proxy_headers "
            "is set to True to confirm it. See docs/production_deployment.md.",
            settings.app_env,
        )


def build_lifespan(
    settings: Settings,
    model: Model | str | None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the FastAPI lifespan context manager for `settings` and `model`.

    Args:
        settings: Application settings, resolved by `create_app` before
            building the app.
        model: LLM model for the chat agent. Forwarded to `build_chat_agent()`
            as-is; when `None`, an eager `FallbackModel` chain is built from
            `settings` during startup instead (Req 10.1).

    Returns:
        An async context manager function suitable for `FastAPI(lifespan=...)`.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Application lifespan manager.

        Handles startup and shutdown of application resources including:
        - Vector store initialization
        - Session store initialization
        - HTTP client setup
        - Agent construction
        - Observability configuration
        - Background cleanup task for expired sessions

        A failure anywhere in `_startup` releases every resource `_startup`
        had already built - through the same `_shutdown` helper the normal
        path uses - before the original exception propagates unchanged
        (Req 4.1, 4.4, 4.5).

        Args:
            app: FastAPI application instance

        Yields:
            None: Control during application lifetime
        """
        try:
            await _startup(app, settings, model)
        except Exception:
            logger.error(
                "Startup failed; releasing resources created so far before propagating",
                exc_info=True,
            )
            await _shutdown(app)
            raise

        yield

        # Cleanup happens here after yield.
        await _shutdown(app)

    return lifespan
