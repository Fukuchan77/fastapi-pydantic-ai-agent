"""FastAPI application factory and lifecycle management."""

import asyncio
import logging
import random
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import logfire
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi.middleware import SlowAPIMiddleware

from app.agents.chat_agent import build_chat_agent
from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.config import get_settings
from app.logging_config import configure_logging
from app.middleware.cors import CORSMiddleware
from app.middleware.rate_limit import add_rate_limiting
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.request_id import request_id_var
from app.middleware.request_size import RequestSizeLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.models.errors import ErrorResponse
from app.observability import configure_logfire
from app.stores.session_store import InMemorySessionStore
from app.stores.vector_store import InMemoryVectorStore


logger = logging.getLogger(__name__)

# Minimum cleanup interval to avoid wasting CPU on frequent cleanups
# Even if session_ttl is very short (e.g., 60 seconds in tests), the cleanup
# interval should not be less than this value.
CLEANUP_INTERVAL_MIN: int = 300  # seconds (5 minutes)


class RetryTransport(httpx.AsyncHTTPTransport):
    """Custom HTTP transport with retry logic and exponential backoff.

    Implements automatic retry for transient failures (network errors,
    5xx server errors) with exponential backoff and jitter. Does NOT retry client
    errors (4xx) as they indicate issues with the request itself.

    Only retries transient 5xx errors {500, 502, 503, 504}.
    Non-transient errors like 501 (Not Implemented) and 505 (HTTP Version Not Supported)
    are permanent configuration issues that will not be resolved by retrying.

    Retry behavior:
    - Network errors (ConnectError, TimeoutException): Retry
    - Transient 5xx errors (500, 502, 503, 504): Retry
    - Non-transient 5xx errors (501, 505, etc.): Do NOT retry
    - 4xx client errors (400-499): Do NOT retry
    - Exponential backoff: delay = base_delay * (2 ** attempt) + random jitter
    - Jitter: random.uniform(0, 1) to prevent thundering herd

    Args:
        max_attempts: Maximum number of retry attempts (from settings)
        base_delay: Base delay in seconds for exponential backoff (from settings)
        **kwargs: Additional arguments passed to AsyncHTTPTransport
    """

    # Define retryable status codes - only transient server errors
    # Use frozenset for immutability (RUF012)
    RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize retry transport with exponential backoff settings.

        Args:
            max_attempts: Maximum number of retry attempts (default: 3)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            **kwargs: Additional arguments for AsyncHTTPTransport
        """
        super().__init__(**kwargs)
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle HTTP request with retry logic.

        Retries transient failures (network errors, transient 5xx) with exponential backoff.
        Does NOT retry client errors (4xx) or non-transient 5xx errors.

        Args:
            request: The HTTP request to execute

        Returns:
            httpx.Response: The HTTP response

        Raises:
            Exception: If all retry attempts are exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                response = await super().handle_async_request(request)

                # Only retry transient 5xx errors {500, 502, 503, 504}
                # Non-transient errors (501, 505, etc.) are permanent and should not be retried
                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt < self.max_attempts - 1
                ):
                    delay = self.base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "HTTP request to %s returned %d (attempt %d/%d), retrying in %.2fs",
                        request.url,
                        response.status_code,
                        attempt + 1,
                        self.max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # 2xx, 3xx, 4xx responses - return immediately (don't retry 4xx)
                return response

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Network errors are transient, retry if attempts remaining
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "HTTP request to %s failed with %s (attempt %d/%d), retrying in %.2fs",
                        request.url,
                        type(e).__name__,
                        attempt + 1,
                        self.max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Last attempt failed, raise the exception
                raise
            except Exception:
                # Non-transient errors (e.g., invalid URL, SSL errors) - raise immediately
                raise

        # All retries exhausted, raise last exception
        if last_exception:
            raise last_exception

        # This should never happen, but satisfy type checker
        raise RuntimeError("Retry logic error: no response or exception")


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

    Args:
        app: FastAPI application instance

    Yields:
        None: Control during application lifetime
    """
    try:
        # Initialize settings from environment variables
        app.state.settings = get_settings()

        # Configure Python logging at startup
        # This must be done early, before any logging occurs
        configure_logging(app.state.settings)
        logger.info("Configured Python logging")

        logger.info("Initialized application settings")

        # Initialize HTTP client for agent tool usage
        # Configure timeout to prevent indefinite hangs
        # Configure connection pooling limits
        # Add retry logic with exponential backoff using custom transport
        retry_transport = RetryTransport(
            max_attempts=app.state.settings.http_retry_max_attempts,
            base_delay=app.state.settings.http_retry_base_delay,
        )
        app.state.http_client = httpx.AsyncClient(
            transport=retry_transport,
            timeout=httpx.Timeout(
                app.state.settings.http_timeout, connect=app.state.settings.http_connect_timeout
            ),
            limits=httpx.Limits(
                max_connections=app.state.settings.http_max_connections,
                max_keepalive_connections=app.state.settings.http_max_keepalive_connections,
            ),
        )
        logger.info(
            "Initialized HTTP client with %ss timeout (%ss connect), "
            "max_connections=%d, max_keepalive=%d, retry_max_attempts=%d, retry_base_delay=%.1fs",
            app.state.settings.http_timeout,
            app.state.settings.http_connect_timeout,
            app.state.settings.http_max_connections,
            app.state.settings.http_max_keepalive_connections,
            app.state.settings.http_retry_max_attempts,
            app.state.settings.http_retry_base_delay,
        )
    except Exception as e:
        logger.error("Failed to initialize app.state.settings or http_client: %s", e, exc_info=True)
        raise

    # Initialize InMemoryVectorStore
    app.state.vector_store = InMemoryVectorStore()
    logger.info("Initialized vector store")

    # Initialize InMemorySessionStore with TTL
    app.state.session_store = InMemorySessionStore()
    logger.info(
        "Initialized session store with TTL of %d seconds",
        app.state.session_store.session_ttl,
    )

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

    # Initialize chat agent
    app.state.chat_agent = build_chat_agent()
    logger.info("Initialized chat agent")

    # Configure Logfire observability
    configure_logfire(app.state.settings)
    logger.info("Configured Logfire observability")

    # Log warning if CORS_ORIGINS contains wildcard "*"
    # Check after logging is configured so warning is properly logged
    if "*" in app.state.settings.cors_origins:
        logger.warning(
            "CORS wildcard '*' detected in CORS_ORIGINS configuration. "
            "This allows requests from ANY origin and may pose a security risk in production. "
            "Consider restricting to specific origins for production deployments."
        )

    yield

    # Cleanup happens here after yield
    # Cancel the cleanup task during shutdown
    if hasattr(app.state, "cleanup_task"):
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            logger.info("Session cleanup task successfully cancelled")

    # Close vector store if it has a close() method (e.g., OllamaEmbeddingVectorStore)
    if hasattr(app.state, "vector_store") and hasattr(app.state.vector_store, "close"):
        await app.state.vector_store.close()
        logger.info("Closed vector store")

    # Close HTTP client during shutdown
    if hasattr(app.state, "http_client"):
        await app.state.http_client.aclose()
        logger.info("Closed HTTP client")


app = FastAPI(
    title="FastAPI Pydantic AI Agent",
    description=(
        "Production-ready agentic AI framework combining FastAPI, "
        "Pydantic AI, and LlamaIndex Workflows. "
        "Features include:\n\n"
        "- **Chat Agent**: Conversational AI with tool-calling capabilities "
        "and session management\n"
        "- **RAG System**: Corrective RAG workflow with TF-IDF vector store "
        "for document retrieval\n"
        "- **Streaming**: Server-Sent Events (SSE) streaming for real-time responses\n"
        "- **Observability**: Integrated Logfire instrumentation for AI-native monitoring\n"
        "- **Security**: API key authentication, CORS, rate limiting, and security headers\n\n"
        "Built with enterprise features: connection pooling, request size limits, "
        "comprehensive error handling, and production-ready configuration management."
    ),
    version="0.1.0",
    lifespan=lifespan,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Get settings for middleware configuration
# This is called at module level before lifespan runs
settings = get_settings()

# Initialize rate limiting (slowapi) with quick workaround
# Quick workaround (Option C): Accept that health endpoints will be rate limited,
# but set a very high limit (1000/minute) that effectively exempts them in practice.
# Trade-off: Health checks get rate limited, but at such a high threshold they won't be affected.
add_rate_limiting(app, default_limits=["1000/minute"])
logger.info("Initialized rate limiting (1000/minute) - applied globally via middleware")

# Add SlowAPIMiddleware to enforce rate limiting on all routes
app.add_middleware(SlowAPIMiddleware)  # type: ignore[arg-type]

# Add security headers middleware
# Added first so it applies to all responses (executes last in the middleware chain)
app.add_middleware(SecurityHeadersMiddleware)  # type: ignore[arg-type]

# Add request size limit middleware BEFORE request ID middleware
# Middleware executes in REVERSE order of addition, so this ensures
# RequestIDMiddleware runs first, adding X-Request-ID to all responses including 413
app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)  # type: ignore[arg-type]

# Add request ID middleware for distributed tracing
app.add_middleware(RequestIDMiddleware)  # type: ignore[arg-type]

# Add CORS middleware for cross-origin requests
# Added last so it executes first (handles preflight requests before other middleware)
# Use cors_origins from settings instead of wildcard
# This prevents CSRF attacks by restricting allowed origins
# Note: cors_origins is validated to always be list[str] by field_validator
app.add_middleware(
    CORSMiddleware,  # type: ignore[arg-type]
    allow_origins=settings.cors_origins
    if isinstance(settings.cors_origins, list)
    else [settings.cors_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Instrument FastAPI with Logfire for HTTP tracing
logfire.instrument_fastapi(app)

# Register routers
app.include_router(health_router)

# Register v1 router
app.include_router(v1_router, prefix="/v1")


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Global exception handler for unhandled exceptions.

    Catches any unhandled exception during request processing and returns
    a structured error response with HTTP 500 status code.

    Security: Returns a generic error message to the client to prevent
    leaking sensitive information (stack traces, database paths, etc.).
    Full exception details are logged internally via background task.

    Logging is performed in a background task to prevent logging
    backend latency from delaying the HTTP response. The traceback is captured
    before creating the background task to ensure it's available when logging runs.

    Args:
        request: The incoming request that caused the exception
        exc: The unhandled exception

    Returns:
        JSONResponse: HTTP 500 response with generic ErrorResponse body
    """
    # Capture exception info immediately while still in exception context
    # This must be done BEFORE creating the background task to preserve traceback
    exc_info = sys.exc_info()
    request_path = request.url.path
    exc_str = str(exc)

    # Define background logging function
    def log_exception() -> None:
        """Log exception details in background to avoid blocking the response."""
        # Include request ID for distributed tracing
        logger.error(
            "Unhandled exception during request to %s: %s",
            request_path,
            exc_str,
            exc_info=exc_info,
            extra={"request_id": request_id_var.get()},
        )

    # Create background tasks and add logging
    background_tasks = BackgroundTasks()
    background_tasks.add_task(log_exception)

    # Return generic error message to client immediately (never expose internal details)
    # Add error code for programmatic error handling
    error_response = ErrorResponse(
        message="Internal server error occurred",
        code="INTERNAL_ERROR",
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
        background=background_tasks,
    )
