"""FastAPI application factory and lifecycle management."""

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import logfire
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

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

# Task 2.7: Minimum cleanup interval to avoid wasting CPU on frequent cleanups
# Even if session_ttl is very short (e.g., 60 seconds in tests), the cleanup
# interval should not be less than this value.
CLEANUP_INTERVAL_MIN: int = 300  # seconds (5 minutes)


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
        # Task 2.6: Initialize settings from environment variables
        app.state.settings = get_settings()

        # Task 16.8: Configure Python logging at startup
        # This must be done early, before any logging occurs
        configure_logging(app.state.settings)
        logger.info("Configured Python logging")

        logger.info("Initialized application settings")

        # Task 2.6: Initialize HTTP client for agent tool usage
        # Task 16.26: Configure timeout to prevent indefinite hangs
        # Task 16.6: Configure connection pooling limits
        app.state.http_client = httpx.AsyncClient(
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
            "max_connections=%d, max_keepalive=%d",
            app.state.settings.http_timeout,
            app.state.settings.http_connect_timeout,
            app.state.settings.http_max_connections,
            app.state.settings.http_max_keepalive_connections,
        )
    except Exception as e:
        logger.error("Failed to initialize app.state.settings or http_client: %s", e, exc_info=True)
        raise

    # Task 8.0: Initialize InMemoryVectorStore
    app.state.vector_store = InMemoryVectorStore()
    logger.info("Initialized vector store")

    # Task 3.11: Initialize InMemorySessionStore with TTL
    app.state.session_store = InMemorySessionStore()
    logger.info(
        "Initialized session store with TTL of %d seconds",
        app.state.session_store.session_ttl,
    )

    # Task 3.11: Start background cleanup task for expired sessions
    async def cleanup_loop() -> None:
        """Background task that periodically cleans up expired sessions.

        Task 16.19: Added comprehensive error handling to prevent cleanup
        task from stopping on transient errors, which would cause memory leaks.
        """
        session_store = app.state.session_store
        # Task 2.7: Ensure cleanup interval has a minimum bound to avoid wasting CPU
        cleanup_interval = max(CLEANUP_INTERVAL_MIN, session_store.session_ttl // 2)
        logger.info("Starting session cleanup task (interval: %d seconds)", cleanup_interval)

        try:
            while True:
                await asyncio.sleep(cleanup_interval)
                try:
                    # Task 3.15: cleanup_expired_sessions is now public
                    removed_count = await session_store.cleanup_expired_sessions()
                    if removed_count > 0:
                        logger.info("Cleaned up %d expired sessions", removed_count)
                except Exception as e:
                    # Task 16.19: Catch all non-CancelledError exceptions
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

    # Task 8.0: Initialize chat agent
    app.state.chat_agent = build_chat_agent()
    logger.info("Initialized chat agent")

    # Task 9.1 & 9.2: Configure Logfire observability
    configure_logfire(app.state.settings)
    logger.info("Configured Logfire observability")

    yield

    # Cleanup happens here after yield
    # Task 3.11: Cancel the cleanup task during shutdown
    if hasattr(app.state, "cleanup_task"):
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            logger.info("Session cleanup task successfully cancelled")

    # Task 2.6: Close HTTP client during shutdown
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
    contact={
        "name": "API Support",
        "url": "https://github.com/yourusername/fastapi-pydantic-ai-agent",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Get settings for middleware configuration
# This is called at module level before lifespan runs
settings = get_settings()

# Task 15.3: Initialize rate limiting (slowapi)
# Default limit: 60 requests per minute per client (by IP address)
# Routes can override this by using @app.state.limiter.limit("custom/limit") decorator
add_rate_limiting(app, default_limits=["60/minute"])
logger.info("Initialized rate limiting (60/minute default)")

# Task 15.4: Add security headers middleware
# Added first so it applies to all responses (executes last in the middleware chain)
app.add_middleware(SecurityHeadersMiddleware)  # type: ignore[arg-type]

# FIX: Add request size limit middleware BEFORE request ID middleware
# Middleware executes in REVERSE order of addition, so this ensures
# RequestIDMiddleware runs first, adding X-Request-ID to all responses including 413
app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)  # type: ignore[arg-type]

# HIGH FIX: Add request ID middleware for distributed tracing
app.add_middleware(RequestIDMiddleware)  # type: ignore[arg-type]

# Task 15.1: Add CORS middleware for cross-origin requests
# Added last so it executes first (handles preflight requests before other middleware)
# CRITICAL FIX: Use cors_origins from settings instead of wildcard
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

# Task 9.2: Instrument FastAPI with Logfire for HTTP tracing
logfire.instrument_fastapi(app)

# Register routers
app.include_router(health_router)

# Task 8.0: Register v1 router
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

    Task 2.5: Logging is performed in a background task to prevent logging
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
        # HIGH FIX: Include request ID for distributed tracing
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
    # Task 16.9: Add error code for programmatic error handling
    error_response = ErrorResponse(
        message="Internal server error occurred",
        code="INTERNAL_ERROR",
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
        background=background_tasks,
    )
