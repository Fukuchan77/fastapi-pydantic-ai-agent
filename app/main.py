"""FastAPI application factory and lifecycle management."""

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from app.agents.chat_agent import build_chat_agent
from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.config import get_settings
from app.models.errors import ErrorResponse
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
        logger.info("Initialized application settings")

        # Task 2.6: Initialize HTTP client for agent tool usage
        app.state.http_client = httpx.AsyncClient()
        logger.info("Initialized HTTP client")
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
        """Background task that periodically cleans up expired sessions."""
        session_store = app.state.session_store
        # Task 2.7: Ensure cleanup interval has a minimum bound to avoid wasting CPU
        cleanup_interval = max(CLEANUP_INTERVAL_MIN, session_store.session_ttl // 2)
        logger.info("Starting session cleanup task (interval: %d seconds)", cleanup_interval)

        try:
            while True:
                await asyncio.sleep(cleanup_interval)
                # Task 3.15: cleanup_expired_sessions is now public
                removed_count = await session_store.cleanup_expired_sessions()
                if removed_count > 0:
                    logger.info("Cleaned up %d expired sessions", removed_count)
        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled during shutdown")
            raise

    # Create and store the cleanup task
    app.state.cleanup_task = asyncio.create_task(cleanup_loop())

    # Task 8.0: Initialize chat agent
    app.state.chat_agent = build_chat_agent()
    logger.info("Initialized chat agent")

    # TODO: Task 9.1 - Configure Logfire when configure_logfire() is implemented
    # configure_logfire(get_settings())

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
    title="fastapi-pydantic-ai-agent",
    description="Agentic AI framework with FastAPI, Pydantic AI, and LlamaIndex Workflows",
    version="0.1.0",
    lifespan=lifespan,
)


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
        logger.error(
            "Unhandled exception during request to %s: %s",
            request_path,
            exc_str,
            exc_info=exc_info,
        )

    # Create background tasks and add logging
    background_tasks = BackgroundTasks()
    background_tasks.add_task(log_exception)

    # Return generic error message to client immediately (never expose internal details)
    error_response = ErrorResponse(message="Internal server error occurred")
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
        background=background_tasks,
    )
