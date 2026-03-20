"""FastAPI application factory and lifecycle management."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.models.errors import ErrorResponse


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Handles startup and shutdown of application resources including:
    - Vector store initialization
    - Session store initialization
    - HTTP client setup
    - Agent construction
    - Observability configuration

    Args:
        app: FastAPI application instance

    Yields:
        None: Control during application lifetime
    """
    # TODO: Task 3.1 - Initialize InMemoryVectorStore when implemented
    # app.state.vector_store = InMemoryVectorStore()

    # TODO: Task 3.2 - Initialize InMemorySessionStore when implemented
    # app.state.session_store = InMemorySessionStore()

    # TODO: Task 6.2 - Initialize chat agent when build_chat_agent() is implemented
    # app.state.chat_agent = build_chat_agent()

    # TODO: Task 9.1 - Configure Logfire when configure_logfire() is implemented
    # configure_logfire(get_settings())

    # TODO: Initialize httpx.AsyncClient
    # async with httpx.AsyncClient() as client:
    #     app.state.http_client = client
    #     yield

    # Placeholder yield for now
    yield

    # Cleanup happens here after yield


app = FastAPI(
    title="fastapi-pydantic-ai-agent",
    description="Agentic AI framework with FastAPI, Pydantic AI, and LlamaIndex Workflows",
    version="0.1.0",
    lifespan=lifespan,
)


# Register routers
app.include_router(health_router)

# TODO: Task 8.5 - Register v1 router when implemented
# app.include_router(v1_router, prefix="/v1")


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
