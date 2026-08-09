"""FastAPI application factory and lifecycle management."""

import logging
import sys

import logfire
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic_ai.models import Model
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.errors import register_error_handlers
from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.config import Settings
from app.config import get_settings
from app.http_client import RetryTransport
from app.lifespan import build_lifespan
from app.middleware.cors import CORSMiddleware
from app.middleware.rate_limit import add_rate_limiting
from app.middleware.request_id import RequestIDMiddleware
from app.middleware.request_id import request_id_var
from app.middleware.request_size import RequestSizeLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.models.errors import ErrorResponse


logger = logging.getLogger(__name__)

# `RetryTransport` moved to `app/http_client.py` (Req 4.3); re-exported here so
# `from app.main import RetryTransport` keeps working for existing callers.
__all__ = ["RetryTransport"]


def create_app(
    settings: Settings | None = None,
    model: Model | str | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application (composition root).

    Constructs the FastAPI app, registers middleware in the existing
    load-bearing order, wires the lifespan manager, and registers routers
    and the global exception handler. Does not call `get_settings()` unless
    `settings` is omitted, so tests can inject a `Settings` instance and a
    test `Model` without relying on environment variables.

    Args:
        settings: Application settings. Resolved via `get_settings()` when
            omitted (used by the production `uvicorn app.main:app` entrypoint).
        model: LLM model for the chat agent. Forwarded to `build_chat_agent()`
            as-is; when `None`, an eager `FallbackModel` chain is built from
            `settings` during lifespan startup instead (Req 10.1).

    Returns:
        Configured FastAPI application instance.
    """
    resolved_settings = settings or get_settings()

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
        lifespan=build_lifespan(resolved_settings, model),
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Initialize rate limiting (slowapi) with quick workaround
    # Quick workaround (Option C): Accept that health endpoints will be rate limited,
    # but set a very high limit (1000/minute) that effectively exempts them in practice.
    # Trade-off: Health checks get rate limited, but at such a high threshold they won't
    # be affected.
    add_rate_limiting(
        app,
        default_limits=["1000/minute"],
        storage_uri=resolved_settings.redis_url,
    )
    logger.info(
        "Initialized rate limiting (1000/minute default, storage=%s)",
        "redis" if resolved_settings.redis_url else "memory",
    )

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
        allow_origins=resolved_settings.cors_origins
        if isinstance(resolved_settings.cors_origins, list)
        else [resolved_settings.cors_origins],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Add host validation LAST so it executes FIRST, ahead of even CORS.
    #
    # Starlette's router rebuilds redirect targets from the request's Host header
    # and `redirect_slashes` defaults to True, so without this middleware a
    # request to any path with a trailing slash (including unauthenticated
    # `/health/`) returns a 307 whose Location points at a caller-supplied host.
    # Rejecting an untrusted Host before any other middleware runs keeps that
    # reflection - and every other Host-derived URL reconstruction - unreachable.
    # `www_redirect` is off: its only job is bouncing `example.com` to an
    # allow-listed `www.example.com`, and it does so by rebuilding the URL from
    # the request scope. This is an API with no www hostname convention, so a
    # rejected host gets a flat 400 and no Host-derived Location is ever emitted.
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=resolved_settings.allowed_hosts
        if isinstance(resolved_settings.allowed_hosts, list)
        else [resolved_settings.allowed_hosts],
        www_redirect=False,
    )

    # Instrument FastAPI with Logfire for HTTP tracing
    logfire.instrument_fastapi(app)

    # Register the flat error-envelope handlers (Req 8): every HTTPException
    # and request-validation failure renders as one flat ErrorResponse body.
    register_error_handlers(app)

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

    return app


# Module-level app instance for `uvicorn app.main:app`.
app = create_app()
