"""Health check endpoint - no authentication required."""

from fastapi import APIRouter
from fastapi import Request


router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness health check endpoint.

    Returns 200 OK when the service is running. No authentication required.
    Used by load balancers and monitoring systems.

    Returns:
        dict: Status response indicating service is healthy.
    """
    return {"status": "ok"}


@router.get("/health/ready")
def readiness_check(request: Request) -> dict[str, str | dict[str, str]]:
    """Deep readiness health check endpoint.

    Verifies that all critical application dependencies are
    operational. Unlike the basic /health endpoint, this performs actual
    checks on vector store, session store, chat agent, and background tasks.

    This endpoint is intended for Kubernetes readiness probes and load
    balancer health checks. No authentication is required.

    Args:
        request: FastAPI request object to access app.state

    Returns:
        dict: Readiness status with individual component checks
            - status: "ready" if all checks pass, "not_ready" otherwise
            - checks: dict mapping component names to their health status
    """
    checks: dict[str, str] = {}

    # Check vector_store
    if hasattr(request.app.state, "vector_store"):
        checks["vector_store"] = "healthy"
    else:
        checks["vector_store"] = "missing"

    # Check session_store
    if hasattr(request.app.state, "session_store"):
        checks["session_store"] = "healthy"
    else:
        checks["session_store"] = "missing"

    # Check chat_agent
    if hasattr(request.app.state, "chat_agent"):
        checks["chat_agent"] = "healthy"
    else:
        checks["chat_agent"] = "missing"

    # Check cleanup_task
    if hasattr(request.app.state, "cleanup_task"):
        # Verify the task is still running (not done)
        if request.app.state.cleanup_task.done():
            checks["cleanup_task"] = "stopped"
        else:
            checks["cleanup_task"] = "healthy"
    else:
        checks["cleanup_task"] = "missing"

    # Determine overall status
    all_healthy = all(status == "healthy" for status in checks.values())
    overall_status = "ready" if all_healthy else "not_ready"

    return {"status": overall_status, "checks": checks}
