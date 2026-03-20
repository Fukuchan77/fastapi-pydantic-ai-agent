"""Health check endpoint - no authentication required."""

from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    This endpoint can be used by load balancers and monitoring systems to
    verify that the service is running and healthy. No authentication is
    required to access this endpoint.

    Returns:
        dict: Status response indicating service is healthy
    """
    return {"status": "ok"}
