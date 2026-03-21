"""Authentication dependencies for API endpoints."""

from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.config import get_settings
from app.models.errors import ErrorResponse


# Define APIKeyHeader security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Depends(api_key_header),
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> None:
    """Verify X-API-Key header matches configured API key.

    This dependency should be applied at the router level to protect endpoints
    while allowing specific routes (like /health) to remain unauthenticated.

    Args:
        api_key: API key from X-API-Key header (None if not provided)
        settings: Application settings containing the expected API key

    Returns:
        None: Dependency succeeds silently when authentication passes

    Raises:
        HTTPException: 401 Unauthorized if key is missing or invalid
    """
    if api_key is None or api_key != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(message="Unauthorized").model_dump(),
        )
