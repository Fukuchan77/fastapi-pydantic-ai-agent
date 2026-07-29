"""Authentication dependencies for API endpoints."""

import logging
import secrets

from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.config import get_settings
from app.middleware.request_id import request_id_var
from app.models.errors import ErrorResponse
from app.security.principal import Principal
from app.security.principal import derive_principal_id


logger = logging.getLogger(__name__)


# Define APIKeyHeader security scheme with description for OpenAPI documentation
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description=(
        "API key for authentication. Include this key in the X-API-Key header "
        "for all requests to protected endpoints. Contact your administrator to obtain an API key."
    ),
)


async def verify_api_key(
    api_key: str | None = Depends(api_key_header),
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Principal:
    """Verify X-API-Key header matches configured API key.

    This dependency should be applied at the router level to protect endpoints
    while allowing specific routes (like /health) to remain unauthenticated.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        api_key: API key from X-API-Key header (None if not provided)
        settings: Application settings containing the expected API key

    Returns:
        Principal: The authenticated caller, derived from the API key (Req 11.1).

    Raises:
        HTTPException: 401 Unauthorized if key is missing or invalid
    """
    # Use constant-time comparison to prevent timing attacks
    # secrets.compare_digest() requires both arguments to be strings
    # Extract secret value from SecretStr for comparison
    if api_key is None or not secrets.compare_digest(api_key, settings.api_key.get_secret_value()):
        # Log authentication failure for security monitoring
        # Do NOT log the actual API key values (security risk)
        request_id = request_id_var.get()
        if api_key is None:
            logger.warning(
                "Authentication failed: missing API key (request_id: %s)",
                request_id,
            )
        else:
            logger.warning(
                "Authentication failed: invalid API key (request_id: %s)",
                request_id,
            )

        # Add error code for programmatic error handling
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(message="Unauthorized", code="UNAUTHORIZED").model_dump(),
        )

    return Principal(id=derive_principal_id(api_key))
