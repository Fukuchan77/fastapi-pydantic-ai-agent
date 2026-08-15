"""Authentication dependencies for API endpoints."""

import logging
import secrets

from fastapi import Depends
from fastapi import HTTPException
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.deps.settings import get_request_settings
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


def _unauthorized() -> HTTPException:
    """Build the single 401 response shared by every rejection reason (Req 5.5).

    Missing, empty, non-ASCII, and wrong-but-ASCII keys all raise this same
    exception so the response shape gives a caller no signal about which
    check failed.
    """
    return HTTPException(
        status_code=401,
        detail=ErrorResponse(message="Unauthorized", code="UNAUTHORIZED").model_dump(),
    )


async def verify_api_key(
    api_key: str | None = Depends(api_key_header),
    settings: Settings = Depends(get_request_settings),  # noqa: B008
) -> Principal:
    """Verify X-API-Key header matches configured API key.

    This dependency should be applied at the router level to protect endpoints
    while allowing specific routes (like /health) to remain unauthenticated.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        api_key: API key from X-API-Key header (None if not provided)
        settings: Application settings containing the expected API key.
            Resolved from `app.state.settings` (the instance
            `create_app(settings=...)` was built with), not process-global
            `get_settings()` - see `app.deps.settings`.

    Returns:
        Principal: The authenticated caller, derived from the API key (Req 11.1).

    Raises:
        HTTPException: 401 Unauthorized for any missing, non-ASCII, or
            incorrect key (Req 5.1, 5.3, 5.5).
    """
    # Do NOT log the actual API key values (security risk)
    request_id = request_id_var.get()

    if api_key is None:
        logger.warning(
            "Authentication failed: missing API key (request_id: %s)",
            request_id,
        )
        raise _unauthorized()

    # secrets.compare_digest() raises TypeError on a non-ASCII str (it only
    # accepts ASCII str or bytes), which would otherwise turn a malformed,
    # non-ASCII key into an unhandled 500 instead of the 401 this dependency
    # promises for any byte sequence a client can present in the header.
    if not api_key.isascii():
        logger.warning(
            "Authentication failed: invalid API key (request_id: %s)",
            request_id,
        )
        raise _unauthorized()

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key.get_secret_value()):
        logger.warning(
            "Authentication failed: invalid API key (request_id: %s)",
            request_id,
        )
        raise _unauthorized()

    return Principal(id=derive_principal_id(api_key))
