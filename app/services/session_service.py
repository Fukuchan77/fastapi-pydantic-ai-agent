"""Server-issued, principal-bound chat session ids (Req 11.1/11.2).

Ownership is embedded in the id itself via an HMAC signature
(`{principal.id}.{token}.{signature}`), so verifying it requires no
additional store lookup or ownership schema.
"""

import hashlib
import hmac
import secrets

from fastapi import HTTPException

from app.config import Settings
from app.models.errors import ErrorResponse
from app.security.principal import Principal


_TOKEN_BYTES = 16
_SIGNATURE_LENGTH = 16
_SESSION_ID_PARTS = 3


def _sign(principal_id: str, token: str, settings: Settings) -> str:
    """Compute the HMAC signature binding `principal_id` and `token` together.

    Args:
        principal_id: The principal id segment of the session id.
        token: The random token segment of the session id.
        settings: Application settings (for `session_signing_key`).

    Returns:
        str: A truncated hex-encoded HMAC-SHA256 signature.
    """
    key = settings.session_signing_key.get_secret_value().encode()
    message = f"{principal_id}.{token}".encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()[:_SIGNATURE_LENGTH]


def _forbidden() -> HTTPException:
    """Build the 403 raised for any session_id ownership/signature failure.

    Returns:
        HTTPException: A 403 with a generic detail message (no id/key leakage).
    """
    return HTTPException(
        status_code=403,
        detail=ErrorResponse(
            message="Session does not belong to this principal",
            code="FORBIDDEN",
        ).model_dump(),
    )


async def start_session(principal: Principal, settings: Settings) -> str:
    """Mint a new signed session_id bound to `principal` (Req 11.1).

    Args:
        principal: The authenticated caller starting a new conversation.
        settings: Application settings (for `session_signing_key`).

    Returns:
        str: The signed `{principal.id}.{token}.{signature}` session id.
    """
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    signature = _sign(principal.id, token, settings)
    return f"{principal.id}.{token}.{signature}"


async def authorize_session(principal: Principal, session_id: str, settings: Settings) -> None:
    """Verify `session_id` is well-formed, signed, and owned by `principal` (Req 11.2).

    Args:
        principal: The authenticated caller presenting `session_id`.
        session_id: The session id to verify ownership of.
        settings: Application settings (for `session_signing_key`).

    Raises:
        HTTPException: 403 if `session_id` is malformed, has an invalid
            signature, or belongs to a different principal.
    """
    parts = session_id.split(".")
    if len(parts) != _SESSION_ID_PARTS:
        raise _forbidden()

    owner_id, token, signature = parts
    expected_signature = _sign(owner_id, token, settings)
    if not secrets.compare_digest(signature, expected_signature):
        raise _forbidden()
    if not secrets.compare_digest(owner_id, principal.id):
        raise _forbidden()
