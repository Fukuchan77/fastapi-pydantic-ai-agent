"""Authenticated principal derivation (Req 11.1).

`Principal.id` is derived from the API key that authenticated the request,
not stored or looked up, so introducing it requires no new store schema.
"""

import hashlib

from pydantic import BaseModel


_PRINCIPAL_ID_LENGTH = 16


class Principal(BaseModel):
    """The authenticated identity a request acts as.

    Attributes:
        id: Stable identifier derived from the API key. No secret is stored.
    """

    id: str


def derive_principal_id(api_key: str) -> str:
    """Derive a stable, non-secret principal id from an authenticated API key.

    Args:
        api_key: The raw API key value that authenticated the request.

    Returns:
        str: A 16-character hex digest derived from the key via SHA-256.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:_PRINCIPAL_ID_LENGTH]
