"""Authenticated principal derivation (Req 11.1).

`Principal.id` is derived from the API key that authenticated the request,
not stored or looked up, so introducing it requires no new store schema.

Single-key today, and two subsystems depend on that
-----------------------------------------------------
`Settings.api_key` holds exactly one key, so every caller derives the same
`Principal.id` and the session-ownership check in
`app/services/session_service.py` is effectively an unguessable-token check
(`secrets.token_urlsafe(16)`) rather than a cross-principal boundary. That is a
sound forward-compatible design - but two subsystems are currently correct
*only* because there is one principal, and both must be revisited in the same
change that introduces a second key:

- **`POST /v1/rag/ingest`** (`app/api/v1/rag.py`) writes into the one
  process-wide `app.state.vector_store` that every RAG query reads from, with no
  principal binding. With two keys that is cross-principal corpus poisoning: one
  caller's ingested text becomes another caller's retrieved context and answer.
- **The Corrective RAG result cache** (`app/workflows/rag_cache.py`) is keyed on
  `(query, max_retries, vector_store.generation)` and on nothing else, so an
  identical query from a different principal is served another principal's
  cached answer.

RAG is deliberately session-less (`app/api/v1/rag.py`, "RAG queries are
session-less"), so Req 11's ownership machinery does not reach either of them.
Adding a second key therefore means partitioning the vector store per principal
and folding `principal.id` into the cache key - not merely widening this
function.
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
