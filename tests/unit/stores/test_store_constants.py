"""Tests for store class constants.

Task 16.14: Verify that magic numbers have been extracted to named class constants
for better maintainability and documentation.
"""

from app.stores.session_store import InMemorySessionStore
from app.stores.vector_store import InMemoryVectorStore


def test_vector_store_has_default_max_documents_constant() -> None:
    """Verify InMemoryVectorStore defines DEFAULT_MAX_DOCUMENTS constant.

    Task 16.14: Magic number extraction - max documents limit.
    """
    assert hasattr(InMemoryVectorStore, "DEFAULT_MAX_DOCUMENTS")
    assert InMemoryVectorStore.DEFAULT_MAX_DOCUMENTS == 1000


def test_vector_store_has_default_max_chunk_size_constant() -> None:
    """Verify InMemoryVectorStore defines DEFAULT_MAX_CHUNK_SIZE constant.

    Task 16.14: Magic number extraction - max chunk size limit.
    """
    assert hasattr(InMemoryVectorStore, "DEFAULT_MAX_CHUNK_SIZE")
    assert InMemoryVectorStore.DEFAULT_MAX_CHUNK_SIZE == 100_000


def test_vector_store_has_max_top_k_constant() -> None:
    """Verify InMemoryVectorStore defines MAX_TOP_K constant.

    Task 16.14: Magic number extraction - max top_k validation limit.
    """
    assert hasattr(InMemoryVectorStore, "MAX_TOP_K")
    assert InMemoryVectorStore.MAX_TOP_K == 1000


def test_vector_store_has_max_query_length_constant() -> None:
    """Verify InMemoryVectorStore defines MAX_QUERY_LENGTH constant.

    Task 16.14: Magic number extraction - max query string length.
    """
    assert hasattr(InMemoryVectorStore, "MAX_QUERY_LENGTH")
    assert InMemoryVectorStore.MAX_QUERY_LENGTH == 10000


def test_vector_store_has_max_query_tokens_constant() -> None:
    """Verify InMemoryVectorStore defines MAX_QUERY_TOKENS constant.

    Task 16.14: Magic number extraction - max query token count.
    """
    assert hasattr(InMemoryVectorStore, "MAX_QUERY_TOKENS")
    assert InMemoryVectorStore.MAX_QUERY_TOKENS == 10000


def test_session_store_has_default_max_messages_constant() -> None:
    """Verify InMemorySessionStore defines DEFAULT_MAX_MESSAGES constant.

    Task 16.14: Magic number extraction - max messages per session.
    """
    assert hasattr(InMemorySessionStore, "DEFAULT_MAX_MESSAGES")
    assert InMemorySessionStore.DEFAULT_MAX_MESSAGES == 1000


def test_session_store_has_default_session_ttl_constant() -> None:
    """Verify InMemorySessionStore defines DEFAULT_SESSION_TTL constant.

    Task 16.14: Magic number extraction - session TTL in seconds.
    """
    assert hasattr(InMemorySessionStore, "DEFAULT_SESSION_TTL")
    assert InMemorySessionStore.DEFAULT_SESSION_TTL == 3600


def test_session_store_has_max_session_id_length_constant() -> None:
    """Verify InMemorySessionStore defines MAX_SESSION_ID_LENGTH constant.

    Task 16.14: Magic number extraction - max session ID length.
    """
    assert hasattr(InMemorySessionStore, "MAX_SESSION_ID_LENGTH")
    assert InMemorySessionStore.MAX_SESSION_ID_LENGTH == 256


def test_session_store_already_has_default_max_sessions_constant() -> None:
    """Verify InMemorySessionStore still defines DEFAULT_MAX_SESSIONS constant.

    This constant already exists - just verify it hasn't been removed.
    """
    assert hasattr(InMemorySessionStore, "DEFAULT_MAX_SESSIONS")
    assert InMemorySessionStore.DEFAULT_MAX_SESSIONS == 10_000
