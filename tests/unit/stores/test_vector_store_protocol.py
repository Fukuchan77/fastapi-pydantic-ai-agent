"""Unit tests for the VectorStore Protocol interface definition.

Split out of test_vector_store.py per `.sdd/steering/file-size-policy.md`.
"""

from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import VectorStore


class TestVectorStoreProtocol:
    """Test VectorStore Protocol interface definition."""

    def test_protocol_has_add_documents_method(self) -> None:
        """VectorStore Protocol defines add_documents method."""
        assert hasattr(VectorStore, "add_documents")

    def test_protocol_has_query_method(self) -> None:
        """VectorStore Protocol defines query method."""
        assert hasattr(VectorStore, "query")

    def test_protocol_has_clear_method(self) -> None:
        """VectorStore Protocol defines clear method."""
        assert hasattr(VectorStore, "clear")

    def test_protocol_has_close_method(self) -> None:
        """VectorStore Protocol defines close method."""
        assert hasattr(VectorStore, "close")

    def test_in_memory_vector_store_implements_protocol(self) -> None:
        """InMemoryVectorStore implements all VectorStore Protocol methods."""
        store = InMemoryVectorStore()
        assert hasattr(store, "add_documents")
        assert hasattr(store, "query")
        assert hasattr(store, "clear")
        assert hasattr(store, "close")
