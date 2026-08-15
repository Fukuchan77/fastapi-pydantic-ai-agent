"""VectorStore Protocol defining the pluggable vector store interface."""

from typing import Protocol

from app.models.rag import RetrievedHit


class VectorStore(Protocol):
    """Protocol defining the vector store interface for RAG operations.

    Implementations must provide document storage, similarity-based retrieval,
    and store clearing capabilities.
    """

    @property
    def generation(self) -> int:
        """Monotonically increasing content-version counter.

        Incremented by the implementation after each successful
        `add_documents()` call and after each `clear()` call, so a consumer
        can key a cache on "which content set is this" without inspecting
        store contents directly. Left unchanged when either operation raises.
        Read-only: callers must never assign to this member.
        """
        ...

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the vector store.

        Args:
            chunks: List of text chunks to store.
        """
        ...

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k most relevant document chunks for a query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of document chunks ranked by relevance score (highest first).
            Returns empty list if no documents are stored.
        """
        ...

    async def query_with_scores(self, query: str, top_k: int = 5) -> list[RetrievedHit]:
        """Retrieve top-k most relevant chunks with stable citation ids and scores.

        Additive counterpart to `query()` that surfaces the data citations need
        (a stable `chunk_id` and the relevance `score`) without changing the
        legacy `query()` contract.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of RetrievedHit ranked by relevance score (highest first).
            Returns empty list if no documents are stored.
        """
        ...

    async def clear(self) -> None:
        """Remove all documents from the vector store."""
        ...

    async def close(self) -> None:
        """Close the vector store and release any resources.

        This method is called during application shutdown to properly clean up
        resources like HTTP clients, database connections, etc. Implementations
        that don't hold external resources can implement this as a no-op.
        """
        ...
