"""OllamaEmbeddingVectorStore: embedding-based semantic search via Ollama's API."""

import httpx

from app.models.rag import RetrievedHit


class OllamaEmbeddingVectorStore:
    """Ollama-backed vector store using embedding-based semantic search.

    This implementation provides semantic search capabilities using Ollama's
    /v1/embeddings API endpoint. Uses cosine similarity for document ranking.
    Suitable for local development and testing with Ollama.

    This class calls the Ollama API directly (POST /v1/embeddings)
    without going through LiteLLM, so the base URL MUST include the /v1 suffix.
    This differs from build_model() in chat_agent.py which uses LiteLLM (which
    auto-appends /v1 for Ollama), so build_model() uses base URL without /v1.

    Requires:
        - Running Ollama instance at base_url
        - Embedding model pulled: ollama pull <embedding_model>

    Args:
        embedding_model: Embedding model name (e.g., "nomic-embed-text:latest").
        base_url: Base URL for Ollama API. Defaults to "http://localhost:11434/v1".
        http_client: Optional HTTP client for requests. If None, creates a new client.

    Example:
        >>> store = OllamaEmbeddingVectorStore(embedding_model="nomic-embed-text:latest")
        >>> await store.add_documents(["Python is a programming language"])
        >>> results = await store.query("coding in Python", top_k=1)
        >>> print(results[0])
        "Python is a programming language"
    """

    DEFAULT_BASE_URL: str = "http://localhost:11434/v1"
    DEFAULT_TIMEOUT: float = 30.0
    SOURCE: str = "ollama"

    def __init__(
        self,
        embedding_model: str,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize Ollama embedding vector store.

        Args:
            embedding_model: Embedding model name (e.g., "nomic-embed-text:latest").
            base_url: Base URL for Ollama API. Defaults to "http://localhost:11434/v1".
            http_client: Optional HTTP client. If None, creates a new client with timeout.
        """
        self._embedding_model = embedding_model
        self._base_url = base_url.rstrip("/")
        # Track whether we own the http_client for proper cleanup
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT)
        self._documents: list[str] = []
        self._embeddings: list[list[float]] = []
        self._generation: int = 0

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Call POST /v1/embeddings and return embedding vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        response = await self._http_client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._embedding_model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()

        # Validate response structure ()
        if "data" not in data:
            raise ValueError(f"Unexpected Ollama embeddings response: {data}")

        # Sort by index to ensure correct order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])

        # Validate each item has 'embedding' key ()
        for item in sorted_data:
            if "embedding" not in item:
                raise ValueError(f"Missing 'embedding' in response item: {item}")

        return [item["embedding"] for item in sorted_data]

    @property
    def generation(self) -> int:
        """Monotonically increasing content-version counter.

        Advanced after each successful `add_documents()` call and after each
        `clear()` call; unchanged when the embedding request raises.
        """
        return self._generation

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the store with automatic embedding generation.

        Args:
            chunks: List of text chunks to add. Empty list is allowed.
        """
        if not chunks:
            self._generation += 1
            return

        # Generate embeddings for all chunks
        embeddings = await self._embed(chunks)

        # Store documents and embeddings
        self._documents.extend(chunks)
        self._embeddings.extend(embeddings)
        self._generation += 1

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k most relevant chunks using cosine similarity.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of up to top_k document chunks, ranked by embedding cosine
            similarity (highest first). Returns empty list if corpus is empty
            or query is empty.

        Raises:
            ValueError: If top_k is less than 1.
        """
        scored = await self._top_k_scored_indices(query, top_k)
        return [self._documents[idx] for idx, _score in scored]

    async def query_with_scores(self, query: str, top_k: int = 5) -> list[RetrievedHit]:
        """Retrieve top-k most relevant chunks with stable citation ids and scores.

        Uses the same embedding cosine similarity ranking as `query()`. Since
        documents are only ever appended (no eviction), a document's position
        in `_documents` is a stable ordinal for its lifetime in the store.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of up to top_k RetrievedHit, ranked by embedding cosine
            similarity (highest first). Returns empty list if corpus is empty
            or query is empty.

        Raises:
            ValueError: If top_k is less than 1.
        """
        scored = await self._top_k_scored_indices(query, top_k)
        return [
            RetrievedHit(
                chunk_id=f"{self.SOURCE}::{idx:04d}",
                text=self._documents[idx],
                score=score,
            )
            for idx, score in scored
        ]

    async def _top_k_scored_indices(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Score every stored document against `query` and return the top-k matches.

        Shared ranking logic behind `query()` and `query_with_scores()` so both
        expose identical results.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (document index, cosine similarity) tuples, sorted by score
            descending and truncated to top_k. Empty if corpus or query is empty.

        Raises:
            ValueError: If top_k is less than 1.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not query.strip() or not self._documents:
            return []

        query_embedding = (await self._embed([query]))[0]
        scores = [_cosine_similarity(query_embedding, emb) for emb in self._embeddings]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(idx, scores[idx]) for idx in top_indices]

    async def clear(self) -> None:
        """Remove all documents and embeddings from the store."""
        self._documents.clear()
        self._embeddings.clear()
        self._generation += 1

    async def ping(self) -> None:
        """Probe Ollama connectivity without mutating the stored corpus.

        Calls the embeddings API directly and discards the result, so a
        connectivity check (startup dry-run or a periodic readiness probe)
        can never wipe live `add_documents()` data. Callers must not use
        `add_documents()` + `clear()` as a probe substitute - `clear()`
        removes the entire corpus, not just a probe document.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
            ValueError: If the response is malformed.
        """
        await self._embed(["ping"])

    async def close(self) -> None:
        """Close the HTTP client if it was created internally.

        Prevents resource leaks by properly closing the AsyncClient
        when the store is no longer needed. Only closes the client if it was
        created by the store itself (not externally provided).

        This method should be called during application shutdown, typically
        in the FastAPI lifespan teardown.
        """
        if self._owns_http_client and self._http_client is not None:
            await self._http_client.aclose()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score in range [-1, 1]. Returns 0 if either
        vector is zero or lengths don't match.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
