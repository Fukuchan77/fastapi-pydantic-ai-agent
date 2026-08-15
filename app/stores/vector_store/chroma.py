"""ChromaVectorStore: embedding-based semantic search backed by ChromaDB."""

import asyncio
import uuid

from app.models.rag import RetrievedHit


class ChromaVectorStore:
    """Chroma-backed vector store using embedding-based semantic search.

    This implementation provides semantic search capabilities using Chroma's
    vector database with embedding-based similarity. Unlike TF-IDF which uses
    token-level matching, embeddings capture semantic meaning and can match
    synonyms, paraphrases, and conceptually similar content.

    Features:
        - Semantic search using sentence embeddings
        - Built-in embedding generation via sentence-transformers
        - Persistent or in-memory storage
        - Support for multiple embedding models

    Args:
        collection_name: Name of the Chroma collection. Defaults to "documents".
        embedding_model: Name of the sentence-transformers model to use for embeddings.
            Defaults to "all-MiniLM-L6-v2" (fast, 384-dimensional embeddings).
        persist_directory: Directory to persist the Chroma database. If None,
            uses in-memory storage (data is lost on restart). Defaults to None.

    Example:
        >>> store = ChromaVectorStore()
        >>> await store.add_documents(["Python is a programming language"])
        >>> results = await store.query("coding in Python", top_k=1)
        >>> print(results[0])
        "Python is a programming language"
    """

    DEFAULT_COLLECTION_NAME: str = "documents"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        persist_directory: str | None = None,
    ) -> None:
        """Initialize Chroma vector store with embedding function.

        Args:
            collection_name: Name of the Chroma collection. Defaults to "documents".
            embedding_model: Sentence-transformers model name. Defaults to "all-MiniLM-L6-v2".
            persist_directory: Directory for persistence. None for in-memory. Defaults to None.
        """
        import chromadb
        from chromadb.utils import embedding_functions

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        # Maps each Chroma document id (uuid) to a stable per-document ordinal,
        # assigned in insertion order. Process-local: not persisted alongside
        # the Chroma collection itself, since deriving it from collection.count()
        # would reintroduce the counter-based race the UUID ids above avoid.
        self._id_to_ordinal: dict[str, int] = {}
        self._next_ordinal: int = 0
        self._generation: int = 0

        # Initialize Chroma client (in-memory or persistent)
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Initialize embedding function using sentence-transformers.
        # `SentenceTransformerEmbeddingFunction` is resolved via chromadb's
        # module-level `__getattr__` lazy-import machinery, which static analysis
        # can't see; confirmed present at runtime (`hasattr(embedding_functions,
        # "SentenceTransformerEmbeddingFunction")` is True on the pinned version).
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(  # ty: ignore[unresolved-attribute]
            model_name=embedding_model
        )

        # Get or create collection with embedding function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

    @property
    def generation(self) -> int:
        """Monotonically increasing content-version counter.

        Advanced after each successful `add_documents()` call and after each
        `clear()` call; unchanged when the underlying Chroma write raises.
        """
        return self._generation

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the store with automatic embedding generation.

        Chroma automatically generates embeddings using the configured embedding
        function. Each document is assigned a unique ID based on insertion order.

        Args:
            chunks: List of text chunks to add. Empty list is allowed.
        """
        if not chunks:
            self._generation += 1
            return

        # Generate unique IDs using UUID4 to prevent race conditions
        # in multi-process deployments with shared persistent Chroma DB.
        # UUID-based IDs eliminate collisions that occur with counter-based IDs.
        ids = [str(uuid.uuid4()) for _ in chunks]
        for doc_id in ids:
            self._id_to_ordinal[doc_id] = self._next_ordinal
            self._next_ordinal += 1

        # Wrap synchronous Chroma operation in executor to prevent blocking event loop
        # Use get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(documents=chunks, ids=ids),
        )
        self._generation += 1

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k most relevant chunks using embedding-based similarity.

        Uses cosine similarity between query embedding and document embeddings
        to find semantically similar content. Unlike TF-IDF, this can match
        synonyms, paraphrases, and conceptually related content.

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
        # Validate top_k parameter
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Wrap synchronous Chroma operations in executor to prevent blocking event loop
        # Use get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()

        # Check collection count (synchronous operation)
        count = await loop.run_in_executor(None, self._collection.count)

        # Return empty list for empty query or empty corpus
        if not query.strip() or count == 0:
            return []

        # Query collection (embeddings generated automatically)
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            ),
        )

        # Extract documents from results
        # results["documents"] is a list of lists: [[doc1, doc2, ...]]
        if results["documents"] and len(results["documents"]) > 0:
            return results["documents"][0]
        return []

    async def query_with_scores(self, query: str, top_k: int = 5) -> list[RetrievedHit]:
        """Retrieve top-k most relevant chunks with stable citation ids and scores.

        Uses the same embedding-based similarity ranking as `query()`.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of up to top_k RetrievedHit, ranked by embedding similarity
            (highest first). Returns empty list if corpus is empty or query
            is empty.

        Raises:
            ValueError: If top_k is less than 1.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(None, self._collection.count)

        if not query.strip() or count == 0:
            return []

        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            ),
        )

        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        # `distances` is an optional key in chromadb's QueryResult, unlike `ids`/
        # `documents` above - narrow through one local instead of re-accessing via
        # `results["distances"]` after a separate `results.get("distances")` guard,
        # which the type checker can't correlate across two different expressions.
        distances_value = results.get("distances")
        distances = distances_value[0] if distances_value else []

        return [
            RetrievedHit(
                chunk_id=f"{self.collection_name}::{self._id_to_ordinal[doc_id]:04d}",
                text=text,
                # Chroma returns a distance (lower = more relevant); negate so
                # higher score means more relevant, matching the other backends.
                score=-distance,
            )
            for doc_id, text, distance in zip(ids, documents, distances, strict=True)
        ]

    async def clear(self) -> None:
        """Remove all documents from the store.

        Deletes the entire collection and recreates it with the same
        configuration (name and embedding function).
        """
        # Wrap synchronous Chroma operations in executor to prevent blocking event loop
        # Use get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()

        # Delete the collection
        await loop.run_in_executor(
            None,
            lambda: self._client.delete_collection(name=self.collection_name),
        )

        # Recreate collection with same configuration
        new_collection = await loop.run_in_executor(
            None,
            lambda: self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
            ),
        )
        self._collection = new_collection
        self._id_to_ordinal = {}
        self._next_ordinal = 0
        self._generation += 1

    async def close(self) -> None:
        """Close the vector store and release any resources.

        ChromaVectorStore doesn't require explicit cleanup as it manages
        its own resources. This is a no-op implementation for Protocol compliance.
        """
        pass
