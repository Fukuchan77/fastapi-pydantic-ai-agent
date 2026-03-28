"""VectorStore Protocol and InMemoryVectorStore implementation with TF-IDF.

This module provides a pluggable vector store interface for the RAG pattern.
The default in-memory implementation uses TF-IDF cosine similarity for retrieval.
"""

import asyncio
import math
import uuid
from collections import Counter
from typing import Protocol

import httpx


class VectorStore(Protocol):
    """Protocol defining the vector store interface for RAG operations.

    Implementations must provide document storage, similarity-based retrieval,
    and store clearing capabilities.
    """

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


class InMemoryVectorStore:
    """In-memory vector store using TF-IDF cosine similarity.

    This implementation provides zero-dependency document retrieval using
    TF-IDF (Term Frequency-Inverse Document Frequency) scoring with cosine
    similarity for ranking.

    Algorithm:
        1. Tokenize query and documents (whitespace split, lowercase)
        2. Build term-frequency vectors
        3. Compute IDF weights from corpus
        4. Calculate cosine similarity scores
        5. Return top-k chunks sorted by score (descending)

    Limitations:
        - Token-level matching only (no stemming or lemmatization)
        - No semantic understanding
        - Suitable for demonstration; production use cases should consider
          embedding-based backends (e.g., Chroma, pgvector)

    Memory Management:
        - Enforces a maximum document limit with FIFO (First-In-First-Out) eviction
        - When document count exceeds max_documents after add_documents(),
          the oldest entries are automatically removed
        - Prevents memory exhaustion in long-running deployments

    Security:
        - Enforces per-chunk size limit (max_chunk_size) to prevent memory
          exhaustion and DoS attacks via oversized document chunks
        - Validation is atomic: all chunks must be valid or none are added
        - Default limit is 100,000 characters per chunk
    """

    # Task 16.14: Extract magic numbers to class constants for maintainability
    DEFAULT_MAX_DOCUMENTS: int = 1000
    DEFAULT_MAX_CHUNK_SIZE: int = 100_000
    MAX_TOP_K: int = 1000
    MAX_QUERY_LENGTH: int = 10000
    MAX_QUERY_TOKENS: int = 10000

    def __init__(
        self,
        max_documents: int = DEFAULT_MAX_DOCUMENTS,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        max_memory_bytes: int | None = None,
    ) -> None:
        """Initialize an empty in-memory vector store.

        Args:
            max_documents: Maximum number of documents to store. When exceeded,
                oldest documents are evicted (FIFO). Defaults to 1000.
            max_chunk_size: Maximum size (in characters) for each document chunk.
                Chunks exceeding this limit will be rejected. Defaults to 100_000.
            max_memory_bytes: Maximum memory usage in bytes. When exceeded,
                oldest documents are evicted (FIFO). None means unlimited. Defaults to None.
        """
        self._documents: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._idf_cache: dict[str, float] | None = None
        self._memory_usage: int = 0  # Track approximate memory usage in bytes
        self.max_documents = max_documents
        self.max_chunk_size = max_chunk_size
        self.max_memory_bytes = max_memory_bytes

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the store.

        When the total document count exceeds max_documents after adding,
        the oldest documents are automatically evicted (FIFO).

        When memory usage exceeds max_memory_bytes after adding,
        the oldest documents are automatically evicted (FIFO).

        Documents are tokenized at index time and cached in _doc_tokens.
        Adding documents invalidates the IDF cache since corpus statistics change.

        Args:
            chunks: List of text chunks to add. Empty list is allowed.

        Raises:
            ValueError: If any chunk exceeds max_chunk_size characters.
        """
        # Validate chunk sizes before adding any documents
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                raise ValueError(f"Document chunk too large (max {self.max_chunk_size} chars)")

        # Tokenize new documents at index time and update memory usage
        for chunk in chunks:
            self._documents.append(chunk)
            tokens = self._tokenize(chunk)
            self._doc_tokens.append(tokens)
            # Estimate memory: document string + tokenized list
            self._memory_usage += self._estimate_memory(chunk, tokens)

        # Apply FIFO eviction if document count exceeds max_documents
        # Must keep _documents and _doc_tokens synchronized
        if len(self._documents) > self.max_documents:
            num_to_evict = len(self._documents) - self.max_documents
            self._evict_oldest(num_to_evict)

        # Apply FIFO eviction if memory usage exceeds max_memory_bytes
        if self.max_memory_bytes is not None:
            while self._memory_usage > self.max_memory_bytes and len(self._documents) > 0:
                self._evict_oldest(1)

        # Invalidate IDF cache since corpus changed
        self._idf_cache = None

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k most relevant chunks using TF-IDF cosine similarity.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.
                Must be at least 1 and cannot exceed 1000.

        Returns:
            List of up to top_k document chunks, ranked by TF-IDF cosine
            similarity score (highest first). Returns empty list if corpus
            is empty or query is empty.

        Raises:
            ValueError: If top_k is less than 1 or greater than 1000.
        """
        # Validate top_k parameter
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > self.MAX_TOP_K:
            raise ValueError(f"top_k cannot exceed {self.MAX_TOP_K}")

        # Validate query length to prevent DoS attacks
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError(f"Query string too long (max {self.MAX_QUERY_LENGTH} chars)")

        if not self._documents or not query.strip():
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Task 3.13: Validate token count to prevent DoS via excessive tokens
        # This is defense-in-depth: with whitespace tokenization, the character
        # limit is more restrictive than the token limit,
        # but this validation guards against future tokenization changes or edge cases.
        if len(query_tokens) > self.MAX_QUERY_TOKENS:
            raise ValueError(f"Query has too many tokens (max {self.MAX_QUERY_TOKENS} tokens)")

        if not query_tokens:
            return []

        # Use cached tokenized documents (no re-tokenization)
        doc_tokens_list = self._doc_tokens

        # Calculate or reuse cached IDF weights
        if self._idf_cache is None:
            # First query after add_documents - compute and cache IDF
            self._idf_cache = self._calculate_idf(doc_tokens_list)
        idf_weights = self._idf_cache

        # Calculate TF-IDF vectors and cosine similarity scores
        query_tfidf = self._calculate_tfidf_vector(query_tokens, idf_weights)

        scores: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(doc_tokens_list):
            doc_tfidf = self._calculate_tfidf_vector(doc_tokens, idf_weights)
            similarity = self._cosine_similarity(query_tfidf, doc_tfidf)
            scores.append((idx, similarity))

        # Sort by score (descending) and take top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in scores[:top_k]]

        return [self._documents[idx] for idx in top_indices]

    async def clear(self) -> None:
        """Remove all documents from the store.

        Also clears the tokenization cache, invalidates IDF cache, and resets memory usage.
        """
        self._documents.clear()
        self._doc_tokens.clear()
        self._idf_cache = None
        self._memory_usage = 0

    async def close(self) -> None:
        """Close the vector store and release any resources.

        InMemoryVectorStore doesn't hold external resources, so this is a no-op.
        Implements the VectorStore Protocol interface for consistency.
        """
        pass

    def get_memory_usage(self) -> int:
        """Get the current estimated memory usage in bytes.

        Returns:
            Approximate memory usage in bytes, including document strings
            and tokenized representations.
        """
        return self._memory_usage

    def _estimate_memory(self, chunk: str, tokens: list[str]) -> int:
        """Estimate memory usage for a document chunk and its tokens.

        Uses a simple heuristic: document size + token list overhead.
        This is an approximation; actual Python object overhead may vary.

        Args:
            chunk: The document chunk string.
            tokens: The tokenized representation.

        Returns:
            Estimated memory usage in bytes.
        """
        # Document string: 2 bytes per character (Python 3 uses compact representation)
        doc_size = len(chunk) * 2

        # Token list: each token string + list overhead
        token_size = sum(len(token) * 2 for token in tokens) + len(tokens) * 8

        # Total: document + tokens + Python object overhead
        return doc_size + token_size + 100  # 100 bytes overhead per document

    def _evict_oldest(self, num_to_evict: int) -> None:
        """Evict the oldest N documents (FIFO eviction).

        Updates memory usage tracking and maintains synchronization between
        _documents and _doc_tokens.

        Args:
            num_to_evict: Number of oldest documents to remove.
        """
        for i in range(min(num_to_evict, len(self._documents))):
            # Subtract memory of evicted document
            evicted_chunk = self._documents[i]
            evicted_tokens = self._doc_tokens[i]
            self._memory_usage -= self._estimate_memory(evicted_chunk, evicted_tokens)

        # Remove from front of lists (oldest documents)
        self._documents = self._documents[num_to_evict:]
        self._doc_tokens = self._doc_tokens[num_to_evict:]

        # Ensure memory usage doesn't go negative due to estimation errors
        self._memory_usage = max(0, self._memory_usage)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text by splitting on whitespace and converting to lowercase.

        Args:
            text: The text to tokenize.

        Returns:
            List of lowercase tokens.
        """
        return text.lower().split()

    def _calculate_idf(self, doc_tokens_list: list[list[str]]) -> dict[str, float]:
        """Calculate IDF (Inverse Document Frequency) weights for the corpus.

        IDF = log(N / df), where:
        - N is the total number of documents
        - df is the number of documents containing the term

        Args:
            doc_tokens_list: List of tokenized documents.

        Returns:
            Dictionary mapping terms to their IDF weights.
        """
        n_docs = len(doc_tokens_list)
        if n_docs == 0:
            return {}

        # Count document frequency (df) for each term
        df: dict[str, int] = {}
        for doc_tokens in doc_tokens_list:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        # Calculate IDF
        idf: dict[str, float] = {}
        for term, doc_freq in df.items():
            idf[term] = math.log(n_docs / doc_freq)

        return idf

    def _calculate_tfidf_vector(
        self, tokens: list[str], idf_weights: dict[str, float]
    ) -> dict[str, float]:
        """Calculate TF-IDF vector for a token list.

        TF-IDF = TF * IDF, where:
        - TF (Term Frequency) = count of term in document / total terms in document
        - IDF (Inverse Document Frequency) = from pre-calculated corpus weights

        Args:
            tokens: List of tokens to calculate TF-IDF for.
            idf_weights: Pre-calculated IDF weights from corpus.

        Returns:
            Dictionary mapping terms to their TF-IDF scores.
        """
        if not tokens:
            return {}

        # Calculate term frequencies
        tf_counter = Counter(tokens)
        total_terms = len(tokens)

        # Calculate TF-IDF
        tfidf: dict[str, float] = {}
        for term, count in tf_counter.items():
            tf = count / total_terms
            idf = idf_weights.get(term, 0.0)
            tfidf[term] = tf * idf

        return tfidf

    def _cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Calculate cosine similarity between two TF-IDF vectors.

        Cosine similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)

        Args:
            vec1: First TF-IDF vector.
            vec2: Second TF-IDF vector.

        Returns:
            Cosine similarity score in range [0, 1]. Returns 0 if either
            vector is empty or has zero magnitude.
        """
        if not vec1 or not vec2:
            return 0.0

        # Calculate dot product
        common_terms = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(val**2 for val in vec1.values()))
        magnitude2 = math.sqrt(sum(val**2 for val in vec2.values()))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


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

        # Initialize Chroma client (in-memory or persistent)
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Initialize embedding function using sentence-transformers
        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(  # type: ignore[attr-defined]
            model_name=embedding_model
        )

        # Get or create collection with embedding function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
        )

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the store with automatic embedding generation.

        Chroma automatically generates embeddings using the configured embedding
        function. Each document is assigned a unique ID based on insertion order.

        Args:
            chunks: List of text chunks to add. Empty list is allowed.
        """
        if not chunks:
            return

        # Task 21.4: Generate unique IDs using UUID4 to prevent race conditions
        # in multi-process deployments with shared persistent Chroma DB.
        # UUID-based IDs eliminate collisions that occur with counter-based IDs.
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Task 21.3: Wrap synchronous Chroma operation in executor to prevent blocking event loop
        # Task 23.1: Use get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(documents=chunks, ids=ids),
        )

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

        # Task 21.3: Wrap synchronous Chroma operations in executor to prevent blocking event loop
        # Task 23.1: Use get_running_loop() instead of deprecated get_event_loop()
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

    async def clear(self) -> None:
        """Remove all documents from the store.

        Deletes the entire collection and recreates it with the same
        configuration (name and embedding function).
        """
        # Task 21.3: Wrap synchronous Chroma operations in executor to prevent blocking event loop
        # Task 23.1: Use get_running_loop() instead of deprecated get_event_loop()
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

    async def close(self) -> None:
        """Close the vector store and release any resources.

        ChromaVectorStore doesn't require explicit cleanup as it manages
        its own resources. This is a no-op implementation for Protocol compliance.
        """
        pass


class OllamaEmbeddingVectorStore:
    """Ollama-backed vector store using embedding-based semantic search.

    This implementation provides semantic search capabilities using Ollama's
    /v1/embeddings API endpoint. Uses cosine similarity for document ranking.
    Suitable for local development and testing with Ollama.

    Task 25.1: This class calls the Ollama API directly (POST /v1/embeddings)
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
        # Task 22.1: Track whether we own the http_client for proper cleanup
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT)
        self._documents: list[str] = []
        self._embeddings: list[list[float]] = []

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

        # Validate response structure (Task 22.2)
        if "data" not in data:
            raise ValueError(f"Unexpected Ollama embeddings response: {data}")

        # Sort by index to ensure correct order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])

        # Validate each item has 'embedding' key (Task 23.3)
        for item in sorted_data:
            if "embedding" not in item:
                raise ValueError(f"Missing 'embedding' in response item: {item}")

        return [item["embedding"] for item in sorted_data]

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the store with automatic embedding generation.

        Args:
            chunks: List of text chunks to add. Empty list is allowed.
        """
        if not chunks:
            return

        # Generate embeddings for all chunks
        embeddings = await self._embed(chunks)

        # Store documents and embeddings
        self._documents.extend(chunks)
        self._embeddings.extend(embeddings)

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
        # Validate top_k parameter
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Return empty list for empty query or empty corpus
        if not query.strip() or not self._documents:
            return []

        # Generate query embedding
        query_embedding = (await self._embed([query]))[0]

        # Calculate cosine similarity scores
        scores = [_cosine_similarity(query_embedding, emb) for emb in self._embeddings]

        # Get top_k indices sorted by score (descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [self._documents[i] for i in top_indices]

    async def clear(self) -> None:
        """Remove all documents and embeddings from the store."""
        self._documents.clear()
        self._embeddings.clear()

    async def close(self) -> None:
        """Close the HTTP client if it was created internally.

        Task 22.1: Prevents resource leaks by properly closing the AsyncClient
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
