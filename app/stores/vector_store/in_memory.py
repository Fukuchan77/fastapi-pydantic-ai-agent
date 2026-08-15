"""InMemoryVectorStore: zero-dependency TF-IDF cosine-similarity vector store."""

import asyncio
import math
from collections import Counter
from dataclasses import dataclass

from app.models.rag import RetrievedHit


@dataclass(frozen=True, slots=True)
class _CorpusSnapshot:
    """Immutable point-in-time view of the corpus a query scores against.

    Captured under `InMemoryVectorStore._lock` and then scored off the event
    loop (`asyncio.to_thread`) without touching any further live instance
    state, so a concurrent `add_documents()`/`clear()` can safely mutate the
    live `_documents`/`_doc_tokens`/`_ordinals` lists while scoring is in
    flight: the scorer only ever sees this snapshot's own tuples/dict.
    """

    documents: tuple[str, ...]
    doc_tokens: tuple[list[str], ...]
    ordinals: tuple[int, ...]
    idf_weights: dict[str, float]
    doc_vectors: tuple[dict[str, float], ...] | None
    doc_norms: tuple[float, ...] | None


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

    Concurrency:
        - All mutation of shared state (documents, tokens, memory accounting,
          IDF cache) is serialized by one `asyncio.Lock`.
        - `query()` takes an immutable snapshot of the corpus under that lock,
          then scores the snapshot off the event loop via `asyncio.to_thread`,
          so a concurrent `add_documents()`/`clear()` can never produce a torn
          read and scoring never blocks other requests.
    """

    # Extract magic numbers to class constants for maintainability
    DEFAULT_MAX_DOCUMENTS: int = 1000
    DEFAULT_MAX_CHUNK_SIZE: int = 100_000
    MAX_TOP_K: int = 1000
    MAX_QUERY_LENGTH: int = 10000
    MAX_QUERY_TOKENS: int = 10000
    SOURCE: str = "memory"

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
        self._ordinals: list[int] = []  # Stable per-document ordinal (survives FIFO eviction)
        self._next_ordinal: int = 0
        self._idf_cache: dict[str, float] | None = None
        # Per-document TF-IDF vectors and their magnitudes, derived from
        # `_idf_cache`. Rebuilt off the event loop on the first query after a
        # corpus change and reused until the next one: without them every query
        # re-derived a vector for every stored document, so each request paid
        # O(corpus) regardless of how few terms it searched for.
        self._doc_vectors: tuple[dict[str, float], ...] | None = None
        self._doc_norms: tuple[float, ...] | None = None
        self._memory_usage: int = 0  # Track approximate memory usage in bytes
        self._generation: int = 0
        self._lock = asyncio.Lock()
        self.max_documents = max_documents
        self.max_chunk_size = max_chunk_size
        self.max_memory_bytes = max_memory_bytes

    @property
    def generation(self) -> int:
        """Monotonically increasing content-version counter.

        Advanced after each successful `add_documents()` call and after each
        `clear()` call; unchanged when `add_documents()` rejects an oversized
        chunk before mutating the store.
        """
        return self._generation

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
        # Validate chunk sizes before adding any documents (no lock needed:
        # this reads only the immutable max_chunk_size setting)
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                raise ValueError(f"Document chunk too large (max {self.max_chunk_size} chars)")

        async with self._lock:
            # Tokenize new documents at index time and update memory usage
            for chunk in chunks:
                self._documents.append(chunk)
                tokens = self._tokenize(chunk)
                self._doc_tokens.append(tokens)
                self._ordinals.append(self._next_ordinal)
                self._next_ordinal += 1
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

            # Invalidate IDF cache and every vector derived from it, since
            # corpus statistics changed
            self._invalidate_derived_caches()
            self._generation += 1

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
        ranked = await self._ranked_hits(query, top_k)
        return [text for text, _ordinal, _score in ranked]

    async def query_with_scores(self, query: str, top_k: int = 5) -> list[RetrievedHit]:
        """Retrieve top-k most relevant chunks with stable citation ids and scores.

        Uses the same TF-IDF cosine similarity ranking as `query()`.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return. Defaults to 5.
                Must be at least 1 and cannot exceed 1000.

        Returns:
            List of up to top_k RetrievedHit, ranked by TF-IDF cosine similarity
            score (highest first). Returns empty list if corpus is empty or
            query is empty.

        Raises:
            ValueError: If top_k is less than 1 or greater than 1000.
        """
        ranked = await self._ranked_hits(query, top_k)
        return [
            RetrievedHit(chunk_id=f"{self.SOURCE}::{ordinal:04d}", text=text, score=score)
            for text, ordinal, score in ranked
        ]

    async def _ranked_hits(self, query: str, top_k: int) -> list[tuple[str, int, float]]:
        """Score every stored document against `query` and return the top-k matches.

        Shared ranking logic behind `query()` and `query_with_scores()` so both
        expose identical results. Validates parameters, then captures an
        immutable snapshot of the corpus under `self._lock` and scores that
        snapshot off the event loop (`asyncio.to_thread`), so a concurrent
        `add_documents()`/`clear()` can never tear the read (Req 6.1-6.6).

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (document text, ordinal, similarity score) tuples, sorted
            by score descending and truncated to top_k. Empty if corpus or
            query is empty.

        Raises:
            ValueError: If top_k is less than 1 or greater than 1000, the query
                string exceeds MAX_QUERY_LENGTH, or the query has too many tokens.
        """
        # Validate top_k parameter
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if top_k > self.MAX_TOP_K:
            raise ValueError(f"top_k cannot exceed {self.MAX_TOP_K}")

        # Validate query length to prevent DoS attacks
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError(f"Query string too long (max {self.MAX_QUERY_LENGTH} chars)")

        async with self._lock:
            if not self._documents or not query.strip():
                return []

            # Tokenize query
            query_tokens = self._tokenize(query)

            # Validate token count to prevent DoS via excessive tokens
            # This is defense-in-depth: with whitespace tokenization, the
            # character limit is more restrictive than the token limit, but
            # this validation guards against future tokenization changes.
            if len(query_tokens) > self.MAX_QUERY_TOKENS:
                raise ValueError(f"Query has too many tokens (max {self.MAX_QUERY_TOKENS} tokens)")

            if not query_tokens:
                return []

            # Calculate or reuse cached IDF weights
            if self._idf_cache is None:
                # First query after add_documents - compute and cache IDF
                self._idf_cache = self._calculate_idf(self._doc_tokens)

            # Immutable snapshot: subsequent add_documents()/clear() rebind or
            # mutate the live lists/cache, never the tuples/dict captured here.
            snapshot = _CorpusSnapshot(
                documents=tuple(self._documents),
                doc_tokens=tuple(self._doc_tokens),
                ordinals=tuple(self._ordinals),
                idf_weights=self._idf_cache,
                doc_vectors=self._doc_vectors,
                doc_norms=self._doc_norms,
            )
            generation = self._generation

        ranked, doc_vectors, doc_norms = await asyncio.to_thread(
            self._score_snapshot, query_tokens, snapshot, top_k
        )

        # Publish the vectors this query had to derive so the next one reuses
        # them. Building them happens inside the worker thread, never on the
        # event loop. The generation guard drops the result if the corpus
        # changed while scoring was in flight - those vectors describe a corpus
        # that no longer exists, and `_invalidate_derived_caches()` has already
        # cleared the fields they would otherwise overwrite.
        if snapshot.doc_vectors is None:
            async with self._lock:
                if self._generation == generation:
                    self._doc_vectors = doc_vectors
                    self._doc_norms = doc_norms

        return ranked

    @staticmethod
    def _score_snapshot(
        query_tokens: list[str], snapshot: _CorpusSnapshot, top_k: int
    ) -> tuple[list[tuple[str, int, float]], tuple[dict[str, float], ...], tuple[float, ...]]:
        """Score a corpus snapshot against `query_tokens` (runs off the event loop).

        Pure function of its arguments only -- touches no live instance state,
        so it is safe to run in a worker thread via `asyncio.to_thread` while
        the event loop concurrently mutates the store.

        Derives the per-document vectors when the snapshot carries none, and
        returns them so the caller can cache them for the next query against the
        same corpus generation. Deriving them here rather than under the store
        lock keeps that O(corpus) work off the event loop even on the first
        query after an ingest.

        Args:
            query_tokens: Tokenized query.
            snapshot: Immutable corpus snapshot to score against.
            top_k: Maximum number of results to return.

        Returns:
            A tuple of (ranked results, document vectors, document norms). The
            ranked results are (document text, ordinal, similarity score)
            tuples, sorted by score descending and truncated to top_k.
        """
        doc_vectors = snapshot.doc_vectors
        doc_norms = snapshot.doc_norms
        if doc_vectors is None or doc_norms is None:
            doc_vectors = tuple(
                InMemoryVectorStore._calculate_tfidf_vector(doc_tokens, snapshot.idf_weights)
                for doc_tokens in snapshot.doc_tokens
            )
            doc_norms = tuple(
                math.sqrt(sum(value**2 for value in vector.values())) for vector in doc_vectors
            )

        query_tfidf = InMemoryVectorStore._calculate_tfidf_vector(
            query_tokens, snapshot.idf_weights
        )
        query_norm = math.sqrt(sum(value**2 for value in query_tfidf.values()))

        # A zero magnitude is not an early exit: a single-document corpus gives
        # every term an IDF of log(N/df) = 0, so every vector is zero and every
        # pair scores 0.0 - but the caller still expects the top_k slice back.
        scores: list[tuple[int, float]] = []
        for idx, doc_vector in enumerate(doc_vectors):
            similarity = InMemoryVectorStore._cosine_similarity_with_norms(
                query_tfidf, query_norm, doc_vector, doc_norms[idx]
            )
            scores.append((idx, similarity))

        # Sort by score (descending) and take top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked = [
            (snapshot.documents[idx], snapshot.ordinals[idx], score)
            for idx, score in scores[:top_k]
        ]
        return ranked, doc_vectors, doc_norms

    @staticmethod
    def _cosine_similarity_with_norms(
        vec1: dict[str, float],
        norm1: float,
        vec2: dict[str, float],
        norm2: float,
    ) -> float:
        """Cosine similarity using pre-computed vector magnitudes.

        Identical to `_cosine_similarity` but skips recomputing the magnitudes,
        which the caller caches alongside the document vectors.

        Args:
            vec1: First TF-IDF vector.
            norm1: Pre-computed magnitude of vec1.
            vec2: Second TF-IDF vector.
            norm2: Pre-computed magnitude of vec2.

        Returns:
            Cosine similarity in [0, 1]; 0.0 if either vector is empty or has
            zero magnitude.
        """
        if not vec1 or not vec2 or norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        # Walk the smaller vector so the intersection scan stays proportional to
        # the query's term count rather than the document's.
        smaller, larger = (vec1, vec2) if len(vec1) <= len(vec2) else (vec2, vec1)
        dot_product = sum(
            weight * larger[term] for term, weight in smaller.items() if term in larger
        )

        return dot_product / (norm1 * norm2)

    async def clear(self) -> None:
        """Remove all documents from the store.

        Also clears the tokenization cache, invalidates IDF cache, and resets memory usage.
        """
        async with self._lock:
            self._documents.clear()
            self._doc_tokens.clear()
            self._ordinals.clear()
            self._next_ordinal = 0
            self._memory_usage = 0
            self._invalidate_derived_caches()
            self._generation += 1

    def _invalidate_derived_caches(self) -> None:
        """Drop every cache derived from the corpus.

        IDF weights depend on the whole corpus and the per-document vectors
        depend on those weights, so all three must be discarded together.
        Callers hold `self._lock`.
        """
        self._idf_cache = None
        self._doc_vectors = None
        self._doc_norms = None

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
        self._ordinals = self._ordinals[num_to_evict:]

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

    @staticmethod
    def _calculate_tfidf_vector(
        tokens: list[str], idf_weights: dict[str, float]
    ) -> dict[str, float]:
        """Calculate TF-IDF vector for a token list.

        TF-IDF = TF * IDF, where:
        - TF (Term Frequency) = count of term in document / total terms in document
        - IDF (Inverse Document Frequency) = from pre-calculated corpus weights

        Static because `_score_snapshot` runs this off the event loop via
        `asyncio.to_thread` and must not touch live instance state.

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

    @staticmethod
    def _cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Calculate cosine similarity between two TF-IDF vectors.

        Cosine similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)

        Static because `_score_snapshot` runs this off the event loop via
        `asyncio.to_thread` and must not touch live instance state.

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
