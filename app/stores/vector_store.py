"""VectorStore Protocol and InMemoryVectorStore implementation with TF-IDF.

This module provides a pluggable vector store interface for the RAG pattern.
The default in-memory implementation uses TF-IDF cosine similarity for retrieval.
"""

import math
from collections import Counter
from typing import Protocol


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

    def __init__(self, max_documents: int = 10000, max_chunk_size: int = 100_000) -> None:
        """Initialize an empty in-memory vector store.

        Args:
            max_documents: Maximum number of documents to store. When exceeded,
                oldest documents are evicted (FIFO). Defaults to 10000.
            max_chunk_size: Maximum size (in characters) for each document chunk.
                Chunks exceeding this limit will be rejected. Defaults to 100_000.
        """
        self._documents: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._idf_cache: dict[str, float] | None = None
        self.max_documents = max_documents
        self.max_chunk_size = max_chunk_size

    async def add_documents(self, chunks: list[str]) -> None:
        """Add document chunks to the store.

        When the total document count exceeds max_documents after adding,
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

        # Tokenize new documents at index time
        for chunk in chunks:
            self._documents.append(chunk)
            self._doc_tokens.append(self._tokenize(chunk))

        # Apply FIFO eviction if document count exceeds max_documents
        # Must keep _documents and _doc_tokens synchronized
        if len(self._documents) > self.max_documents:
            self._documents = self._documents[-self.max_documents :]
            self._doc_tokens = self._doc_tokens[-self.max_documents :]

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
        if top_k > 1000:
            raise ValueError("top_k cannot exceed 1000")

        # Validate query length to prevent DoS attacks
        if len(query) > 10000:
            raise ValueError("Query string too long (max 10000 chars)")

        if not self._documents or not query.strip():
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Task 3.13: Validate token count to prevent DoS via excessive tokens
        # This is defense-in-depth: with whitespace tokenization, the character
        # limit (10000 chars) is more restrictive than the token limit (10000 tokens),
        # but this validation guards against future tokenization changes or edge cases.
        if len(query_tokens) > 10000:
            raise ValueError("Query has too many tokens (max 10000 tokens)")

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

        Also clears the tokenization cache and invalidates IDF cache.
        """
        self._documents.clear()
        self._doc_tokens.clear()
        self._idf_cache = None

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
