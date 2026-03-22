# Vector Store Embeddings Migration Guide

## Overview

This guide provides instructions for migrating from the current TF-IDF based vector store to an embedding-based semantic search system for improved RAG accuracy.

## Current Implementation

The current `InMemoryVectorStore` uses:

- **TF-IDF** (Term Frequency-Inverse Document Frequency) scoring
- **Cosine similarity** on term vectors
- **Whitespace tokenization** (no semantic understanding)

**Limitations:**

- No understanding of synonyms or semantic meaning
- Poor performance on paraphrased queries
- Limited multilingual support
- Cannot capture contextual relationships

## Recommended Approaches

### Option 1: Sentence Transformers (Self-Hosted)

**Pros:** No API costs, fast, good quality
**Cons:** Requires GPU for best performance, increased memory usage

```bash
uv add sentence-transformers torch
```

**Implementation:**

```python
# app/stores/embedding_vector_store.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Protocol

class EmbeddingVectorStore:
    """Vector store using sentence-transformers for semantic search."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",  # 384 dim, fast
        max_documents: int = 10000
    ):
        """Initialize embedding model.

        Popular models:
        - all-MiniLM-L6-v2: Fast, 384 dim, good for English
        - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support
        - all-mpnet-base-v2: Higher quality, 768 dim, slower
        """
        self.model = SentenceTransformer(model_name)
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.max_documents = max_documents

    async def add_documents(self, chunks: list[str]) -> None:
        """Add documents and compute their embeddings."""
        if not chunks:
            return

        # Compute embeddings for new chunks
        new_embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Add to storage
        self.documents.extend(chunks)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Apply FIFO eviction if needed
        if len(self.documents) > self.max_documents:
            self.documents = self.documents[-self.max_documents:]
            self.embeddings = self.embeddings[-self.max_documents:]

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve documents using semantic similarity."""
        if not self.documents or self.embeddings is None:
            return []

        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )[0]

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [self.documents[idx] for idx in top_indices]

    async def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents.clear()
        self.embeddings = None
```

### Option 2: OpenAI Embeddings (Cloud-Based)

**Pros:** High quality, no infrastructure needed, multilingual
**Cons:** API costs, requires internet connection

```bash
uv add openai
```

**Implementation:**

```python
# app/stores/openai_vector_store.py
import numpy as np
from openai import AsyncOpenAI

class OpenAIVectorStore:
    """Vector store using OpenAI embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",  # or text-embedding-3-large
        max_documents: int = 10000
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.max_documents = max_documents

    async def add_documents(self, chunks: list[str]) -> None:
        """Add documents and compute embeddings via OpenAI API."""
        if not chunks:
            return

        # Get embeddings from OpenAI (batch processing for efficiency)
        response = await self.client.embeddings.create(
            input=chunks,
            model=self.model
        )

        new_embeddings = np.array([
            item.embedding for item in response.data
        ])

        # Add to storage
        self.documents.extend(chunks)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # FIFO eviction
        if len(self.documents) > self.max_documents:
            self.documents = self.documents[-self.max_documents:]
            self.embeddings = self.embeddings[-self.max_documents:]

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve documents using semantic similarity."""
        if not self.documents or self.embeddings is None:
            return []

        # Get query embedding
        response = await self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        query_embedding = np.array(response.data[0].embedding)

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.documents[idx] for idx in top_indices]

    async def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents.clear()
        self.embeddings = None
```

### Option 3: Chroma DB (Production-Grade)

For production deployments with persistence and advanced features:

```bash
uv add chromadb
```

**Features:**

- Persistent storage
- Built-in embedding generation
- Metadata filtering
- Multiple collection support

```python
# app/stores/chroma_vector_store.py
import chromadb
from chromadb.config import Settings

class ChromaVectorStore:
    """Production-grade vector store using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "documents"
    ):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, chunks: list[str]) -> None:
        """Add documents to ChromaDB."""
        ids = [f"doc_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids
        )

    async def query(self, query: str, top_k: int = 5) -> list[str]:
        """Query documents from ChromaDB."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results['documents'][0] if results['documents'] else []

    async def clear(self) -> None:
        """Clear all documents."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )
```

## Configuration Updates

### Add to `app/config.py`

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # Vector store configuration
    vector_store_type: str = Field(
        default="tfidf",  # options: tfidf, sentence-transformers, openai, chroma
        description="Vector store implementation to use"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for sentence-transformers"
    )
    openai_embedding_api_key: str | None = Field(
        default=None,
        description="API key for OpenAI embeddings",
        repr=False
    )
```

### Update `app/main.py`

```python
def _create_vector_store(settings: Settings) -> VectorStore:
    """Factory function to create appropriate vector store."""
    if settings.vector_store_type == "sentence-transformers":
        from app.stores.embedding_vector_store import EmbeddingVectorStore
        return EmbeddingVectorStore(model_name=settings.embedding_model)

    elif settings.vector_store_type == "openai":
        from app.stores.openai_vector_store import OpenAIVectorStore
        if not settings.openai_embedding_api_key:
            raise ValueError("OPENAI_EMBEDDING_API_KEY required for OpenAI embeddings")
        return OpenAIVectorStore(api_key=settings.openai_embedding_api_key)

    elif settings.vector_store_type == "chroma":
        from app.stores.chroma_vector_store import ChromaVectorStore
        return ChromaVectorStore()

    else:  # default: tfidf
        from app.stores.vector_store import InMemoryVectorStore
        return InMemoryVectorStore()

# In lifespan function:
app.state.vector_store = _create_vector_store(app.state.settings)
```

## Migration Steps

1. **Choose Implementation**: Start with sentence-transformers for best balance
2. **Add Dependencies**: `uv add sentence-transformers torch`
3. **Create New Store**: Implement `EmbeddingVectorStore`
4. **Update Config**: Add vector store type configuration
5. **Test Locally**: Compare search quality with existing TF-IDF
6. **Gradual Rollout**: Use feature flag to switch between implementations
7. **Monitor Performance**: Track query latency and accuracy
8. **Remove Old Code**: After successful migration, remove TF-IDF implementation

## Performance Comparison

| Implementation        | Speed  | Quality    | Memory | Cost |
| --------------------- | ------ | ---------- | ------ | ---- |
| TF-IDF (current)      | ⚡⚡⚡ | ⭐⭐       | Low    | Free |
| Sentence Transformers | ⚡⚡   | ⭐⭐⭐⭐   | Medium | Free |
| OpenAI Embeddings     | ⚡     | ⭐⭐⭐⭐⭐ | Low    | $$$  |
| ChromaDB              | ⚡⚡   | ⭐⭐⭐⭐   | Medium | Free |

## Testing

```python
# tests/integration/test_embedding_search.py
import pytest

@pytest.mark.parametrize("vector_store_type", [
    "tfidf",
    "sentence-transformers",
    "openai",
    "chroma"
])
async def test_semantic_search_quality(vector_store_type):
    """Test that semantic search finds relevant documents."""
    # Ingest documents
    await vector_store.add_documents([
        "Python is a programming language",
        "Machine learning uses neural networks",
        "FastAPI is a web framework"
    ])

    # Queries that should work better with embeddings
    results = await vector_store.query("coding in Python")
    assert "Python is a programming language" in results[0]

    results = await vector_store.query("AI and deep learning")
    assert "Machine learning uses neural networks" in results[0]
```

## References

- Sentence Transformers: https://www.sbert.net/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- ChromaDB: https://docs.trychroma.com/
