"""Pluggable vector store interface and implementations for the RAG pattern.

Split by backend per `.sdd/steering/file-size-policy.md`
(`protocol.py`, `in_memory.py`, `chroma.py`, `ollama.py`); this package
re-exports the public surface so `from app.stores.vector_store import
VectorStore, InMemoryVectorStore, ChromaVectorStore, OllamaEmbeddingVectorStore`
keeps working unchanged.
"""

from app.stores.vector_store.chroma import ChromaVectorStore
from app.stores.vector_store.in_memory import InMemoryVectorStore
from app.stores.vector_store.ollama import OllamaEmbeddingVectorStore
from app.stores.vector_store.protocol import VectorStore


__all__ = ["ChromaVectorStore", "InMemoryVectorStore", "OllamaEmbeddingVectorStore", "VectorStore"]
