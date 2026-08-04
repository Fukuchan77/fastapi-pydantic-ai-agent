"""Unit tests for the VectorStore Protocol interface definition.

Split out of test_vector_store.py per `.sdd/steering/file-size-policy.md`.

`test_protocol_has_generation_member` and
`test_in_memory_vector_store_remains_identity_hashable` are a boundary
correction for task 3.1 (`.sdd/specs/002-review-roadmap-remediation/tasks.md`):
3.1's own boundary has no test file, so its RED step lands here ahead of task
3.3's formal parametrized widening across all three implementors.

`TestVectorStoreConformance` is that task 3.3 widening: one parametrized
conformance check exercising `InMemoryVectorStore`, `ChromaVectorStore`, and
`OllamaEmbeddingVectorStore` identically, covering the `generation`
content-version counter and the `query_with_scores` scored-query method
(Req 2.1, 2.4). It is additive to, not a replacement for, the per-implementor
generation/query_with_scores tests already living in
`test_in_memory_vector_store.py`, `test_chroma_vector_store.py`, and
`test_ollama_embedding_vector_store.py`.
"""

import weakref
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.models.rag import RetrievedHit
from app.stores.vector_store import ChromaVectorStore
from app.stores.vector_store import InMemoryVectorStore
from app.stores.vector_store import OllamaEmbeddingVectorStore
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

    def test_protocol_has_generation_member(self) -> None:
        """VectorStore Protocol defines a read-only generation member (Req 2.1, 2.2)."""
        assert hasattr(VectorStore, "generation")

    def test_in_memory_vector_store_implements_protocol(self) -> None:
        """InMemoryVectorStore implements all VectorStore Protocol methods."""
        store = InMemoryVectorStore()
        assert hasattr(store, "add_documents")
        assert hasattr(store, "query")
        assert hasattr(store, "clear")
        assert hasattr(store, "close")

    def test_in_memory_vector_store_remains_identity_hashable(self) -> None:
        """Adding `generation` must not break identity hashing (Req 2.1).

        `app/deps/workflow.py`'s `_workflow_cache` is a `WeakKeyDictionary`
        keyed on the store object itself, so any store must stay usable as a
        weak-ref key — which fails immediately if the store gained a custom
        `__eq__`/`__hash__` (e.g. via `@dataclass(eq=True)`).
        """
        store = InMemoryVectorStore()
        cache: weakref.WeakKeyDictionary[InMemoryVectorStore, str] = weakref.WeakKeyDictionary()
        cache[store] = "workflow"
        assert cache[store] == "workflow"


def _make_chroma_store() -> ChromaVectorStore:
    """Construct a ChromaVectorStore with chromadb replaced by an in-process fake.

    Mirrors `tests/unit/stores/test_chroma_vector_store.py`'s `_make_store()`,
    but additionally backs `add`/`count`/`query` with a small in-process fake
    (a plain dict of id -> text) so `query_with_scores()` behaves like the
    real backend for this file's cross-implementor conformance checks,
    without a real Chroma client or embedding model.
    """
    docs: dict[str, str] = {}

    def fake_add(documents: list[str], ids: list[str]) -> None:
        for doc_id, text in zip(ids, documents, strict=True):
            docs[doc_id] = text

    def fake_count() -> int:
        return len(docs)

    def fake_query(query_texts: list[str], n_results: int) -> dict:
        matched_ids = list(docs)[:n_results]
        return {
            "ids": [matched_ids],
            "documents": [[docs[doc_id] for doc_id in matched_ids]],
            "distances": [[0.0 for _ in matched_ids]],
        }

    mock_collection = MagicMock()
    mock_collection.add.side_effect = fake_add
    mock_collection.count.side_effect = fake_count
    mock_collection.query.side_effect = fake_query
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    with (
        patch("chromadb.Client", return_value=mock_client),
        patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"),
    ):
        return ChromaVectorStore()


def _make_ollama_store() -> OllamaEmbeddingVectorStore:
    """Construct an OllamaEmbeddingVectorStore with `_embed` mocked to avoid real network."""
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return store


def _make_failing_chroma_store() -> ChromaVectorStore:
    """A ChromaVectorStore whose next `add_documents()` call raises."""
    store = _make_chroma_store()
    store._collection.add.side_effect = RuntimeError("chroma write failed")
    return store


def _make_failing_ollama_store() -> OllamaEmbeddingVectorStore:
    """An OllamaEmbeddingVectorStore whose next `add_documents()` call raises."""
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._embed = AsyncMock(side_effect=RuntimeError("embedding failed"))
    return store


ALL_STORE_FACTORIES = [
    pytest.param(InMemoryVectorStore, id="in_memory"),
    pytest.param(_make_chroma_store, id="chroma"),
    pytest.param(_make_ollama_store, id="ollama"),
]

FAILING_STORE_FACTORIES = [
    pytest.param(lambda: InMemoryVectorStore(max_chunk_size=1), id="in_memory"),
    pytest.param(_make_failing_chroma_store, id="chroma"),
    pytest.param(_make_failing_ollama_store, id="ollama"),
]


class TestVectorStoreConformance:
    """One parametrized conformance check across all three VectorStore implementors.

    Widens `test_protocol_has_generation_member` and the per-implementor
    generation/query_with_scores tests into a single check that runs
    identically against `InMemoryVectorStore`, `ChromaVectorStore`, and
    `OllamaEmbeddingVectorStore` (task 3.3; Req 2.1, 2.4).
    """

    @pytest.mark.parametrize("make_store", ALL_STORE_FACTORIES)
    def test_implements_full_protocol_surface(self, make_store) -> None:
        """Every implementor exposes every VectorStore Protocol member."""
        store = make_store()
        assert hasattr(store, "add_documents")
        assert hasattr(store, "query")
        assert hasattr(store, "query_with_scores")
        assert hasattr(store, "clear")
        assert hasattr(store, "close")
        assert hasattr(store, "generation")

    @pytest.mark.parametrize("make_store", ALL_STORE_FACTORIES)
    def test_generation_starts_at_zero(self, make_store) -> None:
        """Every implementor starts at generation 0."""
        store = make_store()
        assert store.generation == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_store", ALL_STORE_FACTORIES)
    async def test_generation_increments_after_successful_add(self, make_store) -> None:
        """Req 2.1: a successful add advances generation, identically across implementors."""
        store = make_store()
        await store.add_documents(["doc one"])
        assert store.generation == 1
        await store.add_documents(["doc two"])
        assert store.generation == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_store", ALL_STORE_FACTORIES)
    async def test_generation_increments_after_clear(self, make_store) -> None:
        """A clear() call advances generation, identically across implementors."""
        store = make_store()
        await store.add_documents(["doc one"])
        await store.clear()
        assert store.generation == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_store", FAILING_STORE_FACTORIES)
    async def test_generation_unchanged_on_failed_add(self, make_store) -> None:
        """Req 2.4: a failed add leaves generation unchanged, identically across implementors."""
        store = make_store()
        with pytest.raises((ValueError, RuntimeError)):
            await store.add_documents(["doc"])
        assert store.generation == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_store", ALL_STORE_FACTORIES)
    async def test_query_with_scores_returns_empty_list_on_empty_corpus(
        self, make_store
    ) -> None:
        """The scored-query method returns [] on a fresh store, identically across implementors."""
        store = make_store()
        results = await store.query_with_scores("anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("make_store", ALL_STORE_FACTORIES)
    async def test_query_with_scores_returns_retrieved_hit_instances(self, make_store) -> None:
        """The scored-query method returns RetrievedHit instances, uniformly across implementors."""
        store = make_store()
        await store.add_documents(["doc one"])
        results = await store.query_with_scores("doc one", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedHit)
        assert results[0].text == "doc one"
