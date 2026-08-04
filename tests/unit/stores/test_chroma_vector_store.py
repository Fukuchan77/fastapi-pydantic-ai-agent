"""Unit tests for ChromaVectorStore ID generation and the generation counter.

Tests cover ID generation to prevent race conditions in multi-process deployments.
Replace counter-based IDs with UUID-based IDs.

`TestGeneration` is a boundary correction for task 3.1
(`.sdd/specs/002-review-roadmap-remediation/tasks.md`): 3.1's own boundary has
no test file, so its RED step lands here ahead of task 3.3's formal
parametrized conformance widening across all three implementors.
"""

import re
import uuid
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.stores.vector_store import ChromaVectorStore


def test_doc_counter_id_pattern():
    """Test that counter-based IDs follow the pattern 'doc_N'.

    This test documents the OLD behavior that causes race conditions.
    Counter-based IDs like "doc_0", "doc_1", "doc_2" cause collisions in
    multi-process deployments with shared persistent Chroma DB.
    """
    # Pattern for old counter-based IDs
    counter_pattern = re.compile(r"^doc_\d+$")

    # Examples of problematic counter-based IDs
    assert counter_pattern.match("doc_0")
    assert counter_pattern.match("doc_1")
    assert counter_pattern.match("doc_99")

    # UUID-based IDs should NOT match this pattern
    test_uuid = str(uuid.uuid4())
    assert not counter_pattern.match(test_uuid), "UUID should not match counter pattern"


def test_uuid_format_validation():
    """Test that UUID4 strings can be parsed as valid UUIDs.

    This test validates that UUID4 strings are valid UUIDs
    and can be used as document IDs in Chroma.
    """
    # Generate a UUID
    test_uuid_str = str(uuid.uuid4())

    # Should be parseable as UUID
    parsed = uuid.UUID(test_uuid_str)
    assert isinstance(parsed, uuid.UUID)

    # Should have UUID4 version
    assert parsed.version == 4

    # Should be 36 characters (including hyphens)
    assert len(test_uuid_str) == 36

    # Should not start with "doc_"
    assert not test_uuid_str.startswith("doc_")


def _make_store() -> ChromaVectorStore:
    """Construct a ChromaVectorStore with chromadb's network-touching parts mocked out.

    `ChromaVectorStore.__init__` calls
    `embedding_functions.SentenceTransformerEmbeddingFunction(...)`, which
    downloads a Hugging Face model on first use — the same network dependency
    task 1.2 gates `tests/integration/test_chroma_query_with_scores.py` behind.
    Mocking the client and embedding-function constructors keeps this test
    hermetic and independent of whether that model happens to be cached.
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    with (
        patch("chromadb.Client", return_value=mock_client),
        patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"),
    ):
        return ChromaVectorStore()


class TestGeneration:
    """Test the `generation` content-version counter (Req 2.1, 2.2, 2.4)."""

    def test_generation_starts_at_zero(self) -> None:
        """A freshly constructed store starts at generation 0."""
        store = _make_store()
        assert store.generation == 0

    @pytest.mark.asyncio
    async def test_generation_increments_after_successful_add(self) -> None:
        """A successful add_documents() call advances generation."""
        store = _make_store()
        await store.add_documents(["doc one"])
        assert store.generation == 1
        await store.add_documents(["doc two"])
        assert store.generation == 2

    @pytest.mark.asyncio
    async def test_generation_increments_after_clear(self) -> None:
        """A clear() call advances generation."""
        store = _make_store()
        await store.add_documents(["doc one"])
        await store.clear()
        assert store.generation == 2

    @pytest.mark.asyncio
    async def test_generation_unchanged_on_failed_add(self) -> None:
        """A failed add_documents() call (Chroma write error) leaves generation unchanged."""
        store = _make_store()
        store._collection.add.side_effect = RuntimeError("chroma write failed")
        with pytest.raises(RuntimeError, match="chroma write failed"):
            await store.add_documents(["doc one"])
        assert store.generation == 0
