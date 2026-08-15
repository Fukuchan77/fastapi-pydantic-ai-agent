"""Unit tests for OllamaEmbeddingVectorStore.

Tests cover resource management, error handling, and basic functionality.
"""

import pytest
from httpx import AsyncClient

from app.models.rag import RetrievedHit
from app.stores.vector_store import OllamaEmbeddingVectorStore


@pytest.mark.asyncio
async def test_close_method_closes_internal_http_client():
    """Test that close() method properly closes internally created HTTP client.

    When OllamaEmbeddingVectorStore creates its own AsyncClient
    (http_client=None), calling close() should close the internal client
    to prevent resource leaks.
    """
    # Create store without providing http_client (it will create its own)
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Verify close method exists and is callable
    assert hasattr(store, "close"), "OllamaEmbeddingVectorStore should have close() method"
    assert callable(store.close), "close() should be callable"

    # Call close() - should not raise an exception
    await store.close()

    # Verify the internal client is closed by attempting to use it
    # A closed client should raise an exception
    with pytest.raises(RuntimeError, match="closed"):
        await store._http_client.get("http://test.com")


@pytest.mark.asyncio
async def test_close_method_with_external_http_client():
    """Test that close() does not close externally provided HTTP client.

    When an external HTTP client is provided, close() should
    NOT close it (caller is responsible for lifecycle management).
    """
    # Create an external client
    external_client = AsyncClient()

    # Create store with external client
    store = OllamaEmbeddingVectorStore(
        embedding_model="test-model",
        http_client=external_client,
    )

    # Call close() on store
    await store.close()

    # External client should still be usable (not closed)
    # This should NOT raise an exception
    try:
        # Just checking that the client is still open
        assert not external_client.is_closed
    finally:
        # Clean up the external client ourselves
        await external_client.aclose()


@pytest.mark.asyncio
async def test_add_documents_calls_embed():
    """Test that add_documents calls _embed to generate embeddings.

    Verifies that add_documents properly calls the Ollama API
    to generate embeddings for the provided documents.
    """
    from unittest.mock import AsyncMock
    from unittest.mock import patch

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Mock the _embed method to return test embeddings
    mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch.object(store, "_embed", new=AsyncMock(return_value=mock_embeddings)):
        # Add documents
        await store.add_documents(["doc1", "doc2"])

        # Verify _embed was called with correct arguments
        store._embed.assert_called_once_with(["doc1", "doc2"])

    # Verify documents and embeddings were stored
    assert len(store._documents) == 2
    assert len(store._embeddings) == 2
    assert store._documents == ["doc1", "doc2"]
    assert store._embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    await store.close()


@pytest.mark.asyncio
async def test_query_returns_empty_list_on_empty_corpus():
    """Test that query returns empty list when no documents are stored.

    Verifies graceful handling of queries against empty corpus.
    """
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Query empty store
    results = await store.query("test query", top_k=5)

    # Should return empty list
    assert results == []

    await store.close()


@pytest.mark.asyncio
async def test_query_returns_empty_list_on_empty_query_string():
    """Test that query returns empty list when query string is empty.

    Verifies handling of empty or whitespace-only queries.
    """
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Add a document (so corpus is not empty)
    store._documents = ["doc1"]
    store._embeddings = [[0.1, 0.2, 0.3]]

    # Query with empty string
    results = await store.query("", top_k=5)
    assert results == []

    # Query with whitespace only
    results = await store.query("   ", top_k=5)
    assert results == []

    await store.close()


@pytest.mark.asyncio
async def test_query_top_k_validation():
    """Test that query raises ValueError for invalid top_k values.

    Verifies input validation for top_k parameter.
    """
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # top_k < 1 should raise ValueError
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await store.query("test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await store.query("test", top_k=-1)

    await store.close()


@pytest.mark.asyncio
async def test_clear_resets_state():
    """Test that clear() properly resets store state.

    Verifies that clear() removes all documents and embeddings.
    """
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Manually add some data (bypass API call)
    store._documents = ["doc1", "doc2", "doc3"]
    store._embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    # Verify data is present
    assert len(store._documents) == 3
    assert len(store._embeddings) == 3

    # Clear the store
    await store.clear()

    # Verify state is reset
    assert len(store._documents) == 0
    assert len(store._embeddings) == 0
    assert store._documents == []
    assert store._embeddings == []

    await store.close()


@pytest.mark.asyncio
async def test_embed_malformed_response_raises_error():
    """Test that _embed raises ValueError for malformed API responses.

    & 22.2: Verifies error handling when Ollama API returns
    unexpected response format (missing 'data' key or 'embedding' field).
    """
    from unittest.mock import AsyncMock
    from unittest.mock import Mock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Mock the HTTP response with malformed data (missing 'data' key)
    mock_response = Mock()
    mock_response.json.return_value = {"error": "Model not found"}  # Missing 'data' key
    mock_response.raise_for_status = Mock()

    # Mock the http_client.post method
    store._http_client.post = AsyncMock(return_value=mock_response)

    # Should raise ValueError with descriptive message
    with pytest.raises(ValueError, match="Unexpected Ollama embeddings response"):
        await store.add_documents(["test doc"])

    await store.close()


@pytest.mark.asyncio
async def test_embed_response_with_missing_embedding_field():
    """Test that _embed raises ValueError when response items lack 'embedding' field.

    Verifies that per-item validation raises descriptive error
    when Ollama returns items without 'embedding' key (e.g., model loading error).
    """
    from unittest.mock import AsyncMock
    from unittest.mock import Mock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    # Mock the HTTP response with data missing 'embedding' field
    mock_response = Mock()
    mock_response.json.return_value = {
        "data": [
            {"index": 0},  # Missing 'embedding' field
        ]
    }
    mock_response.raise_for_status = Mock()

    # Mock the http_client.post method
    store._http_client.post = AsyncMock(return_value=mock_response)

    # Should raise ValueError with descriptive message about missing 'embedding' key
    with pytest.raises(ValueError, match="Missing 'embedding' in response item"):
        await store.add_documents(["test doc"])

    await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_returns_retrieved_hit_instances():
    """query_with_scores returns a list of RetrievedHit with the ollama source."""
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._documents = ["doc one", "doc two"]
    store._embeddings = [[1.0, 0.0], [0.0, 1.0]]

    from unittest.mock import AsyncMock

    store._embed = AsyncMock(return_value=[[1.0, 0.0]])

    results = await store.query_with_scores("doc one", top_k=2)

    assert len(results) == 2
    assert all(isinstance(hit, RetrievedHit) for hit in results)
    assert results[0].chunk_id == "ollama::0000"
    assert results[0].text == "doc one"
    assert results[0].score > results[1].score

    await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_chunk_id_is_stable_across_repeated_queries():
    """The same document keeps the same chunk_id across repeated queries."""
    from unittest.mock import AsyncMock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._documents = ["doc one", "doc two", "doc three"]
    store._embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    store._embed = AsyncMock(return_value=[[1.0, 0.0]])

    first_pass = await store.query_with_scores("doc one", top_k=3)
    second_pass = await store.query_with_scores("doc one", top_k=3)

    first_ids = {hit.text: hit.chunk_id for hit in first_pass}
    second_ids = {hit.text: hit.chunk_id for hit in second_pass}
    assert first_ids == second_ids
    assert first_ids["doc one"] == "ollama::0000"
    assert first_ids["doc two"] == "ollama::0001"
    assert first_ids["doc three"] == "ollama::0002"

    await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_returns_empty_list_on_empty_corpus():
    """query_with_scores returns empty list when no documents are stored."""
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    results = await store.query_with_scores("test query", top_k=5)

    assert results == []

    await store.close()


@pytest.mark.asyncio
async def test_query_with_scores_top_k_validation():
    """query_with_scores raises ValueError for invalid top_k values, matching query()."""
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")

    with pytest.raises(ValueError, match="top_k must be at least 1"):
        await store.query_with_scores("test", top_k=0)

    await store.close()


@pytest.mark.asyncio
async def test_legacy_query_contract_unchanged_after_query_with_scores_added():
    """query() keeps returning bare strings after query_with_scores is added."""
    from unittest.mock import AsyncMock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._documents = ["doc one"]
    store._embeddings = [[1.0, 0.0]]
    store._embed = AsyncMock(return_value=[[1.0, 0.0]])

    results = await store.query("doc one", top_k=1)

    assert results == ["doc one"]


# Generation content-version counter (Req 2.1, 2.2, 2.4). Boundary correction
# for task 3.1 (`.sdd/specs/002-review-roadmap-remediation/tasks.md`): 3.1's
# own boundary has no test file, so this section is the RED step ahead of
# task 3.3's formal parametrized conformance widening.


def test_generation_starts_at_zero():
    """A freshly constructed store starts at generation 0."""
    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    assert store.generation == 0


@pytest.mark.asyncio
async def test_generation_increments_after_successful_add():
    """A successful add_documents() call advances generation."""
    from unittest.mock import AsyncMock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._embed = AsyncMock(return_value=[[0.1, 0.2]])

    await store.add_documents(["doc one"])
    assert store.generation == 1
    await store.add_documents(["doc two"])
    assert store.generation == 2

    await store.close()


@pytest.mark.asyncio
async def test_generation_increments_after_clear():
    """A clear() call advances generation."""
    from unittest.mock import AsyncMock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._embed = AsyncMock(return_value=[[0.1, 0.2]])

    await store.add_documents(["doc one"])
    await store.clear()
    assert store.generation == 2

    await store.close()


@pytest.mark.asyncio
async def test_generation_unchanged_on_failed_add():
    """A failed add_documents() call (embedding API error) leaves generation unchanged."""
    from unittest.mock import AsyncMock

    store = OllamaEmbeddingVectorStore(embedding_model="test-model")
    store._embed = AsyncMock(side_effect=ValueError("embedding API failed"))

    with pytest.raises(ValueError, match="embedding API failed"):
        await store.add_documents(["doc one"])
    assert store.generation == 0

    await store.close()

    await store.close()
