"""Unit tests for OllamaEmbeddingVectorStore.

Tests cover resource management, error handling, and basic functionality.
"""

import pytest
from httpx import AsyncClient

from app.stores.vector_store import OllamaEmbeddingVectorStore


@pytest.mark.asyncio
async def test_close_method_closes_internal_http_client():
    """Test that close() method properly closes internally created HTTP client.

    Task 22.1: When OllamaEmbeddingVectorStore creates its own AsyncClient
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

    Task 22.1: When an external HTTP client is provided, close() should
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

    Task 22.4: Verifies that add_documents properly calls the Ollama API
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

    Task 22.4: Verifies graceful handling of queries against empty corpus.
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

    Task 22.4: Verifies handling of empty or whitespace-only queries.
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

    Task 22.4: Verifies input validation for top_k parameter.
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

    Task 22.4: Verifies that clear() removes all documents and embeddings.
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

    Task 22.4 & 22.2: Verifies error handling when Ollama API returns
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

    Task 23.3: Verifies that per-item validation raises descriptive error
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
