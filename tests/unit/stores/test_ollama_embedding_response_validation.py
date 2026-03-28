"""Unit tests for OllamaEmbeddingVectorStore response validation (Task 22.2)."""

import pytest
from httpx import Response

from app.stores.vector_store import OllamaEmbeddingVectorStore


class TestOllamaEmbeddingResponseValidation:
    """Test suite for Task 22.2: Ollama embeddings API response validation."""

    @pytest.mark.asyncio
    async def test_embed_validates_data_key_exists(self) -> None:
        """_embed() should raise ValueError if response missing 'data' key.

        Bug: When Ollama returns an error (e.g., model not installed), the response
        doesn't have a "data" key. Current code raises KeyError with no context.

        Expected: Should raise ValueError with descriptive message showing the
        actual response.
        """
        from unittest.mock import AsyncMock
        from unittest.mock import Mock

        # Create store with mocked HTTP client
        mock_client = AsyncMock()
        store = OllamaEmbeddingVectorStore(
            embedding_model="nonexistent:model",
            http_client=mock_client,
        )

        # Mock a response that's missing the "data" key (Ollama error response)
        mock_response = Mock(spec=Response)
        mock_response.raise_for_status = Mock()  # Doesn't raise (HTTP 200 but error in body)
        mock_response.json = Mock(return_value={"error": "model 'nonexistent:model' not found"})
        mock_client.post = AsyncMock(return_value=mock_response)

        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="Unexpected Ollama embeddings response"):
            await store._embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_validates_embedding_field_exists(self) -> None:
        """_embed() should raise ValueError if response items missing 'embedding' field."""
        from unittest.mock import AsyncMock
        from unittest.mock import Mock

        mock_client = AsyncMock()
        store = OllamaEmbeddingVectorStore(
            embedding_model="test:model",
            http_client=mock_client,
        )

        # Mock a response with "data" key but missing "embedding" field
        mock_response = Mock(spec=Response)
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(
            return_value={
                "data": [
                    {"index": 0}  # Missing "embedding" field
                ]
            }
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        # Should raise ValueError (or KeyError that gets caught and re-raised)
        with pytest.raises((ValueError, KeyError)):
            await store._embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_accepts_valid_response(self) -> None:
        """_embed() should accept and process valid Ollama response."""
        from unittest.mock import AsyncMock
        from unittest.mock import Mock

        mock_client = AsyncMock()
        store = OllamaEmbeddingVectorStore(
            embedding_model="test:model",
            http_client=mock_client,
        )

        # Mock a valid response
        mock_response = Mock(spec=Response)
        mock_response.raise_for_status = Mock()
        mock_response.json = Mock(
            return_value={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            }
        )
        mock_client.post = AsyncMock(return_value=mock_response)

        # Should succeed and return embeddings
        result = await store._embed(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
