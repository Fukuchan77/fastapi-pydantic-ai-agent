"""E2E tests for RAG endpoints.

Tests the POST /v1/rag/ingest and POST /v1/rag/query endpoints through full HTTP stack.
These endpoints require authentication and use the Corrective RAG workflow.
"""

import pytest
from httpx import AsyncClient


class TestRAGEndpoints:
    """E2E tests for RAG ingest and query endpoints."""

    @pytest.mark.asyncio
    async def test_rag_ingest_endpoint(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """RAG ingest endpoint should accept and store documents."""
        # Arrange: Documents to ingest
        request_data = {
            "chunks": [
                "Python is a high-level programming language.",
                "FastAPI is a modern web framework for Python.",
                "Pydantic provides data validation using Python type hints.",
            ]
        }

        # Act: POST to ingest endpoint
        response = await client.post(
            "/v1/rag/ingest",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 200 OK
        assert response.status_code == 200, "Ingest endpoint should return 200 OK"

        # Assert: Response should have ingested count
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
        assert "ingested" in data, "Response should have 'ingested' field"
        assert data["ingested"] == 3, "Should report 3 documents ingested"

    @pytest.mark.asyncio
    async def test_rag_query_endpoint(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """RAG query endpoint should retrieve and answer questions."""
        # Arrange: First ingest some documents
        ingest_data = {
            "chunks": [
                "The capital of France is Paris.",
                "The Eiffel Tower is located in Paris, France.",
                "Paris is known for its art, fashion, and culture.",
            ]
        }
        await client.post("/v1/rag/ingest", json=ingest_data, headers=auth_headers)

        # Arrange: Query about the ingested content
        query_data = {"query": "What is the capital of France?"}

        # Act: POST to query endpoint
        response = await client.post(
            "/v1/rag/query",
            json=query_data,
            headers=auth_headers,
        )

        # Assert: Should return 200 OK
        assert response.status_code == 200, "Query endpoint should return 200 OK"

        # Assert: Response should match RAGQueryResponse schema
        data = response.json()
        assert isinstance(data, dict), "Response should be a JSON object"
        assert "answer" in data, "Response should have 'answer' field"
        assert "context_found" in data, "Response should have 'context_found' field"
        assert "search_count" in data, "Response should have 'search_count' field"

        # Assert: Should find context
        assert data["context_found"] is True, "Should find relevant context"
        assert isinstance(data["answer"], str), "Answer should be a string"
        assert len(data["answer"]) > 0, "Answer should be non-empty"
        assert data["search_count"] >= 1, "Should have performed at least one search"

    @pytest.mark.asyncio
    async def test_rag_query_without_context(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """RAG query should handle queries with no matching context gracefully."""
        # Arrange: Query about something not in the (empty) vector store
        query_data = {"query": "What is quantum computing?"}

        # Act: POST to query endpoint
        response = await client.post(
            "/v1/rag/query",
            json=query_data,
            headers=auth_headers,
        )

        # Assert: Should return 200 OK (graceful handling)
        assert response.status_code == 200, "Query should succeed even without context"

        # Assert: Should indicate no context found
        data = response.json()
        assert data["context_found"] is False, "Should indicate no context found"
        assert isinstance(data["answer"], str), "Should still provide an answer"
        assert len(data["answer"]) > 0, "Should provide graceful no-context message"

    @pytest.mark.asyncio
    async def test_rag_ingest_validates_chunks(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """RAG ingest should validate chunks field."""
        # Arrange: Request with empty chunks array
        request_data = {"chunks": []}

        # Act: POST with empty chunks
        response = await client.post(
            "/v1/rag/ingest",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 422 validation error
        assert response.status_code == 422, "Empty chunks should fail validation"

    @pytest.mark.asyncio
    async def test_rag_query_validates_query(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """RAG query should validate query field."""
        # Arrange: Request with empty query
        request_data = {"query": ""}

        # Act: POST with empty query
        response = await client.post(
            "/v1/rag/query",
            json=request_data,
            headers=auth_headers,
        )

        # Assert: Should return 422 validation error
        assert response.status_code == 422, "Empty query should fail validation"

    @pytest.mark.asyncio
    async def test_rag_query_with_max_retries(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """RAG query should respect max_retries parameter."""
        # Arrange: Query with custom max_retries
        query_data = {
            "query": "Test query",
            "max_retries": 2,
        }

        # Act: POST to query endpoint
        response = await client.post(
            "/v1/rag/query",
            json=query_data,
            headers=auth_headers,
        )

        # Assert: Should return 200 OK
        assert response.status_code == 200

        # Assert: Should respect max_retries limit
        data = response.json()
        assert data["search_count"] <= 2, "Should not exceed max_retries"

    @pytest.mark.asyncio
    async def test_rag_roundtrip(
        self,
        client: AsyncClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Complete RAG roundtrip: ingest documents then query them."""
        # Arrange: Ingest documents about a specific topic
        ingest_data = {
            "chunks": [
                "FastAPI is a modern, fast web framework for building APIs with Python.",
                "Pydantic AI is a Python agent framework built by the team behind Pydantic.",
                "LlamaIndex provides data framework for LLM applications.",
            ]
        }

        # Act: Ingest documents
        ingest_response = await client.post(
            "/v1/rag/ingest",
            json=ingest_data,
            headers=auth_headers,
        )

        # Assert: Ingest should succeed
        assert ingest_response.status_code == 200
        assert ingest_response.json()["ingested"] == 3

        # Act: Query the ingested documents
        query_data = {"query": "What is FastAPI?"}
        query_response = await client.post(
            "/v1/rag/query",
            json=query_data,
            headers=auth_headers,
        )

        # Assert: Query should succeed and find context
        assert query_response.status_code == 200
        query_result = query_response.json()
        assert query_result["context_found"] is True
        assert "FastAPI" in query_result["answer"] or "fast" in query_result["answer"].lower()
