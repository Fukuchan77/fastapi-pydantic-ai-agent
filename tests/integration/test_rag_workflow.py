"""Integration tests for CorrectiveRAGWorkflow.

Tests the full RAG workflow with real stores and FunctionModel LLM.
No HTTP server involved - tests workflow logic directly.
"""

import pytest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from app.config import Settings
from app.stores.vector_store import InMemoryVectorStore
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Provide a fresh in-memory vector store for each test."""
    return InMemoryVectorStore()


@pytest.fixture
def settings() -> Settings:
    """Provide test settings with valid LLM configuration."""
    return Settings(
        api_key="test-api-key-12345",
        llm_model="openai:gpt-4",
        llm_api_key="test-llm-key-12345",
    )


@pytest.fixture
def workflow_with_relevant_eval(
    vector_store: InMemoryVectorStore,
    settings: Settings,
) -> CorrectiveRAGWorkflow:
    """Provide workflow that always evaluates chunks as relevant."""

    def mock_relevance_eval(messages: list, info: AgentInfo) -> ModelResponse:
        """Mock LLM that always returns 'relevant'."""
        return ModelResponse(parts=[TextPart(content="relevant")])

    # Create workflow with FunctionModel for evaluation
    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=FunctionModel(mock_relevance_eval),
    )
    return workflow


@pytest.fixture
def workflow_with_insufficient_eval(
    vector_store: InMemoryVectorStore,
    settings: Settings,
) -> CorrectiveRAGWorkflow:
    """Provide workflow that always evaluates chunks as insufficient."""

    def mock_relevance_eval(messages: list, info: AgentInfo) -> ModelResponse:
        """Mock LLM that always returns 'insufficient'."""
        return ModelResponse(parts=[TextPart(content="insufficient")])

    workflow = CorrectiveRAGWorkflow(
        vector_store=vector_store,
        llm_settings=settings,
        llm_model=FunctionModel(mock_relevance_eval),
    )
    return workflow


class TestCorrectiveRAGWorkflowWithContext:
    """Tests for RAG workflow when relevant context is found."""

    @pytest.mark.asyncio
    async def test_workflow_finds_relevant_context(
        self,
        workflow_with_relevant_eval: CorrectiveRAGWorkflow,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Workflow should return synthesized answer when relevant context found."""
        # Arrange: Ingest test documents
        await vector_store.add_documents(
            [
                "Python is a programming language",
                "FastAPI is a web framework for Python",
                "Pydantic is a data validation library",
            ]
        )

        # Act: Run workflow with matching query
        result = await workflow_with_relevant_eval.run(query="What is FastAPI?")

        # Assert: Check result (workflow.run() returns the result dict directly)
        assert result is not None
        assert isinstance(result, dict)
        # The result should contain answer and metadata
        assert "answer" in result
        assert "context_found" in result
        assert result["context_found"] is True
        assert "search_count" in result
        assert result["search_count"] >= 1

    @pytest.mark.asyncio
    async def test_workflow_with_empty_query_result(
        self,
        workflow_with_relevant_eval: CorrectiveRAGWorkflow,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Workflow should handle empty vector store gracefully."""
        # Arrange: Vector store is empty (no documents ingested)

        # Act: Run workflow
        result = await workflow_with_relevant_eval.run(query="What is FastAPI?")

        # Assert: Should return result even with no chunks
        assert result is not None
        assert isinstance(result, dict)


class TestCorrectiveRAGWorkflowRetries:
    """Tests for RAG workflow retry logic."""

    @pytest.mark.asyncio
    async def test_workflow_exhausts_retries_when_insufficient(
        self,
        workflow_with_insufficient_eval: CorrectiveRAGWorkflow,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Workflow should exhaust retries and return graceful response."""
        # Arrange: Ingest documents but mock LLM always says insufficient
        await vector_store.add_documents(
            [
                "Unrelated content about cats",
                "Unrelated content about dogs",
                "Unrelated content about birds",
            ]
        )

        # Act: Run workflow with query that won't match
        result = await workflow_with_insufficient_eval.run(
            query="What is quantum physics?", max_retries=2
        )

        # Assert: Should return graceful "no context found" response
        assert result is not None
        assert isinstance(result, dict)
        assert result["context_found"] is False
        # max_retries=2 means maximum 2 searches (not initial + 2 retries)
        assert result["search_count"] == 2

    @pytest.mark.asyncio
    async def test_workflow_respects_max_retries_parameter(
        self,
        workflow_with_insufficient_eval: CorrectiveRAGWorkflow,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Workflow should respect custom max_retries parameter."""
        # Arrange: Ingest documents
        await vector_store.add_documents(["Test document"])

        # Act: Run with custom max_retries
        result = await workflow_with_insufficient_eval.run(query="Test query", max_retries=5)

        # Assert: max_retries=5 means maximum 5 searches
        assert result is not None
        assert isinstance(result, dict)
        assert result["search_count"] == 5


class TestCorrectiveRAGWorkflowState:
    """Tests for workflow state management."""

    @pytest.mark.asyncio
    async def test_workflow_tracks_search_count(
        self,
        workflow_with_relevant_eval: CorrectiveRAGWorkflow,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Workflow should accurately track number of searches performed."""
        # Arrange
        await vector_store.add_documents(["Test content"])

        # Act
        result = await workflow_with_relevant_eval.run(query="Test query")

        # Assert
        assert result["search_count"] >= 1

    @pytest.mark.asyncio
    async def test_workflow_isolates_concurrent_runs(
        self,
        workflow_with_relevant_eval: CorrectiveRAGWorkflow,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Concurrent workflow runs should have isolated state."""
        # Arrange
        await vector_store.add_documents(["Test content"])

        # Act: Run two queries concurrently
        import asyncio

        results = await asyncio.gather(
            workflow_with_relevant_eval.run(query="Query 1"),
            workflow_with_relevant_eval.run(query="Query 2"),
        )

        # Assert: Both should succeed independently
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert all(r is not None for r in results)
