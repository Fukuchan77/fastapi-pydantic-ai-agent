"""Unit tests for RAG request/response models."""

import pytest
from pydantic import ValidationError

from app.models.rag import IngestRequest
from app.models.rag import IngestResponse
from app.models.rag import RAGQueryRequest
from app.models.rag import RAGQueryResponse


class TestRAGQueryRequest:
    """Test RAGQueryRequest model validation."""

    def test_valid_rag_query_request_minimal(self) -> None:
        """Test RAGQueryRequest with only required query field."""
        request = RAGQueryRequest(query="What is AI?")
        assert request.query == "What is AI?"
        assert request.max_retries == 3  # default value

    def test_valid_rag_query_request_with_max_retries(self) -> None:
        """Test RAGQueryRequest with custom max_retries."""
        request = RAGQueryRequest(query="What is AI?", max_retries=5)
        assert request.query == "What is AI?"
        assert request.max_retries == 5

    def test_query_min_length_constraint(self) -> None:
        """Test that empty query is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="")
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("query",) and "at least 1 character" in str(err["msg"]).lower()
            for err in errors
        )

    def test_query_max_length_constraint(self) -> None:
        """Test that query exceeding 8,000 characters is rejected."""
        long_query = "x" * 8_001
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query=long_query)
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("query",) and "at most 8000 character" in str(err["msg"]).lower()
            for err in errors
        )

    def test_query_at_max_length_boundary(self) -> None:
        """Test that query with exactly 8,000 characters is accepted."""
        boundary_query = "x" * 8_000
        request = RAGQueryRequest(query=boundary_query)
        assert len(request.query) == 8_000

    def test_max_retries_lower_bound(self) -> None:
        """Test that max_retries cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="test", max_retries=-1)
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("max_retries",)
            and "greater than or equal to 0" in str(err["msg"]).lower()
            for err in errors
        )

    def test_max_retries_upper_bound(self) -> None:
        """Test that max_retries cannot exceed 10."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="test", max_retries=11)
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("max_retries",) and "less than or equal to 10" in str(err["msg"]).lower()
            for err in errors
        )

    def test_max_retries_at_boundaries(self) -> None:
        """Test that max_retries at 0 and 10 are accepted."""
        request_min = RAGQueryRequest(query="test", max_retries=0)
        assert request_min.max_retries == 0

        request_max = RAGQueryRequest(query="test", max_retries=10)
        assert request_max.max_retries == 10

    def test_missing_query_field(self) -> None:
        """Test that query field is required."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("query",) for err in errors)


class TestRAGQueryResponse:
    """Test RAGQueryResponse model validation."""

    def test_valid_rag_query_response_with_context(self) -> None:
        """Test RAGQueryResponse when context was found."""
        response = RAGQueryResponse(
            answer="AI stands for Artificial Intelligence", context_found=True, search_count=1
        )
        assert response.answer == "AI stands for Artificial Intelligence"
        assert response.context_found is True
        assert response.search_count == 1

    def test_valid_rag_query_response_no_context(self) -> None:
        """Test RAGQueryResponse when no context was found."""
        response = RAGQueryResponse(
            answer="No relevant information found", context_found=False, search_count=3
        )
        assert response.answer == "No relevant information found"
        assert response.context_found is False
        assert response.search_count == 3

    def test_missing_required_fields(self) -> None:
        """Test that all fields are required."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryResponse()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        error_locs = {err["loc"] for err in errors}
        assert ("answer",) in error_locs
        assert ("context_found",) in error_locs
        assert ("search_count",) in error_locs


class TestIngestRequest:
    """Test IngestRequest model validation."""

    def test_valid_ingest_request_single_chunk(self) -> None:
        """Test IngestRequest with a single chunk."""
        request = IngestRequest(chunks=["This is a document chunk"])
        assert len(request.chunks) == 1
        assert request.chunks[0] == "This is a document chunk"

    def test_valid_ingest_request_multiple_chunks(self) -> None:
        """Test IngestRequest with multiple chunks."""
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        request = IngestRequest(chunks=chunks)
        assert len(request.chunks) == 3
        assert request.chunks == chunks

    def test_empty_chunks_list_rejected(self) -> None:
        """Test that empty chunks list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(chunks=[])
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("chunks",) and "at least 1" in str(err["msg"]).lower() for err in errors
        )

    def test_missing_chunks_field(self) -> None:
        """Test that chunks field is required."""
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("chunks",) for err in errors)


class TestIngestResponse:
    """Test IngestResponse model validation."""

    def test_valid_ingest_response(self) -> None:
        """Test IngestResponse with ingested count."""
        response = IngestResponse(ingested=5)
        assert response.ingested == 5

    def test_ingest_response_zero_count(self) -> None:
        """Test IngestResponse with zero count."""
        response = IngestResponse(ingested=0)
        assert response.ingested == 0

    def test_missing_ingested_field(self) -> None:
        """Test that ingested field is required."""
        with pytest.raises(ValidationError) as exc_info:
            IngestResponse()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("ingested",) for err in errors)
