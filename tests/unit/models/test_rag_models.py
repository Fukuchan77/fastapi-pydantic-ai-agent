"""Unit tests for RAG request/response models."""

import pytest
from pydantic import ValidationError

from app.models.rag import IngestRequest
from app.models.rag import IngestResponse
from app.models.rag import RAGQueryRequest
from app.models.rag import RAGQueryResponse


class TestIngestRequest:
    """Test suite for IngestRequest validation."""

    def test_ingest_request_with_valid_chunks(self) -> None:
        """IngestRequest should accept list of text chunks."""
        request = IngestRequest(chunks=["chunk1", "chunk2", "chunk3"])
        assert request.chunks == ["chunk1", "chunk2", "chunk3"]
        assert len(request.chunks) == 3

    def test_ingest_request_chunks_is_required(self) -> None:
        """IngestRequest should require chunks field."""
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest()  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("chunks",) and e["type"] == "missing" for e in errors)

    def test_ingest_request_chunks_min_length_one(self) -> None:
        """IngestRequest should reject empty chunks list."""
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(chunks=[])

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("chunks",) and "at least 1 item" in e["msg"].lower() for e in errors
        )

    def test_ingest_request_with_single_chunk(self) -> None:
        """IngestRequest should accept single chunk."""
        request = IngestRequest(chunks=["single chunk"])
        assert len(request.chunks) == 1
        assert request.chunks[0] == "single chunk"

    def test_ingest_request_chunks_must_be_strings(self) -> None:
        """IngestRequest chunks must be list of strings."""
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(chunks=[123, 456])  # type: ignore

        errors = exc_info.value.errors()
        # Should have validation errors for non-string items
        assert len(errors) > 0

    def test_ingest_request_chunks_can_contain_empty_strings(self) -> None:
        """IngestRequest should accept empty strings in chunks list."""
        # No constraint on individual chunk content, only list length
        request = IngestRequest(chunks=["", "valid", ""])
        assert len(request.chunks) == 3
        assert request.chunks[0] == ""


class TestIngestResponse:
    """Test suite for IngestResponse model."""

    def test_ingest_response_with_count(self) -> None:
        """IngestResponse should accept ingested count."""
        response = IngestResponse(ingested=10)
        assert response.ingested == 10

    def test_ingest_response_ingested_is_required(self) -> None:
        """IngestResponse should require ingested field."""
        with pytest.raises(ValidationError) as exc_info:
            IngestResponse()  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("ingested",) and e["type"] == "missing" for e in errors)

    def test_ingest_response_ingested_zero(self) -> None:
        """IngestResponse should accept 0 ingested documents."""
        response = IngestResponse(ingested=0)
        assert response.ingested == 0

    def test_ingest_response_ingested_must_be_int(self) -> None:
        """IngestResponse ingested field must be integer."""
        with pytest.raises(ValidationError) as exc_info:
            IngestResponse(ingested="not-an-int")  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("ingested",) for e in errors)


class TestRAGQueryRequest:
    """Test suite for RAGQueryRequest validation."""

    def test_rag_query_request_with_valid_query(self) -> None:
        """RAGQueryRequest should accept valid query."""
        request = RAGQueryRequest(query="What is AI?")
        assert request.query == "What is AI?"
        assert request.max_retries == 3  # default

    def test_rag_query_request_with_custom_max_retries(self) -> None:
        """RAGQueryRequest should accept custom max_retries."""
        request = RAGQueryRequest(query="Test query", max_retries=5)
        assert request.query == "Test query"
        assert request.max_retries == 5

    def test_rag_query_request_query_is_required(self) -> None:
        """RAGQueryRequest should require query field."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest()  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) and e["type"] == "missing" for e in errors)

    def test_rag_query_request_query_min_length_one(self) -> None:
        """RAGQueryRequest should reject empty query."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="")

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("query",) and "at least 1 character" in e["msg"].lower() for e in errors
        )

    def test_rag_query_request_query_max_length_8000(self) -> None:
        """RAGQueryRequest should reject query longer than 10,000 characters."""
        long_query = "a" * 10_001
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query=long_query)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("query",) and "at most 10000 characters" in e["msg"].lower()
            for e in errors
        )

    def test_rag_query_request_query_exactly_8000_chars(self) -> None:
        """RAGQueryRequest should accept query with exactly 10,000 characters."""
        max_query = "a" * 10_000
        request = RAGQueryRequest(query=max_query)
        assert len(request.query) == 10_000

    def test_rag_query_request_max_retries_defaults_to_three(self) -> None:
        """RAGQueryRequest should default max_retries to 3."""
        request = RAGQueryRequest(query="Test")
        assert request.max_retries == 3

    def test_rag_query_request_max_retries_min_zero(self) -> None:
        """RAGQueryRequest should reject max_retries of 0 (minimum is 1)."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="Test", max_retries=0)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("max_retries",) and "greater than or equal to 1" in e["msg"].lower()
            for e in errors
        )

    def test_rag_query_request_max_retries_cannot_be_negative(self) -> None:
        """RAGQueryRequest should reject negative max_retries."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="Test", max_retries=-1)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("max_retries",) and "greater than or equal to 1" in e["msg"].lower()
            for e in errors
        )

    def test_rag_query_request_max_retries_max_ten(self) -> None:
        """RAGQueryRequest should accept max_retries up to 10."""
        request = RAGQueryRequest(query="Test", max_retries=10)
        assert request.max_retries == 10

    def test_rag_query_request_max_retries_cannot_exceed_ten(self) -> None:
        """RAGQueryRequest should reject max_retries greater than 10."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="Test", max_retries=11)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("max_retries",) and "less than or equal to 10" in e["msg"].lower()
            for e in errors
        )


class TestRAGQueryResponse:
    """Test suite for RAGQueryResponse model."""

    def test_rag_query_response_with_all_fields(self) -> None:
        """RAGQueryResponse should accept all fields."""
        response = RAGQueryResponse(
            answer="Test answer",
            context_found=True,
            search_count=2,
        )
        assert response.answer == "Test answer"
        assert response.context_found is True
        assert response.search_count == 2

    def test_rag_query_response_answer_is_required(self) -> None:
        """RAGQueryResponse should require answer field."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryResponse(context_found=True, search_count=1)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("answer",) and e["type"] == "missing" for e in errors)

    def test_rag_query_response_context_found_is_required(self) -> None:
        """RAGQueryResponse should require context_found field."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryResponse(answer="Test", search_count=1)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("context_found",) and e["type"] == "missing" for e in errors)

    def test_rag_query_response_search_count_is_required(self) -> None:
        """RAGQueryResponse should require search_count field."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryResponse(answer="Test", context_found=True)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("search_count",) and e["type"] == "missing" for e in errors)

    def test_rag_query_response_with_no_context_found(self) -> None:
        """RAGQueryResponse should handle no context found scenario."""
        response = RAGQueryResponse(
            answer="No relevant information found.",
            context_found=False,
            search_count=3,
        )
        assert response.context_found is False
        assert response.search_count == 3

    def test_rag_query_response_search_count_zero(self) -> None:
        """RAGQueryResponse should accept search_count of 0."""
        response = RAGQueryResponse(
            answer="Test",
            context_found=True,
            search_count=0,
        )
        assert response.search_count == 0


class TestRAGModelFieldTypes:
    """Test suite for field type validation."""

    def test_rag_query_request_query_must_be_string(self) -> None:
        """RAGQueryRequest query field must be string."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query=123)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("query",) for e in errors)

    def test_rag_query_request_max_retries_must_be_int(self) -> None:
        """RAGQueryRequest max_retries must be integer."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryRequest(query="Test", max_retries="not-an-int")  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_retries",) for e in errors)

    def test_rag_query_response_context_found_must_be_bool(self) -> None:
        """RAGQueryResponse context_found must be boolean."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryResponse(answer="Test", context_found="not-bool", search_count=1)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("context_found",) for e in errors)

    def test_rag_query_response_search_count_must_be_int(self) -> None:
        """RAGQueryResponse search_count must be integer."""
        with pytest.raises(ValidationError) as exc_info:
            RAGQueryResponse(answer="Test", context_found=True, search_count="not-int")  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("search_count",) for e in errors)
