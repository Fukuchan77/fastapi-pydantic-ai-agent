"""Unit tests for workflow event classes.

Tests event field validation, defaults, and Pydantic constraints.
"""

import pytest
from pydantic import ValidationError

from app.workflows.events import EvaluateEvent
from app.workflows.events import SearchEvent
from app.workflows.events import SynthesizeEvent
from app.workflows.state import WorkflowState


class TestSearchEvent:
    """Tests for SearchEvent."""

    def test_search_event_with_query_only(self) -> None:
        """SearchEvent should accept query and default refined to False."""
        state = WorkflowState(query="test query", search_count=0)
        event = SearchEvent(query="test query", state=state)
        assert event.query == "test query"
        assert event.refined is False
        assert event.state == state

    def test_search_event_with_refined_true(self) -> None:
        """SearchEvent should accept refined=True."""
        state = WorkflowState(query="refined query", search_count=1)
        event = SearchEvent(query="refined query", refined=True, state=state)
        assert event.query == "refined query"
        assert event.refined is True
        assert event.state == state

    def test_search_event_requires_query(self) -> None:
        """SearchEvent should raise ValidationError if query is missing."""
        with pytest.raises(ValidationError) as exc_info:
            SearchEvent()  # type: ignore[call-arg]
        assert "query" in str(exc_info.value)

    def test_search_event_query_must_be_string(self) -> None:
        """SearchEvent query field must be a string."""
        with pytest.raises(ValidationError) as exc_info:
            SearchEvent(query=123)  # type: ignore[arg-type]
        assert "query" in str(exc_info.value)


class TestEvaluateEvent:
    """Tests for EvaluateEvent."""

    def test_evaluate_event_with_valid_data(self) -> None:
        """EvaluateEvent should accept query and chunks."""
        state = WorkflowState(query="test query", search_count=1)
        event = EvaluateEvent(
            query="test query",
            chunks=["chunk1", "chunk2", "chunk3"],
            state=state,
        )
        assert event.query == "test query"
        assert event.chunks == ["chunk1", "chunk2", "chunk3"]
        assert event.state == state

    def test_evaluate_event_with_empty_chunks(self) -> None:
        """EvaluateEvent should accept empty chunks list."""
        state = WorkflowState(query="test query", search_count=1)
        event = EvaluateEvent(query="test query", chunks=[], state=state)
        assert event.query == "test query"
        assert event.chunks == []
        assert event.state == state

    def test_evaluate_event_requires_query(self) -> None:
        """EvaluateEvent should raise ValidationError if query is missing."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluateEvent(chunks=["chunk1"])  # type: ignore[call-arg]
        assert "query" in str(exc_info.value)

    def test_evaluate_event_requires_chunks(self) -> None:
        """EvaluateEvent should raise ValidationError if chunks is missing."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluateEvent(query="test")  # type: ignore[call-arg]
        assert "chunks" in str(exc_info.value)

    def test_evaluate_event_chunks_must_be_list(self) -> None:
        """EvaluateEvent chunks field must be a list."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluateEvent(query="test", chunks="not a list")  # type: ignore[arg-type]
        assert "chunks" in str(exc_info.value)


class TestSynthesizeEvent:
    """Tests for SynthesizeEvent."""

    def test_synthesize_event_with_context_found(self) -> None:
        """SynthesizeEvent should accept all fields with context_found=True."""
        state = WorkflowState(query="test query", search_count=1)
        event = SynthesizeEvent(
            query="test query",
            chunks=["chunk1", "chunk2"],
            context_found=True,
            state=state,
        )
        assert event.query == "test query"
        assert event.chunks == ["chunk1", "chunk2"]
        assert event.context_found is True
        assert event.state == state

    def test_synthesize_event_with_no_context(self) -> None:
        """SynthesizeEvent should accept context_found=False with empty chunks."""
        state = WorkflowState(query="test query", search_count=2)
        event = SynthesizeEvent(
            query="test query",
            chunks=[],
            context_found=False,
            state=state,
        )
        assert event.query == "test query"
        assert event.chunks == []
        assert event.context_found is False
        assert event.state == state

    def test_synthesize_event_requires_query(self) -> None:
        """SynthesizeEvent should raise ValidationError if query is missing."""
        with pytest.raises(ValidationError) as exc_info:
            SynthesizeEvent(chunks=[], context_found=False)  # type: ignore[call-arg]
        assert "query" in str(exc_info.value)

    def test_synthesize_event_requires_chunks(self) -> None:
        """SynthesizeEvent should raise ValidationError if chunks is missing."""
        with pytest.raises(ValidationError) as exc_info:
            SynthesizeEvent(query="test", context_found=True)  # type: ignore[call-arg]
        assert "chunks" in str(exc_info.value)

    def test_synthesize_event_requires_context_found(self) -> None:
        """SynthesizeEvent should raise ValidationError if context_found is missing."""
        with pytest.raises(ValidationError) as exc_info:
            SynthesizeEvent(query="test", chunks=[])  # type: ignore[call-arg]
        assert "context_found" in str(exc_info.value)

    def test_synthesize_event_context_found_coerces_to_bool(self) -> None:
        """SynthesizeEvent context_found field coerces truthy strings to boolean."""
        # Pydantic coerces "yes" and other truthy strings to True by default
        state = WorkflowState(query="test", search_count=1)
        event = SynthesizeEvent(
            query="test",
            chunks=[],
            context_found="yes",  # type: ignore[arg-type]
            state=state,
        )
        assert event.context_found is True
        assert event.state == state
