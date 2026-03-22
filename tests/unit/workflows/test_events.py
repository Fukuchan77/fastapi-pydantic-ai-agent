"""Unit tests for workflow event classes."""

from llama_index.core.workflow import Event

from app.workflows.events import EvaluateEvent
from app.workflows.events import SearchEvent
from app.workflows.events import SynthesizeEvent
from app.workflows.state import WorkflowState


class TestSearchEvent:
    """Test suite for SearchEvent class."""

    def test_search_event_is_event_subclass(self) -> None:
        """SearchEvent should extend Event base class."""
        assert issubclass(SearchEvent, Event)

    def test_search_event_with_required_fields(self) -> None:
        """SearchEvent should be constructable with query and state."""
        state = WorkflowState(query="test query")

        event = SearchEvent(query="test query", state=state)

        assert event.query == "test query"
        assert event.state == state
        assert event.refined is False  # Default value

    def test_search_event_refined_defaults_to_false(self) -> None:
        """SearchEvent refined field should default to False."""
        state = WorkflowState(query="test query")

        event = SearchEvent(query="test query", state=state)

        assert event.refined is False

    def test_search_event_with_refined_true(self) -> None:
        """SearchEvent should accept refined=True for retry queries."""
        state = WorkflowState(query="test query", search_count=1)

        event = SearchEvent(query="refined query", refined=True, state=state)

        assert event.query == "refined query"
        assert event.refined is True
        assert event.state == state

    def test_search_event_query_must_be_string(self) -> None:
        """SearchEvent query field should be string type."""
        state = WorkflowState(query="test")
        event = SearchEvent(query="test query", state=state)

        assert isinstance(event.query, str)

    def test_search_event_state_is_workflow_state(self) -> None:
        """SearchEvent state field should be WorkflowState type."""
        state = WorkflowState(query="test query")
        event = SearchEvent(query="test query", state=state)

        assert isinstance(event.state, WorkflowState)


class TestEvaluateEvent:
    """Test suite for EvaluateEvent class."""

    def test_evaluate_event_is_event_subclass(self) -> None:
        """EvaluateEvent should extend Event base class."""
        assert issubclass(EvaluateEvent, Event)

    def test_evaluate_event_with_required_fields(self) -> None:
        """EvaluateEvent should be constructable with query, chunks, and state."""
        state = WorkflowState(query="test query")
        chunks = ["chunk1", "chunk2", "chunk3"]

        event = EvaluateEvent(query="test query", chunks=chunks, state=state)

        assert event.query == "test query"
        assert event.chunks == chunks
        assert event.state == state

    def test_evaluate_event_with_empty_chunks(self) -> None:
        """EvaluateEvent should accept empty chunks list."""
        state = WorkflowState(query="test query")

        event = EvaluateEvent(query="test query", chunks=[], state=state)

        assert event.chunks == []
        assert isinstance(event.chunks, list)

    def test_evaluate_event_chunks_are_strings(self) -> None:
        """EvaluateEvent chunks should be list of strings."""
        state = WorkflowState(query="test query")
        chunks = ["first chunk", "second chunk"]

        event = EvaluateEvent(query="test query", chunks=chunks, state=state)

        assert all(isinstance(chunk, str) for chunk in event.chunks)

    def test_evaluate_event_query_is_string(self) -> None:
        """EvaluateEvent query field should be string type."""
        state = WorkflowState(query="test")
        event = EvaluateEvent(query="test query", chunks=[], state=state)

        assert isinstance(event.query, str)


class TestSynthesizeEvent:
    """Test suite for SynthesizeEvent class."""

    def test_synthesize_event_is_event_subclass(self) -> None:
        """SynthesizeEvent should extend Event base class."""
        assert issubclass(SynthesizeEvent, Event)

    def test_synthesize_event_with_all_fields(self) -> None:
        """SynthesizeEvent should be constructable with all required fields."""
        state = WorkflowState(query="test query", context_found=True)
        chunks = ["relevant chunk 1", "relevant chunk 2"]

        event = SynthesizeEvent(
            query="test query",
            chunks=chunks,
            context_found=True,
            state=state,
        )

        assert event.query == "test query"
        assert event.chunks == chunks
        assert event.context_found is True
        assert event.state == state

    def test_synthesize_event_context_found_true(self) -> None:
        """SynthesizeEvent with context_found=True should have chunks."""
        state = WorkflowState(query="test query", context_found=True)
        chunks = ["chunk1", "chunk2"]

        event = SynthesizeEvent(
            query="test query",
            chunks=chunks,
            context_found=True,
            state=state,
        )

        assert event.context_found is True
        assert len(event.chunks) > 0

    def test_synthesize_event_context_found_false(self) -> None:
        """SynthesizeEvent with context_found=False should indicate retries exhausted."""
        state = WorkflowState(query="test query", search_count=3, max_retries=3)

        event = SynthesizeEvent(
            query="test query",
            chunks=[],
            context_found=False,
            state=state,
        )

        assert event.context_found is False
        assert event.chunks == []
        assert event.state.search_count >= event.state.max_retries

    def test_synthesize_event_empty_chunks_when_no_context(self) -> None:
        """SynthesizeEvent should have empty chunks when context_found=False."""
        state = WorkflowState(query="test query", context_found=False)

        event = SynthesizeEvent(
            query="test query",
            chunks=[],
            context_found=False,
            state=state,
        )

        assert event.chunks == []
        assert event.context_found is False

    def test_synthesize_event_context_found_is_bool(self) -> None:
        """SynthesizeEvent context_found field should be boolean type."""
        state = WorkflowState(query="test")
        event = SynthesizeEvent(
            query="test",
            chunks=[],
            context_found=True,
            state=state,
        )

        assert isinstance(event.context_found, bool)


class TestEventFieldTypes:
    """Test suite for event field type annotations."""

    def test_search_event_has_correct_field_types(self) -> None:
        """SearchEvent should have correct type annotations for all fields."""
        state = WorkflowState(query="test")
        event = SearchEvent(query="test", state=state)

        # Verify field types through runtime checks
        assert isinstance(event.query, str)
        assert isinstance(event.refined, bool)
        assert isinstance(event.state, WorkflowState)

    def test_evaluate_event_has_correct_field_types(self) -> None:
        """EvaluateEvent should have correct type annotations for all fields."""
        state = WorkflowState(query="test")
        event = EvaluateEvent(query="test", chunks=["a", "b"], state=state)

        # Verify field types through runtime checks
        assert isinstance(event.query, str)
        assert isinstance(event.chunks, list)
        assert isinstance(event.state, WorkflowState)

    def test_synthesize_event_has_correct_field_types(self) -> None:
        """SynthesizeEvent should have correct type annotations for all fields."""
        state = WorkflowState(query="test")
        event = SynthesizeEvent(
            query="test",
            chunks=["a"],
            context_found=True,
            state=state,
        )

        # Verify field types through runtime checks
        assert isinstance(event.query, str)
        assert isinstance(event.chunks, list)
        assert isinstance(event.context_found, bool)
        assert isinstance(event.state, WorkflowState)
