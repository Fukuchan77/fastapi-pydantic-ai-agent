"""Unit tests for WorkflowState Pydantic model.

Tests default values, field mutation, and validation constraints.
"""

import pytest
from pydantic import ValidationError

from app.workflows.state import WorkflowState


class TestWorkflowState:
    """Tests for WorkflowState model."""

    def test_workflow_state_with_query_only(self) -> None:
        """WorkflowState should apply default values when only query is provided."""
        state = WorkflowState(query="test query")
        assert state.query == "test query"
        assert state.search_count == 0
        assert state.max_retries == 3
        assert state.final_answer is None
        assert state.context_found is False

    def test_workflow_state_with_all_fields(self) -> None:
        """WorkflowState should accept all fields with custom values."""
        state = WorkflowState(
            query="test query",
            search_count=2,
            max_retries=5,
            final_answer="Test answer",
            context_found=True,
        )
        assert state.query == "test query"
        assert state.search_count == 2
        assert state.max_retries == 5
        assert state.final_answer == "Test answer"
        assert state.context_found is True

    def test_workflow_state_requires_query(self) -> None:
        """WorkflowState should raise ValidationError if query is missing."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState()  # type: ignore[call-arg]
        assert "query" in str(exc_info.value)

    def test_workflow_state_query_must_be_string(self) -> None:
        """WorkflowState query field must be a string."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState(query=123)  # type: ignore[arg-type]
        assert "query" in str(exc_info.value)

    def test_workflow_state_search_count_must_be_int(self) -> None:
        """WorkflowState search_count field must be an integer."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState(query="test", search_count="not an int")  # type: ignore[arg-type]
        assert "search_count" in str(exc_info.value)

    def test_workflow_state_max_retries_must_be_int(self) -> None:
        """WorkflowState max_retries field must be an integer."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState(query="test", max_retries="not an int")  # type: ignore[arg-type]
        assert "max_retries" in str(exc_info.value)

    def test_workflow_state_context_found_must_be_bool(self) -> None:
        """WorkflowState context_found field coerces truthy values to boolean."""
        # Pydantic coerces truthy strings to boolean
        state = WorkflowState(query="test", context_found="yes")  # type: ignore[arg-type]
        assert state.context_found is True

    def test_workflow_state_field_mutation(self) -> None:
        """WorkflowState fields should be mutable after creation."""
        state = WorkflowState(query="test query")

        # Mutate fields
        state.search_count = 1
        state.max_retries = 5
        state.final_answer = "New answer"
        state.context_found = True

        # Verify mutations
        assert state.search_count == 1
        assert state.max_retries == 5
        assert state.final_answer == "New answer"
        assert state.context_found is True

    def test_workflow_state_increment_search_count(self) -> None:
        """WorkflowState should support incrementing search_count."""
        state = WorkflowState(query="test query")
        assert state.search_count == 0

        state.search_count += 1
        assert state.search_count == 1

        state.search_count += 1
        assert state.search_count == 2

    def test_workflow_state_final_answer_can_be_none(self) -> None:
        """WorkflowState final_answer should accept None value."""
        state = WorkflowState(query="test", final_answer=None)
        assert state.final_answer is None

    def test_workflow_state_final_answer_can_be_string(self) -> None:
        """WorkflowState final_answer should accept string value."""
        state = WorkflowState(query="test", final_answer="Answer text")
        assert state.final_answer == "Answer text"
