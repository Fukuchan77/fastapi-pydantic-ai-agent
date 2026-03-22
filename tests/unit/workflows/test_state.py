"""Unit tests for workflow state model."""

from pydantic import BaseModel

from app.workflows.state import WorkflowState


class TestWorkflowState:
    """Test suite for WorkflowState class."""

    def test_workflow_state_is_pydantic_model(self) -> None:
        """WorkflowState should extend Pydantic BaseModel."""
        assert issubclass(WorkflowState, BaseModel)

    def test_workflow_state_with_required_field_only(self) -> None:
        """WorkflowState should be constructable with just query field."""
        state = WorkflowState(query="test query")

        assert state.query == "test query"
        assert state.search_count == 0  # Default value
        assert state.max_retries == 3  # Default value
        assert state.final_answer is None  # Default value
        assert state.context_found is False  # Default value

    def test_workflow_state_query_is_required(self) -> None:
        """WorkflowState should require query field."""
        state = WorkflowState(query="required query")

        assert state.query == "required query"

    def test_workflow_state_search_count_defaults_to_zero(self) -> None:
        """WorkflowState search_count should default to 0."""
        state = WorkflowState(query="test")

        assert state.search_count == 0

    def test_workflow_state_max_retries_defaults_to_three(self) -> None:
        """WorkflowState max_retries should default to 3."""
        state = WorkflowState(query="test")

        assert state.max_retries == 3

    def test_workflow_state_final_answer_defaults_to_none(self) -> None:
        """WorkflowState final_answer should default to None."""
        state = WorkflowState(query="test")

        assert state.final_answer is None

    def test_workflow_state_context_found_defaults_to_false(self) -> None:
        """WorkflowState context_found should default to False."""
        state = WorkflowState(query="test")

        assert state.context_found is False

    def test_workflow_state_with_all_fields_custom(self) -> None:
        """WorkflowState should accept custom values for all fields."""
        state = WorkflowState(
            query="custom query",
            search_count=2,
            max_retries=5,
            final_answer="custom answer",
            context_found=True,
        )

        assert state.query == "custom query"
        assert state.search_count == 2
        assert state.max_retries == 5
        assert state.final_answer == "custom answer"
        assert state.context_found is True

    def test_workflow_state_can_be_mutated(self) -> None:
        """WorkflowState fields should be mutable after construction."""
        state = WorkflowState(query="initial query")

        # Mutate the state
        state.search_count = 1
        state.context_found = True
        state.final_answer = "synthesized answer"

        assert state.search_count == 1
        assert state.context_found is True
        assert state.final_answer == "synthesized answer"

    def test_workflow_state_search_count_increment(self) -> None:
        """WorkflowState search_count should be incrementable."""
        state = WorkflowState(query="test", search_count=0)

        state.search_count += 1
        assert state.search_count == 1

        state.search_count += 1
        assert state.search_count == 2

    def test_workflow_state_with_custom_max_retries(self) -> None:
        """WorkflowState should accept custom max_retries value."""
        state = WorkflowState(query="test", max_retries=5)

        assert state.max_retries == 5

    def test_workflow_state_retries_exhausted_check(self) -> None:
        """WorkflowState should support checking if retries are exhausted."""
        state = WorkflowState(query="test", search_count=3, max_retries=3)

        # Retries exhausted when search_count >= max_retries
        assert state.search_count >= state.max_retries

    def test_workflow_state_retries_remaining_check(self) -> None:
        """WorkflowState should support checking if retries remain."""
        state = WorkflowState(query="test", search_count=1, max_retries=3)

        # Retries remain when search_count < max_retries
        assert state.search_count < state.max_retries

    def test_workflow_state_final_answer_can_be_set(self) -> None:
        """WorkflowState final_answer should be settable to a string."""
        state = WorkflowState(query="test")

        state.final_answer = "This is the synthesized answer"

        assert state.final_answer == "This is the synthesized answer"
        assert isinstance(state.final_answer, str)

    def test_workflow_state_context_found_can_be_toggled(self) -> None:
        """WorkflowState context_found should be toggleable."""
        state = WorkflowState(query="test", context_found=False)

        state.context_found = True
        assert state.context_found is True

        state.context_found = False
        assert state.context_found is False


class TestWorkflowStateFieldTypes:
    """Test suite for WorkflowState field types."""

    def test_workflow_state_query_is_string(self) -> None:
        """WorkflowState query field should be string type."""
        state = WorkflowState(query="test query")

        assert isinstance(state.query, str)

    def test_workflow_state_search_count_is_int(self) -> None:
        """WorkflowState search_count field should be int type."""
        state = WorkflowState(query="test", search_count=5)

        assert isinstance(state.search_count, int)

    def test_workflow_state_max_retries_is_int(self) -> None:
        """WorkflowState max_retries field should be int type."""
        state = WorkflowState(query="test", max_retries=3)

        assert isinstance(state.max_retries, int)

    def test_workflow_state_final_answer_is_optional_string(self) -> None:
        """WorkflowState final_answer field should be optional string type."""
        state_none = WorkflowState(query="test")
        assert state_none.final_answer is None

        state_str = WorkflowState(query="test", final_answer="answer")
        assert isinstance(state_str.final_answer, str)

    def test_workflow_state_context_found_is_bool(self) -> None:
        """WorkflowState context_found field should be bool type."""
        state = WorkflowState(query="test", context_found=True)

        assert isinstance(state.context_found, bool)


class TestWorkflowStateUsageScenarios:
    """Test suite for common WorkflowState usage scenarios."""

    def test_workflow_state_initial_search_scenario(self) -> None:
        """WorkflowState should support initial search scenario."""
        # Initial state for a new query
        state = WorkflowState(query="What is Python?")

        assert state.query == "What is Python?"
        assert state.search_count == 0
        assert state.max_retries == 3
        assert state.final_answer is None
        assert state.context_found is False

    def test_workflow_state_retry_search_scenario(self) -> None:
        """WorkflowState should support retry search scenario."""
        # State after first search attempt failed
        state = WorkflowState(
            query="What is Python?",
            search_count=1,
            context_found=False,
        )

        assert state.search_count == 1
        assert state.search_count < state.max_retries  # Can retry
        assert state.context_found is False

    def test_workflow_state_successful_synthesis_scenario(self) -> None:
        """WorkflowState should support successful synthesis scenario."""
        # State after successful context retrieval and synthesis
        state = WorkflowState(
            query="What is Python?",
            search_count=1,
            context_found=True,
            final_answer="Python is a high-level programming language.",
        )

        assert state.context_found is True
        assert state.final_answer is not None
        assert len(state.final_answer) > 0

    def test_workflow_state_retries_exhausted_scenario(self) -> None:
        """WorkflowState should support retries exhausted scenario."""
        # State after all retries exhausted without finding context
        state = WorkflowState(
            query="What is Python?",
            search_count=3,
            max_retries=3,
            context_found=False,
            final_answer="No relevant context found.",
        )

        assert state.search_count >= state.max_retries
        assert state.context_found is False
        assert state.final_answer is not None  # Graceful fallback message
