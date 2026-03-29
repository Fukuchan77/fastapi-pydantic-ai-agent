"""Unit tests for WorkflowState max_retries upper bound validation ()."""

import pytest
from pydantic import ValidationError

from app.workflows.state import WorkflowState


class TestWorkflowStateMaxRetriesUpperBound:
    """Test suite for max_retries upper bound validation to prevent DoS."""

    def test_max_retries_accepts_value_of_10(self) -> None:
        """WorkflowState should accept max_retries value of 10 (upper bound)."""
        state = WorkflowState(query="test", max_retries=10)

        assert state.max_retries == 10

    def test_max_retries_accepts_value_below_10(self) -> None:
        """WorkflowState should accept max_retries values below 10."""
        state = WorkflowState(query="test", max_retries=5)

        assert state.max_retries == 5

    def test_max_retries_accepts_value_of_0(self) -> None:
        """WorkflowState should accept max_retries value of 0 (no retries)."""
        state = WorkflowState(query="test", max_retries=0)

        assert state.max_retries == 0

    def test_max_retries_rejects_value_of_11(self) -> None:
        """WorkflowState should reject max_retries value of 11 (exceeds upper bound)."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState(query="test", max_retries=11)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)
        assert errors[0]["type"] == "less_than_equal"

    def test_max_retries_rejects_large_value(self) -> None:
        """WorkflowState should reject large max_retries values to prevent DoS."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState(query="test", max_retries=100)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)
        assert errors[0]["type"] == "less_than_equal"

    def test_max_retries_rejects_negative_value(self) -> None:
        """WorkflowState should reject negative max_retries values."""
        with pytest.raises(ValidationError) as exc_info:
            WorkflowState(query="test", max_retries=-1)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("max_retries",)
        # Should fail on greater_than_equal (ge=0) constraint
        assert errors[0]["type"] == "greater_than_equal"

    def test_max_retries_default_value_is_within_bounds(self) -> None:
        """WorkflowState default max_retries (3) should be within valid bounds."""
        state = WorkflowState(query="test")

        assert state.max_retries == 3
        assert 0 <= state.max_retries <= 10
