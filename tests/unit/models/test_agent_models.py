"""Unit tests for agent request/response models."""

import pytest
from pydantic import ValidationError

from app.models.agent import ChatOutput
from app.models.agent import ChatRequest
from app.models.agent import ChatResponse


class TestChatRequest:
    """Test ChatRequest model validation."""

    def test_valid_chat_request_with_message_only(self) -> None:
        """Test ChatRequest with only required message field."""
        request = ChatRequest(message="Hello, AI!")
        assert request.message == "Hello, AI!"
        assert request.session_id is None

    def test_valid_chat_request_with_session_id(self) -> None:
        """Test ChatRequest with message and session_id."""
        request = ChatRequest(message="Hello, AI!", session_id="session-123")
        assert request.message == "Hello, AI!"
        assert request.session_id == "session-123"

    def test_message_min_length_constraint(self) -> None:
        """Test that empty message is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("message",) and "at least 1 character" in str(err["msg"]).lower()
            for err in errors
        )

    def test_message_max_length_constraint(self) -> None:
        """Test that message exceeding 32,000 characters is rejected."""
        long_message = "x" * 32_001
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=long_message)
        errors = exc_info.value.errors()
        assert any(
            err["loc"] == ("message",) and "at most 32000 character" in str(err["msg"]).lower()
            for err in errors
        )

    def test_message_at_max_length_boundary(self) -> None:
        """Test that message with exactly 32,000 characters is accepted."""
        boundary_message = "x" * 32_000
        request = ChatRequest(message=boundary_message)
        assert len(request.message) == 32_000

    def test_missing_message_field(self) -> None:
        """Test that message field is required."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("message",) for err in errors)

    def test_session_id_is_optional(self) -> None:
        """Test that session_id can be None."""
        request = ChatRequest(message="Test", session_id=None)
        assert request.session_id is None


class TestChatOutput:
    """Test ChatOutput model validation."""

    def test_valid_chat_output_minimal(self) -> None:
        """Test ChatOutput with only required reply field."""
        output = ChatOutput(reply="Here is the answer")
        assert output.reply == "Here is the answer"
        assert output.tool_calls_made == 0

    def test_valid_chat_output_with_tool_calls(self) -> None:
        """Test ChatOutput with tool_calls_made specified."""
        output = ChatOutput(reply="Answer after tools", tool_calls_made=3)
        assert output.reply == "Answer after tools"
        assert output.tool_calls_made == 3

    def test_tool_calls_made_defaults_to_zero(self) -> None:
        """Test that tool_calls_made has default value of 0."""
        output = ChatOutput(reply="Test")
        assert output.tool_calls_made == 0

    def test_missing_reply_field(self) -> None:
        """Test that reply field is required."""
        with pytest.raises(ValidationError) as exc_info:
            ChatOutput()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(err["loc"] == ("reply",) for err in errors)


class TestChatResponse:
    """Test ChatResponse model validation."""

    def test_valid_chat_response_minimal(self) -> None:
        """Test ChatResponse with required fields."""
        response = ChatResponse(reply="Answer", session_id=None, tool_calls_made=0)
        assert response.reply == "Answer"
        assert response.session_id is None
        assert response.tool_calls_made == 0

    def test_valid_chat_response_with_session(self) -> None:
        """Test ChatResponse with all fields populated."""
        response = ChatResponse(
            reply="Answer with context", session_id="session-456", tool_calls_made=2
        )
        assert response.reply == "Answer with context"
        assert response.session_id == "session-456"
        assert response.tool_calls_made == 2

    def test_missing_required_fields(self) -> None:
        """Test that reply and tool_calls_made are required."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse(session_id="test")  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        # Both reply and tool_calls_made should be in errors
        error_locs = {err["loc"] for err in errors}
        assert ("reply",) in error_locs
        assert ("tool_calls_made",) in error_locs
