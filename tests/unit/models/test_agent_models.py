"""Unit tests for agent request/response models."""

import pytest
from pydantic import ValidationError

from app.models.agent import ChatOutput
from app.models.agent import ChatRequest
from app.models.agent import ChatResponse


class TestChatRequest:
    """Test suite for ChatRequest validation."""

    def test_chat_request_with_valid_message(self) -> None:
        """ChatRequest should accept valid message."""
        request = ChatRequest(message="Hello, AI!")
        assert request.message == "Hello, AI!"
        assert request.session_id is None

    def test_chat_request_with_session_id(self) -> None:
        """ChatRequest should accept optional session_id."""
        request = ChatRequest(message="Hello", session_id="session-123")
        assert request.message == "Hello"
        assert request.session_id == "session-123"

    def test_chat_request_message_is_required(self) -> None:
        """ChatRequest should require message field."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest()  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("message",) and e["type"] == "missing" for e in errors)

    def test_chat_request_message_min_length_one(self) -> None:
        """ChatRequest should reject empty message."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("message",) and "at least 1 character" in e["msg"].lower() for e in errors
        )

    def test_chat_request_message_max_length_32000(self) -> None:
        """ChatRequest should reject message longer than 32,000 characters."""
        long_message = "a" * 32_001
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=long_message)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("message",) and "at most 32000 characters" in e["msg"].lower()
            for e in errors
        )

    def test_chat_request_message_exactly_32000_chars(self) -> None:
        """ChatRequest should accept message with exactly 32,000 characters."""
        max_message = "a" * 32_000
        request = ChatRequest(message=max_message)
        assert len(request.message) == 32_000

    def test_chat_request_session_id_can_be_none(self) -> None:
        """ChatRequest should accept None for session_id."""
        request = ChatRequest(message="Hello", session_id=None)
        assert request.session_id is None

    def test_chat_request_session_id_is_optional(self) -> None:
        """ChatRequest should not require session_id."""
        request = ChatRequest(message="Hello")
        assert request.session_id is None

    def test_chat_request_message_whitespace_only_is_valid(self) -> None:
        """ChatRequest should accept whitespace-only messages (min_length=1)."""
        # min_length counts characters, not trimmed content
        request = ChatRequest(message=" ")
        assert request.message == " "


class TestChatOutput:
    """Test suite for ChatOutput model."""

    def test_chat_output_with_reply_only(self) -> None:
        """ChatOutput should work with reply field only."""
        output = ChatOutput(reply="Test response")
        assert output.reply == "Test response"
        assert output.tool_calls_made == 0

    def test_chat_output_with_tool_calls(self) -> None:
        """ChatOutput should accept tool_calls_made count."""
        output = ChatOutput(reply="Result", tool_calls_made=2)
        assert output.reply == "Result"
        assert output.tool_calls_made == 2

    def test_chat_output_reply_is_required(self) -> None:
        """ChatOutput should require reply field."""
        with pytest.raises(ValidationError) as exc_info:
            ChatOutput()  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("reply",) and e["type"] == "missing" for e in errors)

    def test_chat_output_tool_calls_defaults_to_zero(self) -> None:
        """ChatOutput should default tool_calls_made to 0."""
        output = ChatOutput(reply="Test")
        assert output.tool_calls_made == 0

    def test_chat_output_tool_calls_can_be_zero(self) -> None:
        """ChatOutput should accept 0 tool calls."""
        output = ChatOutput(reply="Test", tool_calls_made=0)
        assert output.tool_calls_made == 0


class TestChatResponse:
    """Test suite for ChatResponse model."""

    def test_chat_response_with_all_fields(self) -> None:
        """ChatResponse should accept all fields."""
        response = ChatResponse(
            reply="Test reply",
            session_id="session-123",
            tool_calls_made=1,
        )
        assert response.reply == "Test reply"
        assert response.session_id == "session-123"
        assert response.tool_calls_made == 1

    def test_chat_response_reply_is_required(self) -> None:
        """ChatResponse should require reply field."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse(session_id="123", tool_calls_made=0)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("reply",) and e["type"] == "missing" for e in errors)

    def test_chat_response_tool_calls_is_required(self) -> None:
        """ChatResponse should require tool_calls_made field."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse(reply="Test", session_id="123")  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("tool_calls_made",) and e["type"] == "missing" for e in errors)

    def test_chat_response_session_id_can_be_none(self) -> None:
        """ChatResponse should accept None for session_id."""
        response = ChatResponse(reply="Test", session_id=None, tool_calls_made=0)
        assert response.session_id is None

    def test_chat_response_with_no_session(self) -> None:
        """ChatResponse should work without session_id."""
        response = ChatResponse(reply="Test", session_id=None, tool_calls_made=0)
        assert response.reply == "Test"
        assert response.session_id is None
        assert response.tool_calls_made == 0


class TestAgentModelFieldTypes:
    """Test suite for field type validation."""

    def test_chat_request_message_must_be_string(self) -> None:
        """ChatRequest message field must be string."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=123)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("message",) for e in errors)

    def test_chat_request_session_id_must_be_string_or_none(self) -> None:
        """ChatRequest session_id must be string or None."""
        # Valid: string
        request1 = ChatRequest(message="Test", session_id="abc")
        assert request1.session_id == "abc"

        # Valid: None
        request2 = ChatRequest(message="Test", session_id=None)
        assert request2.session_id is None

        # Invalid: number
        with pytest.raises(ValidationError):
            ChatRequest(message="Test", session_id=123)  # type: ignore

    def test_chat_output_tool_calls_must_be_int(self) -> None:
        """ChatOutput tool_calls_made must be integer."""
        with pytest.raises(ValidationError) as exc_info:
            ChatOutput(reply="Test", tool_calls_made="not-an-int")  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("tool_calls_made",) for e in errors)

    def test_chat_response_tool_calls_must_be_int(self) -> None:
        """ChatResponse tool_calls_made must be integer."""
        with pytest.raises(ValidationError) as exc_info:
            ChatResponse(reply="Test", session_id=None, tool_calls_made="not-an-int")  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("tool_calls_made",) for e in errors)
