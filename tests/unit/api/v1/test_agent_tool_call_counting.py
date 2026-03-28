"""Unit tests for tool call counting logic in agent endpoints.

Task 16.31: Fix tool_calls_made counter to count using
isinstance(m, ModelResponse) and isinstance(p, ToolCallPart).
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.messages import UserPromptPart

from app.deps.auth import verify_api_key
from app.main import app


class TestToolCallCounting:
    """Test correct tool call counting logic."""

    @pytest.mark.asyncio
    async def test_counts_tool_calls_from_model_response_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tool calls should be counted from ToolCallPart instances in ModelResponse.

        RED PHASE: The current implementation counts messages with kind=="tool-call",
        but should count ToolCallPart instances within ModelResponse messages.
        A single ModelResponse can contain multiple ToolCallPart instances.
        """
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")

        # Create a ModelResponse with TWO ToolCallPart instances
        # This tests the edge case where a single response contains multiple tool calls
        response_with_two_tools = ModelResponse(
            parts=[
                ToolCallPart(tool_name="search_web", args={"query": "test"}),
                ToolCallPart(tool_name="calculate", args={"expression": "2+2"}),
            ]
        )

        # Create message history with multiple tool calls
        mock_result = MagicMock()
        mock_result.data = "Final answer"
        mock_result.all_messages = MagicMock(
            return_value=[
                # User request
                ModelRequest(parts=[UserPromptPart(content="Search and calculate")]),
                # Model response with TWO tool calls in ONE message
                response_with_two_tools,
                # Tool returns (two separate messages)
                ModelRequest(parts=[ToolReturnPart(tool_name="search_web", content="result1")]),
                ModelRequest(parts=[ToolReturnPart(tool_name="calculate", content="4")]),
                # Another model response with ONE tool call
                ModelResponse(parts=[ToolCallPart(tool_name="summarize", args={"text": "..."})]),
                # Tool return
                ModelRequest(parts=[ToolReturnPart(tool_name="summarize", content="summary")]),
                # Final response with text (no tool calls)
                ModelResponse(parts=[TextPart(content="Final answer")]),
            ]
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_session_store = AsyncMock()
        mock_session_store.get_history = AsyncMock(return_value=[])

        # Override authentication dependency
        app.dependency_overrides[verify_api_key] = lambda: None

        try:
            with patch("app.api.v1.agent.get_agent_deps") as mock_get_deps:
                mock_deps = MagicMock()
                mock_deps.session_store = mock_session_store
                mock_get_deps.side_effect = AsyncMock(return_value=mock_deps)

                with (
                    patch.object(app.state, "chat_agent", mock_agent, create=True),
                    patch.object(app.state, "http_client", AsyncMock(), create=True),
                    patch.object(app.state, "settings", MagicMock(), create=True),
                    patch.object(app.state, "session_store", mock_session_store, create=True),
                ):
                    client = TestClient(app)

                    response = client.post(
                        "/v1/agent/chat",
                        json={"message": "Search and calculate"},
                        headers={"X-API-Key": "test-api-key-12345"},
                    )

                    assert response.status_code == 200
                    data = response.json()

                    # Should count 3 tool calls total:
                    # - 2 from the first ModelResponse (search_web + calculate)
                    # - 1 from the second ModelResponse (summarize)
                    # NOT just 2 messages with tool calls
                    assert data["tool_calls_made"] == 3, (
                        f"Expected 3 tool calls but got {data['tool_calls_made']}"
                    )
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_ignores_tool_calls_in_non_model_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tool calls should only be counted from ModelResponse, not ModelRequest.

        RED PHASE: Ensure we filter by isinstance(m, ModelResponse) before
        looking at parts.
        """
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")

        mock_result = MagicMock()
        mock_result.data = "Result"
        mock_result.all_messages = MagicMock(
            return_value=[
                ModelRequest(parts=[UserPromptPart(content="Test")]),
                # ModelResponse with tool call (should be counted)
                ModelResponse(parts=[ToolCallPart(tool_name="tool1", args={})]),
                # ModelRequest with ToolReturnPart (should NOT be counted as tool call)
                ModelRequest(parts=[ToolReturnPart(tool_name="tool1", content="result")]),
                # Final response
                ModelResponse(parts=[TextPart(content="Result")]),
            ]
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_session_store = AsyncMock()
        mock_session_store.get_history = AsyncMock(return_value=[])

        # Override authentication dependency
        app.dependency_overrides[verify_api_key] = lambda: None

        try:
            with patch("app.api.v1.agent.get_agent_deps") as mock_get_deps:
                mock_deps = MagicMock()
                mock_deps.session_store = mock_session_store
                mock_get_deps.side_effect = AsyncMock(return_value=mock_deps)

                with (
                    patch.object(app.state, "chat_agent", mock_agent, create=True),
                    patch.object(app.state, "http_client", AsyncMock(), create=True),
                    patch.object(app.state, "settings", MagicMock(), create=True),
                    patch.object(app.state, "session_store", mock_session_store, create=True),
                ):
                    client = TestClient(app)

                    response = client.post(
                        "/v1/agent/chat",
                        json={"message": "Test"},
                        headers={"X-API-Key": "test-api-key-12345"},
                    )

                    assert response.status_code == 200
                    data = response.json()

                    # Should count only 1 tool call (from ModelResponse)
                    assert data["tool_calls_made"] == 1, (
                        f"Expected 1 tool call but got {data['tool_calls_made']}"
                    )
        finally:
            # Clean up override
            app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_counts_zero_when_no_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return 0 when there are no tool calls.

        RED PHASE: Verify edge case of no tool calls.
        """
        monkeypatch.setenv("API_KEY", "test-api-key-12345")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4")
        monkeypatch.setenv("LLM_API_KEY", "test-llm-key-12345")

        mock_result = MagicMock()
        mock_result.data = "Direct answer"
        mock_result.all_messages = MagicMock(
            return_value=[
                ModelRequest(parts=[UserPromptPart(content="Hi")]),
                ModelResponse(parts=[TextPart(content="Direct answer")]),
            ]
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        mock_session_store = AsyncMock()
        mock_session_store.get_history = AsyncMock(return_value=[])

        # Override authentication dependency
        app.dependency_overrides[verify_api_key] = lambda: None

        try:
            with patch("app.api.v1.agent.get_agent_deps") as mock_get_deps:
                mock_deps = MagicMock()
                mock_deps.session_store = mock_session_store
                mock_get_deps.side_effect = AsyncMock(return_value=mock_deps)

                with (
                    patch.object(app.state, "chat_agent", mock_agent, create=True),
                    patch.object(app.state, "http_client", AsyncMock(), create=True),
                    patch.object(app.state, "settings", MagicMock(), create=True),
                    patch.object(app.state, "session_store", mock_session_store, create=True),
                ):
                    client = TestClient(app)

                    response = client.post(
                        "/v1/agent/chat",
                        json={"message": "Hi"},
                        headers={"X-API-Key": "test-api-key-12345"},
                    )

                    assert response.status_code == 200
                    data = response.json()

                    # Should count 0 tool calls
                    assert data["tool_calls_made"] == 0, (
                        f"Expected 0 tool calls but got {data['tool_calls_made']}"
                    )
        finally:
            # Clean up override
            app.dependency_overrides.clear()
