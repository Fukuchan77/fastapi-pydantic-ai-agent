"""Unit tests for SSE stream adapter in agent API."""

from app.api.v1.agent import DefaultSSEAdapter


class TestDefaultSSEAdapter:
    """Test DefaultSSEAdapter SSE format compliance."""

    def test_format_event_has_data_prefix(self) -> None:
        """format_event() must return SSE with ' ' prefix (Task 8.1.1)."""
        adapter = DefaultSSEAdapter()
        result = adapter.format_event("delta", "Hello")

        # SSE standard requires " " prefix
        assert result.startswith("data" + ": "), f"Expected ' ' prefix, got: {result!r}"
        assert '"type": "delta"' in result
        assert '"content": "Hello"' in result
        assert result.endswith("\n\n"), "SSE events must end with double newline"

    def test_format_done_has_data_prefix(self) -> None:
        """format_done() must return SSE with ' ' prefix."""
        adapter = DefaultSSEAdapter()
        result = adapter.format_done()

        assert result.startswith("data" + ": "), f"Expected ' ' prefix, got: {result!r}"
        assert '"type": "done"' in result
        assert result.endswith("\n\n")

    def test_format_error_has_data_prefix(self) -> None:
        """format_error() must return SSE with ' ' prefix."""
        adapter = DefaultSSEAdapter()
        result = adapter.format_error("Something went wrong")

        assert result.startswith("data" + ": "), f"Expected ' ' prefix, got: {result!r}"
        assert '"type": "error"' in result
        assert '"content": "Something went wrong"' in result
        assert result.endswith("\n\n")
