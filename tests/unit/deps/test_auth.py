"""Unit tests for authentication dependency."""

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.deps.auth import verify_api_key
from app.models.errors import ErrorResponse


class TestVerifyApiKey:
    """Test suite for verify_api_key dependency."""

    @pytest.mark.asyncio
    async def test_missing_api_key_raises_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing X-API-Key header raises HTTPException with 401."""
        # Arrange
        monkeypatch.setenv("API_KEY", "test-secret-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123")
        settings = Settings()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key=None, settings=settings)

        assert exc_info.value.status_code == 401
        assert "detail" in exc_info.value.__dict__
        # The detail should be an ErrorResponse or dict with message field
        detail = exc_info.value.detail
        if isinstance(detail, dict):
            assert "message" in detail
            assert detail["message"] == "Unauthorized"
        else:
            # If it's an ErrorResponse model
            assert hasattr(detail, "message")
            assert detail.message == "Unauthorized"

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid X-API-Key value raises HTTPException with 401."""
        # Arrange
        monkeypatch.setenv("API_KEY", "correct-secret-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123")
        settings = Settings()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key="wrong-key", settings=settings)

        assert exc_info.value.status_code == 401
        detail = exc_info.value.detail
        if isinstance(detail, dict):
            assert detail["message"] == "Unauthorized"
        else:
            assert detail.message == "Unauthorized"

    @pytest.mark.asyncio
    async def test_valid_api_key_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that valid X-API-Key allows request to proceed."""
        # Arrange
        monkeypatch.setenv("API_KEY", "correct-secret-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123")
        settings = Settings()

        # Act
        result = await verify_api_key(api_key="correct-secret-key", settings=settings)

        # Assert - dependency should return None when successful
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_string_api_key_raises_401(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty string X-API-Key raises HTTPException with 401."""
        # Arrange
        monkeypatch.setenv("API_KEY", "test-secret-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123")
        settings = Settings()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key="", settings=settings)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_case_sensitive_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API key comparison is case-sensitive."""
        # Arrange
        monkeypatch.setenv("API_KEY", "TestKey123")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123")
        settings = Settings()

        # Act & Assert - wrong case should fail
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key="testkey123", settings=settings)

        assert exc_info.value.status_code == 401

        # Correct case should succeed
        result = await verify_api_key(api_key="TestKey123", settings=settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_error_response_has_correct_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that error response follows ErrorResponse model structure."""
        # Arrange
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
        monkeypatch.setenv("LLM_API_KEY", "sk-test123")
        settings = Settings()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(api_key=None, settings=settings)

        # Verify the detail can be converted to ErrorResponse model
        detail = exc_info.value.detail
        if isinstance(detail, dict):
            # Should be convertible to ErrorResponse
            error_response = ErrorResponse(**detail)
            assert error_response.message == "Unauthorized"
        else:
            # Already an ErrorResponse
            assert isinstance(detail, ErrorResponse)
            assert detail.message == "Unauthorized"
