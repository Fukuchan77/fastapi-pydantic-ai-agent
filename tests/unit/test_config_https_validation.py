"""Unit tests for app/config.py - llm_base_url HTTPS validation."""

import pytest
from pydantic import ValidationError


class TestLLMBaseURLHTTPSValidation:
    """Test HTTPS enforcement for llm_base_url field."""

    def test_llm_base_url_http_non_localhost_raises_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HTTP URLs with non-localhost hostnames must raise ValueError."""
        monkeypatch.setenv("API_KEY", "test-key-32-chars-minimum-length-ok")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        # HTTP with non-localhost hostname - should fail
        monkeypatch.setenv("LLM_BASE_URL", "http://example.com:8080/v1")

        from app.config import Settings

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        errors = exc_info.value.errors()
        assert any(error["loc"] == ("llm_base_url",) for error in errors)
        error_msg = str(errors[0]["msg"])
        assert "https" in error_msg.lower()
        assert "production" in error_msg.lower()

    def test_llm_base_url_https_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTPS URLs must be accepted."""
        monkeypatch.setenv("API_KEY", "test-key-32-chars-minimum-length-ok")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")

        from app.config import Settings

        settings = Settings()
        assert str(settings.llm_base_url) == "https://api.example.com/v1"

    def test_llm_base_url_http_localhost_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP localhost URLs must be accepted for development."""
        monkeypatch.setenv("API_KEY", "test-key-32-chars-minimum-length-ok")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")

        # Test localhost
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
        from app.config import Settings

        settings = Settings()
        assert str(settings.llm_base_url) == "http://localhost:11434/v1"

    def test_llm_base_url_http_127_0_0_1_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HTTP 127.0.0.1 URLs must be accepted for development."""
        monkeypatch.setenv("API_KEY", "test-key-32-chars-minimum-length-ok")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

        from app.config import Settings

        settings = Settings()
        assert str(settings.llm_base_url) == "http://127.0.0.1:11434/v1"

    def test_llm_base_url_none_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """None value (field not set) must be accepted."""
        monkeypatch.setenv("API_KEY", "test-key-32-chars-minimum-length-ok")
        monkeypatch.setenv("LLM_MODEL", "ollama:llama2")
        # Do not set LLM_BASE_URL

        from app.config import Settings

        settings = Settings()
        assert settings.llm_base_url is None
