"""Composed application Settings and cached accessor."""

from functools import cache

from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from app.config.llm import LLMSettingsMixin
from app.config.observability import ObservabilitySettingsMixin
from app.config.security import SecuritySettingsMixin
from app.config.store import StoreSettingsMixin


class Settings(
    LLMSettingsMixin,
    StoreSettingsMixin,
    SecuritySettingsMixin,
    ObservabilitySettingsMixin,
    BaseSettings,
):
    """Application settings loaded from environment variables.

    All settings are loaded from environment variables or a .env file.
    The Settings class uses Pydantic Settings for validation and type safety,
    ensuring configuration errors are caught at startup rather than runtime.
    Fields are composed from domain mixins (`app/config/llm.py`,
    `app/config/store.py`, `app/config/security.py`,
    `app/config/observability.py`) per `.sdd/steering/file-size-policy.md`.

    Security features:
        - API key strength validation (minimum 16 characters, no placeholders)
        - HTTPS enforcement for non-localhost URLs
        - SecretStr for sensitive fields (prevents accidental logging)
        - Extra field prohibition (catches typos in configuration)

    Required fields:
        api_key: API key for X-API-Key authentication (16+ characters)
        session_signing_key: Secret key for signing server-issued session ids (16+ characters)
        llm_model: LLM model identifier in "provider:model" format
            (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet-20241022")

    Optional fields:
        llm_api_key: API key for LLM provider (required for cloud providers,
            optional for local providers like Ollama)
        llm_base_url: Custom base URL for LLM provider (e.g., Azure OpenAI endpoint)
        embedding_model: Embedding model identifier for semantic search
            (e.g., "all-MiniLM-L6-v2")
        embedding_base_url: Custom base URL for embedding provider
            (e.g., Ollama embeddings endpoint)
        max_output_retries: Number of retries for Pydantic AI output validation (0-10)
        logfire_token: Pydantic Logfire token for observability (16+ characters)
        logfire_service_name: Service name for Logfire traces (default: "fastapi-pydantic-ai-agent")
        app_env: Application environment (development, staging, production)
        cors_origins: Allowed CORS origins (comma-separated or JSON array)
        http_timeout: HTTP client timeout in seconds (1-120)
        http_max_connections: Maximum HTTP connections in pool (1-500)
        enable_mock_tools: Enable mock tools for development (forbidden in production)

    Example:
        >>> # Load from environment variables
        >>> settings = get_settings()
        >>> print(settings.llm_model)
        "openai:gpt-4o"

        >>> # Access with validation
        >>> settings.api_key.get_secret_value()  # Extract secret value
        "your-secure-api-key"
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="forbid",
    )


@cache
def get_settings() -> Settings:
    """Get cached Settings instance.

    This function is cached to ensure the same Settings instance is reused
    throughout the application lifecycle. Settings are loaded once from
    environment variables or .env file.

    Returns:
        Settings: Cached application settings

    Raises:
        ValidationError: If required fields are missing or invalid
    """
    return Settings()
