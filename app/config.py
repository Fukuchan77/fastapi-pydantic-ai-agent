"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic import HttpUrl
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Required fields:
        api_key: API key for X-API-Key authentication
        llm_model: LLM model identifier (e.g., "openai:gpt-4o")

    Optional fields:
        llm_api_key: API key for LLM provider (optional for local providers like Ollama)
        llm_base_url: Custom base URL for LLM provider
        max_output_retries: Number of retries for Pydantic AI output validation
        logfire_token: Pydantic Logfire token for observability
        logfire_service_name: Service name for Logfire traces
    """

    # Required fields
    api_key: str = Field(
        ...,
        description="API key for X-API-Key authentication",
        repr=False,
    )
    llm_model: str = Field(..., description="LLM model identifier")

    # Optional fields
    llm_api_key: str | None = Field(
        default=None,
        description="API key for LLM provider (optional for local providers)",
        repr=False,
    )
    llm_base_url: HttpUrl | None = Field(
        default=None,
        description="Custom base URL for LLM provider (e.g., Ollama)",
    )
    max_output_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for Pydantic AI output validation",
    )
    logfire_token: str | None = Field(
        default=None,
        description="Pydantic Logfire token for observability",
        repr=False,
    )
    logfire_service_name: str = Field(
        default="fastapi-pydantic-ai-agent",
        description="Service name for Logfire traces",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache(maxsize=1)
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
