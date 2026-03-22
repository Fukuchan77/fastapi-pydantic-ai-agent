"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import Field
from pydantic import HttpUrl
from pydantic import field_validator
from pydantic import model_validator
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

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model_format(cls, v: str) -> str:
        """Validate llm_model follows 'provider:model' format with valid provider.

        Args:
            v: The llm_model value to validate

        Returns:
            str: The validated llm_model value

        Raises:
            ValueError: If format is invalid or provider is not allowed
        """
        allowed_providers = ["openai", "anthropic", "ollama", "groq"]

        # Check if colon exists
        if ":" not in v:
            raise ValueError(f"llm_model must follow 'provider:model' format, got: {v}")

        # Split into provider and model
        parts = v.split(":", 1)
        provider = parts[0]
        model = parts[1] if len(parts) > 1 else ""

        # Check provider is not empty
        if not provider:
            raise ValueError(
                f"llm_model provider cannot be empty. Must be one of {allowed_providers}"
            )

        # Check model name is not empty
        if not model:
            raise ValueError("llm_model model name cannot be empty. Format: 'provider:model'")

        # Check provider is in allowed list
        if provider not in allowed_providers:
            raise ValueError(
                f"llm_model provider must be one of {allowed_providers}, got: {provider}"
            )

        return v

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

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url_https(cls, v: HttpUrl | None) -> HttpUrl | None:
        """Validate llm_base_url uses HTTPS for non-localhost URLs.

        Args:
            v: The llm_base_url value to validate

        Returns:
            HttpUrl | None: The validated llm_base_url value

        Raises:
            ValueError: If HTTP is used for non-localhost URLs
        """
        if v is None:
            return v

        # Parse URL components
        scheme = v.scheme
        host = v.host

        # Allow HTTP only for localhost or 127.0.0.1
        if scheme == "http" and host not in ["localhost", "127.0.0.1"]:
            raise ValueError(
                "llm_base_url must use HTTPS in production. HTTP is only allowed for localhost."
            )

        return v

    max_output_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for Pydantic AI output validation",
    )
    app_env: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )
    enable_mock_tools: bool = Field(
        default=False,
        description="Enable mock tools (for development only, disable in production)",
    )
    http_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=120.0,  # Maximum 2 minutes to prevent resource exhaustion
        description="HTTP client timeout in seconds",
    )
    http_connect_timeout: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="HTTP client connection timeout in seconds",
    )
    llm_retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for LLM API calls",
    )
    llm_retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay in seconds for exponential backoff retries",
    )
    cors_origins: str | list[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed CORS origins (comma-separated or JSON array)",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse cors_origins from string or list.

        Supports:
        - JSON array string: '["https://app.example.com","https://admin.example.com"]'
        - Comma-separated string: "https://app.example.com,https://admin.example.com"
        - Single URL string: "https://app.example.com"
        - List: ["https://app.example.com"]

        Args:
            v: The cors_origins value to parse

        Returns:
            list[str]: Parsed list of origins
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Try to parse as JSON array first
            import json

            v_stripped = v.strip()
            if v_stripped.startswith("["):
                try:
                    parsed = json.loads(v_stripped)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            # Parse as comma-separated string
            if "," in v:
                return [origin.strip() for origin in v.split(",")]
            # Single origin string
            return [v.strip()]
        return v

    logfire_token: str | None = Field(
        default=None,
        description="Pydantic Logfire token for observability",
        repr=False,
    )
    logfire_service_name: str = Field(
        default="fastapi-pydantic-ai-agent",
        description="Service name for Logfire traces",
    )

    @model_validator(mode="after")
    def validate_cloud_provider_api_key(self) -> "Settings":
        """Validate that cloud providers have llm_api_key set.

        Cloud providers (openai, anthropic, groq) require an API key.
        Local providers (ollama) are exempt from this requirement.

        Returns:
            Settings: The validated settings instance

        Raises:
            ValueError: If a cloud provider is used without llm_api_key
        """
        # Extract provider from llm_model
        provider = self.llm_model.split(":", 1)[0]

        # Define cloud providers that require API key
        cloud_providers = ["openai", "anthropic", "groq"]

        # Check if this is a cloud provider without an API key
        if provider in cloud_providers and self.llm_api_key is None:
            raise ValueError(
                f"llm_api_key is required when using cloud provider '{provider}'. "
                f"Please set the LLM_API_KEY environment variable."
            )

        return self

    @model_validator(mode="after")
    def validate_mock_tools_not_in_production(self) -> "Settings":
        """Validate that enable_mock_tools is not enabled in production.

        Mock tools should only be enabled in development environments to prevent
        security vulnerabilities in production.

        Returns:
            Settings: The validated settings instance

        Raises:
            ValueError: If enable_mock_tools is True and app_env is "production"
        """
        if self.enable_mock_tools and self.app_env == "production":
            raise ValueError(
                "enable_mock_tools cannot be enabled in production environment. "
                "This is a security risk. Set ENABLE_MOCK_TOOLS=false or "
                "change APP_ENV to 'development' or 'staging'."
            )

        return self

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
