"""Application configuration using Pydantic Settings."""

from functools import cache

from pydantic import Field
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings are loaded from environment variables or a .env file.
    The Settings class uses Pydantic Settings for validation and type safety,
    ensuring configuration errors are caught at startup rather than runtime.

    Security features:
        - API key strength validation (minimum 16 characters, no placeholders)
        - HTTPS enforcement for non-localhost URLs
        - SecretStr for sensitive fields (prevents accidental logging)
        - Extra field prohibition (catches typos in configuration)

    Required fields:
        api_key: API key for X-API-Key authentication (16+ characters)
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

    # Required fields
    api_key: SecretStr = Field(
        ...,
        description="API key for X-API-Key authentication",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key_strength(cls, v: SecretStr) -> SecretStr:
        """Validate api_key is not a placeholder and meets minimum strength.

        Args:
            v: The api_key value to validate

        Returns:
            SecretStr: The validated api_key value

        Raises:
            ValueError: If api_key is a placeholder or too weak
        """
        # Extract the secret value for validation
        v_str = v.get_secret_value()
        # Strip whitespace for validation
        v_stripped = v_str.strip()

        # Reject empty or whitespace-only keys
        if not v_stripped:
            raise ValueError("api_key cannot be empty or whitespace only")

        # Define common placeholder values (case-insensitive)
        placeholders = {
            "your-api-key-here",
            "changeme",
            "change-me",
            "test-key",
            "example",
            "replace-me",
            "insert-key-here",
            "api-key-here",
        }

        # Check if the key (lowercased) is a known placeholder
        # This check must come BEFORE length check so placeholders are detected
        # even if they happen to be 16+ characters (e.g., "your-api-key-here" is 19 chars)
        if v_stripped.lower() in placeholders:
            raise ValueError(
                "api_key appears to be a placeholder value. "
                "Please set a strong API key with at least 16 characters."
            )

        # Minimum length check (16 characters minimum for security)
        if len(v_stripped) < 16:
            raise ValueError(
                f"api_key must be at least 16 characters long for security. "
                f"Current length: {len(v_stripped)}"
            )

        return v

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
        provider = parts[0].lower()  # Normalize to lowercase for case-insensitive matching
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

        # Return normalized llm_model with lowercase provider
        return f"{provider}:{model}"

    # Optional fields
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="API key for LLM provider (optional for local providers)",
    )

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key_strength(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate llm_api_key meets minimum strength requirements when provided.

        Args:
            v: The llm_api_key value to validate (can be None for local providers)

        Returns:
            SecretStr | None: The validated llm_api_key value

        Raises:
            ValueError: If llm_api_key is a placeholder or too weak
        """
        # None is allowed for local providers like Ollama
        if v is None:
            return v

        # Extract the secret value for validation
        v_str = v.get_secret_value()
        # Strip whitespace for validation
        v_stripped = v_str.strip()

        # Reject empty or whitespace-only keys
        if not v_stripped:
            raise ValueError("llm_api_key cannot be empty or whitespace only")

        # Define common placeholder values (case-insensitive)
        placeholders = {
            "your-api-key-here",
            "changeme",
            "change-me",
            "test-key",
            "example",
            "replace-me",
            "insert-key-here",
            "api-key-here",
        }

        # Check if the key (lowercased) is a known placeholder
        # This check must come BEFORE length check so placeholders are detected
        # even if they happen to be 16+ characters (e.g., "your-api-key-here" is 19 chars)
        if v_stripped.lower() in placeholders:
            raise ValueError(
                "llm_api_key appears to be a placeholder value. "
                "Please set a strong LLM API key with at least 16 characters."
            )

        # Minimum length check (16 characters minimum for security)
        if len(v_stripped) < 16:
            raise ValueError(
                f"llm_api_key must be at least 16 characters long for security. "
                f"Current length: {len(v_stripped)}"
            )

        return v

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

    embedding_model: str | None = Field(
        default=None,
        description="Embedding model identifier for semantic search (e.g., 'all-MiniLM-L6-v2')",
    )

    embedding_base_url: HttpUrl | None = Field(
        default=None,
        description="Custom base URL for embedding provider (e.g., Ollama embeddings endpoint)",
    )

    @field_validator("embedding_base_url")
    @classmethod
    def validate_embedding_base_url_https(cls, v: HttpUrl | None) -> HttpUrl | None:
        """Validate embedding_base_url uses HTTPS for non-localhost URLs.

        Args:
            v: The embedding_base_url value to validate

        Returns:
            HttpUrl | None: The validated embedding_base_url value

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
                "embedding_base_url must use HTTPS in production. "
                "HTTP is only allowed for localhost."
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
    http_max_connections: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of HTTP connections in the pool",
    )
    http_max_keepalive_connections: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of keep-alive HTTP connections in the pool",
    )
    trusted_proxies: list[str] = Field(
        default=[],
        description="List of trusted proxy IP addresses for X-Forwarded-For validation",
    )
    http_retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for HTTP client requests on transient failures",
    )
    http_retry_base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay in seconds for exponential backoff retries in HTTP client",
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
    llm_agent_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout in seconds for individual LLM agent execution (evaluation/synthesis)",
    )
    rag_workflow_timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Timeout in seconds for entire RAG workflow execution (all steps combined)",
    )
    rag_cache_ttl: int = Field(
        default=300,
        ge=0,
        le=3600,
        description="Time-to-live in seconds for RAG query result cache (0 disables cache)",
    )
    rag_cache_size: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of entries in RAG query result cache (LRU eviction)",
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis connection URL for session store (e.g., redis://localhost:6379/0). "
        "If not set, uses in-memory session store (suitable for development only)",
    )
    redis_session_store_enabled: bool = Field(
        default=False,
        description="Enable Redis-backed session store for multi-instance deployments. "
        "Requires redis_url to be set. If False, uses in-memory store",
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

    logfire_token: SecretStr | None = Field(
        default=None,
        description="Pydantic Logfire token for observability",
    )

    @field_validator("logfire_token")
    @classmethod
    def validate_logfire_token_strength(cls, v: SecretStr | None) -> SecretStr | None:
        """Validate logfire_token meets minimum strength requirements when provided.

        Args:
            v: The logfire_token value to validate (can be None)

        Returns:
            SecretStr | None: The validated logfire_token value

        Raises:
            ValueError: If logfire_token is a placeholder or too weak
        """
        # None is allowed for optional Logfire integration
        if v is None:
            return v

        # Extract the secret value for validation
        v_str = v.get_secret_value()
        # Strip whitespace for validation
        v_stripped = v_str.strip()

        # Reject empty or whitespace-only tokens
        if not v_stripped:
            raise ValueError("logfire_token cannot be empty or whitespace only")

        # Define common placeholder values (case-insensitive)
        placeholders = {
            "your-token-here",
            "your-logfire-token-here",
            "changeme",
            "change-me",
            "test-token",
            "example",
            "replace-me",
            "insert-token-here",
            "token-here",
        }

        # Check if the token (lowercased) is a known placeholder
        if v_stripped.lower() in placeholders:
            raise ValueError(
                "logfire_token appears to be a placeholder value. "
                "Please set a valid Logfire token with at least 16 characters."
            )

        # Minimum length check (16 characters minimum for security)
        if len(v_stripped) < 16:
            raise ValueError(
                f"logfire_token must be at least 16 characters long for security. "
                f"Current length: {len(v_stripped)}"
            )

        return v

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

    @model_validator(mode="after")
    def validate_keepalive_connections_limit(self) -> "Settings":
        """Validate that keepalive connections do not exceed total connections.

        Connection pool configuration validation.
        The number of keepalive connections in the pool cannot exceed the
        maximum total connections, as this would be a logical contradiction.

        Returns:
            Settings: The validated settings instance

        Raises:
            ValueError: If http_max_keepalive_connections > http_max_connections
        """
        if self.http_max_keepalive_connections > self.http_max_connections:
            raise ValueError(
                f"http_max_keepalive_connections ({self.http_max_keepalive_connections}) "
                f"cannot exceed http_max_connections ({self.http_max_connections}). "
                "Keepalive connections are a subset of total connections."
            )

        return self

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
