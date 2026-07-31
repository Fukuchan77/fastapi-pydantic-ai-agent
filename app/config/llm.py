"""LLM provider, fallback chain, and guarded-agent settings."""

from typing import Self

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator


_ALLOWED_LLM_PROVIDERS = ["openai", "anthropic", "ollama", "groq"]


def _validate_provider_model_id(v: str, field_name: str) -> str:
    """Validate a single "provider:model" identifier.

    Shared by `llm_model` and each entry of `llm_fallback_models`.

    Args:
        v: The "provider:model" identifier to validate.
        field_name: The Settings field name, used in error messages.

    Returns:
        str: The validated identifier with a lowercase-normalized provider.

    Raises:
        ValueError: If the format is invalid or the provider is not allowed.
    """
    if ":" not in v:
        raise ValueError(f"{field_name} must follow 'provider:model' format, got: {v}")

    parts = v.split(":", 1)
    provider = parts[0].lower()
    model = parts[1] if len(parts) > 1 else ""

    if not provider:
        raise ValueError(
            f"{field_name} provider cannot be empty. Must be one of {_ALLOWED_LLM_PROVIDERS}"
        )
    if not model:
        raise ValueError(f"{field_name} model name cannot be empty. Format: 'provider:model'")
    if provider not in _ALLOWED_LLM_PROVIDERS:
        raise ValueError(
            f"{field_name} provider must be one of {_ALLOWED_LLM_PROVIDERS}, got: {provider}"
        )

    return f"{provider}:{model}"


class LLMSettingsMixin(BaseModel):
    """LLM provider, fallback chain, and guarded-agent-run settings.

    Composed into `Settings` (`app/config/settings.py`) alongside the other
    domain mixins; not used standalone.
    """

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
        return _validate_provider_model_id(v, "llm_model")

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

    llm_fallback_models: str | list[str] = Field(
        default_factory=list,
        description=(
            "Additional 'provider:model' identifiers tried in order if llm_model's "
            "provider fails (comma-separated or JSON array)"
        ),
    )

    @field_validator("llm_fallback_models", mode="before")
    @classmethod
    def parse_llm_fallback_models(cls, v: str | list[str]) -> list[str]:
        """Parse llm_fallback_models from string or list, mirroring parse_cors_origins.

        Args:
            v: The llm_fallback_models value to parse

        Returns:
            list[str]: Parsed list of "provider:model" identifiers
        """
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            import json

            v_stripped = v.strip()
            if not v_stripped:
                return []
            if v_stripped.startswith("["):
                try:
                    parsed = json.loads(v_stripped)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            if "," in v:
                return [model_id.strip() for model_id in v.split(",")]
            return [v.strip()]
        return v

    @field_validator("llm_fallback_models")
    @classmethod
    def validate_llm_fallback_models_format(cls, v: list[str]) -> list[str]:
        """Validate each llm_fallback_models entry follows 'provider:model' format.

        Args:
            v: The parsed llm_fallback_models list to validate

        Returns:
            list[str]: The validated list with lowercase-normalized providers

        Raises:
            ValueError: If any entry's format is invalid or provider is not allowed
        """
        return [_validate_provider_model_id(model_id, "llm_fallback_models") for model_id in v]

    max_output_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for Pydantic AI output validation",
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
    chat_request_timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Timeout in seconds for the whole POST /v1/agent/chat request; "
        "aborts via asyncio.wait_for rather than hanging indefinitely",
    )
    usage_request_limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of model requests allowed per guarded agent run "
        "(pydantic-ai UsageLimits.request_limit)",
    )
    usage_total_tokens_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum total tokens (input+output) allowed per guarded agent run; "
        "also gates the pre-tool-call budget check before any tool executes. "
        "None disables both checks",
    )

    @model_validator(mode="after")
    def validate_cloud_provider_api_key(self) -> Self:
        """Validate that cloud providers have llm_api_key set.

        Cloud providers (openai, anthropic, groq) require an API key.
        Local providers (ollama) are exempt from this requirement.

        Returns:
            Self: The validated settings instance

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
