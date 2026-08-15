"""Logfire observability settings."""

from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator

from app.config._secret_placeholders import is_placeholder


class ObservabilitySettingsMixin(BaseModel):
    """Logfire observability settings.

    Composed into `Settings` (`app/config/settings.py`) alongside the other
    domain mixins; not used standalone.
    """

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

        # Check whether the token is a known placeholder. The shared shape rule
        # covers the `...-here` spellings; only genuinely token-specific strings
        # need enumerating here.
        if is_placeholder(v_stripped, extra=("insert-token-here",)):
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
    log_sensitive_payloads: bool = Field(
        default=False,
        description=(
            "Disable Logfire scrubbing of prompt/tool_input/tool_output payloads "
            "(for local debugging only; emits an audit warning when enabled)"
        ),
    )
