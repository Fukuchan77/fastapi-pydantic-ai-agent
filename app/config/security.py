"""Auth, CORS, rate-limiting, HTTP client, and SSE resource-limit settings."""

from typing import Self

from pydantic import BaseModel
from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator


class SecuritySettingsMixin(BaseModel):
    """Auth, CORS, rate-limiting, HTTP client, and SSE resource-limit settings.

    Composed into `Settings` (`app/config/settings.py`) alongside the other
    domain mixins; not used standalone.
    """

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

    session_signing_key: SecretStr = Field(
        ...,
        description="Secret key used to HMAC-sign server-issued session ids, binding "
        "them to the authenticated principal (Req 11.1/11.2)",
    )

    @field_validator("session_signing_key")
    @classmethod
    def validate_session_signing_key_strength(cls, v: SecretStr) -> SecretStr:
        """Validate session_signing_key is not a placeholder and meets minimum strength.

        A weak signing key would let an attacker forge session ids and defeat
        the IDOR protection Req 11.1/11.2 depends on, so this uses the same
        strength rule as `api_key` rather than merely requiring non-empty.

        Args:
            v: The session_signing_key value to validate.

        Returns:
            SecretStr: The validated session_signing_key value.

        Raises:
            ValueError: If session_signing_key is a placeholder or too weak.
        """
        v_str = v.get_secret_value()
        v_stripped = v_str.strip()

        if not v_stripped:
            raise ValueError("session_signing_key cannot be empty or whitespace only")

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

        if v_stripped.lower() in placeholders:
            raise ValueError(
                "session_signing_key appears to be a placeholder value. "
                "Please set a strong signing key with at least 16 characters."
            )

        if len(v_stripped) < 16:
            raise ValueError(
                f"session_signing_key must be at least 16 characters long for security. "
                f"Current length: {len(v_stripped)}"
            )

        return v

    app_env: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )
    enable_mock_tools: bool = Field(
        default=False,
        description="Enable mock tools (for development only, disable in production)",
    )

    @model_validator(mode="after")
    def validate_mock_tools_not_in_production(self) -> Self:
        """Validate that enable_mock_tools is not enabled in production.

        Mock tools should only be enabled in development environments to prevent
        security vulnerabilities in production.

        Returns:
            Self: The validated settings instance

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

    trusted_proxies: list[str] = Field(
        default=[],
        description="List of trusted proxy IP addresses for X-Forwarded-For validation",
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
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"cors_origins looks like a JSON array but is not valid "
                        f"JSON: {v_stripped!r}"
                    ) from exc
                if isinstance(parsed, list):
                    return parsed
                raise ValueError(f"cors_origins JSON value must be an array, got: {v_stripped!r}")
            # Parse as comma-separated string
            if "," in v:
                return [origin.strip() for origin in v.split(",")]
            # Single origin string
            return [v.strip()]
        return v

    llm_rate_limit: str = Field(
        default="30/minute",
        description="Rate limit string (e.g. '30/minute') applied per-route to "
        "LLM-invoking endpoints (chat, stream, RAG query), stricter than the "
        "global 1000/minute default (Req 11.3)",
    )

    @field_validator("llm_rate_limit")
    @classmethod
    def validate_llm_rate_limit_format(cls, v: str) -> str:
        """Validate llm_rate_limit follows the '<count>/<period>' format slowapi expects.

        Args:
            v: The llm_rate_limit value to validate.

        Returns:
            str: The validated llm_rate_limit value.

        Raises:
            ValueError: If the format doesn't match '<int>/<second|minute|hour|day>'.
        """
        import re

        if not re.fullmatch(r"\d+/(second|minute|hour|day)", v):
            raise ValueError(
                f"llm_rate_limit must follow '<count>/<second|minute|hour|day>' format, got: {v}"
            )
        return v

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

    @model_validator(mode="after")
    def validate_keepalive_connections_limit(self) -> Self:
        """Validate that keepalive connections do not exceed total connections.

        Connection pool configuration validation.
        The number of keepalive connections in the pool cannot exceed the
        maximum total connections, as this would be a logical contradiction.

        Returns:
            Self: The validated settings instance

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

    sse_max_events: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of SSE events emitted per agent stream request "
        "before the stream stops producing further events",
    )
    sse_heartbeat_interval: int = Field(
        default=15,
        ge=1,
        description="Interval in seconds between SSE heartbeat comments emitted "
        "while the agent stream is idle (no event ready to send)",
    )
    sse_send_timeout: int = Field(
        default=60,
        ge=1,
        description="Timeout in seconds for producing a single SSE event; the "
        "stream aborts with a terminal error event if exceeded",
    )

    @model_validator(mode="after")
    def validate_sse_heartbeat_interval_within_send_timeout(self) -> Self:
        """Validate that sse_heartbeat_interval does not exceed sse_send_timeout.

        The heartbeat loop only gets a chance to fire between waits bounded by
        `sse_send_timeout`; a heartbeat interval larger than the send timeout
        would never actually fire before the stream aborts, defeating its
        purpose of keeping idle connections alive.

        Returns:
            Self: The validated settings instance.

        Raises:
            ValueError: If sse_heartbeat_interval exceeds sse_send_timeout.
        """
        if self.sse_heartbeat_interval > self.sse_send_timeout:
            raise ValueError(
                f"sse_heartbeat_interval ({self.sse_heartbeat_interval}) "
                f"cannot exceed sse_send_timeout ({self.sse_send_timeout}). "
                "A heartbeat that never fires within the send-timeout budget "
                "cannot keep idle connections alive."
            )

        return self
