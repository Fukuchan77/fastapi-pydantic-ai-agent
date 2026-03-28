"""Test connection pooling configuration for httpx.AsyncClient.

Task 16.6: Configure request/connection timeouts and pooling.
Timeouts are already configured. This test verifies connection pool limits.
"""

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings


def test_http_max_connections_default():
    """Test http_max_connections has a reasonable default value."""
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        llm_api_key=SecretStr("test-llm-key-12345"),
    )
    # Default should be 100 (httpx default)
    assert settings.http_max_connections == 100


def test_http_max_keepalive_connections_default():
    """Test http_max_keepalive_connections has a reasonable default value."""
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        llm_api_key=SecretStr("test-llm-key-12345"),
    )
    # Default should be 20 (httpx default)
    assert settings.http_max_keepalive_connections == 20


def test_http_max_connections_validation():
    """Test http_max_connections has appropriate bounds."""
    # Valid range: 1-500
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        llm_api_key=SecretStr("test-llm-key-12345"),
        http_max_connections=50,
    )
    assert settings.http_max_connections == 50

    # Test lower bound
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4",
            llm_api_key=SecretStr("test-llm-key-12345"),
            http_max_connections=0,
        )

    # Test upper bound
    with pytest.raises(ValidationError, match="less than or equal to 500"):
        Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4",
            llm_api_key=SecretStr("test-llm-key-12345"),
            http_max_connections=501,
        )


def test_http_max_keepalive_connections_validation():
    """Test http_max_keepalive_connections has appropriate bounds."""
    # Valid range: 1-100
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        llm_api_key=SecretStr("test-llm-key-12345"),
        http_max_keepalive_connections=10,
    )
    assert settings.http_max_keepalive_connections == 10

    # Test lower bound
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4",
            llm_api_key=SecretStr("test-llm-key-12345"),
            http_max_keepalive_connections=0,
        )

    # Test upper bound
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4",
            llm_api_key=SecretStr("test-llm-key-12345"),
            http_max_keepalive_connections=101,
        )


def test_keepalive_not_greater_than_max_connections():
    """Test http_max_keepalive_connections cannot exceed http_max_connections."""
    # This should fail: keepalive (50) > max_connections (30)
    with pytest.raises(
        ValidationError,
        match=r"http_max_keepalive_connections.*cannot exceed.*http_max_connections",
    ):
        Settings(
            api_key=SecretStr("test-api-key-12345"),
            llm_model="openai:gpt-4",
            llm_api_key=SecretStr("test-llm-key-12345"),
            http_max_connections=30,
            http_max_keepalive_connections=50,
        )

    # This should succeed: keepalive (10) <= max_connections (30)
    settings = Settings(
        api_key=SecretStr("test-api-key-12345"),
        llm_model="openai:gpt-4",
        llm_api_key=SecretStr("test-llm-key-12345"),
        http_max_connections=30,
        http_max_keepalive_connections=10,
    )
    assert settings.http_max_connections == 30
    assert settings.http_max_keepalive_connections == 10
