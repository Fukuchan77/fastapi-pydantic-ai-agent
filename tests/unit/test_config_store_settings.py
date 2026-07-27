"""Unit tests for store-selection Settings fields (Task 3: store factory wiring).

Covers `vector_store_backend` (Req 5.1, 5.3) and the cross-field validators that
guard `redis_session_store_enabled` (Req 5.2) and `vector_store_backend="ollama"`
(Req 5.1/5.3) against missing prerequisite settings.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings


def _build_settings(**overrides: object) -> Settings:
    """Build a valid Settings instance directly, without touching os.environ.

    `embedding_model`/`embedding_base_url`/`redis_url` default to `None`
    explicitly so a developer's local `.env` values for those fields can never
    leak into these tests via pydantic-settings' env-file fallback.
    """
    defaults: dict[str, object] = {
        "api_key": "test-api-key-12345",
        "llm_model": "openai:gpt-4o",
        "llm_api_key": "test-llm-api-key-12345",
        "embedding_model": None,
        "embedding_base_url": None,
        "redis_url": None,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def test_vector_store_backend_default_is_memory() -> None:
    """vector_store_backend defaults to 'memory' when not configured."""
    settings = _build_settings()

    assert settings.vector_store_backend == "memory"


def test_vector_store_backend_accepts_chroma() -> None:
    """vector_store_backend accepts 'chroma' without requiring embedding_model."""
    settings = _build_settings(vector_store_backend="chroma")

    assert settings.vector_store_backend == "chroma"


def test_vector_store_backend_accepts_ollama_with_embedding_model() -> None:
    """vector_store_backend accepts 'ollama' when embedding_model is also set."""
    settings = _build_settings(
        vector_store_backend="ollama",
        embedding_model="nomic-embed-text:latest",
    )

    assert settings.vector_store_backend == "ollama"
    assert settings.embedding_model == "nomic-embed-text:latest"


def test_vector_store_backend_ollama_without_embedding_model_raises() -> None:
    """vector_store_backend='ollama' without embedding_model must fail validation."""
    # embedding_model=None is explicit so this test is deterministic regardless
    # of a developer's local .env EMBEDDING_MODEL value.
    with pytest.raises(ValidationError, match="embedding_model is required"):
        _build_settings(vector_store_backend="ollama", embedding_model=None)


def test_vector_store_backend_rejects_unknown_value() -> None:
    """vector_store_backend only accepts the closed 'memory'/'chroma'/'ollama' vocabulary."""
    with pytest.raises(ValidationError):
        _build_settings(vector_store_backend="pinecone")


def test_redis_session_store_enabled_without_redis_url_raises() -> None:
    """redis_session_store_enabled=True without redis_url must fail validation."""
    # redis_url=None is explicit (not relying on the field default) so this test
    # is deterministic regardless of a developer's local .env REDIS_URL value.
    with pytest.raises(ValidationError, match="redis_url is required"):
        _build_settings(redis_session_store_enabled=True, redis_url=None)


def test_redis_session_store_enabled_with_redis_url_succeeds() -> None:
    """redis_session_store_enabled=True with redis_url set constructs cleanly."""
    settings = _build_settings(
        redis_session_store_enabled=True,
        redis_url="redis://localhost:6379/0",
    )

    assert settings.redis_session_store_enabled is True
    assert settings.redis_url == "redis://localhost:6379/0"


def test_redis_session_store_disabled_without_redis_url_succeeds() -> None:
    """Default (redis disabled, no redis_url) remains valid (no regression)."""
    # redis_url=None is explicit for the same determinism reason as above.
    settings = _build_settings(redis_url=None)

    assert settings.redis_session_store_enabled is False
    assert settings.redis_url is None
