"""Session store, vector store, embedding, and RAG workflow settings."""

from typing import Literal
from typing import Self

from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl
from pydantic import field_validator
from pydantic import model_validator


class StoreSettingsMixin(BaseModel):
    """Session store, vector store, embedding, and RAG workflow settings.

    Composed into `Settings` (`app/config/settings.py`) alongside the other
    domain mixins; not used standalone.
    """

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

    @model_validator(mode="after")
    def validate_redis_session_store_requires_url(self) -> Self:
        """Validate that redis_url is set whenever redis_session_store_enabled is True.

        The store factory (`app/stores/factory.py`) constructs `RedisSessionStore`
        from `redis_url` only when `redis_session_store_enabled` is true, so a
        missing URL is a configuration error, not a runtime one.

        Returns:
            Self: The validated settings instance.

        Raises:
            ValueError: If redis_session_store_enabled is True and redis_url is unset.
        """
        if self.redis_session_store_enabled and not self.redis_url:
            raise ValueError(
                "redis_url is required when redis_session_store_enabled is True. "
                "Please set the REDIS_URL environment variable."
            )

        return self

    vector_store_backend: Literal["memory", "chroma", "ollama"] = Field(
        default="memory",
        description="Vector store backend selection: 'memory' (default, TF-IDF, no "
        "external dependency), 'chroma' (ChromaDB embeddings), or 'ollama' "
        "(Ollama embeddings via embedding_model/embedding_base_url). "
        "'ollama' requires embedding_model to be set",
    )
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

    @model_validator(mode="after")
    def validate_vector_store_backend_requires_embedding_model(self) -> Self:
        """Validate that embedding_model is set whenever vector_store_backend is 'ollama'.

        `OllamaEmbeddingVectorStore` requires a non-empty embedding model name at
        construction time, so a missing value is a configuration error, not a
        runtime one.

        Returns:
            Self: The validated settings instance.

        Raises:
            ValueError: If vector_store_backend is 'ollama' and embedding_model is unset.
        """
        if self.vector_store_backend == "ollama" and not self.embedding_model:
            raise ValueError(
                "embedding_model is required when vector_store_backend is 'ollama'. "
                "Please set the EMBEDDING_MODEL environment variable."
            )

        return self

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
    rag_initial_k: int = Field(
        default=2,
        ge=1,
        le=50,
        description="Number of hits to retrieve on the initial CRAG search attempt",
    )
    rag_widened_k: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Number of hits to retrieve on a CRAG retry after insufficient grading "
        "(widened from rag_initial_k)",
    )
