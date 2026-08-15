"""Request and response models for RAG endpoints."""

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


# Per-chunk character ceiling enforced at the API boundary. Matches
# `InMemoryVectorStore.DEFAULT_MAX_CHUNK_SIZE`, but is declared here rather than
# imported so the wire contract does not depend on which backend happens to be
# configured - the Chroma and Ollama backends enforce no per-chunk limit of
# their own, so without this an oversized chunk reached them unchecked.
MAX_CHUNK_CHARS = 100_000


class IngestRequest(BaseModel):
    """Request model for document ingestion.

    Attributes:
        chunks: List of text chunks to ingest into the vector store (1-1000
            chunks, each at most MAX_CHUNK_CHARS characters).
    """

    chunks: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of text chunks to ingest into the vector store",
    )

    @field_validator("chunks")
    @classmethod
    def validate_chunk_sizes(cls, v: list[str]) -> list[str]:
        """Reject oversized chunks at the API boundary.

        Without this the request reached `VectorStore.add_documents()`, whose
        `ValueError` no handler converts - so a caller-supplied chunk over the
        store's limit surfaced as a 500 with a logged traceback rather than the
        flat 422 envelope every other client input error uses (Req 8).

        Args:
            v: The submitted chunks.

        Returns:
            list[str]: The validated chunks.

        Raises:
            ValueError: If any chunk exceeds MAX_CHUNK_CHARS characters.
        """
        for index, chunk in enumerate(v):
            if len(chunk) > MAX_CHUNK_CHARS:
                raise ValueError(
                    f"chunk at index {index} is {len(chunk)} characters; "
                    f"the maximum is {MAX_CHUNK_CHARS}"
                )
        return v


class IngestResponse(BaseModel):
    """Response model for document ingestion.

    Attributes:
        ingested: Number of chunks successfully ingested into the vector store.
    """

    ingested: int = Field(
        description="Number of chunks successfully ingested into the vector store"
    )


class RAGQueryRequest(BaseModel):
    """Request model for RAG query endpoint.

    Attributes:
        query: User query to search for relevant context (1-10000 chars).
        max_retries: Maximum number of search retries for relevance evaluation (1-10).
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User query to search for relevant context",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of search retries for relevance evaluation",
    )


class RetrievedHit(BaseModel):
    """A single retrieval hit with its stable citation id and relevance score.

    Attributes:
        chunk_id: Stable document-chunk identifier of the form "source::ordinal".
        text: The retrieved document chunk text.
        score: Relevance score assigned by the vector store (higher is more relevant).
    """

    chunk_id: str = Field(description="Stable document-chunk identifier (source::ordinal)")
    text: str = Field(description="Retrieved document chunk text")
    score: float = Field(description="Relevance score assigned by the vector store")


class RelevanceVerdict(BaseModel):
    """Structured sufficiency decision for retrieved RAG context.

    Attributes:
        sufficient: Whether the retrieved chunks are sufficient to answer the query.
        rationale: Reasoning behind the verdict, required and non-empty so the
            contract cannot be satisfied by an absent or blank explanation.
    """

    sufficient: bool = Field(
        description="Whether the retrieved chunks are sufficient to answer the query"
    )
    rationale: str = Field(
        min_length=1,
        description="Reasoning behind the sufficiency verdict",
    )


class RAGQueryResponse(BaseModel):
    """Response model for RAG query endpoint.

    Attributes:
        answer: Generated answer based on retrieved context.
        context_found: Whether relevant context was found in the vector store.
        search_count: Number of search attempts performed during this query.
        citations: Retrieved hits cited in the answer.
    """

    answer: str = Field(description="Generated answer based on retrieved context")
    context_found: bool = Field(
        description="Whether relevant context was found in the vector store"
    )
    search_count: int = Field(description="Number of search attempts performed during this query")
    citations: list[RetrievedHit] = Field(
        default_factory=list,
        description="Retrieved hits cited in the answer",
    )
