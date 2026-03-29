"""Request and response models for RAG endpoints."""

from pydantic import BaseModel
from pydantic import Field


class IngestRequest(BaseModel):
    """Request model for document ingestion.

    Attributes:
        chunks: List of text chunks to ingest into the vector store (1-1000 chunks).
    """

    chunks: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of text chunks to ingest into the vector store",
    )


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


class RAGQueryResponse(BaseModel):
    """Response model for RAG query endpoint.

    Attributes:
        answer: Generated answer based on retrieved context.
        context_found: Whether relevant context was found in the vector store.
        search_count: Number of search attempts performed during this query.
    """

    answer: str = Field(description="Generated answer based on retrieved context")
    context_found: bool = Field(
        description="Whether relevant context was found in the vector store"
    )
    search_count: int = Field(description="Number of search attempts performed during this query")
