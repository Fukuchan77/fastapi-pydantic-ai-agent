"""Request and response models for RAG endpoints."""

from pydantic import BaseModel
from pydantic import Field


class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    chunks: list[str] = Field(..., min_length=1, max_length=1000)


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    ingested: int


class RAGQueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    query: str = Field(..., min_length=1, max_length=10000)
    max_retries: int = Field(default=3, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    answer: str
    context_found: bool
    search_count: int
