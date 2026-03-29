"""RAG API routes for document ingestion and query."""

import asyncio

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request

from app.deps.auth import verify_api_key
from app.deps.workflow import get_rag_workflow
from app.models.rag import IngestRequest
from app.models.rag import IngestResponse
from app.models.rag import RAGQueryRequest
from app.models.rag import RAGQueryResponse
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


# Create router for RAG endpoints
router = APIRouter(tags=["rag"])


@router.post("/rag/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    req: Request,
    _: None = Depends(verify_api_key),
) -> IngestResponse:
    """Ingest document chunks into the vector store.

    This endpoint accepts a list of text chunks and adds them to the
    configured vector store for later retrieval during RAG queries.

    Args:
        request: IngestRequest with list of document chunks.
        req: FastAPI Request object for accessing app.state.
        _: Authentication dependency (validates X-API-Key header).

    Returns:
        IngestResponse with count of ingested chunks.
    """
    # Get vector store from app.state
    vector_store = req.app.state.vector_store

    # Add documents to vector store
    await vector_store.add_documents(request.chunks)

    # Return count of ingested chunks
    return IngestResponse(ingested=len(request.chunks))


@router.post("/rag/query", response_model=RAGQueryResponse)
async def query(
    request: RAGQueryRequest,
    workflow: CorrectiveRAGWorkflow = Depends(get_rag_workflow),  # noqa: B008
    _: None = Depends(verify_api_key),
) -> RAGQueryResponse:
    """Execute a RAG query using the Corrective RAG workflow.

    This endpoint runs the Corrective RAG workflow which:
    1. Retrieves relevant documents from the vector store
    2. Evaluates relevance and retries if insufficient
    3. Synthesizes a final answer from relevant context

    Args:
        request: RAGQueryRequest with query string and optional max_retries.
        workflow: CorrectiveRAGWorkflow instance (per-request).
        _: Authentication dependency (validates X-API-Key header).

    Returns:
        RAGQueryResponse with answer, context_found flag, and search_count.
    """
    # Run the workflow with query and max_retries
    # Wrap workflow execution with timeout to prevent indefinite hangs
    try:
        async with asyncio.timeout(workflow.llm_settings.rag_workflow_timeout):
            result = await workflow.run(
                query=request.query,
                max_retries=request.max_retries,
            )
    except TimeoutError as e:
        # Convert workflow timeout to HTTP 504 Gateway Timeout
        raise HTTPException(
            status_code=504,
            detail="RAG workflow timed out. Please try again with a simpler query.",
        ) from e

    # Extract result data from StopEvent
    result_data = result

    # Return response
    return RAGQueryResponse(
        answer=result_data["answer"],
        context_found=result_data["context_found"],
        search_count=result_data["search_count"],
    )
