"""FastAPI dependencies for workflow injection."""

from fastapi import Request

from app.agents.chat_agent import build_model
from app.config import get_settings
from app.workflows.corrective_rag import CorrectiveRAGWorkflow


def get_rag_workflow(req: Request) -> CorrectiveRAGWorkflow:
    """Create a per-request CorrectiveRAGWorkflow instance.

    Each request gets its own workflow instance to ensure isolation between
    concurrent requests. The workflow itself is stateless - all mutable state
    lives in the per-run LlamaIndex Context object.

    Args:
        req: FastAPI request object containing app.state.vector_store.

    Returns:
        A new CorrectiveRAGWorkflow instance configured with the application's
        vector store and LLM settings.
    """
    settings = get_settings()
    model = build_model(settings)

    return CorrectiveRAGWorkflow(
        vector_store=req.app.state.vector_store,
        llm_settings=settings,
        llm_model=model,
    )
