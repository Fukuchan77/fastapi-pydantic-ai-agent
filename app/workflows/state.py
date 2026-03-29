"""Typed workflow state model for Corrective RAG workflow.

This state is stored in the LlamaIndex workflow Context and shared across
all workflow steps in a single run. Each workflow.run() call gets its own
isolated Context instance.
"""

from pydantic import BaseModel
from pydantic import Field


class WorkflowState(BaseModel):
    """Shared state for Corrective RAG workflow execution.

    This model tracks the progress and results of a single RAG workflow run.
    State is stored in the per-run LlamaIndex Context object to ensure
    isolation between concurrent workflow executions.

    Attributes:
        query: The original user query string.
        search_count: Number of vector store searches performed (incremented on each retry).
        max_retries: Maximum number of search retries allowed before giving up.
            Set to 0 to perform search only once without retries (internal use).
            Public API requires minimum 1.
        final_answer: The synthesized answer from the LLM, or None if not yet generated.
        context_found: Whether relevant context was found (True) or retries were exhausted (False).
    """

    query: str
    search_count: int = 0
    max_retries: int = Field(default=3, ge=0, le=10)
    final_answer: str | None = None
    context_found: bool = False
