"""Event classes for LlamaIndex Corrective RAG workflow.

These events orchestrate the flow between workflow steps:
- SearchEvent: Triggers document retrieval from vector store
- EvaluateEvent: Triggers relevance assessment of retrieved chunks
- SynthesizeEvent: Triggers final answer generation from relevant context

Each event carries the workflow state to enable stateless step functions.
"""

from llama_index.core.workflow import Event
from pydantic import Field

from app.workflows.state import WorkflowState


class SearchEvent(Event):
    """Event to trigger document search in vector store.

    Attributes:
        query: The search query string (1-10000 characters).
        refined: Whether this is a refined retry query (True) or initial query (False).
            Defaults to False.
        state: Current workflow state.
    """

    query: str = Field(..., min_length=1, max_length=10000)
    refined: bool = False
    state: WorkflowState


class EvaluateEvent(Event):
    """Event to trigger relevance evaluation of retrieved chunks.

    Attributes:
        query: The original search query (1-10000 characters).
        chunks: List of retrieved document chunks to evaluate (max 100 chunks).
        state: Current workflow state.
    """

    query: str = Field(..., min_length=1, max_length=10000)
    chunks: list[str] = Field(..., max_length=100)
    state: WorkflowState


class SynthesizeEvent(Event):
    """Event to trigger answer synthesis from relevant context.

    Attributes:
        query: The original search query (1-10000 characters).
        chunks: List of relevant document chunks for synthesis (max 100 chunks).
        context_found: Whether relevant context was found (True) or retries exhausted (False).
        state: Current workflow state.
    """

    query: str = Field(..., min_length=1, max_length=10000)
    chunks: list[str] = Field(..., max_length=100)
    context_found: bool
    state: WorkflowState
