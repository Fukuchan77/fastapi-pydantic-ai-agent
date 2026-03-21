"""Event classes for LlamaIndex Corrective RAG workflow.

These events orchestrate the flow between workflow steps:
- SearchEvent: Triggers document retrieval from vector store
- EvaluateEvent: Triggers relevance assessment of retrieved chunks
- SynthesizeEvent: Triggers final answer generation from relevant context

Each event carries the workflow state to enable stateless step functions.
"""

from llama_index.core.workflow import Event

from app.workflows.state import WorkflowState


class SearchEvent(Event):
    """Event to trigger document search in vector store.

    Attributes:
        query: The search query string.
        refined: Whether this is a refined retry query (True) or initial query (False).
            Defaults to False.
        state: Current workflow state.
    """

    query: str
    refined: bool = False
    state: WorkflowState


class EvaluateEvent(Event):
    """Event to trigger relevance evaluation of retrieved chunks.

    Attributes:
        query: The original search query.
        chunks: List of retrieved document chunks to evaluate.
        state: Current workflow state.
    """

    query: str
    chunks: list[str]
    state: WorkflowState


class SynthesizeEvent(Event):
    """Event to trigger answer synthesis from relevant context.

    Attributes:
        query: The original search query.
        chunks: List of relevant document chunks for synthesis.
        context_found: Whether relevant context was found (True) or retries exhausted (False).
        state: Current workflow state.
    """

    query: str
    chunks: list[str]
    context_found: bool
    state: WorkflowState
