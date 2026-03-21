"""Corrective RAG workflow using LlamaIndex Workflows.

Implements a three-step Corrective RAG pattern:
1. Search: Retrieve documents from vector store
2. Evaluate: Assess relevance and decide if retry needed
3. Synthesize: Generate final answer from relevant context

Each workflow.run() call gets its own isolated Context for state management.
"""

import logfire
from llama_index.core.workflow import Context
from llama_index.core.workflow import StartEvent
from llama_index.core.workflow import StopEvent
from llama_index.core.workflow import Workflow
from llama_index.core.workflow import step
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.config import Settings
from app.stores.vector_store import VectorStore
from app.workflows.events import EvaluateEvent
from app.workflows.events import SearchEvent
from app.workflows.events import SynthesizeEvent
from app.workflows.state import WorkflowState


class CorrectiveRAGWorkflow(Workflow):
    """Corrective RAG workflow with retry logic.

    This workflow implements Corrective RAG: after retrieval, an evaluation
    step assesses relevance. If results are insufficient and retries remain,
    a refined SearchEvent is emitted to trigger a new retrieval cycle.

    Attributes:
        vector_store: Pluggable vector store for document retrieval.
        llm_settings: Configuration for LLM calls (model, API key, etc.).
        llm_model: Optional custom model for testing (e.g., FunctionModel).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_settings: Settings,
        llm_model: Model | str | None = None,
    ) -> None:
        """Initialize the Corrective RAG workflow.

        Args:
            vector_store: Vector store implementation for retrieval.
            llm_settings: Settings containing LLM configuration.
            llm_model: Optional model override (useful for testing with FunctionModel).
        """
        super().__init__()
        self.vector_store = vector_store
        self.llm_settings = llm_settings
        self.llm_model = llm_model

    @step
    async def search(
        self,
        ctx: Context,
        ev: StartEvent | SearchEvent,
    ) -> EvaluateEvent:
        """Retrieve documents from vector store.

        On StartEvent: Initializes WorkflowState with query and max_retries.
        On SearchEvent: Increments search_count and retrieves documents.

        Args:
            ctx: LlamaIndex workflow context (unused in event-based state management).
            ev: Either StartEvent (initial query) or SearchEvent (retry).

        Returns:
            EvaluateEvent with retrieved chunks and updated state.
        """
        with logfire.span("rag_workflow.search"):
            # Initialize or load state
            if isinstance(ev, StartEvent):
                # Extract query and max_retries from StartEvent
                query = ev.get("query")
                max_retries = ev.get("max_retries", 3)

                state = WorkflowState(
                    query=query,
                    search_count=0,
                    max_retries=max_retries,
                )
            else:
                # Load existing state from SearchEvent
                state = ev.state

            # Increment search count
            state.search_count += 1

            # Retrieve documents from vector store
            query = state.query
            chunks = await self.vector_store.query(query, top_k=5)

            logfire.info(
                "Retrieved chunks",
                search_count=state.search_count,
                chunk_count=len(chunks),
            )

            return EvaluateEvent(query=query, chunks=chunks, state=state)

    @step
    async def evaluate(
        self,
        ctx: Context,
        ev: EvaluateEvent,
    ) -> SearchEvent | SynthesizeEvent:
        """Assess relevance of retrieved chunks.

        Uses LLM to classify chunks as relevant or insufficient.
        If insufficient and retries remain, emits SearchEvent for retry.
        Otherwise, emits SynthesizeEvent to generate final answer.

        Args:
            ctx: LlamaIndex workflow context (unused in event-based state management).
            ev: EvaluateEvent with chunks to evaluate and current state.

        Returns:
            SearchEvent (retry) or SynthesizeEvent (proceed to synthesis).
        """
        with logfire.span("rag_workflow.evaluate"):
            state = ev.state

            # If no chunks retrieved, proceed to synthesis with context_found=False
            if not ev.chunks:
                logfire.warn("No chunks retrieved", search_count=state.search_count)
                return SynthesizeEvent(
                    query=ev.query,
                    chunks=[],
                    context_found=False,
                    state=state,
                )

            # Evaluate relevance using LLM
            relevance = await self._evaluate_relevance(ev.chunks, ev.query)

            logfire.info(
                "Evaluated relevance",
                relevance=relevance,
                search_count=state.search_count,
            )

            # If relevant, proceed to synthesis
            if relevance == "relevant":
                return SynthesizeEvent(
                    query=ev.query,
                    chunks=ev.chunks,
                    context_found=True,
                    state=state,
                )

            # If insufficient and retries remain, emit refined search
            if state.search_count < state.max_retries:
                logfire.info(
                    "Insufficient context, refining search",
                    search_count=state.search_count,
                    max_retries=state.max_retries,
                )
                return SearchEvent(query=ev.query, refined=True, state=state)

            # Retries exhausted, proceed to synthesis without context
            logfire.warn(
                "Retries exhausted",
                search_count=state.search_count,
                max_retries=state.max_retries,
            )
            return SynthesizeEvent(
                query=ev.query,
                chunks=ev.chunks,
                context_found=False,
                state=state,
            )

    @step
    async def synthesize(
        self,
        ctx: Context,
        ev: SynthesizeEvent,
    ) -> StopEvent:
        """Generate final answer from relevant context.

        If context_found is False, returns a graceful "no context" response.
        Otherwise, uses LLM to synthesize answer from chunks and query.

        Args:
            ctx: LlamaIndex workflow context (unused in event-based state management).
            ev: SynthesizeEvent with chunks, context_found flag, and current state.

        Returns:
            StopEvent with final answer and metadata.
        """
        with logfire.span("rag_workflow.synthesize"):
            state = ev.state

            # Handle no context found case
            if not ev.context_found:
                logfire.warn("No relevant context found", query=ev.query)
                answer = (
                    "I couldn't find relevant information to answer your question. "
                    "Please try rephrasing or asking a different question."
                )
            else:
                # Synthesize answer from chunks
                answer = await self._synthesize_answer(ev.chunks, ev.query)

            # Update state with final answer
            state.final_answer = answer
            state.context_found = ev.context_found

            logfire.info(
                "Synthesized answer",
                context_found=ev.context_found,
                search_count=state.search_count,
            )

            # Return result with answer and metadata
            return StopEvent(
                result={
                    "answer": answer,
                    "context_found": ev.context_found,
                    "search_count": state.search_count,
                }
            )

    async def _evaluate_relevance(self, chunks: list[str], query: str) -> str:
        """Evaluate relevance of retrieved chunks using LLM.

        Args:
            chunks: Retrieved document chunks.
            query: Original user query.

        Returns:
            "relevant" if chunks are sufficient, "insufficient" otherwise.
        """
        # Build evaluation prompt
        context = "\n\n".join(f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(chunks))
        prompt = f"""Given the following chunks and query, assess if the chunks contain \
relevant information to answer the query.

Query: {query}

Context:
{context}

Respond with exactly one word:
- "relevant" if the chunks contain sufficient information to answer the query
- "insufficient" if the chunks do not contain relevant information

Response:"""

        # Use configured model or override
        model = self.llm_model or self.llm_settings.llm_model

        # Create a simple agent for evaluation
        eval_agent: Agent[None, str] = Agent(
            model=model,
            output_type=str,
        )

        # Run evaluation
        result = await eval_agent.run(prompt)
        response = result.output.strip().lower()

        # Normalize response
        if "relevant" in response:
            return "relevant"
        return "insufficient"

    async def _synthesize_answer(self, chunks: list[str], query: str) -> str:
        """Synthesize final answer from relevant chunks using LLM.

        Args:
            chunks: Relevant document chunks.
            query: Original user query.

        Returns:
            Synthesized answer.
        """
        # Build synthesis prompt
        context = "\n\n".join(f"Source {i + 1}: {chunk}" for i, chunk in enumerate(chunks))
        prompt = f"""Using the following context, provide a clear and concise answer to the query.

Query: {query}

Context:
{context}

Answer:"""

        # Use configured model or override
        model = self.llm_model or self.llm_settings.llm_model

        # Create synthesis agent
        synthesis_agent: Agent[None, str] = Agent(
            model=model,
            output_type=str,
        )

        # Generate answer
        result = await synthesis_agent.run(prompt)
        return result.output.strip()
