"""Corrective RAG workflow using LlamaIndex Workflows.

Implements a three-step Corrective RAG pattern:
1. Search: Retrieve documents from vector store
2. Evaluate: Assess relevance and decide if retry needed
3. Synthesize: Generate final answer from relevant context

Each workflow.run() call gets its own isolated Context for state management.

The result cache (`ResultCacheMixin`), prompt-building helpers
(`PromptBuildingMixin`), and LLM-calling steps (`LLMCallMixin`) are split into
sibling modules per `.sdd/steering/file-size-policy.md`.
"""

import asyncio
import logging
from collections import OrderedDict

import logfire
from llama_index.core.workflow import Context
from llama_index.core.workflow import StartEvent
from llama_index.core.workflow import StopEvent
from llama_index.core.workflow import Workflow
from llama_index.core.workflow import step
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.agents.chat_agent import build_model
from app.config import Settings
from app.models.rag import RelevanceVerdict
from app.models.rag import RetrievedHit
from app.stores.vector_store import VectorStore
from app.workflows.citation import order_hits
from app.workflows.citation import validate_citations
from app.workflows.events import EvaluateEvent
from app.workflows.events import SearchEvent
from app.workflows.events import SynthesizeEvent
from app.workflows.rag_cache import ResultCacheMixin
from app.workflows.rag_llm import LLMCallMixin
from app.workflows.state import WorkflowState


logger = logging.getLogger(__name__)


class CorrectiveRAGWorkflow(ResultCacheMixin, LLMCallMixin, Workflow):  # ty: ignore[invalid-method-override]
    """Corrective RAG workflow with retry logic and result caching.

    `ResultCacheMixin.run`'s signature is deliberately incompatible with
    `Workflow.run`'s at the type level (see `rag_cache.py`'s `super().run()`
    sites) - the mixin overrides `run` to interpose caching, not to satisfy
    `Workflow`'s own contract independently of this composition. Tracked as
    design debt (Protocol-typed mixin bases), not fixed here.

    This workflow implements Corrective RAG: after retrieval, an evaluation
    step assesses relevance. If results are insufficient and retries remain,
    a refined SearchEvent is emitted to trigger a new retrieval cycle.

    Implements TTL-based LRU cache for query results to reduce
    redundant LLM calls and vector store queries for identical requests
    (`ResultCacheMixin`, `app/workflows/rag_cache.py`).

    Attributes:
        vector_store: Pluggable vector store for document retrieval.
        llm_settings: Configuration for LLM calls (model, API key, etc.).
        llm_model: Optional custom model for testing (e.g., FunctionModel).
        cache_stats: Dictionary containing cache hit/miss statistics.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_settings: Settings,
        llm_model: Model | str | None = None,
    ) -> None:
        """Initialize the Corrective RAG workflow with caching.

        Args:
            vector_store: Vector store implementation for retrieval.
            llm_settings: Settings containing LLM configuration.
            llm_model: Optional model override (useful for testing with FunctionModel).
        """
        super().__init__()
        self.vector_store = vector_store
        self.llm_settings = llm_settings
        self.llm_model = llm_model

        # Create Agent instances once during initialization
        # instead of recreating them on every LLM call (which happens in retry loops)
        # This reduces overhead from max_retries x 2 instances to just 2 instances total
        #
        # Req 11.1/11.2: an unsupplied model must resolve through build_model(),
        # the same builder the chat path uses, so LiteLLM provider routing and
        # Ollama base-URL handling stay identical. Passing llm_settings.llm_model
        # (a raw "provider:model" string) straight to Agent() would bypass both,
        # since Agent's own model inference has no knowledge of settings.llm_base_url.
        resolved_model = llm_model or build_model(llm_settings)
        # Req 10.1/10.3: the sufficiency decision is a validated model, not
        # prose, with its own output-retry budget (distinct from
        # `_run_agent_with_retry`'s transient-retry loop - see the nested
        # retry budgets note in `rag_llm.py`).
        self._eval_agent: Agent[None, RelevanceVerdict] = Agent(
            model=resolved_model,
            output_type=RelevanceVerdict,
            retries={"output": llm_settings.max_output_retries},
        )
        self._synth_agent: Agent[None, str] = Agent(
            model=resolved_model,
            output_type=str,
        )

        # Initialize cache data structures (read/mutated by ResultCacheMixin)
        # OrderedDict provides O(1) access and maintains insertion order for LRU
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Initialize cache lock to protect concurrent access
        # Prevents race conditions when multiple coroutines access cache simultaneously
        self._cache_lock: asyncio.Lock = asyncio.Lock()

        # Track in-flight requests to prevent thundering herd
        # Maps cache_key -> Future for requests currently executing
        self._pending_futures: dict[str, asyncio.Future[dict]] = {}

    @step
    async def search(
        self,
        ctx: Context,
        ev: StartEvent | SearchEvent,
    ) -> EvaluateEvent:
        """Retrieve documents from vector store.

        On StartEvent: Initializes WorkflowState with query and max_retries, and
            retrieves with `rag_initial_k` hits.
        On SearchEvent: Increments search_count and retrieves with the widened
            `rag_widened_k` hit count (AC 3.2).

        Args:
            ctx: LlamaIndex workflow context (unused in event-based state management).
            ev: Either StartEvent (initial query) or SearchEvent (retry).

        Returns:
            EvaluateEvent with retrieved hits and updated state.
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
                top_k = self.llm_settings.rag_initial_k
            else:
                # Load existing state from SearchEvent
                state = ev.state
                top_k = self.llm_settings.rag_widened_k

            # Increment search count
            state.search_count += 1

            # Retrieve documents from vector store
            query = state.query
            hits = await self.vector_store.query_with_scores(query, top_k=top_k)
            state.retrieved_hit_ids |= {hit.chunk_id for hit in hits}

            logfire.info(
                "Retrieved hits",
                search_count=state.search_count,
                top_k=top_k,
                hit_count=len(hits),
            )

            return EvaluateEvent(query=query, hits=hits, state=state)

    @step
    async def evaluate(
        self,
        ctx: Context,
        ev: EvaluateEvent,
    ) -> SearchEvent | SynthesizeEvent:
        """Assess relevance of retrieved hits.

        Uses LLM to classify hits as relevant or insufficient.
        If insufficient and retries remain, emits SearchEvent for retry.
        Otherwise, emits SynthesizeEvent to generate final answer.

        Args:
            ctx: LlamaIndex workflow context (unused in event-based state management).
            ev: EvaluateEvent with hits to evaluate and current state.

        Returns:
            SearchEvent (retry) or SynthesizeEvent (proceed to synthesis).
        """
        with logfire.span("rag_workflow.evaluate"):
            state = ev.state

            # AC 3.1: zero hits skip the LLM call entirely and terminate early.
            if not ev.hits:
                logfire.warn("No hits retrieved", search_count=state.search_count)
                return SynthesizeEvent(
                    query=ev.query,
                    hits=[],
                    context_found=False,
                    state=state,
                )

            # Evaluate relevance using LLM
            verdict = await self._evaluate_relevance([hit.text for hit in ev.hits], ev.query)

            logfire.info(
                "Evaluated relevance",
                sufficient=verdict.sufficient,
                rationale=verdict.rationale,
                search_count=state.search_count,
            )

            # If sufficient, proceed to synthesis
            if verdict.sufficient:
                return SynthesizeEvent(
                    query=ev.query,
                    hits=ev.hits,
                    context_found=True,
                    state=state,
                )

            # If insufficient and retries remain, emit refined search (widened k)
            if state.search_count < state.max_retries:
                logfire.info(
                    "Insufficient context, refining search",
                    search_count=state.search_count,
                    max_retries=state.max_retries,
                )
                return SearchEvent(query=ev.query, refined=True, state=state)

            # AC 3.6: retries exhausted but hits exist — proceed to a degraded,
            # grounded-subset synthesis instead of a generic "no context" message.
            logfire.warn(
                "Retries exhausted",
                search_count=state.search_count,
                max_retries=state.max_retries,
            )
            return SynthesizeEvent(
                query=ev.query,
                hits=ev.hits,
                context_found=False,
                state=state,
            )

    @step
    async def synthesize(
        self,
        ctx: Context,
        ev: SynthesizeEvent,
    ) -> StopEvent:
        """Generate final answer from relevant or grounded-subset context.

        If no hits were ever retrieved, returns a graceful "no context" response
        without calling the LLM. Otherwise (context_found=True, or retries
        exhausted with hits still present per AC 3.6), synthesizes an answer from
        the deterministically-ordered hits and cites exactly the hits used.

        Args:
            ctx: LlamaIndex workflow context (unused in event-based state management).
            ev: SynthesizeEvent with hits, context_found flag, and current state.

        Returns:
            StopEvent with final answer, citations, and metadata.
        """
        with logfire.span("rag_workflow.synthesize"):
            state = ev.state
            citations: list[RetrievedHit] = []

            if not ev.hits:
                # No hits were ever retrieved this run — nothing to ground an answer on.
                logfire.warn("No relevant context found", query=ev.query)
                answer = (
                    "I couldn't find relevant information to answer your question. "
                    "Please try rephrasing or asking a different question."
                )
            else:
                # Order deterministically before truncation (AC 3.8), then synthesize
                # from whatever survives the character budget — the grounded subset.
                ordered_hits = order_hits(ev.hits)
                citations = self._truncate_hits(ordered_hits, max_chars=15000)
                answer = await self._synthesize_answer(citations, ev.query)

                # Defense-in-depth: every citation is drawn from this run's own
                # retrieved hits by construction, so this should never raise.
                validate_citations(
                    cited_ids={hit.chunk_id for hit in citations},
                    hit_ids=state.retrieved_hit_ids,
                )

            # Update state with final answer
            state.final_answer = answer
            state.context_found = ev.context_found

            logfire.info(
                "Synthesized answer",
                context_found=ev.context_found,
                search_count=state.search_count,
                citation_count=len(citations),
            )

            # Return result with answer and metadata
            return StopEvent(
                result={
                    "answer": answer,
                    "context_found": ev.context_found,
                    "search_count": state.search_count,
                    "citations": citations,
                }
            )
