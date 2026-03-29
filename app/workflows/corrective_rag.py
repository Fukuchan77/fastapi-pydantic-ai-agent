"""Corrective RAG workflow using LlamaIndex Workflows.

Implements a three-step Corrective RAG pattern:
1. Search: Retrieve documents from vector store
2. Evaluate: Assess relevance and decide if retry needed
3. Synthesize: Generate final answer from relevant context

Each workflow.run() call gets its own isolated Context for state management.
"""

import asyncio
import hashlib
import html
import logging
import random
import time
from collections import OrderedDict

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
from app.workflows.exceptions import RAGWorkflowError
from app.workflows.state import WorkflowState


logger = logging.getLogger(__name__)


class CorrectiveRAGWorkflow(Workflow):
    """Corrective RAG workflow with retry logic and result caching.

    This workflow implements Corrective RAG: after retrieval, an evaluation
    step assesses relevance. If results are insufficient and retries remain,
    a refined SearchEvent is emitted to trigger a new retrieval cycle.

    Task 17.1: Implements TTL-based LRU cache for query results to reduce
    redundant LLM calls and vector store queries for identical requests.

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

        # MEDIUM FIX: Create Agent instances once during initialization
        # instead of recreating them on every LLM call (which happens in retry loops)
        # This reduces overhead from max_retries x 2 instances to just 2 instances total
        resolved_model = llm_model or llm_settings.llm_model
        self._eval_agent: Agent[None, str] = Agent(
            model=resolved_model,
            output_type=str,
        )
        self._synth_agent: Agent[None, str] = Agent(
            model=resolved_model,
            output_type=str,
        )

        # Task 17.1: Initialize cache data structures
        # OrderedDict provides O(1) access and maintains insertion order for LRU
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Task 21.1: Initialize cache lock to protect concurrent access
        # Prevents race conditions when multiple coroutines access cache simultaneously
        self._cache_lock: asyncio.Lock = asyncio.Lock()

        # Task 25.2: Track in-flight requests to prevent thundering herd
        # Maps cache_key -> Future for requests currently executing
        self._pending_futures: dict[str, asyncio.Future[dict]] = {}

    def _generate_cache_key(self, query: str, max_retries: int) -> str:
        """Generate cache key from query and max_retries parameter.

        Task 17.1: Cache key includes both query and max_retries because
        the same query with different max_retries may produce different results.

        Args:
            query: User query string.
            max_retries: Maximum retry attempts.

        Returns:
            str: SHA256 hash of query + max_retries for cache key.
        """
        # Combine query and max_retries into a single string
        key_material = f"{query}|{max_retries}"
        # Generate SHA256 hash for consistent key length
        return hashlib.sha256(key_material.encode()).hexdigest()

    def _evict_expired_entries(self) -> None:
        """Remove expired cache entries based on TTL.

        Task 17.1: Removes entries where current_time - cached_time > ttl.
        """
        if self.llm_settings.rag_cache_ttl == 0:
            return  # Cache disabled

        current_time = time.time()
        ttl = self.llm_settings.rag_cache_ttl
        expired_keys = [
            key for key, (_, cached_time) in self._cache.items() if current_time - cached_time > ttl
        ]
        for key in expired_keys:
            del self._cache[key]

    def _evict_lru_entry(self) -> None:
        """Remove least recently used entry to maintain size limit.

        Task 17.1: OrderedDict maintains insertion order, so the first item
        is the least recently used (after move_to_end on cache hits).
        """
        if self._cache:
            self._cache.popitem(last=False)  # Remove first (oldest) item

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics for monitoring.

        Task 17.1: Exposes cache hit/miss/size metrics for observability.

        Returns:
            dict: Dictionary with 'hits', 'misses', and 'size' keys.
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    async def run(self, query: str, max_retries: int = 3) -> dict:  # type: ignore[override]
        """Run the workflow with caching support.

        Task 17.1: Overrides parent run() to check cache before executing workflow.
        If cache hit, returns cached result immediately. Otherwise, executes workflow
        and caches the result.

        Args:
            query: User query string.
            max_retries: Maximum retry attempts for relevance evaluation.

        Returns:
            dict: Workflow result with answer, context_found, and search_count.
        """
        # Task 17.1: Check if caching is disabled (ttl=0)
        if self.llm_settings.rag_cache_ttl == 0:
            # Cache disabled, execute workflow directly
            result = await super().run(query=query, max_retries=max_retries)
            return result

        # Task 17.1: Generate cache key
        cache_key = self._generate_cache_key(query, max_retries)

        # Task 21.1 & 25.2: Use double-check locking to prevent thundering herd
        async with self._cache_lock:
            # Task 17.1: Evict expired entries before checking cache
            self._evict_expired_entries()

            # Task 17.1: Check cache for existing result
            if cache_key in self._cache:
                # Cache hit - move to end (mark as recently used) and return cached result
                cached_result, _ = self._cache[cache_key]
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                logfire.info("RAG cache hit", query=query[:50], cache_key=cache_key[:16])
                # Task 25.4: Verify cached result is a dict before copying
                if not isinstance(cached_result, dict):
                    raise TypeError(f"Expected dict from cache, got {type(cached_result).__name__}")
                # Task 21.5: Return a copy to prevent callers from mutating the cached dict
                return dict(cached_result)

            # Task 25.2: Check if there's a pending future for this query (thundering herd fix)
            if cache_key in self._pending_futures:
                # Another request is already executing this workflow - wait for it
                pending_future = self._pending_futures[cache_key]
                self._cache_hits += 1
                logfire.info(
                    "RAG pending request found, awaiting",
                    query=query[:50],
                    cache_key=cache_key[:16],
                )
                # Await OUTSIDE the lock to allow other operations
                # Note: We must release lock before awaiting
            else:
                # No cache hit and no pending future - this request will execute the workflow
                # Create future BEFORE releasing lock to prevent other requests
                # from also creating one
                self._cache_misses += 1
                logfire.info("RAG cache miss", query=query[:50], cache_key=cache_key[:16])
                future: asyncio.Future[dict] = asyncio.Future()
                self._pending_futures[cache_key] = future
                pending_future = None

        # Task 25.2: If we found a pending future, await it OUTSIDE the lock
        if pending_future is not None:
            result = await pending_future
            # Task 25.4: Verify result from pending future is a dict before copying
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict from workflow, got {type(result).__name__}")
            return dict(result)

        # Task 21.1: Execute workflow OUTSIDE the lock to allow concurrent workflow execution
        # Only cache operations need to be protected, not the actual LLM calls
        try:
            result = await super().run(query=query, max_retries=max_retries)

            # Task 25.4: Verify result from workflow is a dict before caching
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict from workflow, got {type(result).__name__}")

            # Task 21.1: Re-acquire lock to store result
            async with self._cache_lock:
                # Task 17.1: Store result in cache with current timestamp
                # Task 21.5: Store a copy to prevent the returned result from mutating the cache
                current_time = time.time()
                self._cache[cache_key] = (dict(result), current_time)

                # Task 17.1: Enforce cache size limit with LRU eviction
                if len(self._cache) > self.llm_settings.rag_cache_size:
                    self._evict_lru_entry()
                    logfire.info(
                        "RAG cache eviction",
                        cache_size=len(self._cache),
                        max_size=self.llm_settings.rag_cache_size,
                    )

                # Task 25.2: Resolve future and remove from pending
                future.set_result(result)
                del self._pending_futures[cache_key]

            return result

        except Exception as e:
            # Task 25.2: If workflow fails, reject future and remove from pending
            async with self._cache_lock:
                if cache_key in self._pending_futures:
                    future.set_exception(e)
                    del self._pending_futures[cache_key]
            raise

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

    def _truncate_chunks(self, chunks: list[str], max_chars: int = 15000) -> list[str]:
        """Truncate chunks to fit within character limit.

        MEDIUM FIX: DRY helper to avoid code duplication between _evaluate_relevance
        and _synthesize_answer. Truncates based on actual character count, not just
        "first N chunks" which could still exceed the limit.

        Args:
            chunks: List of text chunks to truncate.
            max_chars: Maximum total character count (default: 15000).

        Returns:
            List of chunks that fit within max_chars. Returns at least the first
            chunk even if it exceeds the limit (to avoid empty context).
        """
        if not chunks:
            return []

        total = 0
        result: list[str] = []
        for chunk in chunks:
            chunk_len = len(chunk)
            # If adding this chunk would exceed limit and we have at least one chunk, stop
            if total + chunk_len > max_chars and result:
                break
            result.append(chunk)
            total += chunk_len

        # Always return at least the first chunk (even if it exceeds the limit)
        # to ensure we have some context
        return result if result else chunks[:1]

    def _build_prompt(
        self,
        query: str,
        chunks: list[str],
        instruction: str,
        chunk_label: str = "Chunk",
    ) -> str:
        """Build prompt with HTML-escaped query and chunks.

        DRY helper to avoid code duplication between _evaluate_relevance()
        and _synthesize_answer(). Handles HTML escaping and XML tag formatting.

        Args:
            query: User query to escape and include.
            chunks: Document chunks to escape and include.
            instruction: Instruction text to prepend to prompt.
            chunk_label: Label prefix for chunks (default: "Chunk").

        Returns:
            Formatted prompt string with instruction, XML-tagged query, and context.
        """
        sanitized_query = html.escape(query)
        context = "\n\n".join(
            f"{chunk_label} {i + 1}: {html.escape(chunk)}" for i, chunk in enumerate(chunks)
        )
        return f"{instruction}\n\n<query>{sanitized_query}</query>\n\n<context>{context}</context>"

    async def _evaluate_relevance(self, chunks: list[str], query: str) -> str:
        """Evaluate relevance of retrieved chunks using LLM.

        Task 16.24: Added comprehensive error handling and retry logic
        for transient LLM API failures.

        Args:
            chunks: Retrieved document chunks.
            query: Original user query.

        Returns:
            "relevant" if chunks are sufficient, "insufficient" otherwise.
        """
        # MEDIUM FIX: Use helper method to truncate chunks based on actual character count
        original_count = len(chunks)
        chunks = self._truncate_chunks(chunks, max_chars=15000)
        if len(chunks) < original_count:
            logger.warning(
                "Context length exceeded 15000 chars, truncated from %d to %d chunks",
                original_count,
                len(chunks),
            )

        # Task 27.2 & 28.3: HTML escaping for XML tag safety
        #
        # SECURITY NOTE: html.escape() limitations
        # -----------------------------------------
        # We use html.escape() to sanitize user input before inserting it into
        # XML tags (<query> and <context>). This prevents XML injection attacks
        # where malicious input could break out of the tags.
        #
        # HOWEVER, this approach has limitations:
        # 1. LLMs can decode HTML entities - they understand that &lt; means <
        # 2. This means prompt injection is still possible despite HTML escaping
        # 3. Example: A user could inject "&lt;query&gt;malicious&lt;/query&gt;"
        #    which the LLM would interpret as actual XML tags
        #
        # Real defense strategy:
        # ----------------------
        # The fundamental protection against prompt injection comes from:
        # - Using the LLM's messages API with proper system/user role separation
        # - System messages define behavior; user messages are treated as data
        # - Modern LLMs enforce this boundary to prevent prompt injection
        #
        # Current approach (defense-in-depth):
        # - XML tags provide structure for the prompt
        # - html.escape() prevents accidental XML parsing issues
        # - But don't rely on it as primary security mechanism
        #
        # Future improvement (Task 28.3):
        # - Migrate to LLM messages API with explicit role separation
        # - Use system message for instructions, user message for query/context
        # - This provides stronger isolation than text-based XML tags
        #
        # DRY refactoring: Use _build_prompt() helper to avoid code duplication
        instruction = """Given the following chunks and query, assess if the chunks contain \
relevant information to answer the query."""
        prompt = self._build_prompt(query, chunks, instruction, chunk_label="Chunk")
        prompt += """

Respond with exactly one word:
- "relevant" if the chunks contain sufficient information to answer the query
- "insufficient" if the chunks do not contain relevant information

Response:"""

        # Task 16.24: Retry logic with exponential backoff for transient errors
        # Use configurable retry parameters from settings
        max_retries = self.llm_settings.llm_retry_max_attempts
        base_delay = self.llm_settings.llm_retry_base_delay

        for attempt in range(max_retries):
            try:
                # Run evaluation using pre-initialized agent with timeout
                # Task 19.3: Wrap agent execution with timeout to prevent indefinite hangs
                result = await asyncio.wait_for(
                    self._eval_agent.run(prompt),
                    timeout=self.llm_settings.llm_agent_timeout,
                )
                response = result.output.strip().lower()

                # Normalize response
                if "relevant" in response:
                    return "relevant"
                return "insufficient"

            except TimeoutError:
                # Task 20.6: asyncio.TimeoutError indicates the LLM is consistently too slow,
                # not a transient failure. Return graceful fallback immediately (no retries).
                logger.error(
                    "LLM evaluation timed out after %ds (attempt %d/%d): "
                    "LLM is too slow, not retrying",
                    self.llm_settings.llm_agent_timeout,
                    attempt + 1,
                    max_retries,
                )
                return "insufficient"

            except Exception as e:
                # Use explicit error classification from RAGWorkflowError
                is_transient = RAGWorkflowError.is_error_transient(e)

                if attempt < max_retries - 1 and is_transient:
                    # Exponential backoff with jitter for transient errors
                    # Task 17.7: Add jitter to prevent thundering herd
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "Transient error in LLM evaluation (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    # Permanent error or max retries exhausted
                    error_type = "transient" if is_transient else "permanent"
                    logger.error(
                        "LLM evaluation failed after %d attempts (%s error): %s",
                        attempt + 1,
                        error_type,
                        e,
                        exc_info=True,
                    )
                    # Return "insufficient" as safe fallback (graceful error handling)
                    return "insufficient"

        # Fallback (should not reach here)
        return "insufficient"

    async def _synthesize_answer(self, chunks: list[str], query: str) -> str:
        """Synthesize final answer from relevant chunks using LLM.

        Task 16.24: Added comprehensive error handling and retry logic
        for transient LLM API failures.

        Args:
            chunks: Relevant document chunks.
            query: Original user query.

        Returns:
            Synthesized answer.
        """
        # MEDIUM FIX: Use helper method to truncate chunks based on actual character count
        original_count = len(chunks)
        chunks = self._truncate_chunks(chunks, max_chars=15000)
        if len(chunks) < original_count:
            logger.warning(
                "Context length exceeded 15000 chars, truncated from %d to %d chunks",
                original_count,
                len(chunks),
            )

        # Task 27.2 & 28.3: HTML escaping for XML tag safety
        #
        # SECURITY NOTE: html.escape() limitations
        # -----------------------------------------
        # We use html.escape() to sanitize user input before inserting it into
        # XML tags (<query> and <context>). This prevents XML injection attacks
        # where malicious input could break out of the tags.
        #
        # HOWEVER, this approach has limitations:
        # 1. LLMs can decode HTML entities - they understand that &lt; means <
        # 2. This means prompt injection is still possible despite HTML escaping
        # 3. Example: A user could inject "&lt;query&gt;malicious&lt;/query&gt;"
        #    which the LLM would interpret as actual XML tags
        #
        # Real defense strategy:
        # ----------------------
        # The fundamental protection against prompt injection comes from:
        # - Using the LLM's messages API with proper system/user role separation
        # - System messages define behavior; user messages are treated as data
        # - Modern LLMs enforce this boundary to prevent prompt injection
        #
        # Current approach (defense-in-depth):
        # - XML tags provide structure for the prompt
        # - html.escape() prevents accidental XML parsing issues
        # - But don't rely on it as primary security mechanism
        #
        # Future improvement (Task 28.3):
        # - Migrate to LLM messages API with explicit role separation
        # - Use system message for instructions, user message for query/context
        # - This provides stronger isolation than text-based XML tags
        #
        # DRY refactoring: Use _build_prompt() helper to avoid code duplication
        instruction = (
            "Using the following context, provide a clear and concise answer to the query."
        )
        prompt = self._build_prompt(query, chunks, instruction, chunk_label="Source")
        prompt += "\n\nAnswer:"

        # Task 16.24: Retry logic with exponential backoff for transient errors
        # Use configurable retry parameters from settings
        max_retries = self.llm_settings.llm_retry_max_attempts
        base_delay = self.llm_settings.llm_retry_base_delay

        for attempt in range(max_retries):
            try:
                # Generate answer using pre-initialized agent with timeout
                # Task 19.3: Wrap agent execution with timeout to prevent indefinite hangs
                result = await asyncio.wait_for(
                    self._synth_agent.run(prompt),
                    timeout=self.llm_settings.llm_agent_timeout,
                )
                return result.output.strip()

            except TimeoutError:
                # Task 20.6: asyncio.TimeoutError indicates the LLM is consistently too slow,
                # not a transient failure. Return graceful error message immediately (no retries).
                logger.error(
                    "LLM synthesis timed out after %ds (attempt %d/%d): "
                    "LLM is too slow, not retrying",
                    self.llm_settings.llm_agent_timeout,
                    attempt + 1,
                    max_retries,
                )
                return (
                    "I encountered an error while processing your question. "
                    "Please try again or rephrase your question."
                )

            except Exception as e:
                # Use explicit error classification from RAGWorkflowError
                is_transient = RAGWorkflowError.is_error_transient(e)

                if attempt < max_retries - 1 and is_transient:
                    # Exponential backoff with jitter for transient errors
                    # Task 17.7: Add jitter to prevent thundering herd
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "Transient error in LLM synthesis (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    # Permanent error or max retries exhausted
                    error_type = "transient" if is_transient else "permanent"
                    logger.error(
                        "LLM synthesis failed after %d attempts (%s error): %s",
                        attempt + 1,
                        error_type,
                        e,
                        exc_info=True,
                    )
                    # Return graceful error message (graceful error handling)
                    return (
                        "I encountered an error while processing your question. "
                        "Please try again or rephrase your question."
                    )

        # Fallback (should not reach here)
        return (
            "I encountered an error while processing your question. "
            "Please try again or rephrase your question."
        )
