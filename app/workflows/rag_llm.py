"""LLM-calling steps (evaluation, synthesis) for `CorrectiveRAGWorkflow`.

Split out of `corrective_rag.py` per `.sdd/steering/file-size-policy.md`;
not used standalone (methods read `self._eval_agent`/`self._synth_agent`/
`self.llm_settings` set up by `CorrectiveRAGWorkflow.__init__`, and call
`self._truncate_chunks`/`self._truncate_hits`/`self._build_prompt` from
`PromptBuildingMixin`).
"""

import asyncio
import logging
import random

from pydantic_ai import Agent

from app.config import Settings
from app.models.rag import RetrievedHit
from app.workflows.exceptions import RAGWorkflowError
from app.workflows.rag_prompts import PromptBuildingMixin


logger = logging.getLogger(__name__)


class LLMCallMixin(PromptBuildingMixin):
    """Relevance-evaluation and answer-synthesis LLM calls with retry/timeout handling.

    Composed into `CorrectiveRAGWorkflow` (`app/workflows/corrective_rag.py`);
    the class-level annotations below declare the attributes that class's
    `__init__` must set up before any of this mixin's methods run — they are
    not defaults, just the interface this mixin depends on (`ty` strict needs
    them to resolve `self._eval_agent`/`self._synth_agent`/`self.llm_settings`
    statically). Inherits `PromptBuildingMixin` for `_truncate_chunks`/
    `_truncate_hits`/`_build_prompt`, used by both methods below.
    """

    llm_settings: Settings
    _eval_agent: Agent[None, str]
    _synth_agent: Agent[None, str]

    async def _evaluate_relevance(self, chunks: list[str], query: str) -> str:
        """Evaluate relevance of retrieved chunks using LLM.

        Uses configurable retry logic with exponential backoff for transient
        LLM API failures. Returns "insufficient" as safe fallback on error.

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

        # See MODULE-LEVEL SECURITY NOTE in rag_prompts.py for html.escape() limitations
        # DRY refactoring: Use _build_prompt() helper to avoid code duplication
        instruction = """Given the following chunks and query, assess if the chunks contain \
relevant information to answer the query."""
        prompt = self._build_prompt(query, chunks, instruction, chunk_label="Chunk")
        prompt += """

Respond with exactly one word:
- "relevant" if the chunks contain sufficient information to answer the query
- "insufficient" if the chunks do not contain relevant information

Response:"""

        # Retry logic with exponential backoff for transient errors
        # Use configurable retry parameters from settings
        max_retries = self.llm_settings.llm_retry_max_attempts
        base_delay = self.llm_settings.llm_retry_base_delay

        for attempt in range(max_retries):
            try:
                # Run evaluation using pre-initialized agent with timeout
                # Wrap agent execution with timeout to prevent indefinite hangs
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
                # asyncio.TimeoutError indicates the LLM is consistently too slow,
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
                    # Add jitter to prevent thundering herd
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

    async def _synthesize_answer(self, hits: list[RetrievedHit], query: str) -> str:
        """Synthesize final answer from relevant hits using LLM.

        Uses configurable retry logic with exponential backoff for transient
        LLM API failures. Returns graceful error message as fallback on error.

        Args:
            hits: Relevant retrieved hits to ground the answer in.
            query: Original user query.

        Returns:
            Synthesized answer.
        """
        # MEDIUM FIX: Use helper method to truncate hits based on actual character count
        # (defense-in-depth; callers such as synthesize() already truncate before this call)
        original_count = len(hits)
        hits = self._truncate_hits(hits, max_chars=15000)
        if len(hits) < original_count:
            logger.warning(
                "Context length exceeded 15000 chars, truncated from %d to %d chunks",
                original_count,
                len(hits),
            )

        # See MODULE-LEVEL SECURITY NOTE in rag_prompts.py for html.escape() limitations
        # DRY refactoring: Use _build_prompt() helper to avoid code duplication
        instruction = (
            "Using the following context, provide a clear and concise answer to the query."
        )
        prompt = self._build_prompt(
            query, [hit.text for hit in hits], instruction, chunk_label="Source"
        )
        prompt += "\n\nAnswer:"

        # Retry logic with exponential backoff for transient errors
        # Use configurable retry parameters from settings
        max_retries = self.llm_settings.llm_retry_max_attempts
        base_delay = self.llm_settings.llm_retry_base_delay

        for attempt in range(max_retries):
            try:
                # Generate answer using pre-initialized agent with timeout
                # Wrap agent execution with timeout to prevent indefinite hangs
                result = await asyncio.wait_for(
                    self._synth_agent.run(prompt),
                    timeout=self.llm_settings.llm_agent_timeout,
                )
                return result.output.strip()

            except TimeoutError:
                # asyncio.TimeoutError indicates the LLM is consistently too slow,
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
                    # Add jitter to prevent thundering herd
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
