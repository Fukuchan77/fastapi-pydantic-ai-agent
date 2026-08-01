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

    async def _run_agent_with_retry(
        self,
        agent: Agent[None, str],
        prompt: str,
        *,
        op_name: str,
        fallback: str,
    ) -> str:
        """Run an LLM agent with timeout-then-no-retry, transient-then-backoff policy.

        Shared by `_evaluate_relevance` and `_synthesize_answer`, which need the
        identical retry policy and differ only in the agent, prompt, and the
        value to fall back to on failure.

        Args:
            agent: Pre-initialized pydantic-ai agent to run.
            prompt: Full prompt to send to the agent.
            op_name: Human-readable operation name for log messages (e.g. "evaluation").
            fallback: Value returned on timeout, permanent error, or retry exhaustion.

        Returns:
            str: The agent's stripped output, or `fallback` on failure.
        """
        max_retries = self.llm_settings.llm_retry_max_attempts
        base_delay = self.llm_settings.llm_retry_base_delay

        for attempt in range(max_retries):
            try:
                # Wrap agent execution with timeout to prevent indefinite hangs
                result = await asyncio.wait_for(
                    agent.run(prompt),
                    timeout=self.llm_settings.llm_agent_timeout,
                )
                return result.output.strip()

            except TimeoutError:
                # asyncio.TimeoutError indicates the LLM is consistently too slow,
                # not a transient failure. Return the fallback immediately (no retries).
                logger.error(
                    "LLM %s timed out after %ds (attempt %d/%d): LLM is too slow, not retrying",
                    op_name,
                    self.llm_settings.llm_agent_timeout,
                    attempt + 1,
                    max_retries,
                )
                return fallback

            except Exception as e:
                # Use explicit error classification from RAGWorkflowError
                is_transient = RAGWorkflowError.is_error_transient(e)

                if attempt < max_retries - 1 and is_transient:
                    # Exponential backoff with jitter to prevent thundering herd
                    delay = base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "Transient error in LLM %s (attempt %d/%d), retrying in %.1fs: %s",
                        op_name,
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
                        "LLM %s failed after %d attempts (%s error): %s",
                        op_name,
                        attempt + 1,
                        error_type,
                        e,
                        exc_info=True,
                    )
                    return fallback

        # Fallback (should not reach here: loop always returns within the try/except)
        return fallback

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

        response = await self._run_agent_with_retry(
            self._eval_agent, prompt, op_name="evaluation", fallback="insufficient"
        )
        normalized = response.strip().lower()
        # The prompt asks for exactly one word, so try an exact match first -
        # this also protects a genuinely positive verdict that happens to
        # *mention* a negation word (e.g. "relevant, not irrelevant") from
        # being misread by the substring fallback below.
        if normalized in ("relevant", "insufficient"):
            return normalized
        # Free-form fallback: "irrelevant" and "not relevant" both contain
        # the substring "relevant", so a naive `"relevant" in normalized`
        # misreads the LLM's negative verdict as positive and skips the
        # widened-retrieval retry that CRAG relies on for grounding - check
        # negations before the plain substring check below.
        negations = ("insufficient", "irrelevant", "not relevant")
        if any(term in normalized for term in negations):
            return "insufficient"
        return "relevant" if "relevant" in normalized else "insufficient"

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

        fallback = (
            "I encountered an error while processing your question. "
            "Please try again or rephrase your question."
        )
        return await self._run_agent_with_retry(
            self._synth_agent, prompt, op_name="synthesis", fallback=fallback
        )
