"""Prompt-building helpers for `CorrectiveRAGWorkflow`.

Split out of `corrective_rag.py` per `.sdd/steering/file-size-policy.md`.
These helpers are stateless (no `self` attribute reads), but stay as mixin
methods rather than free functions because `tests/unit/workflows/
test_rag_build_prompt_dry.py` calls `workflow._build_prompt(...)` directly.
"""

import html

from app.models.rag import RetrievedHit


# ==============================================================================
# MODULE-LEVEL SECURITY NOTE: html.escape() limitations
# ==============================================================================
# We use html.escape() throughout this module to sanitize user input before
# inserting it into XML tags (<query> and <context>). This prevents XML
# injection attacks where malicious input could break out of the tags.
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
# Future improvement:
# - Migrate to LLM messages API with explicit role separation
# - Use system message for instructions, user message for query/context
# - This provides stronger isolation than text-based XML tags
# ==============================================================================


class PromptBuildingMixin:
    """Prompt truncation/formatting helpers for RAG evaluation and synthesis.

    Composed into `CorrectiveRAGWorkflow` (`app/workflows/corrective_rag.py`);
    stateless (no shared attributes required from the composing class).
    """

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

    def _truncate_hits(
        self, hits: list[RetrievedHit], max_chars: int = 15000
    ) -> list[RetrievedHit]:
        """Truncate ordered hits to fit within a character budget.

        Mirrors `_truncate_chunks()` but preserves the `RetrievedHit` objects
        (chunk_id/score) so the surviving hits can be cited, instead of
        collapsing to plain text.

        Args:
            hits: Hits to truncate, already ordered (highest priority first).
            max_chars: Maximum total character count across hit texts.

        Returns:
            The leading hits whose text fits within max_chars. Always returns
            at least the first hit (even if it exceeds the limit) when the
            input is non-empty, to avoid an ungrounded answer.
        """
        if not hits:
            return []

        total = 0
        result: list[RetrievedHit] = []
        for hit in hits:
            hit_len = len(hit.text)
            if total + hit_len > max_chars and result:
                break
            result.append(hit)
            total += hit_len

        return result if result else hits[:1]

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
