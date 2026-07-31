"""TTL + LRU result-cache mixin for `CorrectiveRAGWorkflow`.

Provides thundering-herd-safe caching around `Workflow.run()`. Split out of
`corrective_rag.py` per `.sdd/steering/file-size-policy.md`; not used
standalone (methods read `self.llm_settings`/`self._cache*` attributes set up
by `CorrectiveRAGWorkflow.__init__`).
"""

import asyncio
import hashlib
import time
from collections import OrderedDict

import logfire

from app.config import Settings


class ResultCacheMixin:
    """TTL + LRU result cache with thundering-herd protection.

    Composed into `CorrectiveRAGWorkflow` (`app/workflows/corrective_rag.py`);
    the class-level annotations below declare the attributes that class's
    `__init__` must set up before any of this mixin's methods run — they are
    not defaults, just the interface this mixin depends on (`ty` strict needs
    them to resolve `self.llm_settings`/`self._cache*` statically).
    """

    llm_settings: Settings
    _cache: OrderedDict[str, tuple[dict, float]]
    _cache_hits: int
    _cache_misses: int
    _cache_lock: asyncio.Lock
    _pending_futures: dict[str, asyncio.Future[dict]]

    def _generate_cache_key(self, query: str, max_retries: int) -> str:
        """Generate cache key from query and max_retries parameter.

        Cache key includes both query and max_retries because
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

        Removes entries where current_time - cached_time > ttl.
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

        OrderedDict maintains insertion order, so the first item
        is the least recently used (after move_to_end on cache hits).
        """
        if self._cache:
            self._cache.popitem(last=False)  # Remove first (oldest) item

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics for monitoring.

        Exposes cache hit/miss/size metrics for observability.

        Returns:
            dict: Dictionary with 'hits', 'misses', and 'size' keys.
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    async def run(self, query: str, max_retries: int = 3) -> dict:
        """Run the workflow with caching support.

        Overrides parent run() to check cache before executing workflow.
        If cache hit, returns cached result immediately. Otherwise, executes workflow
        and caches the result.

        Args:
            query: User query string.
            max_retries: Maximum retry attempts for relevance evaluation.

        Returns:
            dict: Workflow result with answer, context_found, and search_count.
        """
        # Check if caching is disabled (ttl=0)
        if self.llm_settings.rag_cache_ttl == 0:
            # Cache disabled, execute workflow directly
            result = await super().run(query=query, max_retries=max_retries)  # type: ignore[misc]
            return result

        # Generate cache key
        cache_key = self._generate_cache_key(query, max_retries)

        # Use double-check locking to prevent thundering herd
        async with self._cache_lock:
            # Evict expired entries before checking cache
            self._evict_expired_entries()

            # Check cache for existing result
            if cache_key in self._cache:
                # Cache hit - move to end (mark as recently used) and return cached result
                cached_result, _ = self._cache[cache_key]
                self._cache.move_to_end(cache_key)
                self._cache_hits += 1
                logfire.info("RAG cache hit", query=query[:50], cache_key=cache_key[:16])
                # Verify cached result is a dict before copying
                if not isinstance(cached_result, dict):
                    raise TypeError(f"Expected dict from cache, got {type(cached_result).__name__}")
                # Return a copy to prevent callers from mutating the cached dict
                return dict(cached_result)

            # Check if there's a pending future for this query (thundering herd fix)
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

        # If we found a pending future, await it OUTSIDE the lock
        if pending_future is not None:
            result = await pending_future
            # Verify result from pending future is a dict before copying
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict from workflow, got {type(result).__name__}")
            return dict(result)

        # Execute workflow OUTSIDE the lock to allow concurrent workflow execution
        # Only cache operations need to be protected, not the actual LLM calls
        try:
            result = await super().run(query=query, max_retries=max_retries)  # type: ignore[misc]

            # Verify result from workflow is a dict before caching
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict from workflow, got {type(result).__name__}")

            # Re-acquire lock to store result
            async with self._cache_lock:
                # Store result in cache with current timestamp
                # Store a copy to prevent the returned result from mutating the cache
                current_time = time.time()
                self._cache[cache_key] = (dict(result), current_time)

                # Enforce cache size limit with LRU eviction
                if len(self._cache) > self.llm_settings.rag_cache_size:
                    self._evict_lru_entry()
                    logfire.info(
                        "RAG cache eviction",
                        cache_size=len(self._cache),
                        max_size=self.llm_settings.rag_cache_size,
                    )

                # Resolve future and remove from pending
                future.set_result(result)
                del self._pending_futures[cache_key]

            return result

        except Exception as e:
            # If workflow fails, reject future and remove from pending
            async with self._cache_lock:
                if cache_key in self._pending_futures:
                    future.set_exception(e)
                    del self._pending_futures[cache_key]
            raise
