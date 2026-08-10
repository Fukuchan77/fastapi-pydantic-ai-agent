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
from app.stores.vector_store import VectorStore


class ResultCacheMixin:
    """TTL + LRU result cache with thundering-herd protection.

    Composed into `CorrectiveRAGWorkflow` (`app/workflows/corrective_rag.py`);
    the class-level annotations below declare the attributes that class's
    `__init__` must set up before any of this mixin's methods run — they are
    not defaults, just the interface this mixin depends on (`ty` strict needs
    them to resolve `self.llm_settings`/`self._cache*` statically).
    """

    llm_settings: Settings
    vector_store: VectorStore
    _cache: OrderedDict[str, tuple[dict, float]]
    _cache_hits: int
    _cache_misses: int
    _cache_lock: asyncio.Lock
    _pending_futures: dict[str, asyncio.Future[dict]]

    def _generate_cache_key(self, query: str, max_retries: int) -> str:
        """Generate cache key from query, max_retries, and the store's content version.

        The store's `generation` (Req 2.1/2.2) is included so an ingest - which
        advances it - makes every pre-ingest entry miss on the next identical
        query, without disturbing a request already in flight under the
        pre-ingest key (Req 2.3): `run()` derives one key and reuses it for
        both `_cache` and `_pending_futures`, so a generation bump separates
        the two generations' keys automatically.

        Args:
            query: User query string.
            max_retries: Maximum retry attempts.

        Returns:
            str: SHA256 hash of query + max_retries + store generation for cache key.
        """
        # Combine query, max_retries, and the store's content version into a
        # single string so a post-ingest query never resolves from a
        # pre-ingest entry.
        key_material = f"{query}|{max_retries}|{self.vector_store.generation}"
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
            # `super().run()` resolves to `Workflow.run` only at the composition
            # site (`CorrectiveRAGWorkflow(ResultCacheMixin, LLMCallMixin,
            # Workflow)`); `ResultCacheMixin` itself doesn't inherit `Workflow`; a
            # sibling can't be declared without inheriting it directly, which
            # would break the mixin split. Tracked as design debt (Protocol-typed
            # mixin bases), not fixed here.
            result = await super().run(query=query, max_retries=max_retries)  # ty: ignore[unresolved-attribute]
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
            # `shield` is load-bearing, not defensive: cancelling a task that is
            # awaiting a future cancels that future (`Task.cancel()` cancels its
            # `_fut_waiter`). Without the shield, a follower whose own
            # `asyncio.timeout` fires - or whose client disconnects - would
            # cancel the future it *shares* with the leader, and the leader's
            # later `set_result()` would raise `InvalidStateError`, failing a
            # run that had already succeeded and populated the cache.
            result = await asyncio.shield(pending_future)
            # Verify result from pending future is a dict before copying
            if not isinstance(result, dict):
                raise TypeError(f"Expected dict from workflow, got {type(result).__name__}")
            return dict(result)

        # Execute workflow OUTSIDE the lock to allow concurrent workflow execution
        # Only cache operations need to be protected, not the actual LLM calls
        try:
            # `super().run()` resolves to `Workflow.run` only at the composition
            # site (`CorrectiveRAGWorkflow(ResultCacheMixin, LLMCallMixin,
            # Workflow)`); `ResultCacheMixin` itself doesn't inherit `Workflow`; a
            # sibling can't be declared without inheriting it directly, which
            # would break the mixin split. Tracked as design debt (Protocol-typed
            # mixin bases), not fixed here.
            result = await super().run(query=query, max_retries=max_retries)  # ty: ignore[unresolved-attribute]

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

            # Resolve future and remove from pending. The `done()` guard pairs
            # with the `shield` above: any path that resolves the shared future
            # before the leader gets here must not turn a successful run into an
            # `InvalidStateError`.
            if not future.done():
                future.set_result(result)
            self._pending_futures.pop(cache_key, None)
            return result

        except BaseException as e:
            # Catches both `Exception` (genuine workflow failures) and
            # `CancelledError` (raised by a timeout via
            # asyncio.timeout/wait_for, or a client disconnect).
            # `CancelledError` is a `BaseException`, not an `Exception`, so a
            # plain `except Exception` never sees it - without this branch,
            # `future` and its `_pending_futures` entry were left stuck
            # forever, so every later request for the same query would await
            # a future nobody would ever resolve, and time out too.
            #
            # This block is entirely await-free (no `async with` on
            # `_cache_lock`), so it cannot itself be interrupted by a second
            # cancellation and leave the entry orphaned. The dict pop/del
            # below don't need the lock for the same reason: with no
            # `await` between the check and the mutation, no other coroutine
            # can be scheduled in between regardless of whether the lock is
            # held.
            if self._pending_futures.get(cache_key) is future:
                del self._pending_futures[cache_key]

            if not future.done():
                if isinstance(e, Exception):
                    # Genuine workflow failure - other requests awaiting
                    # this future should see the same error, matching the
                    # pre-existing behavior for non-cancellation failures.
                    future.set_exception(e)
                else:
                    # CancelledError (or any other non-Exception
                    # BaseException): never propagate it as-is to the
                    # shared future. A follower already awaiting this
                    # future (thundering-herd de-dup, see the pending-future
                    # branch above) would otherwise see a CancelledError
                    # raised out of its OWN `run()` call, which the FastAPI
                    # route has no chance to map to a response - Starlette
                    # treats an unhandled CancelledError as a dropped
                    # connection and sends nothing back at all. A
                    # TimeoutError instead lets the follower's own
                    # `asyncio.timeout` block (`app/api/v1/rag.py`) catch it
                    # and return its normal 504, exactly as if it had run
                    # the workflow itself and hit its own timeout.
                    future.set_exception(TimeoutError("RAG leader run was cancelled; retry"))
                # If no follower ever joined (no one is awaiting `future`),
                # nothing consumes the exception set above - asyncio logs a
                # spurious "exception was never retrieved" warning when the
                # future is garbage-collected. The done-callback consumes it.
                future.add_done_callback(lambda f: f.exception())

            raise
