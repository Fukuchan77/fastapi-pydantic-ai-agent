"""Regression tests for `ResultCacheMixin.run()` cancellation handling.

Covers the adversarial-review CRITICAL finding: the leader's `super().run()`
being cancelled (e.g. by `asyncio.timeout` in `app/api/v1/rag.py`, or a
client disconnect) must not `cancel()` the shared `_pending_futures` entry,
because any follower already awaiting it (thundering-herd de-dup) would then
have that same `CancelledError` raised out of its own `run()` call — which
Starlette treats as a dropped connection and answers with nothing at all,
instead of the follower's own `asyncio.timeout` block converting it to a
normal 504. It also must not leak the `_pending_futures` entry itself.
"""

import asyncio
from collections import OrderedDict

import pytest

from app.config import Settings
from app.workflows.rag_cache import ResultCacheMixin
from tests.conftest import build_test_settings


class _HangingBase:
    """Stands in for `Workflow.run()` in the MRO below `ResultCacheMixin`."""

    _delay: float

    async def run(self, query: str, max_retries: int = 3) -> dict:
        await asyncio.sleep(self._delay)
        return {"answer": "ok", "query": query}


class _StubVectorStore:
    """Real int `.generation`, so no double leaks a Mock into a cache key."""

    @property
    def generation(self) -> int:
        return 0


class _CachedWorkflow(ResultCacheMixin, _HangingBase):
    def __init__(self, settings: Settings, delay: float = 10.0) -> None:
        self.llm_settings = settings
        self.vector_store = _StubVectorStore()
        self._cache: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = asyncio.Lock()
        self._pending_futures: dict[str, asyncio.Future[dict]] = {}
        self._delay = delay


class TestLeaderCancellationDoesNotDropFollower:
    """A cancelled leader must resolve the shared future with TimeoutError, not cancel it."""

    @pytest.mark.asyncio
    async def test_follower_gets_timeout_error_not_cancelled_error(self) -> None:
        """A leader cancelled by its own timeout must not cancel the follower's await.

        Mirrors app/api/v1/rag.py: each request wraps workflow.run() in its
        own `asyncio.timeout`. The leader's timeout fires first; the
        follower (already awaiting the leader's pending future) must see a
        `TimeoutError` it can map to its own 504 - not a `CancelledError`
        that would abort its request with no HTTP response at all.
        """
        settings = build_test_settings(llm_model="openai:gpt-4o", rag_cache_ttl=300)
        workflow = _CachedWorkflow(settings, delay=10.0)

        async def leader() -> str:
            try:
                async with asyncio.timeout(0.1):
                    await workflow.run(query="q")
            except TimeoutError:
                return "leader:timeout"
            return "leader:unexpected-success"

        async def follower() -> str:
            await asyncio.sleep(0.02)  # join after the leader registers the pending future
            try:
                async with asyncio.timeout(5.0):
                    await workflow.run(query="q")
            except TimeoutError:
                return "follower:timeout"
            except asyncio.CancelledError:
                return "follower:cancelled"
            return "follower:unexpected-success"

        leader_result, follower_result = await asyncio.gather(leader(), follower())

        assert leader_result == "leader:timeout"
        assert follower_result == "follower:timeout", (
            "the follower observed the leader's CancelledError directly instead of "
            "a TimeoutError it could map to its own 504 response"
        )

    @pytest.mark.asyncio
    async def test_pending_future_is_cleaned_up_after_cancellation(self) -> None:
        """A cancelled leader must not leave a permanently-unresolved pending future."""
        settings = build_test_settings(llm_model="openai:gpt-4o", rag_cache_ttl=300)
        workflow = _CachedWorkflow(settings, delay=10.0)
        cache_key = workflow._generate_cache_key("q", 3)

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await workflow.run(query="q")

        assert cache_key not in workflow._pending_futures, (
            "the pending future was leaked after cancellation - every future request "
            "for this query would await a future nobody will ever resolve"
        )

    @pytest.mark.asyncio
    async def test_genuine_workflow_exception_still_propagates_to_follower(self) -> None:
        """A real (non-cancellation) workflow failure must still reach the follower as itself."""

        class _FailingBase:
            async def run(self, query: str, max_retries: int = 3) -> dict:
                await asyncio.sleep(0.02)
                raise ValueError("boom")

        class _FailingWorkflow(ResultCacheMixin, _FailingBase):
            def __init__(self, settings: Settings) -> None:
                self.llm_settings = settings
                self.vector_store = _StubVectorStore()
                self._cache = OrderedDict()
                self._cache_hits = 0
                self._cache_misses = 0
                self._cache_lock = asyncio.Lock()
                self._pending_futures = {}

        settings = build_test_settings(llm_model="openai:gpt-4o", rag_cache_ttl=300)
        workflow = _FailingWorkflow(settings)

        async def leader() -> str:
            try:
                await workflow.run(query="q")
            except ValueError as e:
                return f"leader:{e}"
            return "leader:unexpected-success"

        async def follower() -> str:
            await asyncio.sleep(0.005)
            try:
                await workflow.run(query="q")
            except ValueError as e:
                return f"follower:{e}"
            return "follower:unexpected-success"

        leader_result, follower_result = await asyncio.gather(leader(), follower())

        assert leader_result == "leader:boom"
        assert follower_result == "follower:boom"
        assert workflow._pending_futures == {}
