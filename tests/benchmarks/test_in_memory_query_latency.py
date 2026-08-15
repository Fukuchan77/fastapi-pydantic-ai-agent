"""Benchmark: contiguous event-loop-blocking time of `InMemoryVectorStore.query()`.

Task 11.3 (`.sdd/specs/002-review-roadmap-remediation/tasks.md`, Req 6.1).
Task 11.1 offloaded TF-IDF scoring to a worker thread via `asyncio.to_thread`
so the event loop stays responsive during scoring, per plan.md L4.4's NFR:
"no more than 50ms of contiguous synchronous work at the 1000-document
limit". The offload satisfies that NFR by construction only if nothing else
in `query()`'s path blocks for longer than the budget -- this measures that
directly instead of assuming the offload alone is sufficient.

`query()`'s only synchronous (non-offloaded) work is the code that runs
inside `_ranked_hits`'s `async with self._lock:` block: parameter validation,
query tokenization, IDF computation (cold on the first query after an
ingest -- the worst case exercised here), and the snapshot copy. None of
that contains an `await`, so it runs as one uninterruptible stretch on the
event loop before `asyncio.to_thread` hands scoring to a worker thread and
yields control back.

A concurrent heartbeat coroutine that ticks via `asyncio.sleep(0)` is the
probe: while the event loop is free, ticks are near-instantaneous, so the
longest observed gap between ticks is (within scheduling noise) exactly the
longest stretch the event loop was unable to service anything else -- the
"contiguous synchronous work" the budget bounds.

Run with: `mise run test:benchmark` (this module is not part of `test:ci`,
matching the rest of `tests/benchmarks/`).
"""

import asyncio
import time

import pytest

from app.stores.vector_store import InMemoryVectorStore


# NFR budget from plan.md L4.4: no more than 50ms of contiguous synchronous
# work at the 1000-document limit.
MAX_CONTIGUOUS_BLOCK_SECONDS = 0.050

DOCUMENT_LIMIT = InMemoryVectorStore.DEFAULT_MAX_DOCUMENTS


async def _max_heartbeat_gap(coro) -> tuple[object, float]:
    """Run `coro` while probing event-loop responsiveness with a heartbeat.

    Args:
        coro: The coroutine to run concurrently with the heartbeat probe.

    Returns:
        A tuple of (coro's result, longest gap in seconds between
        consecutive heartbeat ticks observed while coro was running).
    """
    gaps: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        result = await coro
    finally:
        stop.set()
        await heartbeat_task

    return result, max(gaps)


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_query_contiguous_sync_work_at_document_limit_under_budget() -> None:
    """A query at the 1000-document limit must not block the event loop for >50ms.

    Exercises the worst case for the synchronous portion of `query()`: the
    first query after ingesting up to `max_documents`, which pays for a cold
    IDF-cache computation over the full corpus inside the same lock hold as
    validation and the snapshot copy -- all of it before `asyncio.to_thread`
    hands scoring off the event loop.
    """
    store = InMemoryVectorStore(max_documents=DOCUMENT_LIMIT)
    await store.add_documents(
        [f"document {i} discusses topic {i % 50} in some detail" for i in range(DOCUMENT_LIMIT)]
    )

    result, max_gap = await _max_heartbeat_gap(store.query("topic 10", top_k=5))

    assert result, "expected at least one match for a term present in the corpus"
    print("\n=== InMemoryVectorStore.query() Contiguous Block Benchmark ===")
    print(f"Documents: {DOCUMENT_LIMIT}")
    print(f"Max event-loop-blocking gap: {max_gap * 1000:.3f}ms")
    print(f"Budget: {MAX_CONTIGUOUS_BLOCK_SECONDS * 1000:.0f}ms")

    assert max_gap < MAX_CONTIGUOUS_BLOCK_SECONDS, (
        f"query() blocked the event loop for {max_gap * 1000:.3f}ms at the "
        f"{DOCUMENT_LIMIT}-document limit, exceeding the "
        f"{MAX_CONTIGUOUS_BLOCK_SECONDS * 1000:.0f}ms budget (plan.md L4.4, Req 6.1)"
    )


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_query_contiguous_sync_work_with_warm_idf_cache_under_budget() -> None:
    """A repeat query (warm IDF cache) at the document limit stays well under budget.

    Complements the cold-cache case above: once the IDF cache is populated,
    the only remaining synchronous work is validation, tokenization, and the
    snapshot copy, so this should be strictly cheaper than the cold-cache
    measurement -- included as a baseline, not because it is the binding case.
    """
    store = InMemoryVectorStore(max_documents=DOCUMENT_LIMIT)
    await store.add_documents(
        [f"document {i} discusses topic {i % 50} in some detail" for i in range(DOCUMENT_LIMIT)]
    )
    await store.query("topic 10", top_k=5)  # Warm the IDF cache before measuring.

    result, max_gap = await _max_heartbeat_gap(store.query("topic 20", top_k=5))

    assert result
    print("\n=== InMemoryVectorStore.query() Warm-Cache Contiguous Block Benchmark ===")
    print(f"Max event-loop-blocking gap: {max_gap * 1000:.3f}ms")

    assert max_gap < MAX_CONTIGUOUS_BLOCK_SECONDS, (
        f"query() blocked the event loop for {max_gap * 1000:.3f}ms with a warm "
        f"IDF cache at the {DOCUMENT_LIMIT}-document limit, exceeding the "
        f"{MAX_CONTIGUOUS_BLOCK_SECONDS * 1000:.0f}ms budget (plan.md L4.4, Req 6.1)"
    )
