"""Performance benchmark tests for fastapi-pydantic-ai-agent.

Task 17.5: Baseline latency, throughput, and cache hit rate metrics.

These tests measure performance characteristics:
- Latency: Time taken for individual requests
- Throughput: Requests processed per second
- Cache hit rate: Effectiveness of RAG result caching

Run with: pytest tests/benchmarks/ -v
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import AsyncClient


# Benchmark thresholds (in seconds)
AGENT_CHAT_LATENCY_THRESHOLD = 2.0  # Max acceptable latency for agent chat
RAG_QUERY_LATENCY_THRESHOLD = 3.0  # Max acceptable latency for RAG query
MIN_THROUGHPUT = 5.0  # Minimum requests per second
MIN_CACHE_HIT_RATE = 0.5  # 50% cache hit rate for repeated queries


@pytest_asyncio.fixture
async def benchmark_client(client: AsyncClient) -> AsyncIterator[AsyncClient]:
    """Create test client with fast mock LLM for benchmarking.

    Reuses the client fixture from conftest.py which already has:
    - FunctionModel for fast, deterministic responses
    - Proper lifespan management
    - Test authentication configured

    This fixture adds benchmark-specific setup like populating the vector store.
    """
    # Populate vector store with test data for benchmarking
    # Access the app through the client's transport
    from app.main import app

    await app.state.vector_store.add_documents(
        [
            "FastAPI is a modern web framework for Python",
            "Pydantic AI provides type-safe AI agents",
            "LlamaIndex Workflows enable event-driven RAG",
        ]
    )

    # Add authentication header to all requests
    client.headers["X-API-Key"] = "test-api-key-12345"

    yield client


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_agent_chat_latency(benchmark_client):
    """Measure baseline latency for agent chat endpoint.

    Task 17.5: Establishes baseline latency metric for POST /v1/agent/chat.
    Measures time from request to response completion.

    Acceptance Criteria:
    - Latency should be below AGENT_CHAT_LATENCY_THRESHOLD
    - Reports p50, p95, p99 percentiles
    """
    # Import benchmark utilities (will fail until implemented)
    from tests.benchmarks.utils import BenchmarkResults
    from tests.benchmarks.utils import measure_latency

    # Measure latency over multiple requests
    latencies = await measure_latency(
        client=benchmark_client,
        method="POST",
        url="/v1/agent/chat",
        json={"message": "What is FastAPI?"},
        num_requests=20,
    )

    # Calculate statistics
    results = BenchmarkResults.from_latencies(latencies)

    # Assert performance meets threshold
    assert results.p50 < AGENT_CHAT_LATENCY_THRESHOLD, (
        f"Agent chat p50 latency {results.p50:.3f}s exceeds "
        f"threshold {AGENT_CHAT_LATENCY_THRESHOLD}s"
    )

    # Print results for baseline tracking
    print("\n=== Agent Chat Latency Benchmark ===")
    print(f"Requests: {len(latencies)}")
    print(f"p50: {results.p50:.3f}s")
    print(f"p95: {results.p95:.3f}s")
    print(f"p99: {results.p99:.3f}s")
    print(f"Min: {results.min:.3f}s")
    print(f"Max: {results.max:.3f}s")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_rag_query_latency(benchmark_client):
    """Measure baseline latency for RAG query endpoint.

    Task 17.5: Establishes baseline latency metric for POST /v1/rag/query.
    Measures full workflow: search -> evaluate -> synthesize.

    Acceptance Criteria:
    - Latency should be below RAG_QUERY_LATENCY_THRESHOLD
    - Reports p50, p95, p99 percentiles
    """
    from tests.benchmarks.utils import BenchmarkResults
    from tests.benchmarks.utils import measure_latency

    # Measure latency over multiple requests
    latencies = await measure_latency(
        client=benchmark_client,
        method="POST",
        url="/v1/rag/query",
        json={"query": "What is FastAPI?", "max_retries": 3},
        num_requests=20,
    )

    # Calculate statistics
    results = BenchmarkResults.from_latencies(latencies)

    # Assert performance meets threshold
    assert results.p50 < RAG_QUERY_LATENCY_THRESHOLD, (
        f"RAG query p50 latency {results.p50:.3f}s exceeds threshold {RAG_QUERY_LATENCY_THRESHOLD}s"
    )

    # Print results for baseline tracking
    print("\n=== RAG Query Latency Benchmark ===")
    print(f"Requests: {len(latencies)}")
    print(f"p50: {results.p50:.3f}s")
    print(f"p95: {results.p95:.3f}s")
    print(f"p99: {results.p99:.3f}s")
    print(f"Min: {results.min:.3f}s")
    print(f"Max: {results.max:.3f}s")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_throughput_agent_chat(benchmark_client):
    """Measure throughput (requests per second) for agent chat endpoint.

    Task 17.5: Measures how many concurrent requests the system can handle.

    Acceptance Criteria:
    - Throughput should exceed MIN_THROUGHPUT requests/second
    - Reports total requests, duration, and requests/second
    """
    from tests.benchmarks.utils import measure_throughput

    # Measure throughput with concurrent requests
    result = await measure_throughput(
        client=benchmark_client,
        method="POST",
        url="/v1/agent/chat",
        json={"message": "What is FastAPI?"},
        num_requests=50,
        max_concurrency=10,
    )

    # Assert performance meets threshold
    assert result.requests_per_second >= MIN_THROUGHPUT, (
        f"Throughput {result.requests_per_second:.2f} req/s is below minimum {MIN_THROUGHPUT} req/s"
    )

    # Print results for baseline tracking
    print("\n=== Agent Chat Throughput Benchmark ===")
    print(f"Total requests: {result.total_requests}")
    print(f"Duration: {result.duration:.3f}s")
    print(f"Throughput: {result.requests_per_second:.2f} req/s")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_cache_hit_rate(benchmark_client):
    """Measure cache hit rate for repeated RAG queries.

    Task 17.5: Validates that query result caching (Task 17.1) improves
    performance by reducing redundant LLM calls.

    Acceptance Criteria:
    - Cache hit rate should exceed MIN_CACHE_HIT_RATE for repeated queries
    - Reports total queries, cache hits, misses, and hit rate
    """
    from tests.benchmarks.utils import measure_cache_hit_rate

    # Measure cache hit rate with repeated queries
    result = await measure_cache_hit_rate(
        client=benchmark_client,
        url="/v1/rag/query",
        queries=[
            {"query": "What is FastAPI?", "max_retries": 3},
            {"query": "What is FastAPI?", "max_retries": 3},  # Repeat (should hit cache)
            {"query": "What is Pydantic AI?", "max_retries": 3},
            {"query": "What is Pydantic AI?", "max_retries": 3},  # Repeat (should hit cache)
            {"query": "What is FastAPI?", "max_retries": 3},  # Repeat again
        ],
    )

    # Assert cache hit rate meets threshold
    assert result.hit_rate >= MIN_CACHE_HIT_RATE, (
        f"Cache hit rate {result.hit_rate:.1%} is below minimum {MIN_CACHE_HIT_RATE:.1%}"
    )

    # Print results for baseline tracking
    print("\n=== RAG Cache Hit Rate Benchmark ===")
    print(f"Total queries: {result.total_queries}")
    print(f"Unique queries: {result.unique_queries}")
    print(f"Cache hits: {result.cache_hits}")
    print(f"Cache misses: {result.cache_misses}")
    print(f"Hit rate: {result.hit_rate:.1%}")
    print(f"Average cached latency: {result.avg_cached_latency:.3f}s")
    print(f"Average uncached latency: {result.avg_uncached_latency:.3f}s")
    print(f"Speedup: {result.speedup:.1f}x")


@pytest.mark.asyncio
@pytest.mark.benchmark
async def test_end_to_end_performance_profile(benchmark_client):
    """Comprehensive performance profile across all endpoints.

    Task 17.5: Generates complete performance baseline covering
    agent chat, RAG query, and streaming endpoints.

    Reports:
    - Latency percentiles for each endpoint
    - Throughput metrics
    - Cache effectiveness
    - Resource usage trends
    """
    from tests.benchmarks.utils import generate_performance_profile

    # Generate comprehensive performance profile
    profile = await generate_performance_profile(
        client=benchmark_client,
        endpoints=[
            {"method": "POST", "url": "/v1/agent/chat", "json": {"message": "Test"}},
            {"method": "POST", "url": "/v1/rag/query", "json": {"query": "Test", "max_retries": 3}},
        ],
        num_requests_per_endpoint=30,
        max_concurrency=10,
    )

    # Print comprehensive report
    print("\n=== End-to-End Performance Profile ===")
    for endpoint_result in profile.endpoints:
        print(f"\nEndpoint: {endpoint_result.endpoint}")
        print(f"  Latency p50: {endpoint_result.latency_p50:.3f}s")
        print(f"  Latency p95: {endpoint_result.latency_p95:.3f}s")
        print(f"  Throughput: {endpoint_result.throughput:.2f} req/s")
        print(f"  Success rate: {endpoint_result.success_rate:.1%}")

    print("\nOverall:")
    print(f"  Total requests: {profile.total_requests}")
    print(f"  Duration: {profile.duration:.3f}s")
    print(f"  Average throughput: {profile.avg_throughput:.2f} req/s")

    # Assert overall performance is acceptable
    assert profile.avg_throughput >= MIN_THROUGHPUT, (
        f"Overall throughput {profile.avg_throughput:.2f} req/s is below minimum"
    )
