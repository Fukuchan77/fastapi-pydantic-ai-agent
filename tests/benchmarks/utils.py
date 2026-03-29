"""Benchmark utilities for performance testing.

Utilities for measuring latency, throughput, and cache hit rates.
"""

import asyncio
import statistics
import time
from dataclasses import dataclass

from httpx import AsyncClient


@dataclass
class BenchmarkResults:
    """Results from latency benchmark measurements.

    Attributes:
        p50: 50th percentile (median) latency in seconds
        p95: 95th percentile latency in seconds
        p99: 99th percentile latency in seconds
        min: Minimum latency in seconds
        max: Maximum latency in seconds
    """

    p50: float
    p95: float
    p99: float
    min: float
    max: float

    @classmethod
    def from_latencies(cls, latencies: list[float]) -> "BenchmarkResults":
        """Create BenchmarkResults from a list of latency measurements.

        Args:
            latencies: List of latency measurements in seconds

        Returns:
            BenchmarkResults with calculated percentiles
        """
        if not latencies:
            return cls(p50=0.0, p95=0.0, p99=0.0, min=0.0, max=0.0)

        sorted_latencies = sorted(latencies)
        return cls(
            p50=statistics.quantiles(sorted_latencies, n=100)[49],  # 50th percentile
            p95=statistics.quantiles(sorted_latencies, n=100)[94],  # 95th percentile
            p99=statistics.quantiles(sorted_latencies, n=100)[98],  # 99th percentile
            min=min(sorted_latencies),
            max=max(sorted_latencies),
        )


async def measure_latency(
    client: AsyncClient,
    method: str,
    url: str,
    json: dict | None = None,
    num_requests: int = 20,
) -> list[float]:
    """Measure latency for multiple sequential requests.

    Args:
        client: HTTP client to use for requests
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        json: Optional JSON body for POST requests
        num_requests: Number of requests to make

    Returns:
        List of latency measurements in seconds
    """
    latencies = []

    for _ in range(num_requests):
        start_time = time.perf_counter()

        if method.upper() == "POST":
            await client.post(url, json=json)
        elif method.upper() == "GET":
            await client.get(url)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    return latencies


@dataclass
class ThroughputResult:
    """Results from throughput benchmark measurements.

    Attributes:
        total_requests: Total number of requests made
        duration: Total duration in seconds
        requests_per_second: Throughput (requests per second)
        successful: Number of successful requests (2xx status)
        failed: Number of failed requests (non-2xx status)
    """

    total_requests: int
    duration: float
    requests_per_second: float
    successful: int
    failed: int


async def measure_throughput(
    client: AsyncClient,
    method: str,
    url: str,
    json: dict | None = None,
    num_requests: int = 50,
    max_concurrency: int = 10,
) -> ThroughputResult:
    """Measure throughput with concurrent requests.

    Args:
        client: HTTP client to use for requests
        method: HTTP method (GET, POST, etc.)
        url: URL to request
        json: Optional JSON body for POST requests
        num_requests: Total number of requests to make
        max_concurrency: Maximum concurrent requests

    Returns:
        ThroughputResult with throughput metrics
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    successful = 0
    failed = 0

    async def make_request() -> bool:
        """Make a single request and return success status."""
        nonlocal successful, failed
        async with semaphore:
            try:
                if method.upper() == "POST":
                    response = await client.post(url, json=json)
                elif method.upper() == "GET":
                    response = await client.get(url)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if 200 <= response.status_code < 300:
                    successful += 1
                    return True
                else:
                    failed += 1
                    return False
            except Exception:
                failed += 1
                return False

    start_time = time.perf_counter()

    # Execute all requests concurrently (up to max_concurrency at a time)
    await asyncio.gather(*[make_request() for _ in range(num_requests)])

    end_time = time.perf_counter()
    duration = end_time - start_time

    return ThroughputResult(
        total_requests=num_requests,
        duration=duration,
        requests_per_second=num_requests / duration if duration > 0 else 0.0,
        successful=successful,
        failed=failed,
    )


@dataclass
class CacheHitRateResult:
    """Results from cache hit rate measurements.

    Attributes:
        total_queries: Total number of queries made
        unique_queries: Number of unique queries
        cache_hits: Number of cache hits (repeated queries)
        cache_misses: Number of cache misses (new queries)
        hit_rate: Cache hit rate as a fraction (0.0 to 1.0)
        avg_cached_latency: Average latency for cached queries in seconds
        avg_uncached_latency: Average latency for uncached queries in seconds
        speedup: Speedup factor (uncached / cached latency)
    """

    total_queries: int
    unique_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_cached_latency: float
    avg_uncached_latency: float
    speedup: float


async def measure_cache_hit_rate(
    client: AsyncClient,
    url: str,
    queries: list[dict],
) -> CacheHitRateResult:
    """Measure cache hit rate for repeated queries.

    Args:
        client: HTTP client to use for requests
        url: URL to POST queries to
        queries: List of query payloads (dicts with 'query' and 'max_retries')

    Returns:
        CacheHitRateResult with cache statistics
    """
    # Track which queries we've seen before (for cache hit detection)
    seen_queries = set()
    cache_hits = 0
    cache_misses = 0
    cached_latencies = []
    uncached_latencies = []

    for query_payload in queries:
        # Generate a cache key from the query
        query_text = query_payload.get("query", "")
        max_retries = query_payload.get("max_retries", 3)
        cache_key = f"{query_text}|{max_retries}"

        # Measure latency for this query
        start_time = time.perf_counter()
        await client.post(url, json=query_payload)
        end_time = time.perf_counter()
        latency = end_time - start_time

        # Determine if this was a cache hit or miss
        if cache_key in seen_queries:
            cache_hits += 1
            cached_latencies.append(latency)
        else:
            cache_misses += 1
            uncached_latencies.append(latency)
            seen_queries.add(cache_key)

    total_queries = len(queries)
    unique_queries = len(seen_queries)
    hit_rate = cache_hits / total_queries if total_queries > 0 else 0.0

    avg_cached_latency = statistics.mean(cached_latencies) if cached_latencies else 0.0
    avg_uncached_latency = statistics.mean(uncached_latencies) if uncached_latencies else 0.0
    speedup = avg_uncached_latency / avg_cached_latency if avg_cached_latency > 0 else 1.0

    return CacheHitRateResult(
        total_queries=total_queries,
        unique_queries=unique_queries,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hit_rate=hit_rate,
        avg_cached_latency=avg_cached_latency,
        avg_uncached_latency=avg_uncached_latency,
        speedup=speedup,
    )


@dataclass
class EndpointResult:
    """Performance results for a single endpoint.

    Attributes:
        endpoint: Endpoint URL
        latency_p50: 50th percentile latency in seconds
        latency_p95: 95th percentile latency in seconds
        throughput: Requests per second
        success_rate: Fraction of successful requests (0.0 to 1.0)
    """

    endpoint: str
    latency_p50: float
    latency_p95: float
    throughput: float
    success_rate: float


@dataclass
class PerformanceProfile:
    """Comprehensive performance profile across multiple endpoints.

    Attributes:
        endpoints: List of EndpointResult for each tested endpoint
        total_requests: Total number of requests across all endpoints
        duration: Total duration in seconds
        avg_throughput: Average throughput across all endpoints
    """

    endpoints: list[EndpointResult]
    total_requests: int
    duration: float
    avg_throughput: float


async def generate_performance_profile(
    client: AsyncClient,
    endpoints: list[dict],
    num_requests_per_endpoint: int = 30,
    max_concurrency: int = 10,
) -> PerformanceProfile:
    """Generate comprehensive performance profile across multiple endpoints.

    Args:
        client: HTTP client to use for requests
        endpoints: List of endpoint configs (dicts with 'method', 'url', 'json')
        num_requests_per_endpoint: Number of requests per endpoint
        max_concurrency: Maximum concurrent requests

    Returns:
        PerformanceProfile with comprehensive performance data
    """
    endpoint_results = []
    total_requests = 0
    total_start_time = time.perf_counter()

    for endpoint_config in endpoints:
        method = endpoint_config["method"]
        url = endpoint_config["url"]
        json_body = endpoint_config.get("json")

        # Measure latency
        latencies = await measure_latency(
            client=client,
            method=method,
            url=url,
            json=json_body,
            num_requests=num_requests_per_endpoint,
        )

        # Calculate latency percentiles
        benchmark_results = BenchmarkResults.from_latencies(latencies)

        # Measure throughput
        throughput_result = await measure_throughput(
            client=client,
            method=method,
            url=url,
            json=json_body,
            num_requests=num_requests_per_endpoint,
            max_concurrency=max_concurrency,
        )

        success_rate = (
            throughput_result.successful / throughput_result.total_requests
            if throughput_result.total_requests > 0
            else 0.0
        )

        endpoint_results.append(
            EndpointResult(
                endpoint=f"{method} {url}",
                latency_p50=benchmark_results.p50,
                latency_p95=benchmark_results.p95,
                throughput=throughput_result.requests_per_second,
                success_rate=success_rate,
            )
        )

        total_requests += throughput_result.total_requests + num_requests_per_endpoint

    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time

    avg_throughput = total_requests / total_duration if total_duration > 0 else 0.0

    return PerformanceProfile(
        endpoints=endpoint_results,
        total_requests=total_requests,
        duration=total_duration,
        avg_throughput=avg_throughput,
    )
