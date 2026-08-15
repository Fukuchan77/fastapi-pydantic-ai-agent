"""Tests for the `/health/ready` probe cache.

`/health/ready` is unauthenticated by design (load balancers and Kubernetes
must reach it) and its LLM check is a real, billable request to the configured
provider. Uncached, one inbound HTTP request becomes one outbound provider
request, bounded only by the global 1000/minute default that `create_app()` sets
that high *specifically so health routes are effectively exempt* - an
unauthenticated cost amplifier sitting outside `llm_rate_limit` (Req 11.3),
which only guards authenticated LLM routes.

`ReadinessProbeCache` is what bounds it: provider traffic becomes a constant
rate (1 probe per TTL) no matter how much traffic arrives.
"""

import asyncio

import pytest

from app.api.health import ReadinessProbeCache


class _CountingProbe:
    """Probe stub counting how many times the real dependencies were hit."""

    def __init__(self, result: dict[str, str] | None = None, delay: float = 0.0) -> None:
        """Record the result to return and how long each probe should take."""
        self.calls = 0
        self._result = result or {"llm_provider": "healthy"}
        self._delay = delay

    async def __call__(self) -> dict[str, str]:
        """Count the call and return the canned result."""
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return dict(self._result)


class TestCachingBoundsProviderTraffic:
    """Repeated requests within the TTL must not repeat the provider request."""

    async def test_second_call_within_ttl_does_not_probe_again(self) -> None:
        """The whole point: inbound volume stops driving outbound volume."""
        probe = _CountingProbe()
        cache = ReadinessProbeCache(ttl=60)

        first = await cache.get(probe)
        second = await cache.get(probe)

        assert probe.calls == 1
        assert first == second == {"llm_provider": "healthy"}

    async def test_many_sequential_calls_still_probe_once(self) -> None:
        """A burst of readiness checks costs one provider request, not N."""
        probe = _CountingProbe()
        cache = ReadinessProbeCache(ttl=60)

        for _ in range(50):
            await cache.get(probe)

        assert probe.calls == 1

    async def test_concurrent_callers_share_one_probe(self) -> None:
        """Concurrency must not defeat the cache.

        The lock is held across the probe, so callers arriving while one is in
        flight wait for it instead of each starting their own - without that, a
        simultaneous burst would issue N provider requests despite the cache.
        """
        probe = _CountingProbe(delay=0.05)
        cache = ReadinessProbeCache(ttl=60)

        results = await asyncio.gather(*(cache.get(probe) for _ in range(10)))

        assert probe.calls == 1
        assert all(result == {"llm_provider": "healthy"} for result in results)


class TestExpiryAndDisabling:
    """The cache must still surface changes, and must be switchable off."""

    async def test_expired_entry_probes_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A readiness endpoint that never re-probes would be useless."""
        probe = _CountingProbe()
        cache = ReadinessProbeCache(ttl=10)

        clock = {"now": 1000.0}
        monkeypatch.setattr("app.api.health.time.monotonic", lambda: clock["now"])

        await cache.get(probe)
        clock["now"] += 9.0
        await cache.get(probe)
        assert probe.calls == 1

        clock["now"] += 2.0  # now past the 10s TTL
        await cache.get(probe)
        assert probe.calls == 2

    async def test_status_change_is_observed_after_expiry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dependency going unreachable must surface once the entry expires."""
        results = [{"llm_provider": "healthy"}, {"llm_provider": "unreachable"}]

        async def probe() -> dict[str, str]:
            return results.pop(0)

        cache = ReadinessProbeCache(ttl=10)
        clock = {"now": 500.0}
        monkeypatch.setattr("app.api.health.time.monotonic", lambda: clock["now"])

        assert await cache.get(probe) == {"llm_provider": "healthy"}
        clock["now"] += 11.0
        assert await cache.get(probe) == {"llm_provider": "unreachable"}

    async def test_ttl_zero_probes_every_time(self) -> None:
        """`readiness_probe_cache_ttl=0` restores live probing per request."""
        probe = _CountingProbe()
        cache = ReadinessProbeCache(ttl=0)

        await cache.get(probe)
        await cache.get(probe)

        assert probe.calls == 2


class TestCachedValueIsolation:
    """A caller must not be able to corrupt the cached mapping."""

    async def test_mutating_a_returned_mapping_does_not_affect_the_cache(self) -> None:
        """The endpoint builds a response body from this dict; copies are returned."""
        probe = _CountingProbe()
        cache = ReadinessProbeCache(ttl=60)

        first = await cache.get(probe)
        first["llm_provider"] = "tampered"

        assert await cache.get(probe) == {"llm_provider": "healthy"}
