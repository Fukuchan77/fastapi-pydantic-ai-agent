"""Shared helpers for gating tests on real Redis server reachability.

Lives alongside `tests/support/chroma.py`, `tests/support/docker.py`, and
`tests/support/ollama.py`. Redis's dependency is a reachable service rather
than a one-time model download, so - unlike Chroma's env-var opt-in - the
gate here is a live probe (mirrors `tests/support/docker.py`'s daemon probe
and `tests/support/ollama.py`'s server-reachability check). No in-process
Redis substitute is introduced: the lane exists precisely to exercise a real
client against a real server (Req 6.8).
"""

import redis.asyncio as redis


REDIS_URL = "redis://localhost:6379/0"
"""Connection URL probed by the real-server verification lane."""

REDIS_UNREACHABLE_SKIP_REASON = (
    f"No Redis server reachable at {REDIS_URL}; the redis-marked lane requires "
    "a real server (start one, or run 'mise run test:redis' in an environment "
    "that has one, e.g. CI's Redis service container)."
)
"""Stated reason reported when the reachability probe finds no server (Req 6.3)."""

REDIS_LIVE_TEST_COUNT = 7
"""Number of tests gated behind Redis reachability (Req 6.2).

Pass `EXPECT_LIVE_TESTS=$REDIS_LIVE_TEST_COUNT` alongside a reachable server
(e.g. `EXPECT_LIVE_TESTS=7 mise run test:redis`) so a lane that silently
collects zero live cases - for example a marker expression that no longer
matches anything - fails instead of reporting success (Req 13.8).
`test_redis_test_gating.py` guards this value against drift as tests are
added to or removed from the gated module, and against the CI step's
`EXPECT_LIVE_TESTS` literal in `.github/workflows/pr.yml`.
"""


async def redis_reachable(redis_url: str = REDIS_URL) -> bool:
    """Probe whether a real Redis server answers PING at `redis_url` (Req 6.2).

    Args:
        redis_url: Redis connection URL to probe.

    Returns:
        True if the server responded to PING; False on any connection
        failure, so an unreachable server skips the gated lane instead of
        failing it (Req 6.3).
    """
    client = redis.from_url(redis_url)
    try:
        await client.ping()
    except Exception:
        return False
    else:
        return True
    finally:
        await client.aclose()
