"""Health check endpoints - no authentication required."""

import asyncio
import logging
import time
from collections.abc import Awaitable
from collections.abc import Callable

from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import Model
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models import infer_model
from pydantic_ai.settings import ModelSettings

from app.stores.factory import StoreDryRunError
from app.stores.factory import dry_run_stores
from app.stores.session_store import RedisSessionStore
from app.stores.session_store import SessionStore
from app.stores.vector_store import OllamaEmbeddingVectorStore
from app.stores.vector_store import VectorStore


router = APIRouter()
logger = logging.getLogger(__name__)

# A single-token completion is enough to prove the provider round-trip works
# without paying for a full generation.
_PROBE_MESSAGES: list[ModelMessage] = [ModelRequest.user_text_prompt("ping")]
_PROBE_SETTINGS: ModelSettings = {"max_tokens": 1}


class ReadinessProbeCache:
    """Short-TTL cache in front of the `/health/ready` dependency probes.

    `/health/ready` is deliberately unauthenticated so load balancers and
    Kubernetes can reach it, and its LLM check is a real request to the
    configured provider. Uncached, that makes one inbound HTTP request cost one
    outbound provider request, bounded only by the global 1000/minute default -
    which `create_app()` sets that high precisely so health routes are
    effectively exempt. That is an unauthenticated cost amplifier sitting
    entirely outside `llm_rate_limit` (Req 11.3), the control that bounds LLM
    spend on every *authenticated* route.

    Caching the probe outcome for a few seconds bounds provider traffic to a
    constant rate no matter how much traffic arrives, without authenticating the
    endpoint or throttling legitimate probers. A readiness probe firing every
    5-10 seconds does not need a fresh round-trip per call; it needs a recent
    one.

    One instance is created per application in `app.lifespan._startup` and read
    from `app.state.readiness_cache`, so nothing is shared between applications
    in a single test session.
    """

    def __init__(self, ttl: int) -> None:
        """Initialize an empty cache.

        Args:
            ttl: Seconds a probe result stays fresh. 0 disables caching, so
                every request probes live.
        """
        self._ttl = ttl
        self._cached: dict[str, str] | None = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get(self, probe: Callable[[], Awaitable[dict[str, str]]]) -> dict[str, str]:
        """Return a fresh-enough probe result, running `probe` only when needed.

        The lock is held across the probe so concurrent callers arriving on a
        cold or expired cache share a single probe run rather than each starting
        their own - without it, a burst of readiness checks would defeat the
        whole point of caching.

        Args:
            probe: Zero-argument coroutine function producing a checks mapping.

        Returns:
            dict: The checks mapping, copied so a caller cannot mutate the
                cached value.
        """
        if self._ttl == 0:
            return await probe()

        async with self._lock:
            now = time.monotonic()
            if self._cached is not None and now < self._expires_at:
                return dict(self._cached)

            checks = await probe()
            self._cached = dict(checks)
            self._expires_at = now + self._ttl
            return checks


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness health check endpoint.

    Returns 200 OK when the service is running. No authentication required.
    Used by load balancers and monitoring systems.

    Returns:
        dict: Status response indicating service is healthy.
    """
    return {"status": "ok"}


async def _probe_session_store(session_store: SessionStore) -> str:
    """Probe the session store's connectivity, iff it is Redis-backed.

    In-memory session stores have no external dependency to probe (Req 13.1
    scopes the Redis probe to "only when the Redis session store is
    enabled") and are reported as "skipped" rather than "healthy".

    Args:
        session_store: The active session store from `app.state`.

    Returns:
        "healthy", "unreachable", or "skipped".
    """
    if not isinstance(session_store, RedisSessionStore):
        return "skipped"
    try:
        await dry_run_stores(session_store)
    except StoreDryRunError:
        logger.warning("Readiness probe: session store unreachable", exc_info=True)
        return "unreachable"
    return "healthy"


async def _probe_vector_store(vector_store: VectorStore) -> str:
    """Probe the active vector store backend's connectivity.

    In-memory and embedded-Chroma backends have no network dependency to
    probe (same reasoning as the startup dry-run, Task 3) and are reported
    as "skipped" rather than "healthy"; only the Ollama-embedding backend
    makes a real round-trip.

    Args:
        vector_store: The active vector store from `app.state`.

    Returns:
        "healthy", "unreachable", or "skipped".
    """
    if not isinstance(vector_store, OllamaEmbeddingVectorStore):
        return "skipped"
    try:
        await dry_run_stores(vector_store)
    except StoreDryRunError:
        logger.warning("Readiness probe: vector store unreachable", exc_info=True)
        return "unreachable"
    return "healthy"


async def _probe_llm_provider(model: Model) -> str:
    """Probe the active LLM provider with a minimal, single-token request.

    Args:
        model: The chat agent's configured model (a concrete `Model`,
            including a `FallbackModel` chain in production).

    Returns:
        "healthy" or "unreachable".
    """
    try:
        await model.request(_PROBE_MESSAGES, _PROBE_SETTINGS, ModelRequestParameters())
    except Exception:
        logger.warning("Readiness probe: LLM provider unreachable", exc_info=True)
        return "unreachable"
    return "healthy"


@router.get("/health/ready")
async def readiness_check(request: Request) -> JSONResponse:
    """Deep readiness health check endpoint.

    Performs live connectivity probes against every store or provider
    currently selected by configuration: the Redis session store (only when
    enabled), the active vector store backend, and the LLM provider
    (Req 13.1). Probes run concurrently to minimize added latency.

    This endpoint is intended for Kubernetes readiness probes and load
    balancer health checks. No authentication is required - which is why the
    result is served through `ReadinessProbeCache` (see its docstring): the
    LLM probe is a billable provider request, and an unauthenticated route
    must not turn inbound request volume into outbound provider volume.

    An application whose lifespan did not run (a bare `FastAPI()` test
    harness) has no cache on `app.state` and probes live on every request.

    Args:
        request: FastAPI request object to access app.state

    Returns:
        JSONResponse: 200 with `{"status": "ready", "checks": {...}}` when
            every probe passes; 503 with `{"status": "not_ready", "checks":
            {...}}` naming the failed dependency otherwise (Req 13.2).
    """
    state = request.app.state

    async def probe() -> dict[str, str]:
        """Run every configured dependency probe concurrently."""
        session_result, vector_result, llm_result = await asyncio.gather(
            _probe_session_store(state.session_store),
            _probe_vector_store(state.vector_store),
            _probe_llm_provider(infer_model(state.chat_agent.model)),
        )
        return {
            "session_store": session_result,
            "vector_store": vector_result,
            "llm_provider": llm_result,
        }

    cache = getattr(state, "readiness_cache", None)
    checks = await cache.get(probe) if isinstance(cache, ReadinessProbeCache) else await probe()

    is_ready = all(status != "unreachable" for status in checks.values())
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={"status": "ready" if is_ready else "not_ready", "checks": checks},
    )
