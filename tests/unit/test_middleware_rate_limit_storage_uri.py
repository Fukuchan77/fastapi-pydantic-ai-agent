"""Unit tests for rate limiter storage_uri wiring and graceful fallback (Req 11.4)."""

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.rate_limit import add_rate_limiting


def test_add_rate_limiting_defaults_to_in_memory_storage() -> None:
    """No storage_uri configured means requests still succeed (in-memory storage)."""
    app = FastAPI()
    limiter = add_rate_limiting(app, default_limits=["3/minute"])

    @app.get("/test")
    @limiter.limit("3/minute")
    async def test_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200


def test_add_rate_limiting_with_unreachable_redis_falls_back_to_memory() -> None:
    """An unreachable configured Redis storage_uri degrades to in-memory, not a crash.

    Port 1 is unassigned, so the connection is refused instantly (no real
    network needed, matching the `test_store_dry_run_startup.py` precedent for
    OllamaEmbeddingVectorStore) - `in_memory_fallback_enabled=True` must catch
    that failure and let the request through rather than raising.
    """
    app = FastAPI()
    limiter = add_rate_limiting(
        app,
        default_limits=["3/minute"],
        storage_uri="redis://localhost:1/0",
    )

    @app.get("/test")
    @limiter.limit("3/minute")
    async def test_endpoint(request: Request) -> JSONResponse:
        return JSONResponse(content={"status": "ok"})

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200
