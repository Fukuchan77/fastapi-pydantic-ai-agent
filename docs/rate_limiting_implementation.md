# Rate Limiting Implementation Guide

## Overview

This guide provides instructions for implementing rate limiting on API endpoints to prevent DoS attacks and control API costs.

## Recommended Approach: slowapi

Use `slowapi`, a rate limiting library specifically designed for FastAPI.

## Installation

```bash
uv add slowapi
```

## Implementation Steps

### 1. Update `app/config.py`

Add rate limit configuration:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    rate_limit_per_minute: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum requests per minute per IP address",
    )
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting (disable for development)",
    )
```

### 2. Update `app/main.py`

Add slowapi initialization:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"]
)

app = FastAPI(...)

# Add limiter to app state
app.state.limiter = limiter

# Add exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

### 3. Apply to Endpoints

**Option A: Apply to specific endpoints**

```python
from slowapi import Limiter
from fastapi import Request

@router.post("/agent/chat")
@limiter.limit("10/minute")  # Override default for this endpoint
async def chat(request: Request, ...):
    ...
```

**Option B: Apply to entire router**

```python
# In app/api/v1/router.py
from slowapi import Limiter

router = APIRouter(
    prefix="/v1",
    tags=["v1"],
    dependencies=[Depends(limiter.limit("20/minute"))]
)
```

### 4. Different Limits for Different Endpoints

```python
# Expensive operations: lower limit
@router.post("/agent/stream")
@limiter.limit("5/minute")
async def stream_agent(...):
    ...

# Cheap operations: higher limit
@router.post("/rag/ingest")
@limiter.limit("50/minute")
async def ingest(...):
    ...
```

### 5. Per-User Rate Limiting

For authenticated users, use API key instead of IP:

```python
def get_api_key(request: Request) -> str:
    """Extract API key from request for rate limiting."""
    return request.headers.get("X-API-Key", get_remote_address(request))

limiter = Limiter(key_func=get_api_key)
```

## Testing

```python
# tests/unit/test_rate_limiting.py
import pytest
from fastapi.testclient import TestClient

def test_rate_limit_enforced(client: TestClient):
    """Test that rate limit is enforced."""
    # Make requests up to limit
    for _ in range(10):
        response = client.post("/v1/agent/chat", ...)
        assert response.status_code == 200

    # Next request should be rate limited
    response = client.post("/v1/agent/chat", ...)
    assert response.status_code == 429
```

## Production Considerations

1. **Redis Backend**: For multi-instance deployments, use Redis for shared rate limit state:

```python
from slowapi.util import get_remote_address
import redis

redis_client = redis.from_url("redis://localhost:6379")

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)
```

2. **Per-Customer Limits**: Store limits in database per customer/API key
3. **Monitoring**: Log rate limit hits to detect potential attacks
4. **Graceful Degradation**: Return helpful error messages with retry-after headers

## Environment Variables

Add to `.env.example`:

```bash
RATE_LIMIT_PER_MINUTE=10
RATE_LIMIT_ENABLED=true
```

## References

- slowapi documentation: https://slowapi.readthedocs.io/
- FastAPI rate limiting guide: https://fastapi.tiangolo.com/advanced/middleware/
