"""Shared HTTP client: retrying transport and the `httpx.AsyncClient` builder.

Split out of `app/main.py` (Req 4.3) so the composition root stays focused on
wiring; `RetryTransport` moved verbatim, and `build_http_client()` follows the
`app/stores/factory.py` builder pattern (settings -> instance, no logging).
"""

import asyncio
import logging
import random
from typing import Any

import httpx

from app.config import Settings


logger = logging.getLogger(__name__)


class RetryTransport(httpx.AsyncHTTPTransport):
    """Custom HTTP transport with retry logic and exponential backoff.

    Implements automatic retry for transient failures (network errors,
    5xx server errors) with exponential backoff and jitter. Does NOT retry client
    errors (4xx) as they indicate issues with the request itself.

    Only retries transient 5xx errors {500, 502, 503, 504}.
    Non-transient errors like 501 (Not Implemented) and 505 (HTTP Version Not Supported)
    are permanent configuration issues that will not be resolved by retrying.

    Retry behavior:
    - Network errors (ConnectError, TimeoutException): Retry
    - Transient 5xx errors (500, 502, 503, 504): Retry
    - Non-transient 5xx errors (501, 505, etc.): Do NOT retry
    - 4xx client errors (400-499): Do NOT retry
    - Exponential backoff: delay = base_delay * (2 ** attempt) + random jitter
    - Jitter: random.uniform(0, 1) to prevent thundering herd

    Args:
        max_attempts: Maximum number of retry attempts (from settings)
        base_delay: Base delay in seconds for exponential backoff (from settings)
        **kwargs: Additional arguments passed to AsyncHTTPTransport
    """

    # Define retryable status codes - only transient server errors
    # Use frozenset for immutability (RUF012)
    RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize retry transport with exponential backoff settings.

        Args:
            max_attempts: Maximum number of retry attempts (default: 3)
            base_delay: Base delay for exponential backoff in seconds (default: 1.0)
            **kwargs: Additional arguments for AsyncHTTPTransport
        """
        super().__init__(**kwargs)
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Handle HTTP request with retry logic.

        Retries transient failures (network errors, transient 5xx) with exponential backoff.
        Does NOT retry client errors (4xx) or non-transient 5xx errors.

        Args:
            request: The HTTP request to execute

        Returns:
            httpx.Response: The HTTP response

        Raises:
            Exception: If all retry attempts are exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                response = await super().handle_async_request(request)

                # Only retry transient 5xx errors {500, 502, 503, 504}
                # Non-transient errors (501, 505, etc.) are permanent and should not be retried
                if (
                    response.status_code in self.RETRYABLE_STATUS_CODES
                    and attempt < self.max_attempts - 1
                ):
                    delay = self.base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "HTTP request to %s returned %d (attempt %d/%d), retrying in %.2fs",
                        request.url,
                        response.status_code,
                        attempt + 1,
                        self.max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                # 2xx, 3xx, 4xx responses - return immediately (don't retry 4xx)
                return response

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                # Network errors are transient, retry if attempts remaining
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.base_delay * (2**attempt) + random.uniform(0, 1)  # noqa: S311
                    logger.warning(
                        "HTTP request to %s failed with %s (attempt %d/%d), retrying in %.2fs",
                        request.url,
                        type(e).__name__,
                        attempt + 1,
                        self.max_attempts,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # Last attempt failed, raise the exception
                raise
            except Exception:
                # Non-transient errors (e.g., invalid URL, SSL errors) - raise immediately
                raise

        # All retries exhausted, raise last exception
        if last_exception:
            raise last_exception

        # This should never happen, but satisfy type checker
        raise RuntimeError("Retry logic error: no response or exception")


def build_http_client(settings: Settings) -> httpx.AsyncClient:
    """Construct the shared `httpx.AsyncClient` used for agent tool calls.

    Args:
        settings: Application settings.

    Returns:
        An `httpx.AsyncClient` wired to `RetryTransport` with the timeout and
        connection-pool limits from settings.
    """
    retry_transport = RetryTransport(
        max_attempts=settings.http_retry_max_attempts,
        base_delay=settings.http_retry_base_delay,
    )
    return httpx.AsyncClient(
        transport=retry_transport,
        timeout=httpx.Timeout(
            settings.http_timeout,
            connect=settings.http_connect_timeout,
        ),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
        ),
    )
