"""Unit tests for RetryTransport should not retry non-transient 5xx errors.

RetryTransport currently retries ALL 5xx errors (500-599) including
non-transient errors like 501 (Not Implemented) and 505 (HTTP Version Not Supported).
These are permanent errors that should not be retried.

This test verifies the fix: only {500, 502, 503, 504} should be retried.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import httpx
import pytest

from app.main import RetryTransport


@pytest.mark.asyncio
async def test_retry_transport_does_not_retry_501_not_implemented() -> None:
    """Test that RetryTransport does NOT retry 501 Not Implemented.

    501 indicates the server does not support the functionality
    required to fulfill the request. This is a permanent error that will
    not be resolved by retrying.

    Expected behavior (after fix):
    - Request is made once (no retries)
    - 501 response is returned immediately

    Current behavior (before fix - this test will FAIL):
    - Request is retried 3 times (incorrectly)
    - 501 response is returned after exhausting retries
    """
    # Create mock response
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 501
    mock_response.content = b"Not Implemented"

    # Patch the parent class method to return our mock response
    with patch.object(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_parent_call:
        # Create RetryTransport instance
        retry_transport = RetryTransport(max_attempts=3, base_delay=0.01)

        # Make request
        request = httpx.Request("GET", "https://example.com/api")

        # Execute - should NOT retry 501
        response = await retry_transport.handle_async_request(request)

        # Verify response is 501
        assert response.status_code == 501

        # Verify parent transport was called only ONCE (no retries)
        # This will FAIL with current implementation (which retries 501)
        assert mock_parent_call.call_count == 1, (
            f"Expected 1 call (no retries for 501), got {mock_parent_call.call_count}. "
            f"Current implementation incorrectly retries non-transient 5xx errors."
        )


@pytest.mark.asyncio
async def test_retry_transport_does_not_retry_505_http_version_not_supported() -> None:
    """Test that RetryTransport does NOT retry 505 HTTP Version Not Supported.

    505 indicates the server does not support the HTTP version
    used in the request. This is a permanent configuration error that will
    not be resolved by retrying.

    Expected behavior (after fix):
    - Request is made once (no retries)
    - 505 response is returned immediately

    Current behavior (before fix - this test will FAIL):
    - Request is retried 3 times (incorrectly)
    - 505 response is returned after exhausting retries
    """
    # Create mock response
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 505
    mock_response.content = b"HTTP Version Not Supported"

    # Patch the parent class method to return our mock response
    with patch.object(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_parent_call:
        # Create RetryTransport instance
        retry_transport = RetryTransport(max_attempts=3, base_delay=0.01)

        # Make request
        request = httpx.Request("GET", "https://example.com/api")

        # Execute - should NOT retry 505
        response = await retry_transport.handle_async_request(request)

        # Verify response is 505
        assert response.status_code == 505

        # Verify parent transport was called only ONCE (no retries)
        # This will FAIL with current implementation (which retries 505)
        assert mock_parent_call.call_count == 1, (
            f"Expected 1 call (no retries for 505), got {mock_parent_call.call_count}. "
            f"Current implementation incorrectly retries non-transient 5xx errors."
        )


@pytest.mark.asyncio
async def test_retry_transport_retries_transient_5xx_errors() -> None:
    """Test that RetryTransport DOES retry transient 5xx errors.

    Only {500, 502, 503, 504} should be retried as these
    are typically transient server issues that may resolve on retry.

    Expected behavior:
    - 500 Internal Server Error: Retry (general server error)
    - 502 Bad Gateway: Retry (upstream server error)
    - 503 Service Unavailable: Retry (temporary overload)
    - 504 Gateway Timeout: Retry (upstream timeout)

    This test should PASS both before and after the fix.
    """
    for status_code in [500, 502, 503, 504]:
        # Create mock response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = status_code
        mock_response.content = b"Server Error"

        # Patch the parent class method to return our mock response
        with patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_parent_call:
            # Create RetryTransport instance
            retry_transport = RetryTransport(max_attempts=3, base_delay=0.01)

            # Make request
            request = httpx.Request("GET", "https://example.com/api")

            # Execute - should retry up to max_attempts
            response = await retry_transport.handle_async_request(request)

            # Verify response has the expected status code
            assert response.status_code == status_code

            # Verify parent transport was called max_attempts times (all retries exhausted)
            assert mock_parent_call.call_count == 3, (
                f"Expected 3 calls (with retries) for {status_code}, "
                f"got {mock_parent_call.call_count}"
            )
