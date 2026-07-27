"""Unit tests for the SSE stream lifecycle guards (app/api/v1/_stream.py).

Exercises `_run_with_lifecycle_guards()` directly against fake async event
sources so the cap/disconnect/timeout/cancel/error/aclose mechanics can be
verified deterministically, without depending on pydantic-ai internals.
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from app.api.v1._stream import _run_with_lifecycle_guards
from app.patterns.sse import Completed
from app.patterns.sse import Error
from app.patterns.sse import SSEEvent
from app.patterns.sse import Token
from app.patterns.sse import parse_sse_events
from tests.conftest import build_test_settings


class _FakeRequest:
    """Minimal stand-in for fastapi.Request exposing only is_disconnected()."""

    def __init__(self, *, disconnected: bool = False) -> None:
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected


class _TrackingAsyncGen:
    """Wraps an async generator of SSEEvents and records whether aclose() was called."""

    def __init__(self, agen: AsyncGenerator[SSEEvent]) -> None:
        self._agen = agen
        self.closed = False

    def __aiter__(self) -> "_TrackingAsyncGen":
        return self

    async def __anext__(self) -> SSEEvent:
        return await self._agen.__anext__()

    async def aclose(self) -> None:
        self.closed = True
        await self._agen.aclose()


async def _events(*events: SSEEvent) -> AsyncGenerator[SSEEvent]:
    for event in events:
        yield event


async def _hanging_forever() -> AsyncGenerator[SSEEvent]:
    await asyncio.Event().wait()
    yield Token(content="unreachable")  # pragma: no cover


async def _raises(exc: BaseException) -> AsyncGenerator[SSEEvent]:
    raise exc
    yield Token(content="unreachable")  # pragma: no cover


@pytest.mark.asyncio
async def test_yields_sse_wire_text_for_each_event() -> None:
    """Each upstream event is serialized to SSE wire text via to_sse()."""
    settings = build_test_settings()
    agen = _TrackingAsyncGen(_events(Token(content="a"), Completed()))

    wires = [w async for w in _run_with_lifecycle_guards(_FakeRequest(), agen, settings)]

    parsed = parse_sse_events("".join(wires))
    assert parsed == [Token(content="a"), Completed()]


@pytest.mark.asyncio
async def test_aclose_called_on_normal_completion() -> None:
    """The underlying async generator is closed after normal exhaustion."""
    settings = build_test_settings()
    agen = _TrackingAsyncGen(_events(Completed()))

    async for _ in _run_with_lifecycle_guards(_FakeRequest(), agen, settings):
        pass

    assert agen.closed


@pytest.mark.asyncio
async def test_stops_when_client_disconnected() -> None:
    """When request.is_disconnected() is True, no events are emitted and agen is closed."""
    settings = build_test_settings()
    agen = _TrackingAsyncGen(_events(Token(content="never seen")))

    wires = [
        w
        async for w in _run_with_lifecycle_guards(
            _FakeRequest(disconnected=True), agen, settings
        )
    ]

    assert wires == []
    assert agen.closed


@pytest.mark.asyncio
async def test_stops_producing_further_events_at_sse_max_events_cap() -> None:
    """Once sse_max_events is reached, the stream stops even if more events remain."""
    settings = build_test_settings(sse_max_events=2)
    agen = _TrackingAsyncGen(
        _events(Token(content="a"), Token(content="b"), Token(content="c"))
    )

    wires = [w async for w in _run_with_lifecycle_guards(_FakeRequest(), agen, settings)]

    parsed = parse_sse_events("".join(wires))
    assert parsed == [Token(content="a"), Token(content="b")]
    assert agen.closed


@pytest.mark.asyncio
async def test_terminal_error_event_on_unexpected_exception() -> None:
    """An unexpected exception mid-stream yields a generic terminal error event."""
    settings = build_test_settings()
    agen = _TrackingAsyncGen(_raises(RuntimeError("boom")))

    wires = [w async for w in _run_with_lifecycle_guards(_FakeRequest(), agen, settings)]

    parsed = parse_sse_events("".join(wires))
    assert parsed == [Error(message="An unexpected error occurred")]
    assert agen.closed


@pytest.mark.asyncio
async def test_cancelled_error_is_re_raised_after_cleanup() -> None:
    """Cancelling the stream's own consumer re-raises CancelledError and closes agen.

    Models a real client disconnect: the task iterating `_run_with_lifecycle_guards`
    gets cancelled while waiting on an event that never arrives.
    """
    settings = build_test_settings()
    agen = _TrackingAsyncGen(_hanging_forever())

    async def consume() -> None:
        async for _ in _run_with_lifecycle_guards(_FakeRequest(), agen, settings):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)  # let the producer start waiting on the hanging agen
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert agen.closed


@pytest.mark.asyncio
async def test_send_timeout_yields_terminal_error_and_stops() -> None:
    """A single event that never arrives within sse_send_timeout aborts with an error."""
    settings = build_test_settings(sse_send_timeout=1, sse_heartbeat_interval=1)
    agen = _TrackingAsyncGen(_hanging_forever())

    wires = [w async for w in _run_with_lifecycle_guards(_FakeRequest(), agen, settings)]

    parsed = parse_sse_events("".join(wires))
    assert parsed == [Error(message="Stream timed out")]
    assert agen.closed


@pytest.mark.asyncio
async def test_no_events_emitted_after_error() -> None:
    """Once an error event is yielded, no further events (e.g. from cap) are produced."""
    settings = build_test_settings()
    agen = _TrackingAsyncGen(_raises(ValueError("nope")))

    wires = [w async for w in _run_with_lifecycle_guards(_FakeRequest(), agen, settings)]

    assert len(wires) == 1
    assert "error" in wires[0]
