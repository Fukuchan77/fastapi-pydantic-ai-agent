r"""Unit tests for the SSE heartbeat wrapper (app/api/v1/agent.py::_with_heartbeat).

`_with_heartbeat()` interleaves `: heartbeat\n\n` comments while the wrapped
generator is idle, without ever cancelling the in-flight upstream call (Req 2.8).
"""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from app.api.v1.agent import _with_heartbeat


async def _immediate(*values: str) -> AsyncGenerator[str]:
    for value in values:
        yield value


async def _slow_then_value(delay: float, value: str) -> AsyncGenerator[str]:
    await asyncio.sleep(delay)
    yield value


@pytest.mark.asyncio
async def test_passes_through_immediately_available_values() -> None:
    """No heartbeat is emitted when the upstream generator has no idle gaps."""
    wires = [w async for w in _with_heartbeat(_immediate("a", "b"), interval=10)]

    assert wires == ["a", "b"]


@pytest.mark.asyncio
async def test_emits_heartbeat_while_idle_then_the_delayed_value() -> None:
    """A slow upstream value is preceded by heartbeat comments, not lost or duplicated."""
    wires = [
        w async for w in _with_heartbeat(_slow_then_value(0.25, "late"), interval=0.1)
    ]

    assert wires[-1] == "late"
    heartbeats = [w for w in wires[:-1]]
    assert len(heartbeats) >= 1
    assert all(w == ": heartbeat\n\n" for w in heartbeats)


@pytest.mark.asyncio
async def test_heartbeat_tick_does_not_cancel_the_pending_upstream_call() -> None:
    """The upstream generator's in-flight call survives multiple heartbeat ticks."""
    cancelled = False

    async def upstream() -> AsyncGenerator[str]:
        nonlocal cancelled
        try:
            await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            cancelled = True
            raise
        yield "survived"

    wires = [w async for w in _with_heartbeat(upstream(), interval=0.1)]

    assert wires[-1] == "survived"
    assert not cancelled


@pytest.mark.asyncio
async def test_empty_upstream_yields_nothing() -> None:
    """An upstream generator with no values produces no output at all."""
    wires = [w async for w in _with_heartbeat(_immediate(), interval=10)]

    assert wires == []
