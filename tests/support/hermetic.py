"""Structural network-egress guard for hermetic unit tests (Req 9.1, 9.2).

Intercepts outbound `socket.socket.connect` calls so a missed mock in a unit
test surfaces as a loud, immediate failure instead of a slow or flaky real
network hit. `AF_UNIX` connections (e.g. asyncio's self-pipe) pass through
unaffected.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


_BLOCKED_FAMILIES = {socket.AF_INET, socket.AF_INET6}


class NetworkBlockedError(RuntimeError):
    """Raised when a unit test attempts a real outbound network connection."""


@contextmanager
def block_network() -> Iterator[None]:
    """Loud-fail outbound `AF_INET`/`AF_INET6` `socket.connect`; let `AF_UNIX` through.

    Yields:
        None. Restores the original `socket.socket.connect` on exit, even if
        the wrapped block raises.
    """
    real_connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        if sock.family in _BLOCKED_FAMILIES:
            raise NetworkBlockedError(
                f"Blocked real outbound network connection to {address!r}. "
                "Unit tests must not perform network I/O - mock the client/store "
                "instead of hitting a real socket."
            )
        real_connect(sock, address)

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
