"""Structural network-egress guard for hermetic unit tests (Req 9.1, 9.2).

Intercepts outbound `socket.socket.connect`/`connect_ex` calls so a missed
mock in a unit test surfaces as a loud, immediate failure instead of a slow
or flaky real network hit. `socket.getaddrinfo` is intercepted too — DNS
resolution alone (with no subsequent `connect`) is otherwise a hole in this
guard, since `getaddrinfo` itself performs a real network round-trip against
a resolver. `AF_UNIX` connections (e.g. asyncio's self-pipe) pass through
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
    """Loud-fail outbound `AF_INET`/`AF_INET6` sockets and DNS lookups.

    Guards `socket.socket.connect`, `socket.socket.connect_ex`, and
    `socket.getaddrinfo`; `AF_UNIX` connections pass through untouched.

    Yields:
        None. Restores the three original callables on exit, even if the
        wrapped block raises.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock: socket.socket, address: Any) -> None:
        if sock.family in _BLOCKED_FAMILIES:
            raise NetworkBlockedError(
                f"Blocked real outbound network connection to {address!r}. "
                "Unit tests must not perform network I/O - mock the client/store "
                "instead of hitting a real socket."
            )
        real_connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family in _BLOCKED_FAMILIES:
            raise NetworkBlockedError(
                f"Blocked real outbound network connection to {address!r}. "
                "Unit tests must not perform network I/O - mock the client/store "
                "instead of hitting a real socket."
            )
        return real_connect_ex(sock, address)

    def guarded_getaddrinfo(*args: Any, **kwargs: Any) -> Any:
        raise NetworkBlockedError(
            f"Blocked real DNS resolution for {args!r}. "
            "Unit tests must not perform network I/O - mock the client/store "
            "instead of resolving a real hostname."
        )

    socket.socket.connect = guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = guarded_connect_ex  # type: ignore[method-assign]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = real_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = real_getaddrinfo  # type: ignore[assignment]
