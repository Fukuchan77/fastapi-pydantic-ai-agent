"""Self-test proving the `block_network` hermetic guard actually triggers (Req 9.3)."""

import socket

import pytest

from tests.support.hermetic import NetworkBlockedError


def _ipv6_available() -> bool:
    """Probe for a usable IPv6 stack by attempting real socket construction.

    `socket.has_ipv6` only reports build-time support; a host can be built
    with IPv6 support and still have no usable IPv6 stack at runtime (the
    construction itself raises `OSError` in that case).
    """
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    sock.close()
    return True


_IPV6_UNAVAILABLE_REASON = "Host has no usable IPv6 stack (AF_INET6 socket construction failed)"


def test_block_network_blocks_af_inet_connect() -> None:
    """A real outbound AF_INET connection attempt is loud-failed, not silently allowed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("93.184.216.34", 80))
    finally:
        sock.close()


@pytest.mark.skipif(not _ipv6_available(), reason=_IPV6_UNAVAILABLE_REASON)
def test_block_network_blocks_af_inet6_connect() -> None:
    """A real outbound AF_INET6 connection attempt is loud-failed too."""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("::1", 80))
    finally:
        sock.close()


def test_block_network_allows_af_unix_connect() -> None:
    """AF_UNIX connections pass through untouched (e.g. the asyncio self-pipe)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with pytest.raises(FileNotFoundError):
            sock.connect("/nonexistent/path/for/block-network-test.sock")
    finally:
        sock.close()


def test_block_network_blocks_af_inet_connect_ex() -> None:
    """The non-raising `connect_ex` variant is loud-failed too, not silently allowed.

    `connect_ex` is a distinct C-level entry point from `connect` (it returns
    an errno instead of raising), so guarding `connect` alone leaves it as an
    unguarded escape hatch for real outbound I/O.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            sock.connect_ex(("93.184.216.34", 80))
    finally:
        sock.close()


def test_block_network_blocks_getaddrinfo() -> None:
    """A real DNS lookup with no subsequent `connect` is loud-failed too.

    `getaddrinfo` itself performs a network round-trip against a resolver, so
    guarding only `connect`/`connect_ex` leaves DNS resolution as a hole in
    this guard.
    """
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 80)


def test_ipv6_probe_reports_unavailable_when_socket_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe treats a failed AF_INET6 construction as unusable.

    `socket.has_ipv6` only reports build-time support, not whether the
    runtime stack actually works, so the probe must attempt construction.
    """

    def _raise_eafnosupport(family: int, kind: int) -> socket.socket:
        raise OSError("Address family not supported by protocol")

    monkeypatch.setattr(socket, "socket", _raise_eafnosupport)

    assert _ipv6_available() is False
