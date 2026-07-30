"""Self-test proving the `block_network` hermetic guard actually triggers (Req 9.3)."""

import socket

import pytest

from tests.support.hermetic import NetworkBlockedError


def test_block_network_blocks_af_inet_connect() -> None:
    """A real outbound AF_INET connection attempt is loud-failed, not silently allowed."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlockedError):
            sock.connect(("93.184.216.34", 80))
    finally:
        sock.close()


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
