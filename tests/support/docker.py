"""Shared helper for probing Docker daemon reachability (Req 13.8, 13.9).

Lives alongside `tests/support/ollama.py` and `tests/support/chroma.py`. A bare
`shutil.which("docker")` check only tells you the CLI is on `PATH` - it cannot
distinguish "Docker Desktop/Rancher Desktop installed but not started" from
"no Docker at all", so `test_docker_deployment.py`'s `pytestmark` needs a live
daemon probe instead. Split into its own module so the probe's decision logic
- not the real subprocess call - can be unit-tested via dependency injection,
without ever invoking a real `docker info` at import time.
"""

import shutil
import subprocess
from collections.abc import Callable


def _run_docker_info() -> subprocess.CompletedProcess[bytes]:
    """Invoke `docker info` for real, capturing output and bounding the wait."""
    return subprocess.run(["docker", "info"], capture_output=True, timeout=10)


def probe_docker_daemon(
    which: Callable[[str], str | None] = shutil.which,
    run_docker_info: Callable[[], subprocess.CompletedProcess[bytes]] = _run_docker_info,
) -> tuple[bool, str]:
    """Probe whether a Docker daemon is reachable.

    Distinguishes CLI absence from a CLI present but daemon not running (e.g.
    Docker Desktop installed but not started), each with its own skip reason,
    so a machine with the CLI installed but no running daemon skips cleanly
    instead of a real `docker build` call erroring out.

    Args:
        which: Injectable stand-in for `shutil.which`, for testing.
        run_docker_info: Injectable stand-in for the real `docker info` call.

    Returns:
        A `(reachable, skip_reason)` tuple. `skip_reason` is empty when
        `reachable` is True.
    """
    if which("docker") is None:
        return False, "Docker CLI not found on PATH - these tests run in CI with Docker"

    try:
        result = run_docker_info()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Docker daemon not reachable ('docker info' failed: {exc})"

    if result.returncode != 0:
        return (
            False,
            "Docker CLI found but daemon not reachable (docker info failed) - "
            "is the daemon running?",
        )

    return True, ""
