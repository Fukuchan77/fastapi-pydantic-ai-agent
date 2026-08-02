"""Unit tests for the Docker daemon reachability probe (`tests/support/docker.py`).

Boundary correction: Task 1.5's declared boundary is
`tests/integration/test_docker_deployment.py` and `tests/support/chroma.py`,
but TDD (Critical Constraints) requires a failing test written before the
implementation, and the probe's decision logic must be exercised without
invoking a real `docker info` subprocess at import time - which
`test_docker_deployment.py`'s module-level `pytestmark` computation always
does on import. The probe is therefore extracted to its own support module,
mirroring the existing `tests/support/ollama.py` and `tests/support/chroma.py`
gating helpers, with its decision logic covered directly here via dependency
injection - the same "boundary correction" precedent already used by
`test_expect_live_tests_plugin.py` and `test_local_test_gating.py`.
"""

import subprocess

from tests.support.docker import probe_docker_daemon


def test_probe_reports_unreachable_when_cli_absent() -> None:
    """No `docker` on PATH is reported unreachable with a CLI-specific reason."""
    reachable, reason = probe_docker_daemon(which=lambda _: None)

    assert reachable is False
    assert "PATH" in reason


def test_probe_reports_unreachable_when_daemon_down() -> None:
    """CLI present but `docker info` exits nonzero is reported daemon-down, not CLI-absent."""

    def _daemon_down() -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=["docker", "info"], returncode=1)

    reachable, reason = probe_docker_daemon(
        which=lambda _: "/usr/bin/docker", run_docker_info=_daemon_down
    )

    assert reachable is False
    assert "daemon" in reason.lower()


def test_probe_reports_unreachable_when_docker_info_raises() -> None:
    """An `OSError`/timeout from the probe itself is reported unreachable, not raised."""

    def _raise() -> subprocess.CompletedProcess[bytes]:
        raise OSError("boom")

    reachable, reason = probe_docker_daemon(
        which=lambda _: "/usr/bin/docker", run_docker_info=_raise
    )

    assert reachable is False
    assert reason


def test_probe_reports_reachable_when_daemon_up() -> None:
    """CLI present and `docker info` exits zero is reported reachable with no reason."""

    def _daemon_up() -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=["docker", "info"], returncode=0)

    reachable, reason = probe_docker_daemon(
        which=lambda _: "/usr/bin/docker", run_docker_info=_daemon_up
    )

    assert reachable is True
    assert reason == ""
