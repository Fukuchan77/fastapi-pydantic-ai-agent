"""Integration tests for Docker deployment.

This module tests the Dockerfile multi-stage build and verifies:
- Image builds successfully
- Container starts and runs as non-root user
- Health endpoint is accessible
- Environment variables are properly injected

Note: These tests require Docker to be installed and available.
They will be skipped if Docker is not found in the system PATH.
"""

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest


# Check if Docker is available
DOCKER_AVAILABLE = shutil.which("docker") is not None
SKIP_REASON = "Docker not available - these tests run in CI with Docker"

# Mark all tests in this module to require Docker
pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_REASON),
]

_STARTUP_TIMEOUT = 30  # seconds to wait for container health


def _find_free_port() -> int:
    """Find a free local port dynamically to avoid port conflicts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout: int = _STARTUP_TIMEOUT) -> None:
    """Poll the health endpoint until it responds or timeout expires.

    Args:
        port: Host port to poll
        timeout: Maximum seconds to wait

    Raises:
        RuntimeError: If health endpoint does not respond within timeout
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
        time.sleep(1)
    raise RuntimeError(
        f"Container health endpoint did not respond within {timeout}s (last error: {last_error})"
    )


def _start_container(image_name: str, host_port: int) -> str:
    """Start a container and wait until healthy.

    Args:
        image_name: Docker image to run
        host_port: Host port to map to container port 8000

    Returns:
        Container ID

    Raises:
        RuntimeError: If container fails to start or become healthy
    """
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            f"{host_port}:8000",
            "-e",
            "API_KEY=test-docker-key",
            "-e",
            "LLM_MODEL=openai:gpt-4o",
            "-e",
            "LLM_API_KEY=sk-test-key-docker",
            image_name,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start container:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    container_id = result.stdout.strip()

    try:
        _wait_for_health(host_port)
    except RuntimeError:
        logs = subprocess.run(
            ["docker", "logs", container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
        raise RuntimeError(
            f"Container did not become healthy:\nLogs:\n{logs.stdout}\n{logs.stderr}"
        ) from None

    return container_id


def _stop_container(container_id: str) -> None:
    """Stop and remove a container (best-effort cleanup)."""
    subprocess.run(["docker", "stop", container_id], capture_output=True, timeout=30)
    subprocess.run(["docker", "rm", container_id], capture_output=True, timeout=30)


@pytest.fixture(scope="module")
def docker_image_name() -> str:
    """Docker image name for testing."""
    return "fastapi-pydantic-ai-agent:test"


@pytest.fixture(scope="module")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def build_docker_image(project_root: Path, docker_image_name: str) -> Iterator[str]:
    """Build the Docker image and remove it after the test module completes.

    Args:
        project_root: Project root directory
        docker_image_name: Name for the Docker image

    Yields:
        Image name

    Raises:
        RuntimeError: If build fails
    """
    result = subprocess.run(
        ["docker", "build", "-t", docker_image_name, "."],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Docker build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    yield docker_image_name

    # Cleanup: remove test image to prevent disk accumulation across CI runs
    subprocess.run(
        ["docker", "rmi", "-f", docker_image_name],
        capture_output=True,
        timeout=30,
    )


@pytest.fixture(scope="module")
def container_port() -> int:
    """Allocate a free host port for the shared running container."""
    return _find_free_port()


@pytest.fixture(scope="module")
def running_container(build_docker_image: str, container_port: int) -> Iterator[str]:
    """Start a shared Docker container for the test module.

    Waits for the health endpoint to respond before yielding — no fixed sleeps.

    Args:
        build_docker_image: Ensures the image is built first
        container_port: Host port to expose

    Yields:
        Container ID while running
    """
    container_id = _start_container(build_docker_image, container_port)
    yield container_id
    _stop_container(container_id)


def test_docker_image_builds(build_docker_image: str) -> None:
    """Test that Docker image builds successfully."""
    assert build_docker_image == "fastapi-pydantic-ai-agent:test"


def test_container_runs_as_nonroot_user(running_container: str) -> None:
    """Test that container runs as non-root user.

    Requirements: 9.3
    """
    result = subprocess.run(
        ["docker", "exec", running_container, "whoami"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    username = result.stdout.strip()
    assert username == "app", f"Expected user 'app', got '{username}'"
    assert username != "root", "Container must not run as root user"


def test_container_health_endpoint_responds(running_container: str, container_port: int) -> None:
    """Test that health endpoint is accessible.

    Requirements: 7.5, 9.4
    """
    with httpx.Client(timeout=10.0) as client:
        response = client.get(f"http://localhost:{container_port}/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_environment_variables_injected(running_container: str) -> None:
    """Test that environment variables are properly injected.

    Requirements: 9.5
    """
    result = subprocess.run(
        ["docker", "exec", running_container, "printenv", "API_KEY"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "test-docker-key"


def test_virtual_environment_in_path(running_container: str) -> None:
    """Test that virtual environment is in PATH.

    Requirements: 9.1
    """
    result = subprocess.run(
        ["docker", "exec", running_container, "which", "uvicorn"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    uvicorn_path = result.stdout.strip()
    assert "/app/.venv/bin/uvicorn" in uvicorn_path


@pytest.fixture
def sigterm_container(build_docker_image: str) -> Iterator[str]:
    """Start a dedicated container for the SIGTERM test.

    Uses its own container so stopping it does not affect the shared
    running_container fixture used by other tests.

    Yields:
        Container ID
    """
    port = _find_free_port()
    container_id = _start_container(build_docker_image, port)
    yield container_id
    # Force-remove: handles both running and stopped states
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=30)


def test_sigterm_handling(sigterm_container: str) -> None:
    """Test that container handles SIGTERM gracefully.

    Requirements: 9.4

    Uses a dedicated container so this test does not affect others,
    regardless of execution order.
    """
    result = subprocess.run(
        ["docker", "stop", "-t", "10", sigterm_container],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0

    # Verify container stopped (not killed)
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", sigterm_container],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.stdout.strip() == "false"
