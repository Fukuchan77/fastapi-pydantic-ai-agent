"""Behavioral tests for the availability-gated pre-push hook (AC 1.6, 1.7).

The hook shells out to `curl` (Ollama reachability probe) and `mise` (the
actual test/evals runners). Both are stubbed with fake executables placed
first on PATH so the test never touches the network or a real mise install.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest


HOOK_PATH = Path(".githooks/pre-push").resolve()

_FAKE_CURL = """#!/usr/bin/env bash
echo "curl $*" >> "$CALL_LOG"
exit "$FAKE_CURL_EXIT"
"""

_FAKE_MISE = """#!/usr/bin/env bash
echo "mise $*" >> "$CALL_LOG"
case "$*" in
  *test:local*) exit "$FAKE_MISE_TEST_LOCAL_EXIT" ;;
  *evals*) exit "$FAKE_MISE_EVALS_EXIT" ;;
esac
exit 0
"""


def _make_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """Directory of stub `curl`/`mise` executables, prepended to the hook's PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _make_executable(bin_dir / "curl", _FAKE_CURL)
    _make_executable(bin_dir / "mise", _FAKE_MISE)
    return bin_dir


@pytest.fixture
def call_log(tmp_path: Path) -> Path:
    """Path the stub executables append their invocation args to."""
    return tmp_path / "calls.log"


def _run_hook(
    fake_bin: Path,
    call_log: Path,
    *,
    curl_exit: int,
    test_local_exit: int = 0,
    evals_exit: int = 0,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["CALL_LOG"] = str(call_log)
    env["FAKE_CURL_EXIT"] = str(curl_exit)
    env["FAKE_MISE_TEST_LOCAL_EXIT"] = str(test_local_exit)
    env["FAKE_MISE_EVALS_EXIT"] = str(evals_exit)
    return subprocess.run(
        ["bash", str(HOOK_PATH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_hook_runs_local_tests_and_evals_when_ollama_reachable(
    fake_bin: Path, call_log: Path
) -> None:
    """AC 1.6: Ollama reachable -> runs both local tests and evals, exits 0."""
    result = _run_hook(fake_bin, call_log, curl_exit=0)

    assert result.returncode == 0
    calls = call_log.read_text()
    assert "run test:local" in calls
    assert "run evals" in calls


def test_hook_blocks_push_when_local_tests_fail(fake_bin: Path, call_log: Path) -> None:
    """AC 1.6: a failing `mise run test:local` blocks the push (non-zero exit)."""
    result = _run_hook(fake_bin, call_log, curl_exit=0, test_local_exit=1)

    assert result.returncode != 0
    assert "run test:local" in call_log.read_text()


def test_hook_blocks_push_when_evals_fail(fake_bin: Path, call_log: Path) -> None:
    """AC 1.6: a failing `mise run evals` also blocks the push (non-zero exit)."""
    result = _run_hook(fake_bin, call_log, curl_exit=0, evals_exit=1)

    assert result.returncode != 0
    calls = call_log.read_text()
    assert "run test:local" in calls
    assert "run evals" in calls


def test_hook_skips_and_warns_when_ollama_unreachable(fake_bin: Path, call_log: Path) -> None:
    """AC 1.7: Ollama unreachable -> warns, skips mise entirely, allows the push."""
    result = _run_hook(fake_bin, call_log, curl_exit=1)

    assert result.returncode == 0
    assert not call_log.exists() or "mise" not in call_log.read_text()
    assert "ollama" in result.stderr.lower() or "ollama" in result.stdout.lower()


def test_hook_probes_ollama_api_tags_endpoint(fake_bin: Path, call_log: Path) -> None:
    """The reachability probe hits the same `/api/tags` endpoint as tests/local/conftest.py."""
    _run_hook(fake_bin, call_log, curl_exit=0)

    assert "/api/tags" in call_log.read_text()
