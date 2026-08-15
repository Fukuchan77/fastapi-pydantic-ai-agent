"""Behavioral tests for the availability-gated pre-push hook (AC 1.6, 1.7).

The hook shells out to `curl` (Ollama reachability probe) and `mise` (the
actual test/evals runners). Both are stubbed with fake executables placed
first on PATH so the test never touches the network or a real mise install.

Boundary correction: Task 1.6's declared boundary is `.githooks/pre-push`
only, but TDD requires a failing test before the implementation - the same
"boundary correction" precedent used elsewhere in this spec. This file
already behaviorally covers the hook, so the `EXPECT_LIVE_TESTS` wiring
(Req 13.8) extends it rather than adding a new module.
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
echo "mise $* EXPECT_LIVE_TESTS=${EXPECT_LIVE_TESTS-<unset>}" >> "$CALL_LOG"
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


def _mise_call_line(calls: str, *, matching: str) -> str:
    """Return the single logged `mise` invocation line containing `matching`."""
    lines = [line for line in calls.splitlines() if matching in line]
    assert len(lines) == 1, f"expected exactly one call matching {matching!r}, got: {lines}"
    return lines[0]


def test_hook_sets_expect_live_tests_for_test_local(fake_bin: Path, call_log: Path) -> None:
    """Req 13.8: `test:local` runs with `EXPECT_LIVE_TESTS` pinned to a real count.

    Without this, a lane that silently collects zero live cases (e.g. the
    `ollama` marker matching nothing) would report success.
    """
    _run_hook(fake_bin, call_log, curl_exit=0)

    line = _mise_call_line(call_log.read_text(), matching="test:local")
    assert "EXPECT_LIVE_TESTS=<unset>" not in line
    assert "EXPECT_LIVE_TESTS=" in line


def test_hook_does_not_set_expect_live_tests_for_evals(fake_bin: Path, call_log: Path) -> None:
    """Req 13.8: `evals` invocation leaves `EXPECT_LIVE_TESTS` unset.

    `evals` runs `evals/runner.py` directly, not pytest, so the plugin has no
    session to attach to there - setting the variable would be inert.
    `evals/runner.py`'s own golden-set exit-code check is that lane's
    anti-false-green guard instead.
    """
    _run_hook(fake_bin, call_log, curl_exit=0)

    line = _mise_call_line(call_log.read_text(), matching="evals")
    assert "EXPECT_LIVE_TESTS=<unset>" in line
