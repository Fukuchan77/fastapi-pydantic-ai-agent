"""Unit tests for the `EXPECT_LIVE_TESTS` anti-false-green guard (Req 14.3).

Boundary correction: Task 15's declared boundary for this subtask is
`tests/conftest.py` only, but TDD (Critical Constraints) requires a failing
test written first, and conftest.py files are not collected as test modules
by pytest - a `def test_*` inside `tests/conftest.py` itself would never run.
Testing `LiveTestCountGuard` directly (imported from `tests.conftest`, which
`tests/__init__.py` makes a regular importable package) exercises the exact
counting/comparison logic the real `pytest_runtest_logreport`/
`pytest_sessionfinish` hooks delegate to, without needing a nested pytest
subprocess/`pytester` session - a necessary consequence of the TDD mandate,
not scope creep, mirroring the "boundary correction" precedent used
throughout this spec's earlier tasks (e.g. Task 3's `.env.example`, Task 9's
`session_store.py` regex widening).
"""

from tests.conftest import LiveTestCountGuard


def test_records_only_call_phase_reports() -> None:
    """Only reports whose `when` is `"call"` increment the count."""
    guard = LiveTestCountGuard()
    guard.record("setup")
    guard.record("call")
    guard.record("teardown")
    guard.record("call")
    assert guard.call_count == 2


def test_check_returns_none_when_count_matches_expected() -> None:
    """No failure message when the actual count equals `EXPECT_LIVE_TESTS`."""
    guard = LiveTestCountGuard()
    guard.record("call")
    guard.record("call")
    assert guard.check(2) is None


def test_check_returns_a_message_when_a_gated_lane_was_silently_skipped() -> None:
    """A failure message names both the expected and actual counts on mismatch."""
    guard = LiveTestCountGuard()
    guard.record("call")
    error = guard.check(5)
    assert error is not None
    assert "EXPECT_LIVE_TESTS=5" in error
    assert "1 test(s) actually executed" in error


def test_check_fails_when_more_tests_ran_than_expected() -> None:
    """A count higher than expected is also reported, not just a shortfall."""
    guard = LiveTestCountGuard()
    for _ in range(3):
        guard.record("call")
    error = guard.check(1)
    assert error is not None
    assert "EXPECT_LIVE_TESTS=1" in error
