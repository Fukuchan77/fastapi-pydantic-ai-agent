"""Repo-guard test: the Redis live-test count declared in `tests/support/redis.py` (Req 6.7).

Asserts `REDIS_LIVE_TEST_COUNT` matches both places that restate it: the
number of `redis`-marked tests in the gated module, and the
`EXPECT_LIVE_TESTS` literal the PR CI step supplies in
`.github/workflows/pr.yml`. Mirrors `test_chroma_test_gating.py`'s
precedent for the first restatement, extended to the second because the
change unit that bumps the session-key prefix (Requirement 11) adds a case
to this lane and therefore moves the constant - without the second
assertion, that unit would break CI on a count mismatch with nothing in the
repo to catch it before a real PR run failed.

Boundary correction: Task 7.3's declared boundary is
`tests/integration/test_redis_session_store_live.py` only, but TDD requires
a failing test written before the constant is added - the same "boundary
correction" precedent already used by `test_expect_live_tests_plugin.py` and
`test_local_test_gating.py`.
"""

import inspect
from pathlib import Path

import yaml

from tests.integration import test_redis_session_store_live
from tests.support.redis import REDIS_LIVE_TEST_COUNT


PR_WORKFLOW = Path(".github/workflows/pr.yml")


def test_redis_live_test_count_matches_gated_module() -> None:
    """The declared count equals the number of `test_*` functions in the gated module."""
    actual = sum(
        1
        for name, obj in vars(test_redis_session_store_live).items()
        if name.startswith("test_") and inspect.isfunction(obj)
    )

    assert actual == REDIS_LIVE_TEST_COUNT


def test_redis_live_test_count_matches_ci_step_literal() -> None:
    """The PR CI step's `EXPECT_LIVE_TESTS` literal for the redis lane matches the constant."""
    workflow = yaml.safe_load(PR_WORKFLOW.read_text())

    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "test:redis" in step.get("run", "")
    ]

    assert steps, "expected a PR CI step invoking the redis lane via 'mise run test:redis'"
    assert len(steps) == 1, "expected exactly one PR CI step invoking the redis lane"

    expect_live_tests = steps[0].get("env", {}).get("EXPECT_LIVE_TESTS")
    assert expect_live_tests is not None, "expected the redis lane step to set EXPECT_LIVE_TESTS"
    assert int(expect_live_tests) == REDIS_LIVE_TEST_COUNT
