"""Repo-guard test: the Chroma live-test count declared in chroma.py.

Asserts it matches the actual number of tests gated behind
`RUN_CHROMA_INTEGRATION_TESTS` (Req 13.8).

`EXPECT_LIVE_TESTS` only checks what a caller passes in - it has no way to know
on its own that the Chroma-gated lane's "true" count changed, so the constant a
developer would pass alongside the opt-in (e.g.
`RUN_CHROMA_INTEGRATION_TESTS=1 EXPECT_LIVE_TESTS=$CHROMA_LIVE_TEST_COUNT`) can
silently go stale the moment a test is added to or removed from the gated
module. This guard fails loudly instead, mirroring the `test_pytest_config.py`
precedent of asserting directly on a declared value rather than relying on
suite-wide behaviour to catch drift.

Boundary correction: Task 1.5's declared boundary for this concern is
`tests/support/chroma.py` only, but TDD requires a failing test before the
constant is added - the same "boundary correction" precedent already used by
`test_expect_live_tests_plugin.py` and `test_local_test_gating.py`.
"""

import inspect

from tests.integration import test_chroma_query_with_scores
from tests.support.chroma import CHROMA_LIVE_TEST_COUNT


def test_chroma_live_test_count_matches_gated_module() -> None:
    """The declared count equals the number of `test_*` functions in the gated module."""
    actual = sum(
        1
        for name, obj in vars(test_chroma_query_with_scores).items()
        if name.startswith("test_") and inspect.isfunction(obj)
    )

    assert actual == CHROMA_LIVE_TEST_COUNT
