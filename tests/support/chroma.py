"""Shared helpers for gating tests on explicit opt-in for Chroma integration tests.

Lives alongside `tests/support/ollama.py` and mirrors its skip mechanics, but the
gate itself reads an environment variable rather than probing a server: Chroma's
network dependency is a one-time Hugging Face sentence-transformers model
download, not a reachable service, so there is nothing to probe. Whether the
model happens to be cached locally is machine-state that must not leak into
whether the default lanes run these tests (Req 13.3-13.5).
"""

import os


RUN_CHROMA_INTEGRATION_TESTS_ENV_VAR = "RUN_CHROMA_INTEGRATION_TESTS"
"""Opt-in env var: unset/"0"/"false" (case-insensitive) skips, any other value runs."""

CHROMA_SKIP_REASON = (
    "Chroma integration tests require a Hugging Face sentence-transformers model "
    f"download; set {RUN_CHROMA_INTEGRATION_TESTS_ENV_VAR}=1 to opt in."
)
"""Stated reason reported when the opt-in variable is unset (Req 13.4, 13.5)."""

CHROMA_LIVE_TEST_COUNT = 6
"""Number of tests gated behind `RUN_CHROMA_INTEGRATION_TESTS`.

Pass `EXPECT_LIVE_TESTS=$CHROMA_LIVE_TEST_COUNT` alongside the opt-in (e.g.
`RUN_CHROMA_INTEGRATION_TESTS=1 EXPECT_LIVE_TESTS=6 uv run pytest
tests/integration/test_chroma_query_with_scores.py`) so a lane that silently
collects zero live cases - for example a marker expression that no longer
matches anything - fails instead of reporting success (Req 13.8).
`test_chroma_test_gating.py` guards this value against drift as tests are
added to or removed from the gated module.
"""


def chroma_integration_tests_enabled() -> bool:
    """Report whether the explicit opt-in for Chroma integration tests is set.

    Returns:
        False if `RUN_CHROMA_INTEGRATION_TESTS` is unset, `"0"`, or `"false"`
        (case-insensitive); True for any other value.
    """
    value = os.environ.get(RUN_CHROMA_INTEGRATION_TESTS_ENV_VAR, "")
    return value.lower() not in ("", "0", "false")
