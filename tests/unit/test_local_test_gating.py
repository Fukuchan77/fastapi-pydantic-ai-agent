"""Unit tests for the Ollama model gating helper (`tests/support/ollama.py`).

`tests/local/` only runs when Ollama is reachable, so its gating logic is never
exercised by CI. These hermetic tests cover `skip_unless_model_pulled` directly
so a regression in the gating shows up in the normal unit lane instead of
resurfacing as a confusing `ModelHTTPError` on a developer's machine.
"""

import pytest

from tests.support.ollama import OLLAMA_BASE_URL
from tests.support.ollama import skip_unless_model_pulled


def test_skip_unless_model_pulled_allows_pulled_model() -> None:
    """A model present in the pulled set should not skip the test."""
    skip_unless_model_pulled(frozenset({"granite4.1:8b", "llama3.2:latest"}), "granite4.1:8b")


def test_skip_unless_model_pulled_skips_absent_model() -> None:
    """A model missing from the pulled set should raise pytest's Skipped outcome."""
    with pytest.raises(pytest.skip.Exception) as exc_info:
        skip_unless_model_pulled(frozenset({"llama3.2:latest"}), "granite4.1:8b")

    message = str(exc_info.value)
    assert "granite4.1:8b" in message, "Skip reason should name the missing model"
    assert "ollama pull granite4.1:8b" in message, "Skip reason should show the remedy"
    assert OLLAMA_BASE_URL in message, "Skip reason should name the probed Ollama URL"
    assert "llama3.2:latest" in message, "Skip reason should list what is actually pulled"


def test_skip_unless_model_pulled_reports_empty_pulled_set() -> None:
    """With nothing pulled, the skip reason should say so rather than show a blank list."""
    with pytest.raises(pytest.skip.Exception) as exc_info:
        skip_unless_model_pulled(frozenset(), "granite4.1:8b")

    assert "(none)" in str(exc_info.value), "Empty pulled set should render as '(none)'"


def test_skip_unless_model_pulled_requires_exact_tag_match() -> None:
    """Matching is exact: a bare name must not satisfy a tagged requirement.

    Ollama's /api/tags reports fully tagged names, and LiteLLM routes the tag
    verbatim, so `granite4.1` must not be accepted for `granite4.1:8b`.
    """
    with pytest.raises(pytest.skip.Exception):
        skip_unless_model_pulled(frozenset({"granite4.1"}), "granite4.1:8b")
