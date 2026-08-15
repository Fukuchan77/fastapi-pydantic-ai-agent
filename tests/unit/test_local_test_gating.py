"""Unit tests for the Ollama model gating helper (`tests/support/ollama.py`).

`tests/local/` only runs when Ollama is reachable, so its gating logic is never
exercised by CI. These hermetic tests cover `skip_unless_model_pulled` directly
so a regression in the gating shows up in the normal unit lane instead of
resurfacing as a confusing `ModelHTTPError` on a developer's machine.

This module also carries the Ollama lane's live-test-count drift guard (Req
6.7, 9.3). Unlike the Chroma and Redis lanes, whose live cases each live in
one gated module (`test_chroma_test_gating.py`, `test_redis_test_gating.py`
introspect that single module directly), the `ollama` marker is applied per
test **function**, never as a module-level `pytestmark`, and is scattered
across however many modules `tests/local/` holds. A single-module
introspection shape would therefore silently undercount the moment a second
live module exists, so the guard below walks the whole package instead of
importing one hardcoded module - a future module (e.g. a readiness-probe live
test) is picked up automatically with no further edit here.

The count constant this guard pins is `tests/support/ollama.py::
OLLAMA_LIVE_TEST_COUNT`. The guard was written deliberately ahead of that
constant (Req 9.3 - the test must demonstrably fail before the change it
verifies exists, not merely land alongside it), so its red phase was an
`ImportError` at collection rather than an `AssertionError`; the constant
has since landed and this module collects normally.

The guard also asserts the constant against every place that restates it
outside `tests/local/` itself (Req 6.7's "with the drift guard asserting both
restatements green"): the pre-push hook's `EXPECT_LIVE_TESTS` literal, and its
prose restatements in `CLAUDE.md` and `AGENTS.md`. `docs/adapter-probe-report*.md`
also states a count and is deliberately excluded - Req 6.8 forbids editing a
gate-evidence artifact when this count rises.

Boundary correction: the raise of those three restatements is declared
against `.githooks/pre-push`, `CLAUDE.md`, and `AGENTS.md` only, but TDD
requires a failing test written before the raise - the same "boundary
correction" precedent already used by `test_chroma_test_gating.py` and
`test_redis_test_gating.py` for their own restatement guards.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest

import tests.local
from tests.support.ollama import OLLAMA_BASE_URL
from tests.support.ollama import OLLAMA_LIVE_TEST_COUNT
from tests.support.ollama import skip_unless_model_pulled


PRE_PUSH_HOOK = Path(".githooks/pre-push")
CLAUDE_MD = Path("CLAUDE.md")
AGENTS_MD = Path("AGENTS.md")

_EXPECT_LIVE_TESTS_RE = re.compile(r"EXPECT_LIVE_TESTS=(\d+)\s+mise run test:local")


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


def _iter_local_test_modules() -> list:
    """Import and return every `test_*` module under `tests/local/`.

    Discovered by walking the package rather than importing a fixed list of
    module names, so a module added later is counted without editing this
    guard.
    """
    return [
        importlib.import_module(module_info.name)
        for module_info in pkgutil.iter_modules(
            tests.local.__path__, prefix=f"{tests.local.__name__}."
        )
        if module_info.name.rsplit(".", 1)[-1].startswith("test_")
    ]


def _count_ollama_marked_functions() -> int:
    """Count `ollama`-marked test functions across all of `tests/local/`'s modules."""
    return sum(
        1
        for module in _iter_local_test_modules()
        for name, obj in vars(module).items()
        if name.startswith("test_")
        and inspect.isfunction(obj)
        and any(mark.name == "ollama" for mark in getattr(obj, "pytestmark", []))
    )


def test_ollama_live_test_count_matches_gated_modules() -> None:
    """The declared count equals the number of `ollama`-marked functions in `tests/local/`."""
    assert _count_ollama_marked_functions() == OLLAMA_LIVE_TEST_COUNT


def _extract_expect_live_tests(path: Path) -> int:
    """Extract the `EXPECT_LIVE_TESTS=N mise run test:local` literal from `path`'s text."""
    match = _EXPECT_LIVE_TESTS_RE.search(path.read_text())
    assert match is not None, (
        f"expected an 'EXPECT_LIVE_TESTS=N mise run test:local' literal in {path}"
    )
    return int(match.group(1))


def test_ollama_live_test_count_matches_pre_push_hook_literal() -> None:
    """The pre-push hook's `EXPECT_LIVE_TESTS` literal for `test:local` matches the constant."""
    assert _extract_expect_live_tests(PRE_PUSH_HOOK) == OLLAMA_LIVE_TEST_COUNT


def test_ollama_live_test_count_matches_claude_md_restatement() -> None:
    """`CLAUDE.md`'s pre-push prose restatement of the count matches the constant."""
    assert _extract_expect_live_tests(CLAUDE_MD) == OLLAMA_LIVE_TEST_COUNT


def test_ollama_live_test_count_matches_agents_md_restatement() -> None:
    """`AGENTS.md`'s condensed pre-push restatement of the count matches the constant."""
    assert _extract_expect_live_tests(AGENTS_MD) == OLLAMA_LIVE_TEST_COUNT
