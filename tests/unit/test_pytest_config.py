"""Repo-guard test pinning pytest's asyncio collection settings.

Req 13.1/13.2: task 1.1's `asyncio_mode = "auto"` setting is measured as
byte-identical in effect on today's suite (no test observing suite behaviour
changed pass/fail counts). That means no test exercising suite *behaviour*
could ever catch this setting being reverted or dropped in a later merge —
only a direct assertion on `pyproject.toml` itself can.
"""

import tomllib
from pathlib import Path


def test_asyncio_mode_is_auto() -> None:
    """Verify [tool.pytest.ini_options] pins asyncio_mode to "auto".

    Without this, a coroutine test missing an explicit @pytest.mark.asyncio
    marker collects as a passing test without ever being awaited.
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    ini_options = pyproject["tool"]["pytest"]["ini_options"]

    assert ini_options["asyncio_mode"] == "auto", (
        "pyproject.toml's [tool.pytest.ini_options] must set asyncio_mode = "
        '"auto" (Req 13.1/13.2) so an unmarked coroutine test cannot pass '
        "without being awaited."
    )


def test_asyncio_default_fixture_loop_scope_is_function() -> None:
    """Verify [tool.pytest.ini_options] pins the default async fixture loop scope.

    Required alongside asyncio_mode = "auto": without an explicit scope,
    auto mode falls back to a session-scoped default event loop shared
    across tests instead of a fresh one per test function.
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    ini_options = pyproject["tool"]["pytest"]["ini_options"]

    assert ini_options["asyncio_default_fixture_loop_scope"] == "function", (
        "pyproject.toml's [tool.pytest.ini_options] must set "
        'asyncio_default_fixture_loop_scope = "function" (Req 13.1/13.2).'
    )


def test_filterwarnings_declares_error_on_deprecation_warning() -> None:
    """Verify [tool.pytest.ini_options] declares the deprecation-warning filter.

    Req 7.2: without `error::DeprecationWarning`, a `pydantic_ai` deprecation
    warning collects as a passing test instead of failing the session. The
    filter is demonstrated (not merely declared) by
    tests/unit/test_deprecation_filter.py; this guard only pins that the key
    cannot be silently dropped from configuration.
    """
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    ini_options = pyproject["tool"]["pytest"]["ini_options"]

    assert "error::DeprecationWarning" in ini_options["filterwarnings"], (
        "pyproject.toml's [tool.pytest.ini_options] must declare "
        '"error::DeprecationWarning" in filterwarnings (Req 7.2).'
    )
