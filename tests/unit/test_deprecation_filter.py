"""Repo-guard test demonstrating the pytest deprecation-warning filter actually fires (Req 7.4).

Criterion 7.2 requires `pyproject.toml` to convert a `pydantic_ai` deprecation
warning into a test failure, but pytest's warning-filter `module` field
matches the module the warning is *attributed to* -- the caller identified by
`warnings.warn`'s `stacklevel` argument -- not the module whose code calls
`warnings.warn`. A filter keyed on `pydantic_ai`'s own module name would
therefore never fire for a deprecated call attributed to first-party code
(the exact shape of a real `pydantic_ai` deprecation, which warns from deep
inside the library but attributes the warning to its caller). This test
proves the filter fires for that attribution rather than assuming it from
the filter's mere presence in `pyproject.toml` (Principle 5, Req 13.5).

The scenario runs a nested pytest session against a throwaway test module
(outside this repository's test tree, so none of this repo's fixtures or
`conftest.py` files apply) using this repository's own `pyproject.toml` as
its config via `-c`. That module calls a helper which raises a
`DeprecationWarning` with `stacklevel=2`, attributing the warning to the
*scenario test module* -- standing in for first-party code -- rather than to
the helper module that calls `warnings.warn`.
"""

import subprocess
import sys
from pathlib import Path


_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"

_HELPER_SOURCE = '''\
"""Throwaway helper for test_deprecation_filter.py's nested pytest scenario."""

import warnings


def deprecated_call() -> None:
    """Warn with DeprecationWarning, attributed to the caller via stacklevel=2."""
    warnings.warn("simulated first-party deprecation", DeprecationWarning, stacklevel=2)
'''

_SCENARIO_SOURCE = '''\
"""Throwaway scenario module for test_deprecation_filter.py's nested pytest run."""

from helper import deprecated_call


def test_trigger_deprecation() -> None:
    """Call the helper so the warning is attributed to this module."""
    deprecated_call()
'''


def _run_scenario(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the throwaway scenario under this repo's pytest config and return the result."""
    (tmp_path / "helper.py").write_text(_HELPER_SOURCE)
    (tmp_path / "test_scenario.py").write_text(_SCENARIO_SOURCE)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_PYPROJECT_PATH),
            "-p",
            "no:cacheprovider",
            str(tmp_path / "test_scenario.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_deprecation_warning_attributed_to_first_party_code_fails_the_session(
    tmp_path: Path,
) -> None:
    """The configured filter must fail a session on a DeprecationWarning from first-party code.

    Before criterion 7.2's filter is declared, this fails (the nested run
    exits 0): that RED observation is what makes the later GREEN a
    demonstration rather than an assumption (Req 7.4).
    """
    result = _run_scenario(tmp_path)

    assert result.returncode != 0, (
        "Expected the configured `error::DeprecationWarning` filter to fail "
        "the nested pytest run over a warning attributed to first-party "
        f"code; it exited 0 instead.\nstdout:\n{result.stdout}"
    )
    assert "DeprecationWarning" in result.stdout, (
        "Expected the nested run's failure to be attributed to the "
        f"DeprecationWarning.\nstdout:\n{result.stdout}"
    )
