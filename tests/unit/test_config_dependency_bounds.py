"""Test that all dependencies in pyproject.toml have version upper bounds.

This test ensures that all dependencies specify upper bounds (e.g., <2.0, <1.0)
to prevent breaking changes from future major version releases.
"""

import re
import tomllib
from pathlib import Path


def test_all_dependencies_have_upper_bounds() -> None:
    """Verify all production dependencies have version upper bounds.

    Security & Stability Requirement (Task 16.4):
    - All dependencies MUST specify upper bounds to prevent breaking changes
    - Format: "package>=X.Y.Z,<NEXT_MAJOR" or "package>=X.Y.Z,<NEXT_MINOR"
    - Example: "fastapi>=0.135.1,<1.0" or "pydantic-settings>=2.13.1,<3.0"

    This prevents:
    - Automatic upgrades to incompatible major versions
    - Breaking API changes in production
    - Supply chain attacks via malicious version bumps
    """
    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dependencies = pyproject["project"]["dependencies"]

    # Track dependencies without upper bounds
    missing_upper_bounds: list[str] = []

    # Pattern to detect upper bounds: must contain '<' (e.g., <2.0, <1.0)
    # Valid formats:
    # - "package>=1.0.0,<2.0"
    # - "package>=1.0,<2"
    # - "package[extras]>=1.0,<2"
    upper_bound_pattern = re.compile(r"<\d+")

    for dep in dependencies:
        # Parse dependency spec (may contain extras like package[extra]>=1.0)
        # We only care about the version constraint part
        if not upper_bound_pattern.search(dep):
            missing_upper_bounds.append(dep)

    # Assert all dependencies have upper bounds
    assert not missing_upper_bounds, (
        f"The following dependencies are missing upper bounds:\n"
        f"{'  - ' + chr(10) + '  - '.join(missing_upper_bounds)}\n\n"
        f"Add upper bounds to prevent breaking changes. Examples:\n"
        f"  - 'fastapi>=0.135.1' → 'fastapi>=0.135.1,<1.0'\n"
        f"  - 'httpx>=0.28.1' → 'httpx>=0.28.1,<1.0'\n"
        f"  - 'pydantic-settings>=2.13.1' → 'pydantic-settings>=2.13.1,<3.0'\n"
    )


def test_dev_dependencies_have_upper_bounds() -> None:
    """Verify all dev dependencies have version upper bounds.

    Dev dependencies should also specify upper bounds to ensure:
    - Reproducible test results across environments
    - Consistent CI/CD behavior
    - No surprise breakages from linter/test framework updates
    """
    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    dev_dependencies = pyproject["dependency-groups"]["dev"]

    # Track dependencies without upper bounds
    missing_upper_bounds: list[str] = []

    # Pattern to detect upper bounds
    upper_bound_pattern = re.compile(r"<\d+")

    for dep in dev_dependencies:
        if not upper_bound_pattern.search(dep):
            missing_upper_bounds.append(dep)

    # Assert all dev dependencies have upper bounds
    assert not missing_upper_bounds, (
        f"The following dev dependencies are missing upper bounds:\n"
        f"{'  - ' + chr(10) + '  - '.join(missing_upper_bounds)}\n\n"
        f"Add upper bounds to prevent breaking changes. Examples:\n"
        f"  - 'pytest>=9.0.2' → 'pytest>=9.0.2,<10.0'\n"
        f"  - 'ruff>=0.15.7' → 'ruff>=0.15.7,<1.0'\n"
    )


def test_upper_bounds_are_reasonable() -> None:
    """Verify upper bounds are not overly restrictive.

    Upper bounds should generally be:
    - Next major version for stable packages (e.g., >=2.0,<3.0)
    - Next minor version for 0.x packages (e.g., >=0.28,<0.29)

    This test checks that bounds are not too tight (e.g., patch-level bounds).
    """
    # Read pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    all_dependencies = pyproject["project"]["dependencies"] + pyproject["dependency-groups"]["dev"]

    # Pattern to detect overly restrictive bounds (patch-level)
    # Example: >=1.2.3,<1.2.4 (BAD - too restrictive)
    # Should be: >=1.2.3,<2.0 or >=1.2.3,<1.3
    overly_restrictive_pattern = re.compile(r">=(\d+)\.(\d+)\.(\d+),<\1\.\2\.(\d+)$")

    overly_restrictive: list[str] = []

    for dep in all_dependencies:
        # Extract version constraint part (after package name)
        match = overly_restrictive_pattern.search(dep)
        if match:
            # Check if upper bound is only one patch version ahead
            lower_patch = int(match.group(3))
            upper_patch = int(match.group(4))
            if upper_patch == lower_patch + 1:
                overly_restrictive.append(dep)

    assert not overly_restrictive, (
        f"The following dependencies have overly restrictive upper bounds:\n"
        f"{'  - ' + chr(10) + '  - '.join(overly_restrictive)}\n\n"
        f"Use major or minor version bounds instead of patch-level bounds.\n"
    )
