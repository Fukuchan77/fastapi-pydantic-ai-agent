"""Static validation of the Dependabot configuration.

Enforces spec Requirement 4 (Dependabot configuration that cannot propose a
rate-limit-breaking bump, criterion 4.7) by parsing `.github/dependabot.yml`
and failing when any required ignore entry, the minor/patch group, or the
open-pull-requests limit is absent. Mirrors the parsing approach
`test_ci_workflows.py` uses for the CI workflow YAML, so Requirement 4 is
enforceable rather than merely reviewable.
"""

from pathlib import Path

import yaml


DEPENDABOT_CONFIG = Path(".github/dependabot.yml")


def _load_config() -> dict:
    return yaml.safe_load(DEPENDABOT_CONFIG.read_text())


def _uv_update() -> dict:
    config = _load_config()
    for update in config["updates"]:
        if update["package-ecosystem"] == "uv":
            return update
    raise AssertionError("expected a uv package-ecosystem entry in dependabot.yml")


def _ignore_entry(update: dict, dependency_name: str) -> dict:
    for entry in update.get("ignore", []):
        if entry.get("dependency-name") == dependency_name:
            return entry
    raise AssertionError(f"no ignore entry for {dependency_name!r}")


def test_starlette_major_updates_are_ignored() -> None:
    """AC 4.1: a starlette major bump can never be proposed again."""
    entry = _ignore_entry(_uv_update(), "starlette")
    assert "version-update:semver-major" in entry["update-types"]


def test_fastapi_minor_and_major_updates_are_ignored() -> None:
    """AC 4.2: fastapi 0.x classifies 0.136->0.137 as a minor bump.

    Both minor and major must be ignored or the rate-limit-breaking bump
    still gets through.
    """
    entry = _ignore_entry(_uv_update(), "fastapi")
    assert "version-update:semver-minor" in entry["update-types"]
    assert "version-update:semver-major" in entry["update-types"]


def test_chromadb_and_redis_major_updates_are_ignored() -> None:
    """AC 4.3: chromadb/redis majors are deliberately shelved, not unsupported."""
    update = _uv_update()
    for name in ("chromadb", "redis"):
        entry = _ignore_entry(update, name)
        assert "version-update:semver-major" in entry["update-types"]


def test_minor_and_patch_updates_are_grouped_into_one_group() -> None:
    """AC 4.4: minor+patch updates land in one group.

    Previously they accumulated as 13 separate PRs.
    """
    groups = _uv_update().get("groups", {})
    assert len(groups) == 1
    (group,) = groups.values()
    assert set(group["update-types"]) == {"minor", "patch"}


def test_open_pull_requests_limit_is_declared_explicitly() -> None:
    """AC 4.4: an explicit limit is declared.

    This stops uv-ecosystem PRs from silently accumulating past
    Dependabot's default again.
    """
    limit = _uv_update().get("open-pull-requests-limit")
    assert isinstance(limit, int)
    assert limit > 0
