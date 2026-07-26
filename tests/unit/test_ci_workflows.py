"""Static validation of GitHub Actions CI workflows and their config wiring.

Enforces spec Requirement 1 (CI Pipeline and Coverage Enforcement, AC 1.1-1.9)
without needing to actually execute on GitHub Actions infrastructure: parses
the workflow YAML and the project config files that back it.
"""

import re
import tomllib
from pathlib import Path

import yaml


PR_WORKFLOW = Path(".github/workflows/pr.yml")
SECURITY_WORKFLOW = Path(".github/workflows/security.yml")
PYPROJECT = Path("pyproject.toml")
MISE_TOML = Path("mise.toml")

_FULL_SHA_PIN_RE = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _load_workflow(path: Path) -> dict:
    """Load a GitHub Actions workflow, tolerating the PyYAML `on:` bool quirk."""
    data = yaml.safe_load(path.read_text())
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


def _all_uses_refs(workflow: dict) -> list[str]:
    refs = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            uses = step.get("uses")
            if uses:
                refs.append(uses)
    return refs


def _all_run_commands(workflow: dict) -> list[str]:
    commands = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if run:
                commands.append(run)
    return commands


def test_pr_workflow_triggers_on_pull_request_to_default_branch() -> None:
    """AC 1.1: the PR workflow runs when a PR targets the default branch."""
    workflow = _load_workflow(PR_WORKFLOW)
    pull_request = workflow["on"]["pull_request"]
    assert "main" in pull_request.get("branches", [])


def test_pr_workflow_has_cancel_in_progress_concurrency_group() -> None:
    """AC 1.9: superseded runs for the same PR are cancelled via concurrency."""
    workflow = _load_workflow(PR_WORKFLOW)
    concurrency = workflow["concurrency"]
    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is True


def test_pr_workflow_actions_are_pinned_to_commit_sha() -> None:
    """AC 1.9: third-party actions are pinned to a full commit SHA, not a tag."""
    workflow = _load_workflow(PR_WORKFLOW)
    refs = _all_uses_refs(workflow)
    assert refs, "expected at least one `uses:` step in pr.yml"
    for ref in refs:
        assert _FULL_SHA_PIN_RE.match(ref), f"{ref} is not pinned to a 40-char commit SHA"


def test_pr_workflow_runs_lint() -> None:
    """AC 1.1: the workflow runs `mise run lint` and can fail the run on it."""
    workflow = _load_workflow(PR_WORKFLOW)
    commands = _all_run_commands(workflow)
    assert any("mise run lint" in cmd for cmd in commands)


def test_pr_workflow_runs_pip_audit() -> None:
    """AC 1.4: the workflow runs pip-audit via the `audit` mise task."""
    workflow = _load_workflow(PR_WORKFLOW)
    commands = _all_run_commands(workflow)
    assert any("mise run audit" in cmd for cmd in commands)


def test_pr_workflow_runs_unit_integration_e2e_test_stage() -> None:
    """AC 1.2: the test stage runs unit+integration+e2e (TestModel/FunctionModel only)."""
    workflow = _load_workflow(PR_WORKFLOW)
    commands = _all_run_commands(workflow)
    assert any("mise run test:ci" in cmd for cmd in commands)


def test_pr_workflow_never_runs_ollama_or_evals() -> None:
    """AC 1.2, 1.8: GitHub Actions never invokes Ollama-dependent tests or evals."""
    workflow = _load_workflow(PR_WORKFLOW)
    commands = _all_run_commands(workflow)
    joined = "\n".join(commands).lower()
    assert "ollama" not in joined
    assert "test:local" not in joined
    assert "run evals" not in joined


def test_security_workflow_is_schedule_only() -> None:
    """AC 1.5: the nightly cron workflow is not triggered by pushes or PRs."""
    workflow = _load_workflow(SECURITY_WORKFLOW)
    on = workflow["on"]
    assert "schedule" in on
    assert "pull_request" not in on
    assert "push" not in on


def test_security_workflow_actions_are_pinned_to_commit_sha() -> None:
    """AC 1.9: the cron workflow also pins its third-party actions to a SHA."""
    workflow = _load_workflow(SECURITY_WORKFLOW)
    refs = _all_uses_refs(workflow)
    assert refs, "expected at least one `uses:` step in security.yml"
    for ref in refs:
        assert _FULL_SHA_PIN_RE.match(ref), f"{ref} is not pinned to a 40-char commit SHA"


def test_security_workflow_runs_pip_audit_and_gitleaks_only() -> None:
    """AC 1.5: the cron workflow runs only pip-audit and gitleaks."""
    workflow = _load_workflow(SECURITY_WORKFLOW)
    commands = _all_run_commands(workflow)
    refs = _all_uses_refs(workflow)
    assert any("mise run audit" in cmd for cmd in commands)
    assert any("gitleaks" in ref.lower() for ref in refs)


def test_security_workflow_never_runs_ollama_or_evals() -> None:
    """AC 1.5, 1.8: the nightly cron never runs Ollama tests or the evals suite."""
    workflow = _load_workflow(SECURITY_WORKFLOW)
    commands = _all_run_commands(workflow)
    joined = "\n".join(commands).lower()
    assert "ollama" not in joined
    assert "test:local" not in joined
    assert "run evals" not in joined


def test_coverage_fail_under_is_80_percent() -> None:
    """AC 1.3: line coverage below 80% fails the PR test stage."""
    config = tomllib.loads(PYPROJECT.read_text())
    assert config["tool"]["coverage"]["report"]["fail_under"] == 80


def test_mise_audit_task_runs_pip_audit() -> None:
    """AC 1.4: `mise run audit` is the task the PR workflow invokes for pip-audit."""
    config = tomllib.loads(MISE_TOML.read_text())
    audit_task = config["tasks"]["audit"]
    assert "pip-audit" in audit_task["run"]


def test_mise_test_ci_task_covers_only_unit_integration_e2e() -> None:
    """AC 1.2: `mise run test:ci` covers exactly unit+integration+e2e with coverage."""
    config = tomllib.loads(MISE_TOML.read_text())
    ci_task = config["tasks"]["test:ci"]
    run = ci_task["run"]
    for directory in ("tests/unit", "tests/integration", "tests/e2e"):
        assert directory in run
    assert "tests/local" not in run
    assert "tests/benchmarks" not in run
    assert "--cov=app" in run


def test_mise_evals_task_placeholder_exists() -> None:
    """AC 1.6 (placeholder): `mise run evals` exists as a no-op until task 13.2."""
    config = tomllib.loads(MISE_TOML.read_text())
    assert "evals" in config["tasks"]
