"""Tests for the offline golden-set runner (Task 13.2, Req 6.5).

Hermetic: the agent-under-test and judge are both injected fakes, so these
tests never reach a real model or the network (the unit tier's autouse
`block_network` fixture would fail them if they tried).
"""

import json
import logging
from pathlib import Path

import pytest

from evals.graders import Axis
from evals.graders import Rating
from evals.runner import EvalReport
from evals.runner import GoldenCase
from evals.runner import _build_judge_model
from evals.runner import load_golden_cases
from evals.runner import run_evals
from tests.conftest import build_test_settings


class _StubJudge:
    """Returns a pre-configured rating per (case_id, axis), regardless of output."""

    def __init__(self, ratings_by_case: dict[str, dict[Axis, Rating]]) -> None:
        self._ratings_by_case = ratings_by_case

    async def grade(self, axis: Axis, case: GoldenCase, agent_output: str) -> Rating:
        """Look up the pre-configured rating for `case.id`/`axis`."""
        return self._ratings_by_case[case.id][axis]


def _write_golden_file(directory: Path, name: str, cases: list[dict[str, str]]) -> None:
    (directory / name).write_text(json.dumps(cases))


class TestLoadGoldenCases:
    """Req 6.5: golden cases are loaded from `evals/golden/*.json`."""

    def test_loads_all_cases_from_all_files(self, tmp_path: Path) -> None:
        """Every case across every JSON file in the directory is loaded."""
        _write_golden_file(
            tmp_path,
            "a.json",
            [{"id": "a1", "prompt": "What is 2+2?", "expected": "4"}],
        )
        _write_golden_file(
            tmp_path,
            "b.json",
            [{"id": "b1", "prompt": "Capital of Japan?", "expected": "Tokyo"}],
        )

        cases = load_golden_cases(tmp_path)

        assert {case.id for case in cases} == {"a1", "b1"}
        assert all(isinstance(case, GoldenCase) for case in cases)

    def test_empty_directory_yields_no_cases(self, tmp_path: Path) -> None:
        """A directory with no golden files loads to an empty list."""
        assert load_golden_cases(tmp_path) == []

    def test_default_golden_directory_has_at_least_one_case(self) -> None:
        """The shipped `evals/golden/` dataset is non-empty (Req 6.5)."""
        assert len(load_golden_cases()) > 0


class TestRunEvals:
    """Req 6.1/6.4/6.5: run every case through the agent under test and grade it."""

    @pytest.mark.asyncio
    async def test_runs_every_case_and_aggregates_scores(self) -> None:
        """Each case is run, graded, and both axes are aggregated."""
        cases = [
            GoldenCase(id="c1", prompt="p1", expected="e1"),
            GoldenCase(id="c2", prompt="p2", expected="e2"),
        ]
        judge = _StubJudge(
            {
                "c1": {
                    "outcome": Rating(score=5, rationale="Correct."),
                    "behavior": Rating(score=4, rationale="Fine."),
                },
                "c2": {
                    "outcome": Rating(score=3, rationale="Partially correct."),
                    "behavior": Rating(score="Unknown", rationale="No tools involved."),
                },
            }
        )

        async def agent_runner(case: GoldenCase) -> str:
            return f"answer for {case.id}"

        report = await run_evals(cases, agent_runner, judge)

        assert [result.case_id for result in report.results] == ["c1", "c2"]
        assert report.results[0].agent_output == "answer for c1"
        assert report.outcome_aggregate == pytest.approx((5 + 3) / 2)
        assert report.behavior_aggregate == pytest.approx(4.0)  # "Unknown" excluded

    @pytest.mark.asyncio
    async def test_empty_case_list_yields_empty_report(self) -> None:
        """No cases means no results and `None` aggregates."""

        async def agent_runner(case: GoldenCase) -> str:
            raise AssertionError("should not be called")

        report = await run_evals([], agent_runner, _StubJudge({}))

        assert report.results == []
        assert report.outcome_aggregate is None
        assert report.behavior_aggregate is None


class TestEvalReportPassed:
    """Req 6.5 (pre-push gate): the report's pass/fail verdict."""

    def test_passes_when_both_aggregates_meet_the_threshold(self) -> None:
        """Aggregates at or above the minimum passing score pass."""
        report = EvalReport(results=[], outcome_aggregate=3.0, behavior_aggregate=5.0)
        assert report.passed is True

    def test_fails_when_either_aggregate_is_below_the_threshold(self) -> None:
        """A single sub-threshold axis fails the whole run."""
        report = EvalReport(results=[], outcome_aggregate=2.9, behavior_aggregate=5.0)
        assert report.passed is False

    def test_none_aggregate_does_not_fail_the_run(self) -> None:
        """An axis with nothing numeric to score (`None`) does not fail the run."""
        report = EvalReport(results=[], outcome_aggregate=None, behavior_aggregate=None)
        assert report.passed is True


class TestBuildJudgeModel:
    """Req 6.4: the judge model must be independent of the agent-under-test's model.

    Otherwise the judge grades with itself, defeating self-evaluation-bias mitigation.
    """

    def test_uses_judge_model_when_configured(self) -> None:
        """An independent judge_model is used verbatim, not llm_model."""
        settings = build_test_settings(
            llm_model="openai:gpt-4",
            judge_model="anthropic:claude-3-5-sonnet-20241022",
        )
        model = _build_judge_model(settings)
        assert model.model_name == "anthropic/claude-3-5-sonnet-20241022"

    def test_falls_back_to_llm_model_and_warns_when_unconfigured(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without judge_model, the judge self-evaluates and this is logged loudly."""
        settings = build_test_settings(llm_model="openai:gpt-4", judge_model=None)
        with caplog.at_level(logging.WARNING, logger="evals.runner"):
            model = _build_judge_model(settings)
        assert model.model_name == "openai/gpt-4"
        assert any("AUDIT" in record.message for record in caplog.records)

    def test_cross_provider_judge_model_does_not_inherit_llm_base_url(self) -> None:
        """A judge_model on a different provider must not reuse llm_model's base_url.

        `llm_base_url` is provider-specific (e.g. a self-hosted LiteLLM proxy
        or Azure OpenAI endpoint) - carrying it over unmodified would point
        the judge's (different-provider) client at the wrong endpoint.
        """
        settings = build_test_settings(
            llm_model="ollama:granite3.3",
            llm_base_url="http://localhost:11434",
            judge_model="anthropic:claude-3-5-sonnet-20241022",
        )
        model = _build_judge_model(settings)
        assert model.model_name == "anthropic/claude-3-5-sonnet-20241022"
        assert model.settings is None or "litellm_api_base" not in model.settings

    def test_same_provider_judge_model_keeps_llm_base_url(self) -> None:
        """A judge_model on the SAME provider as llm_model should keep the shared base_url."""
        settings = build_test_settings(
            llm_model="ollama:granite3.3",
            llm_base_url="http://localhost:11434",
            judge_model="ollama:llama3.1",
        )
        model = _build_judge_model(settings)
        assert model.settings is not None
        assert model.settings["litellm_api_base"] == "http://localhost:11434/"

    def test_cloud_judge_model_without_llm_api_key_raises(self) -> None:
        """A cloud judge_model with no llm_api_key must fail loudly, not build unauthenticated.

        This check lives in `_build_judge_model`, not `Settings`, so a
        misconfigured judge_model (evals-only) can never fail production
        `uvicorn` startup (which never reads judge_model).
        """
        settings = build_test_settings(
            llm_model="ollama:granite3.3",
            llm_api_key=None,
            judge_model="anthropic:claude-3-5-sonnet-20241022",
        )
        with pytest.raises(ValueError, match=r"(?i)llm_api_key.*anthropic"):
            _build_judge_model(settings)
