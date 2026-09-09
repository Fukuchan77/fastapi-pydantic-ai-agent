"""Tests for the offline PR-gate metrics module (X-8, `evals/pr_gate.py`).

Fully hermetic: every test builds `EvalReport`/`CaseResult` objects directly
(or writes them to `tmp_path` as JSON), never invoking `run_evals()` or any
LLM. `compute_pr_gate_metrics()` and `read_baseline_sample()` are pure/IO-only
functions with no network reach, so the unit tier's autouse `block_network`
fixture would fail these tests instantly if that assumption were wrong.
"""

import json
import logging
from pathlib import Path

import pytest

from evals.graders import Rating
from evals.pr_gate import compute_pr_gate_metrics
from evals.pr_gate import main
from evals.pr_gate import read_baseline_sample
from evals.runner import CaseResult
from evals.runner import EvalReport


def _case(
    case_id: str,
    *,
    outcome_score: int | str = 5,
    behavior_score: int | str = "Unknown",
    skipped: bool = False,
    total_tokens: int | None = None,
    duration_ms: float | None = None,
) -> CaseResult:
    """Build one `CaseResult`; both axes default to a pass (5 / Unknown)."""
    return CaseResult(
        case_id=case_id,
        agent_output="output",
        outcome=Rating(score=outcome_score, rationale="r"),
        behavior=Rating(score=behavior_score, rationale="r"),
        skipped=skipped,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
    )


def _report(cases: list[CaseResult]) -> EvalReport:
    """Wrap `cases` in an `EvalReport`; aggregates are irrelevant to pr_gate."""
    return EvalReport(results=cases, outcome_aggregate=None, behavior_aggregate=None)


class TestNoBaseline:
    """A single report with no `previous` to diff against."""

    def test_previous_fields_are_none(self) -> None:
        """No baseline means every baseline-derived field is `None`/empty."""
        report = compute_pr_gate_metrics(_report([_case("a")]), None)

        assert report.pass_rate.previous is None
        assert report.pass_rate.delta is None
        assert report.previous is None
        assert report.trigger_balance.over_trigger_count == 0
        assert report.trigger_balance.under_trigger_count == 0
        assert report.has_new_case_failure is False

    def test_current_pass_rate_reflects_this_run_alone(self) -> None:
        """The current pass rate is computed even with no baseline to diff."""
        report = compute_pr_gate_metrics(
            _report([_case("a", outcome_score=5), _case("b", outcome_score=1)]), None
        )

        assert report.pass_rate.current == pytest.approx(0.5)
        assert report.has_case_failure is True


class TestRegressionAndImprovement:
    """Over-trigger (regression) and under-trigger (improvement) detection."""

    def test_case_that_passed_and_now_fails_is_an_over_trigger(self) -> None:
        """A pass-to-fail flip on an existing case is an over-trigger."""
        previous = _report([_case("a", outcome_score=5)])
        current = _report([_case("a", outcome_score=1)])

        report = compute_pr_gate_metrics(current, previous)

        assert report.trigger_balance.over_trigger_ids == ["a"]
        assert report.trigger_balance.over_trigger_count == 1
        assert report.trigger_balance.under_trigger_count == 0
        assert report.has_case_failure is True

    def test_case_that_failed_and_now_passes_is_an_under_trigger(self) -> None:
        """A fail-to-pass flip on an existing case is an under-trigger."""
        previous = _report([_case("a", outcome_score=1)])
        current = _report([_case("a", outcome_score=5)])

        report = compute_pr_gate_metrics(current, previous)

        assert report.trigger_balance.under_trigger_ids == ["a"]
        assert report.trigger_balance.under_trigger_count == 1
        assert report.trigger_balance.over_trigger_count == 0
        assert report.has_case_failure is False

    def test_mixed_regression_and_improvement_are_both_reported(self) -> None:
        """A PR that regresses one case and fixes another shows both, not a wash."""
        previous = _report(
            [_case("regresses", outcome_score=5), _case("improves", outcome_score=1)]
        )
        current = _report([_case("regresses", outcome_score=1), _case("improves", outcome_score=5)])

        report = compute_pr_gate_metrics(current, previous)

        assert report.trigger_balance.over_trigger_ids == ["regresses"]
        assert report.trigger_balance.under_trigger_ids == ["improves"]

    def test_new_case_that_fails_is_flagged_but_not_an_over_trigger(self) -> None:
        """A case absent from the baseline can't have "regressed" - it's new."""
        previous = _report([_case("a", outcome_score=5)])
        current = _report([_case("a", outcome_score=5), _case("new", outcome_score=1)])

        report = compute_pr_gate_metrics(current, previous)

        assert report.trigger_balance.over_trigger_ids == []
        assert report.has_new_case_failure is True
        assert report.has_case_failure is True

    def test_pass_rate_delta_is_current_minus_previous(self) -> None:
        """`delta` is signed: negative when the pass rate dropped."""
        previous = _report([_case("a", outcome_score=5), _case("b", outcome_score=5)])
        current = _report([_case("a", outcome_score=1), _case("b", outcome_score=5)])

        report = compute_pr_gate_metrics(current, previous)

        assert report.pass_rate.previous == pytest.approx(1.0)
        assert report.pass_rate.current == pytest.approx(0.5)
        assert report.pass_rate.delta == pytest.approx(-0.5)


class TestReportOnlyThreshold:
    """Below `_MIN_CASES_FOR_BLOCKING` (20), a run never blocks."""

    def test_a_failing_small_golden_set_is_report_only(self) -> None:
        """A failure on a tiny golden set is visible but never blocks."""
        report = compute_pr_gate_metrics(_report([_case("a", outcome_score=1)]), None)

        assert report.report_only is True
        assert report.has_case_failure is True
        assert report.should_block is False

    def test_a_failing_large_golden_set_blocks(self) -> None:
        """A failure on a golden set at/above the threshold blocks."""
        cases = [_case(f"c{i}", outcome_score=5) for i in range(19)]
        cases.append(_case("c19", outcome_score=1))  # 20 total, one failing

        report = compute_pr_gate_metrics(_report(cases), None)

        assert report.golden_set_size == 20
        assert report.report_only is False
        assert report.should_block is True

    def test_a_passing_large_golden_set_does_not_block(self) -> None:
        """A fully-passing golden set at the threshold never blocks."""
        cases = [_case(f"c{i}", outcome_score=5) for i in range(20)]

        report = compute_pr_gate_metrics(_report(cases), None)

        assert report.report_only is False
        assert report.should_block is False


class TestSkippedCasesAreExcluded:
    """A skipped case contributes no pass/fail signal and no cost signal."""

    def test_skipped_case_does_not_count_toward_pass_rate_or_failure(self) -> None:
        """A failing skipped case doesn't drag down the pass rate or trip a failure."""
        report = compute_pr_gate_metrics(
            _report([_case("a", outcome_score=5), _case("skipped", outcome_score=1, skipped=True)]),
            None,
        )

        assert report.pass_rate.current == pytest.approx(1.0)
        assert report.has_case_failure is False

    def test_skipped_case_is_excluded_from_averages(self) -> None:
        """A skipped case's cost figures don't skew the token/duration averages."""
        report = compute_pr_gate_metrics(
            _report(
                [
                    _case("a", total_tokens=100, duration_ms=10.0),
                    _case("skipped", total_tokens=999, duration_ms=999.0, skipped=True),
                ]
            ),
            None,
        )

        assert report.current.average_tokens == pytest.approx(100)
        assert report.current.average_duration_ms == pytest.approx(10.0)


class TestAveragesSkipUnsetValues:
    """`total_tokens`/`duration_ms` average only over cases where the field is set."""

    def test_average_is_none_when_no_case_has_the_field_set(self) -> None:
        """No case carrying a value yields `None`, not a misleading `0.0`."""
        report = compute_pr_gate_metrics(_report([_case("a"), _case("b")]), None)

        assert report.current.average_tokens is None
        assert report.current.average_duration_ms is None

    def test_average_ignores_cases_missing_the_field(self) -> None:
        """A mix of set and unset values averages over only the set ones."""
        report = compute_pr_gate_metrics(
            _report([_case("a", total_tokens=100), _case("b", total_tokens=None)]), None
        )

        assert report.current.average_tokens == pytest.approx(100)


class TestReadBaselineSample:
    """`read_baseline_sample()` tolerates absence and corruption without raising."""

    def test_none_path_yields_none(self) -> None:
        """No path configured yields `None`."""
        assert read_baseline_sample(None) is None

    def test_missing_file_yields_none_silently(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing file is the expected first-run case - no warning logged."""
        with caplog.at_level(logging.WARNING):
            result = read_baseline_sample(tmp_path / "does-not-exist.json")

        assert result is None
        assert caplog.records == []

    def test_corrupt_file_yields_none_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A file that fails to parse warns, unlike a merely-missing one."""
        path = tmp_path / "baseline.json"
        path.write_text("not valid json {{{")

        with caplog.at_level(logging.WARNING, logger="evals.pr_gate"):
            result = read_baseline_sample(path)

        assert result is None
        assert any("Corrupt" in record.message for record in caplog.records)

    def test_valid_file_round_trips(self, tmp_path: Path) -> None:
        """A well-formed file parses back to an equivalent `EvalReport`."""
        path = tmp_path / "baseline.json"
        original = _report([_case("a")])
        path.write_text(original.model_dump_json())

        result = read_baseline_sample(path)

        assert result is not None
        assert result.results[0].case_id == "a"


class TestMainCli:
    """`main()` wires `--current`/`--baseline` file I/O to the pure compute function."""

    def test_exit_code_0_when_should_not_block(self, tmp_path: Path) -> None:
        """A passing run exits 0."""
        current_path = tmp_path / "current.json"
        current_path.write_text(_report([_case("a", outcome_score=5)]).model_dump_json())

        exit_code = main(["--current", str(current_path)])

        assert exit_code == 0

    def test_exit_code_1_when_should_block(self, tmp_path: Path) -> None:
        """A failing run at/above the blocking threshold exits 1."""
        current_path = tmp_path / "current.json"
        cases = [_case(f"c{i}", outcome_score=5) for i in range(19)]
        cases.append(_case("c19", outcome_score=1))
        current_path.write_text(_report(cases).model_dump_json())

        exit_code = main(["--current", str(current_path)])

        assert exit_code == 1

    def test_baseline_argument_is_read_and_diffed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`--baseline` is actually loaded and diffed against, not just accepted."""
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(_report([_case("a", outcome_score=5)]).model_dump_json())
        current_path = tmp_path / "current.json"
        current_path.write_text(_report([_case("a", outcome_score=1)]).model_dump_json())

        # A regression on a 1-case golden set is still report-only (exit 0),
        # but the logged report is inspected to prove --baseline was
        # actually read and diffed: without it, "a" could not appear as an
        # over-trigger.
        with caplog.at_level(logging.INFO, logger="evals.pr_gate"):
            exit_code = main(["--current", str(current_path), "--baseline", str(baseline_path)])

        assert exit_code == 0
        logged_report = json.loads(caplog.records[-1].message)
        assert logged_report["trigger_balance"]["over_trigger_ids"] == ["a"]
