"""Offline PR-gate metrics over two evals runs (X-8, docs/cross-repo-adoption-backlog.md).

Ported from `vaz-ai-next/packages/evals/src/pr-gate.ts`'s design (baseline
diff, over/under-trigger balance, per-case cost/latency averages, a
report-only threshold below a minimum case count) - the *compute* half only.
This module makes no LLM calls and is entirely offline: it compares two
already-computed `EvalReport` snapshots (`evals.runner`'s report, persisted
via its `--output` flag), never runs the agent or the judge itself.

**Not wired into CI.** The golden set (`evals/golden/*.json`) currently has
3 cases, well below `_MIN_CASES_FOR_BLOCKING` - `report_only=True` is the
honest state today regardless of what CI does with this module's output, so
wiring a blocking CI job ahead of growing the golden set past that threshold
would be gating on noise. `mise run evals:pr-gate` runs this by hand for now.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from evals.graders import Rating
from evals.runner import _MIN_PASSING_SCORE
from evals.runner import CaseResult
from evals.runner import EvalReport


logger = logging.getLogger(__name__)

_MIN_CASES_FOR_BLOCKING = 20
"""Below this many golden cases, a run is report-only and never blocks (X-8).

Mirrors `vaz-ai-next`'s `PR_GATE_MIN_CASES_FOR_BLOCKING`: a handful of cases
is too noisy a sample to gate a merge on.
"""


class PassRate(BaseModel):
    """The current run's per-case pass rate, and its delta from a baseline."""

    current: float
    previous: float | None = None
    delta: float | None = None


class TriggerBalance(BaseModel):
    """Which cases flipped pass/fail relative to the baseline, in each direction.

    "Over-trigger" (a regression: passed at the baseline, fails now) and
    "under-trigger" (an improvement: failed at the baseline, passes now) are
    reported separately rather than netted against each other, so a PR that
    fixes one case while regressing another is visible as both, not a wash.
    """

    over_trigger_ids: list[str]
    over_trigger_count: int
    under_trigger_ids: list[str]
    under_trigger_count: int


class CaseAverages(BaseModel):
    """Per-case cost/latency averages over one run's graded (non-skipped) cases."""

    average_tokens: float | None
    average_duration_ms: float | None


class PrGateReport(BaseModel):
    """The full result of one PR-gate comparison (X-8)."""

    golden_set_size: int
    report_only: bool
    should_block: bool
    has_case_failure: bool
    has_new_case_failure: bool
    pass_rate: PassRate
    trigger_balance: TriggerBalance
    current: CaseAverages
    previous: CaseAverages | None = None


def _axis_passed(rating: Rating) -> bool:
    """Whether one axis rating meets the passing bar.

    `"Unknown"` (nothing numeric to judge) does not fail a case, mirroring
    `EvalReport.passed`'s treatment of an all-`"Unknown"` aggregate.
    """
    return rating.score == "Unknown" or rating.score >= _MIN_PASSING_SCORE


def _case_passed(result: CaseResult) -> bool:
    """Whether one graded case passed - both axes at/above the passing bar."""
    return _axis_passed(result.outcome) and _axis_passed(result.behavior)


def _graded(results: Sequence[CaseResult]) -> list[CaseResult]:
    """Filter out skipped cases - they carry no pass/fail or cost signal."""
    return [result for result in results if not result.skipped]


def _compute_averages(results: Sequence[CaseResult]) -> CaseAverages:
    """Average `total_tokens`/`duration_ms` over `results`, skipping unset values.

    A field being unset on every result (e.g. `total_tokens`, which
    `evals.runner.run_evals()` does not currently populate - see
    `CaseResult`'s docstring) yields `None` rather than a misleading `0.0`.
    """
    tokens = [result.total_tokens for result in results if result.total_tokens is not None]
    durations = [result.duration_ms for result in results if result.duration_ms is not None]
    return CaseAverages(
        average_tokens=(sum(tokens) / len(tokens)) if tokens else None,
        average_duration_ms=(sum(durations) / len(durations)) if durations else None,
    )


def compute_pr_gate_metrics(current: EvalReport, previous: EvalReport | None) -> PrGateReport:
    """Compare `current` against an optional `previous` baseline run (X-8).

    Pure function: no I/O, no LLM calls. Blocking is gated on both the
    golden-set size (`_MIN_CASES_FOR_BLOCKING`) and an actual case failure
    this run - a failure that already existed at the baseline still blocks,
    not only a newly-introduced one, so a chronically red case can't let
    every subsequent PR through unblocked.

    Args:
        current: This run's report (e.g. from `evals.runner --output`).
        previous: A prior run's report to diff against, or `None` when no
            baseline is available (e.g. the first run, or a corrupted
            baseline per `read_baseline_sample`) - regression/trigger-balance
            fields degrade to their empty/`None` form rather than raising.

    Returns:
        The full comparison, including whether this run should block.
    """
    graded_current = _graded(current.results)
    current_pass_map = {result.case_id: _case_passed(result) for result in graded_current}
    pass_count = sum(current_pass_map.values())
    pass_rate_current = pass_count / len(graded_current) if graded_current else 1.0
    has_case_failure = pass_count < len(graded_current)

    trigger_balance = TriggerBalance(
        over_trigger_ids=[], over_trigger_count=0, under_trigger_ids=[], under_trigger_count=0
    )
    pass_rate_previous: float | None = None
    pass_rate_delta: float | None = None
    previous_averages: CaseAverages | None = None
    has_new_case_failure = False

    if previous is not None:
        graded_previous = _graded(previous.results)
        previous_pass_map = {result.case_id: _case_passed(result) for result in graded_previous}
        prev_pass_count = sum(previous_pass_map.values())
        pass_rate_previous = prev_pass_count / len(graded_previous) if graded_previous else 1.0
        pass_rate_delta = pass_rate_current - pass_rate_previous
        previous_averages = _compute_averages(graded_previous)

        over_trigger_ids = sorted(
            case_id
            for case_id, passed in current_pass_map.items()
            # `is True` (not a truthy default) excludes a case absent from
            # the baseline: a brand-new failing case hasn't "regressed" -
            # there is nothing to regress from - it surfaces via
            # `has_new_case_failure` instead, not as an over-trigger.
            if not passed and previous_pass_map.get(case_id) is True
        )
        under_trigger_ids = sorted(
            case_id
            for case_id, passed in current_pass_map.items()
            if passed and previous_pass_map.get(case_id) is False
        )
        trigger_balance = TriggerBalance(
            over_trigger_ids=over_trigger_ids,
            over_trigger_count=len(over_trigger_ids),
            under_trigger_ids=under_trigger_ids,
            under_trigger_count=len(under_trigger_ids),
        )
        has_new_case_failure = any(
            case_id not in previous_pass_map
            for case_id, passed in current_pass_map.items()
            if not passed
        )

    golden_set_size = len(current.results)
    report_only = golden_set_size < _MIN_CASES_FOR_BLOCKING
    should_block = (not report_only) and has_case_failure

    return PrGateReport(
        golden_set_size=golden_set_size,
        report_only=report_only,
        should_block=should_block,
        has_case_failure=has_case_failure,
        has_new_case_failure=has_new_case_failure,
        pass_rate=PassRate(
            current=pass_rate_current, previous=pass_rate_previous, delta=pass_rate_delta
        ),
        trigger_balance=trigger_balance,
        current=_compute_averages(graded_current),
        previous=previous_averages,
    )


def read_baseline_sample(path: Path | None) -> EvalReport | None:
    """Load a prior run's `EvalReport` from disk, tolerating absence or corruption.

    Args:
        path: Path to a JSON file written by `evals.runner`'s `--output`
            flag, or `None` when no baseline is configured.

    Returns:
        The parsed report, or `None` when: `path` is `None`; the file
        doesn't exist (expected on the first run, or after a CI cache
        eviction - silent, not a warning); or the file's content fails to
        parse as an `EvalReport` (a corrupted baseline disables regression
        blocking for this run rather than crashing the gate - logged as a
        warning, since unlike a missing file this is unexpected).
    """
    if path is None or not path.exists():
        return None
    try:
        return EvalReport.model_validate_json(path.read_text())
    except (ValueError, TypeError):
        logger.warning("Corrupt PR-gate baseline at %s, ignoring it", path, exc_info=True)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: `mise run evals:pr-gate` (X-8).

    Args:
        argv: Command-line arguments, or `None` to read `sys.argv`.

    Returns:
        `1` when `should_block` is true, `0` otherwise (including every
        report-only run, regardless of what it found).
    """
    parser = argparse.ArgumentParser(
        description="Compare an evals run against a baseline (X-8, offline, no LLM calls)."
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to the current run's EvalReport JSON (evals.runner --output).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to a prior run's EvalReport JSON, if any.",
    )
    args = parser.parse_args(argv)

    current = EvalReport.model_validate_json(args.current.read_text())
    baseline = read_baseline_sample(args.baseline)
    report = compute_pr_gate_metrics(current, baseline)

    logger.info("%s", report.model_dump_json(indent=2))
    return 1 if report.should_block else 0


if __name__ == "__main__":
    raise SystemExit(main())
