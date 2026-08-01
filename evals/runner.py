"""Offline golden-set runner for the evals harness (Req 6.5).

Wired to `mise run evals`, invoked only from the availability-gated
`pre-push` hook (Req 1.6/1.7) - never from GitHub Actions (Req 1.8), since
it makes real LLM calls against both the agent under test and the judge.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import httpx
from pydantic import BaseModel
from pydantic_ai.models import Model

from app.agents.chat_agent import ChatOutput
from app.agents.chat_agent import build_chat_agent
from app.agents.chat_agent import build_model
from app.agents.deps import AgentDeps
from app.config import Settings
from app.config import get_settings
from app.llm.factory import settings_for_model_id
from app.stores.session_store import InMemorySessionStore
from evals.graders import Judge
from evals.graders import LLMJudge
from evals.graders import Rating
from evals.graders import aggregate_ratings
from evals.graders import grade_case


logger = logging.getLogger(__name__)

GOLDEN_DIR = Path(__file__).parent / "golden"
"""Default directory `load_golden_cases()` reads `*.json` files from."""

_MIN_PASSING_SCORE = 3.0
"""An axis aggregate below this fails the run (and blocks the push, Req 1.6)."""


class GoldenCase(BaseModel):
    """One golden-set input and its expected outcome (Req 6.5).

    Attributes:
        id: Stable identifier for this case, used to report results.
        prompt: The user message sent to the agent under test.
        expected: A natural-language description of what a correct,
            complete response should contain - read by the judge, not
            compared programmatically.
    """

    id: str
    prompt: str
    expected: str


class CaseResult(BaseModel):
    """One graded golden case: the agent's output plus both axis ratings."""

    case_id: str
    agent_output: str
    outcome: Rating
    behavior: Rating


class EvalReport(BaseModel):
    """The full result of one offline evals run (Req 6.5)."""

    results: list[CaseResult]
    outcome_aggregate: float | None
    behavior_aggregate: float | None

    @property
    def passed(self) -> bool:
        """Whether every graded axis met the minimum passing score.

        An aggregate of `None` (every rating on that axis was `"Unknown"`,
        or there were no cases at all) does not fail the run - there is
        nothing numeric to fail on.
        """
        return (
            self.outcome_aggregate is None or self.outcome_aggregate >= _MIN_PASSING_SCORE
        ) and (self.behavior_aggregate is None or self.behavior_aggregate >= _MIN_PASSING_SCORE)


AgentRunner = Callable[[GoldenCase], Awaitable[str]]
"""Runs the agent under test against one golden case, returning its text output."""


def load_golden_cases(golden_dir: Path = GOLDEN_DIR) -> list[GoldenCase]:
    """Load every golden case from the `*.json` files under `golden_dir` (Req 6.5).

    Args:
        golden_dir: Directory containing golden-set JSON files, each a JSON
            array of case objects.

    Returns:
        All parsed golden cases, in file-then-array order.
    """
    cases: list[GoldenCase] = []
    for path in sorted(golden_dir.glob("*.json")):
        raw_cases = json.loads(path.read_text())
        cases.extend(GoldenCase.model_validate(raw_case) for raw_case in raw_cases)
    return cases


async def run_evals(
    cases: Sequence[GoldenCase],
    agent_runner: AgentRunner,
    judge: Judge[GoldenCase],
) -> EvalReport:
    """Run every case through the agent under test and grade it (Req 6.1, 6.5).

    Args:
        cases: The golden cases to run.
        agent_runner: Produces the agent-under-test's output for one case.
        judge: The independent judge grading each case's output (Req 6.4).

    Returns:
        An eval report with per-case results and both axis aggregates.
    """
    results: list[CaseResult] = []
    for case in cases:
        output = await agent_runner(case)
        graded = await grade_case(judge, case, output)
        results.append(
            CaseResult(
                case_id=case.id,
                agent_output=output,
                outcome=graded.outcome,
                behavior=graded.behavior,
            )
        )
    return EvalReport(
        results=results,
        outcome_aggregate=aggregate_ratings([result.outcome for result in results]),
        behavior_aggregate=aggregate_ratings([result.behavior for result in results]),
    )


def _log_report(report: EvalReport) -> None:
    """Log a per-case and aggregate summary of `report` (Req 6.5)."""
    for result in report.results:
        logger.info(
            "[%s] outcome=%s behavior=%s",
            result.case_id,
            result.outcome.score,
            result.behavior.score,
        )
        logger.info("  outcome rationale: %s", result.outcome.rationale)
        logger.info("  behavior rationale: %s", result.behavior.rationale)
    logger.info("outcome aggregate: %s", report.outcome_aggregate)
    logger.info("behavior aggregate: %s", report.behavior_aggregate)
    logger.info("evals %s", "PASSED" if report.passed else "FAILED")


_CLOUD_PROVIDERS = {"openai", "anthropic", "groq"}
"""Providers that require `llm_api_key`; mirrors `app/config/llm.py`'s check.

Kept here rather than in `Settings.validate_cloud_provider_api_key()`: that
validator runs on every `Settings()` construction, including production
`uvicorn` startup, which never reads `judge_model` - an evals-only
misconfiguration (a cloud `JUDGE_MODEL` without `LLM_API_KEY`) should not be
able to fail the production API's startup.
"""


def _build_judge_model(settings: Settings) -> Model:
    """Build the judge's model, independent of the agent-under-test's when configured.

    Args:
        settings: Application settings; `judge_model` selects the judge's
            model, mirroring how `build_fallback_model()` builds each
            fallback via `model_copy()` (Req 6.4).

    Returns:
        Model: The judge's model - a distinct one when `judge_model` is set,
        otherwise `llm_model` again (self-evaluation, logged loudly below).

    Raises:
        ValueError: If `judge_model` names a cloud provider and
            `llm_api_key` is unset (there is only one configured LLM API
            key, shared by `llm_model`/`llm_fallback_models`/`judge_model`).
    """
    if settings.judge_model is None:
        logger.warning(
            "AUDIT: judge_model is not configured - the evals judge will grade "
            "the agent under test with its own model (%s). This defeats the "
            "self-evaluation-bias mitigation Judge[T] exists to provide; set "
            "JUDGE_MODEL to an independent 'provider:model' to fix.",
            settings.llm_model,
        )
        return build_model(settings)

    judge_provider = settings.judge_model.split(":", 1)[0]
    if judge_provider in _CLOUD_PROVIDERS and settings.llm_api_key is None:
        raise ValueError(
            f"llm_api_key is required when using cloud provider '{judge_provider}' "
            f"(from judge_model '{settings.judge_model}'). Please set the "
            f"LLM_API_KEY environment variable."
        )
    return build_model(settings_for_model_id(settings, settings.judge_model))


async def _run_against_live_agent() -> EvalReport:
    """Build the real agent-under-test and judge from settings, then run.

    Both the agent under test and the judge make real LLM calls, so this
    is only ever invoked from `main()` (i.e. `mise run evals`), never
    imported at module-load time.
    """
    settings = get_settings()
    agent = build_chat_agent(settings=settings)
    judge = LLMJudge[GoldenCase](_build_judge_model(settings))
    cases = load_golden_cases()

    async with httpx.AsyncClient() as http_client:
        deps = AgentDeps(
            http_client=http_client,
            settings=settings,
            session_store=InMemorySessionStore(),
        )

        async def agent_runner(case: GoldenCase) -> str:
            result = await agent.run(case.prompt, deps=deps)
            return (
                result.output.reply if isinstance(result.output, ChatOutput) else str(result.output)
            )

        return await run_evals(cases, agent_runner, judge)


def main() -> int:
    """Entry point for `mise run evals` (Req 6.5).

    Returns:
        `0` when every graded axis meets the minimum passing score, `1`
        otherwise - the process exit code the pre-push hook blocks on
        (Req 1.6).
    """
    report = asyncio.run(_run_against_live_agent())
    _log_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
