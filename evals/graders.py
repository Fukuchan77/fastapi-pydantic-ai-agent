"""Two-axis LLM-judge grader for the offline evals harness (Req 6.1-6.4).

Grades an agent output on two axes - Outcome (correctness, completeness) and
Behavior (tool-use discipline, faithfulness) - each on a 1-5 scale or
`"Unknown"`, with a required rationale. `Judge[T]` is a structural protocol
so a golden-set run can inject an independent judge model, avoiding
self-evaluation bias (Req 6.4); `LLMJudge` is the default implementation,
backed by a dedicated pydantic-ai `Agent`.
"""

import asyncio
from typing import Literal
from typing import Protocol

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic_ai import Agent
from pydantic_ai.models import Model


Axis = Literal["outcome", "behavior"]
"""The two grading axes (Req 6.1): outcome = correctness/completeness,
behavior = tool-use discipline/faithfulness."""

_AXIS_DESCRIPTIONS: dict[Axis, str] = {
    "outcome": "Outcome - the correctness and completeness of the final answer.",
    "behavior": "Behavior - tool-use discipline and faithfulness to available context.",
}

_JUDGE_OUTPUT_RETRIES = 2


class Rating(BaseModel):
    """A single 1-5 (or `Unknown`) rating with a required rationale (Req 6.2/6.3)."""

    score: Literal[1, 2, 3, 4, 5, "Unknown"]
    rationale: str = Field(min_length=1)

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_stringified_digit_score(cls, value: object) -> object:
        """Coerce a stringified digit score (e.g. `'4'`) to its integer form.

        A small local judge model has been observed returning `score` as a
        stringified digit rather than an integer. Only the exact strings
        `"1"`-`"5"` are coerced; every other value - including `"Unknown"`
        and out-of-range or malformed strings - passes through unchanged,
        so the closed `Literal` domain still rejects it (Req 3.1, 3.2).
        """
        if isinstance(value, str) and value in {"1", "2", "3", "4", "5"}:
            return int(value)
        return value


class TwoAxisResult(BaseModel):
    """Outcome + Behavior ratings for one graded agent output (Req 6.1)."""

    outcome: Rating
    behavior: Rating


class Judge[T](Protocol):
    """Independent grading judge, injectable to avoid self-evaluation bias (Req 6.4)."""

    async def grade(self, axis: Axis, case: T, agent_output: str) -> Rating:
        """Grade one axis of one case's agent output.

        Args:
            axis: Which axis to grade.
            case: The golden case being graded (prompt + expectations).
            agent_output: The text produced by the agent under test.

        Returns:
            A rating with a 1-5 score (or `"Unknown"`) and a rationale.
        """
        ...


async def grade_case[T](judge: Judge[T], case: T, agent_output: str) -> TwoAxisResult:
    """Grade both axes of one case's agent output concurrently (Req 6.1).

    Args:
        judge: The independent judge to grade with.
        case: The golden case being graded.
        agent_output: The text produced by the agent under test.

    Returns:
        The outcome and behavior ratings for `agent_output`.
    """
    outcome, behavior = await asyncio.gather(
        judge.grade("outcome", case, agent_output),
        judge.grade("behavior", case, agent_output),
    )
    return TwoAxisResult(outcome=outcome, behavior=behavior)


def aggregate_ratings(ratings: list[Rating]) -> float | None:
    """Average numeric scores, excluding `Unknown` ratings (Req 6.2).

    Args:
        ratings: Ratings to aggregate.

    Returns:
        The mean of the numeric scores, or `None` when every rating is
        `"Unknown"` (or `ratings` is empty).
    """
    numeric_scores = [rating.score for rating in ratings if rating.score != "Unknown"]
    if not numeric_scores:
        return None
    return sum(numeric_scores) / len(numeric_scores)


class LLMJudge[T]:
    """Default `Judge[T]` implementation, backed by a pydantic-ai `Agent`.

    Wraps `model` in a dedicated grading agent so the caller can inject a
    model distinct from the agent under test (Req 6.4) - e.g. a stronger or
    otherwise independent model, to avoid self-evaluation bias.
    """

    def __init__(self, model: Model | str) -> None:
        """Build the judge's grading agent.

        Args:
            model: The model used to grade - independent of the agent under
                test, per Req 6.4.
        """
        self._agent = Agent[object, Rating](
            model=model,
            output_type=Rating,
            # See app/agents/chat_agent.py: `output_retries=` is deprecated in
            # pydantic-ai 1.x; the mapping form is the supported spelling.
            retries={"output": _JUDGE_OUTPUT_RETRIES},
        )

    async def grade(self, axis: Axis, case: T, agent_output: str) -> Rating:
        """Grade one axis of one case's agent output via the wrapped agent.

        Args:
            axis: Which axis to grade.
            case: The golden case being graded, rendered via `str()`.
            agent_output: The text produced by the agent under test.

        Returns:
            The judge model's rating.
        """
        prompt = (
            "You are an impartial evaluator grading an AI agent's response.\n\n"
            f"Grading axis: {_AXIS_DESCRIPTIONS[axis]}\n\n"
            f"Case:\n{case}\n\n"
            f"Agent output:\n{agent_output}\n\n"
            "Rate this response 1 (worst) to 5 (best) on this axis alone. "
            'If this axis cannot be judged from the given information, use "Unknown". '
            "Always provide a rationale."
        )
        result = await self._agent.run(prompt)
        return result.output
