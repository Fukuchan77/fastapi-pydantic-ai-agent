"""Tests for the two-axis LLM-judge grader (Task 13.1, Req 6.1-6.4)."""

from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.models.function import FunctionModel

from evals.graders import Axis
from evals.graders import LLMJudge
from evals.graders import Rating
from evals.graders import TwoAxisResult
from evals.graders import aggregate_ratings
from evals.graders import grade_case


class TestRating:
    """Req 6.2/6.3: 1-5 or Unknown score, with a required rationale."""

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5, "Unknown"])
    def test_accepts_valid_scores(self, score: int | str) -> None:
        """Every value in the closed 1-5-or-Unknown vocabulary validates."""
        rating = Rating(score=score, rationale="Because.")
        assert rating.score == score

    @pytest.mark.parametrize(
        ("stringified_score", "expected_score"),
        [("1", 1), ("2", 2), ("3", 3), ("4", 4), ("5", 5)],
    )
    def test_coerces_stringified_digit_scores_to_int(
        self, stringified_score: str, expected_score: int
    ) -> None:
        """A score arriving as a stringified digit is accepted as the corresponding integer.

        Reproduces, hermetically and through the rating model directly
        rather than through a live judge, the exact brittleness that
        exhausted output retries against a small local judge model: it
        returned `Rating.score` as the string `'4'` instead of the
        integer `4`. (Req 3.1, 3.5)
        """
        rating = Rating(score=stringified_score, rationale="Because.")  # type: ignore[arg-type]
        assert rating.score == expected_score
        assert isinstance(rating.score, int)

    def test_leaves_unknown_score_unchanged(self) -> None:
        """`"Unknown"` is untouched by the digit-coercion validator (Req 3.2).

        The coercion added for Req 3.1 only recognizes the exact strings
        `"1"`-`"5"`; `"Unknown"` must fall through it unchanged rather than
        being coerced to some numeric value, since it is not a stringified
        digit at all.
        """
        rating = Rating(score="Unknown", rationale="Because.")
        assert rating.score == "Unknown"
        assert isinstance(rating.score, str)

    @pytest.mark.parametrize("score", [0, 6, -1, "unknown", "unrated", "4.5", "four", ""])
    def test_rejects_invalid_scores(self, score: object) -> None:
        """Values outside the closed vocabulary are rejected exactly as today (Req 3.3).

        `"4.5"`, `"four"`, and `""` are not in the coercion validator's exact
        `"1"`-`"5"` set, so they reach the closed `Literal` domain unchanged
        and are rejected there, the same as the pre-existing `0`/`6`/`-1`
        cases.
        """
        with pytest.raises(ValidationError):
            Rating(score=score, rationale="Because.")  # type: ignore[arg-type]

    def test_requires_rationale(self) -> None:
        """A rationale is mandatory for every rating (Req 6.3)."""
        with pytest.raises(ValidationError):
            Rating(score=3)  # type: ignore[call-arg]

    def test_rejects_empty_rationale(self) -> None:
        """An empty-string rationale does not satisfy the "required" contract."""
        with pytest.raises(ValidationError):
            Rating(score=3, rationale="")


class _RecordingJudge:
    """A duck-typed `Judge[T]` that records every `grade()` call.

    Deliberately does not inherit from `evals.graders.Judge` - Req 6.4 only
    requires structural conformance, not a concrete base class.
    """

    def __init__(self, ratings: dict[Axis, Rating]) -> None:
        self.ratings = ratings
        self.calls: list[tuple[Axis, Any, str]] = []

    async def grade(self, axis: Axis, case: Any, agent_output: str) -> Rating:
        """Record the call and return the pre-configured rating for `axis`."""
        self.calls.append((axis, case, agent_output))
        return self.ratings[axis]


class TestGradeCase:
    """Req 6.1/6.4: two-axis grading via an injected judge."""

    @pytest.mark.asyncio
    async def test_grades_both_axes_via_injected_judge(self) -> None:
        """Both axes are graded, each with the judge's per-axis rating."""
        judge = _RecordingJudge(
            {
                "outcome": Rating(score=5, rationale="Fully correct."),
                "behavior": Rating(score=2, rationale="Called an unneeded tool."),
            }
        )

        result = await grade_case(judge, case="golden-case-1", agent_output="The answer is 42.")

        assert isinstance(result, TwoAxisResult)
        assert result.outcome.score == 5
        assert result.behavior.score == 2
        assert {call[0] for call in judge.calls} == {"outcome", "behavior"}
        assert all(call[1] == "golden-case-1" for call in judge.calls)
        assert all(call[2] == "The answer is 42." for call in judge.calls)

    @pytest.mark.asyncio
    async def test_accepts_any_structurally_conforming_judge(self) -> None:
        """A plain object with an async `grade()` method is injectable (Req 6.4)."""

        class MinimalJudge:
            async def grade(self, axis: Axis, case: Any, agent_output: str) -> Rating:
                return Rating(score="Unknown", rationale=f"No opinion on {axis}.")

        result = await grade_case(MinimalJudge(), case=object(), agent_output="anything")

        assert result.outcome.score == "Unknown"
        assert result.behavior.score == "Unknown"


class TestAggregateRatings:
    """Req 6.2: aggregate scores exclude `Unknown` ratings."""

    def test_excludes_unknown_from_average(self) -> None:
        """`Unknown` ratings are dropped before averaging."""
        ratings = [
            Rating(score=4, rationale="Good."),
            Rating(score="Unknown", rationale="Could not tell."),
            Rating(score=2, rationale="Missed a detail."),
        ]
        assert aggregate_ratings(ratings) == pytest.approx(3.0)

    def test_returns_none_when_every_rating_is_unknown(self) -> None:
        """An all-`Unknown` sequence aggregates to `None`, not zero (Req 3.4).

        `"Unknown"` is excluded from the aggregate rather than coerced to a
        number, so a fully-`"Unknown"` axis is `None`, not `0.0`.
        """
        ratings = [Rating(score="Unknown", rationale="N/A") for _ in range(3)]
        assert aggregate_ratings(ratings) is None

    def test_returns_none_for_empty_sequence(self) -> None:
        """No ratings at all also aggregates to `None`."""
        assert aggregate_ratings([]) is None

    def test_single_numeric_rating(self) -> None:
        """A single numeric rating aggregates to itself."""
        assert aggregate_ratings([Rating(score=3, rationale="Adequate.")]) == pytest.approx(3.0)


class TestLLMJudge:
    """The default `Judge[T]` implementation, backed by a pydantic-ai `Agent`."""

    def test_init_wires_judge_output_retries_onto_the_agent(self) -> None:
        """`_JUDGE_OUTPUT_RETRIES` must actually reach the constructed Agent.

        Pins the `retries={"output": _JUDGE_OUTPUT_RETRIES}` mapping form
        (pydantic-ai 1.x deprecated `output_retries=`) against silently
        regressing to `retries={}` or dropping the kwarg - both would still
        construct successfully, so only checking the wired value catches it.
        `_max_output_retries` is pydantic-ai's only place this is observable
        post-construction; there is no public accessor.
        """
        from evals.graders import _JUDGE_OUTPUT_RETRIES

        def respond(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[])

        judge: LLMJudge[str] = LLMJudge(FunctionModel(respond))

        assert judge._agent._max_output_retries == _JUDGE_OUTPUT_RETRIES

    @pytest.mark.asyncio
    async def test_grade_returns_parsed_rating(self) -> None:
        """A structured tool-call response is parsed into a `Rating`."""

        def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool_name, args={"score": 4, "rationale": "Solid."})]
            )

        judge: LLMJudge[str] = LLMJudge(FunctionModel(respond))

        rating = await judge.grade("outcome", "some-case", "The agent's answer.")

        assert rating == Rating(score=4, rationale="Solid.")

    @pytest.mark.asyncio
    async def test_grade_prompt_mentions_the_requested_axis(self) -> None:
        """The judge's prompt distinguishes outcome grading from behavior grading."""
        captured_prompts: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured_prompts.append(str(messages[-1].parts[-1].content))  # type: ignore[union-attr]
            tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool_name, args={"score": 3, "rationale": "Ok."})]
            )

        judge: LLMJudge[str] = LLMJudge(FunctionModel(respond))

        await judge.grade("outcome", "case", "output text")
        await judge.grade("behavior", "case", "output text")

        assert "outcome" in captured_prompts[0].lower()
        assert "behavior" in captured_prompts[1].lower()
