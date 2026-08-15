"""Two-layer assertion of every `UsageLimitExceeded` message template pydantic_ai emits (Req 1.8).

`classify_usage_limit_exceeded` (`app/agents/guardrails.py`) derives the
`StopReason` from the raised exception's message text, because
`UsageLimitExceeded` carries no structured attribute distinguishing which
limit fired (see that function's docstring). `pydantic_ai.usage.UsageLimits`'s
raise sites are enumerated one per test in `TestUsageLimitMessageTemplates`
below, and the classifier only recognises the `request_limit`/`tool_calls_limit`
substrings among the messages those sites raise. Each test drives its real
raise site directly against the installed library and asserts through
`assert_usage_limit_message` (`tests.support.usage_limits`) rather than by
exact string equality: a pinned leading-sentence prefix plus keying-substring
containment, positive at the sites the classifier keys on and negative
everywhere else (Req 1.1-1.3), proven not loosened too far by
`TestAssertionFormProof` (Req 1.6). A silent upstream reword of any raise
site's leading sentence still fails this suite rather than letting
`classify_usage_limit_exceeded` quietly return the wrong `StopReason`.
"""

from decimal import Decimal
from typing import ClassVar

import pytest
from pydantic_ai import RunUsage
from pydantic_ai import UsageLimitExceeded
from pydantic_ai import UsageLimits

from app.agents.guardrails import classify_usage_limit_exceeded
from tests.support.usage_limits import assert_usage_limit_message


class TestUsageLimitMessageTemplates:
    """Drive each of `UsageLimits`' ten raise sites and pin its message (Req 1.1-1.5).

    v2 widened the site set from the seven `003` recorded: a `cost_limit`
    branch inside `check_before_request`, and two wholly new methods -
    `check_cost` and `check_per_request_input_tokens` - each with their own
    raise site (Req 5.4).

    Each assertion goes through `assert_usage_limit_message` rather than exact
    string equality: pinned prefix at every site, plus the keying-substring
    disposition `classify_usage_limit_exceeded` inspects -
    `request_limit`/`tool_calls_limit` positive only at the two sites that
    must classify as `max_iterations`, both negative at the other eight, whose
    absence is what makes the classifier fall through to `budget_exceeded`.
    """

    def test_check_before_request_request_limit(self) -> None:
        """`check_before_request` raises with the request_limit template."""
        limits = UsageLimits(request_limit=1)
        usage = RunUsage(requests=1)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="The next request would exceed the request_limit of 1",
            keying_substrings={"request_limit": True, "tool_calls_limit": False},
        )

    def test_check_before_request_input_tokens_limit(self) -> None:
        """`check_before_request` raises with the pre-flight input_tokens_limit template."""
        limits = UsageLimits(request_limit=None, input_tokens_limit=5)
        usage = RunUsage(input_tokens=6)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="The next request would exceed the input_tokens_limit of 5 (input_tokens=6)",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_before_request_total_tokens_limit(self) -> None:
        """`check_before_request` raises with the pre-flight total_tokens_limit template."""
        limits = UsageLimits(request_limit=None, total_tokens_limit=5)
        usage = RunUsage(input_tokens=3, output_tokens=3)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="The next request would exceed the total_tokens_limit of 5 (total_tokens=6)",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_tokens_input_tokens_limit(self) -> None:
        """`check_tokens` raises with the post-response input_tokens_limit template."""
        limits = UsageLimits(input_tokens_limit=5)
        usage = RunUsage(input_tokens=6)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_tokens(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="Exceeded the input_tokens_limit of 5 (input_tokens=6)",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_tokens_output_tokens_limit(self) -> None:
        """`check_tokens` raises with the post-response output_tokens_limit template."""
        limits = UsageLimits(output_tokens_limit=5)
        usage = RunUsage(output_tokens=6)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_tokens(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="Exceeded the output_tokens_limit of 5 (output_tokens=6)",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_tokens_total_tokens_limit(self) -> None:
        """`check_tokens` raises with the post-response total_tokens_limit template."""
        limits = UsageLimits(total_tokens_limit=5)
        usage = RunUsage(input_tokens=3, output_tokens=3)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_tokens(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="Exceeded the total_tokens_limit of 5 (total_tokens=6)",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_before_tool_call_tool_calls_limit(self) -> None:
        """`check_before_tool_call` raises with the tool_calls_limit template."""
        limits = UsageLimits(tool_calls_limit=1)
        projected_usage = RunUsage(tool_calls=2)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_tool_call(projected_usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix=("The next tool call(s) would exceed the tool_calls_limit of 1 (tool_calls=2)."),
            keying_substrings={"request_limit": False, "tool_calls_limit": True},
        )

    def test_check_before_request_cost_limit(self) -> None:
        """`check_before_request` raises with the pre-flight cost_limit template (v2, Req 5.4)."""
        limits = UsageLimits(request_limit=None, cost_limit=Decimal("5"))
        usage = RunUsage(cost=Decimal("6"))

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="The next request would exceed the `cost_limit` of 5 (`cost`=Decimal('6'))",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_cost_cost_limit(self) -> None:
        """`check_cost` raises with the post-response cost_limit template (v2, Req 5.4)."""
        limits = UsageLimits(cost_limit=Decimal("5"))
        usage = RunUsage(cost=Decimal("6"))

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_cost(usage)

        assert_usage_limit_message(
            exc_info.value,
            prefix="Exceeded the `cost_limit` of 5 (`usage.cost`=Decimal('6'))",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )

    def test_check_per_request_input_tokens_per_request_input_tokens_limit(self) -> None:
        """`check_per_request_input_tokens` raises with its own template (v2, Req 5.4)."""
        limits = UsageLimits(per_request_input_tokens_limit=5)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_per_request_input_tokens(6)

        assert_usage_limit_message(
            exc_info.value,
            prefix="Exceeded the per_request_input_tokens_limit of 5 (request_input_tokens=6)",
            keying_substrings={"request_limit": False, "tool_calls_limit": False},
        )


class TestDistinctStopReasonsForDistinctLimits:
    """Req 7.7: an iteration limit and a token limit map to distinct stop reasons."""

    def test_iteration_limit_and_token_limit_classify_distinctly(self) -> None:
        """A request-count breach and a token-budget breach resolve to different `StopReason`s.

        Both exceptions are the real objects `UsageLimits` raises - not
        hand-typed strings - so this proves the classifier's iteration/budget
        split holds against the library's actual templates, not merely
        against wording this test happens to guess.
        """
        with pytest.raises(UsageLimitExceeded) as request_limit_exc_info:
            UsageLimits(request_limit=1).check_before_request(RunUsage(requests=1))

        with pytest.raises(UsageLimitExceeded) as token_limit_exc_info:
            UsageLimits(total_tokens_limit=5).check_tokens(
                RunUsage(input_tokens=3, output_tokens=3)
            )

        iteration_stop_reason = classify_usage_limit_exceeded(request_limit_exc_info.value)
        budget_stop_reason = classify_usage_limit_exceeded(token_limit_exc_info.value)

        assert iteration_stop_reason == "max_iterations"
        assert budget_stop_reason == "budget_exceeded"
        assert iteration_stop_reason != budget_stop_reason


class TestAssertionFormProof:
    """Req 1.6: prove the loosened prefix-plus-containment form still has teeth.

    Drives `tests.support.usage_limits.assert_usage_limit_message` against
    synthetic messages rather than a real raise site, so what is demonstrated
    is the assertion form the suite runs, not merely the classifier it
    protects. A policy loosened from exact equality without this proof would
    be indistinguishable from one loosened too far (Req 1.4, 1.5).
    """

    _PREFIX = "The next request would exceed the request_limit of 1"
    _KEYING_SUBSTRINGS: ClassVar[dict[str, bool]] = {
        "request_limit": True,
        "tool_calls_limit": False,
    }

    def test_appended_suffix_still_passes(self) -> None:
        """An upstream sentence appended after the pinned prefix must not fail (Req 1.4)."""
        message = f"{self._PREFIX} (retry after the cooldown window)."

        assert_usage_limit_message(
            message,
            prefix=self._PREFIX,
            keying_substrings=self._KEYING_SUBSTRINGS,
        )

    def test_rewritten_leading_sentence_fails(self) -> None:
        """A rewritten leading sentence must fail, even keeping the keying substring (Req 1.5)."""
        message = "Your allotted request_limit of 1 has already been used up for this run."

        with pytest.raises(AssertionError):
            assert_usage_limit_message(
                message,
                prefix=self._PREFIX,
                keying_substrings=self._KEYING_SUBSTRINGS,
            )
