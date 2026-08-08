"""Pin every `UsageLimitExceeded` message template pydantic_ai emits (Req 7.7).

`classify_usage_limit_exceeded` (`app/agents/guardrails.py`) derives the
`StopReason` from the raised exception's message text, because
`UsageLimitExceeded` carries no structured attribute distinguishing which
limit fired (see that function's docstring). `pydantic_ai.usage.UsageLimits`
has exactly seven raise sites - three in `check_before_request`, three in
`check_tokens`, one in `check_before_tool_call` - and the classifier only
recognises the `request_limit`/`tool_calls_limit` substrings. A silent
upstream reword of any of those seven messages would make the classifier
silently misclassify a stop reason instead of failing loudly, so this module
drives each real raise site directly against the installed library and pins
its exact message: if pydantic_ai changes wording, this test fails the suite
rather than letting `classify_usage_limit_exceeded` quietly return the wrong
`StopReason`.
"""

import pytest
from pydantic_ai import RunUsage
from pydantic_ai import UsageLimitExceeded
from pydantic_ai import UsageLimits

from app.agents.guardrails import classify_usage_limit_exceeded


class TestUsageLimitMessageTemplates:
    """Drive each of `UsageLimits`' seven raise sites and pin its exact message."""

    def test_check_before_request_request_limit(self) -> None:
        """`check_before_request` raises with the request_limit template."""
        limits = UsageLimits(request_limit=1)
        usage = RunUsage(requests=1)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert str(exc_info.value) == "The next request would exceed the request_limit of 1"

    def test_check_before_request_input_tokens_limit(self) -> None:
        """`check_before_request` raises with the pre-flight input_tokens_limit template."""
        limits = UsageLimits(request_limit=None, input_tokens_limit=5)
        usage = RunUsage(input_tokens=6)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert (
            str(exc_info.value)
            == "The next request would exceed the input_tokens_limit of 5 (input_tokens=6)"
        )

    def test_check_before_request_total_tokens_limit(self) -> None:
        """`check_before_request` raises with the pre-flight total_tokens_limit template."""
        limits = UsageLimits(request_limit=None, total_tokens_limit=5)
        usage = RunUsage(input_tokens=3, output_tokens=3)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_request(usage)

        assert (
            str(exc_info.value)
            == "The next request would exceed the total_tokens_limit of 5 (total_tokens=6)"
        )

    def test_check_tokens_input_tokens_limit(self) -> None:
        """`check_tokens` raises with the post-response input_tokens_limit template."""
        limits = UsageLimits(input_tokens_limit=5)
        usage = RunUsage(input_tokens=6)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_tokens(usage)

        assert str(exc_info.value) == "Exceeded the input_tokens_limit of 5 (input_tokens=6)"

    def test_check_tokens_output_tokens_limit(self) -> None:
        """`check_tokens` raises with the post-response output_tokens_limit template."""
        limits = UsageLimits(output_tokens_limit=5)
        usage = RunUsage(output_tokens=6)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_tokens(usage)

        assert str(exc_info.value) == "Exceeded the output_tokens_limit of 5 (output_tokens=6)"

    def test_check_tokens_total_tokens_limit(self) -> None:
        """`check_tokens` raises with the post-response total_tokens_limit template."""
        limits = UsageLimits(total_tokens_limit=5)
        usage = RunUsage(input_tokens=3, output_tokens=3)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_tokens(usage)

        assert str(exc_info.value) == "Exceeded the total_tokens_limit of 5 (total_tokens=6)"

    def test_check_before_tool_call_tool_calls_limit(self) -> None:
        """`check_before_tool_call` raises with the tool_calls_limit template."""
        limits = UsageLimits(tool_calls_limit=1)
        projected_usage = RunUsage(tool_calls=2)

        with pytest.raises(UsageLimitExceeded) as exc_info:
            limits.check_before_tool_call(projected_usage)

        assert (
            str(exc_info.value)
            == "The next tool call(s) would exceed the tool_calls_limit of 1 (tool_calls=2)."
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
