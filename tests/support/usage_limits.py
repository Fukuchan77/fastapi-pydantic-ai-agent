"""Two-layer assertion form for `UsageLimitExceeded` message templates.

Lives alongside `tests/support/redis.py`, `tests/support/chroma.py`, and
`tests/support/ollama.py`. `tests/unit/agents/test_usage_limit_templates.py`
drives each of `pydantic_ai.usage.UsageLimits`' raise sites against the
installed library and asserts through `assert_usage_limit_message` rather than
by exact string equality (Req 1.1-1.3): a pinned-prefix comparison survives an
upstream release appending a help-text sentence, while a keying-substring
containment check still catches a rewrite of the leading sentence that
`classify_usage_limit_exceeded` (`app/agents/guardrails.py`) keys its
`StopReason` on. This module owns only the assertion form itself - neither the
per-raise-site template strings nor how many raise sites exist.
"""

from collections.abc import Mapping


def assert_usage_limit_message(
    raised: Exception | str,
    *,
    prefix: str,
    keying_substrings: Mapping[str, bool],
) -> None:
    """Assert a `UsageLimitExceeded` message against the two-layer policy.

    Three layers, checked in order: the message starts with the pinned
    template prefix (Req 1.1); each keying substring expected present is
    contained in the message (Req 1.2); each keying substring expected absent
    is not contained in the message (Req 1.3). `test_appended_suffix_still_passes`
    and `test_rewritten_leading_sentence_fails` demonstrate this form against
    synthetic messages rather than assuming it (Req 1.6).

    Args:
        raised: The raised exception, or its already-rendered message, to
            check. An exception is rendered via `str()` before comparison.
        prefix: The pinned leading template text the message must start with.
        keying_substrings: Maps each substring the classifier inspects to
            whether the message is expected to contain it (`True`) or must
            not contain it (`False`).

    Raises:
        AssertionError: Naming which layer failed - the prefix comparison, a
            missing expected-present substring, or a present expected-absent
            substring.
    """
    message = str(raised)

    if not message.startswith(prefix):
        raise AssertionError(
            f"prefix layer failed: message {message!r} does not start with pinned prefix {prefix!r}"
        )

    for substring, expected_present in keying_substrings.items():
        is_present = substring in message
        if expected_present and not is_present:
            raise AssertionError(
                f"positive keying-substring layer failed: expected {substring!r} "
                f"in message {message!r}"
            )
        if not expected_present and is_present:
            raise AssertionError(
                f"negative keying-substring layer failed: unexpected {substring!r} "
                f"found in message {message!r}"
            )
