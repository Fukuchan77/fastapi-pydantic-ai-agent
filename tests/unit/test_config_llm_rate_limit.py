"""Unit tests for the llm_rate_limit setting (Req 11.3)."""

import pytest
from pydantic import ValidationError

from tests.conftest import build_test_settings


def test_llm_rate_limit_default_value() -> None:
    """The default llm_rate_limit is a stricter limit than the global 1000/minute."""
    settings = build_test_settings()
    assert settings.llm_rate_limit == "30/minute"


@pytest.mark.parametrize("value", ["10/second", "5/minute", "100/hour", "1000/day"])
def test_llm_rate_limit_accepts_valid_formats(value: str) -> None:
    """Valid '<count>/<period>' strings are accepted."""
    settings = build_test_settings(llm_rate_limit=value)
    assert settings.llm_rate_limit == value


@pytest.mark.parametrize(
    "value",
    ["30 per minute", "thirty/minute", "30/fortnight", "30", "/minute", "-5/minute"],
)
def test_llm_rate_limit_rejects_invalid_formats(value: str) -> None:
    """Malformed rate-limit strings are rejected at startup, not at first request."""
    with pytest.raises(ValidationError, match=r"(?i)llm_rate_limit"):
        build_test_settings(llm_rate_limit=value)
