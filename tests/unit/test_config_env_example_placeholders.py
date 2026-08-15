"""Guard that every secret placeholder shipped in `.env.example` fails validation.

`SESSION_SIGNING_KEY=your-session-signing-key-here` used to validate
successfully: `validate_session_signing_key_strength` reused `api_key`'s
enumerated placeholder set, which does not contain that string, and at 30
characters the value cleared the 16-character floor as well. A deployment that
copied `.env.example` and fixed only `API_KEY` (whose placeholder *is*
enumerated, so it fails loudly) would therefore run on a signing key published
in this repository - and `session_signing_key` is the only secret binding a
session id to its principal, so a known value makes every session id forgeable
and defeats Req 11.1/11.2 entirely.

This module tests the class of defect, not just that one string: it reads the
committed `.env.example` and asserts each secret value there is rejected.
"""

import re
from pathlib import Path

import pytest
from pydantic import SecretStr
from pydantic import ValidationError

from app.config import Settings
from app.config._secret_placeholders import is_placeholder


_ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# Settings fields backed by a SecretStr whose placeholder must never validate.
_SECRET_ENV_VARS = ("API_KEY", "SESSION_SIGNING_KEY", "LLM_API_KEY", "LOGFIRE_TOKEN")


def _env_example_values() -> dict[str, str]:
    """Parse `KEY=value` assignments out of the committed `.env.example`."""
    values: dict[str, str] = {}
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line.strip())
        if match:
            values[match.group(1)] = match.group(2)
    return values


def _base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set every required setting to a valid value."""
    monkeypatch.setenv("API_KEY", "test-api-key-12345")
    monkeypatch.setenv("SESSION_SIGNING_KEY", "test-signing-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "openai:gpt-4o")
    monkeypatch.setenv("LLM_API_KEY", "sk-test123456789")


class TestEnvExamplePlaceholdersAreRejected:
    """Every secret placeholder in `.env.example` must fail startup validation."""

    @pytest.mark.parametrize("var", _SECRET_ENV_VARS)
    def test_placeholder_is_rejected(self, monkeypatch: pytest.MonkeyPatch, var: str) -> None:
        """Copying `.env.example` unedited must not produce a usable secret."""
        example = _env_example_values()
        assert var in example, f"{var} is missing from .env.example"

        value = example[var]
        if not value:
            pytest.skip(f"{var} is empty in .env.example (optional setting)")

        _base_env(monkeypatch)
        monkeypatch.setenv(var, value)

        with pytest.raises(ValidationError, match=r"(?i)placeholder|at least 16 characters"):
            Settings()

    def test_session_signing_key_placeholder_shape_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The specific regression: a long `...-here` signing key must not pass."""
        _base_env(monkeypatch)
        monkeypatch.setenv("SESSION_SIGNING_KEY", "your-session-signing-key-here")

        with pytest.raises(ValidationError, match=r"(?i)placeholder"):
            Settings()

    def test_strong_signing_key_still_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real generated key is unaffected by the shape rule."""
        _base_env(monkeypatch)
        generated = "9f2c1d7e4a8b60335f1e2d9c8b7a6543210fedcba98765432100abcdef123456"
        monkeypatch.setenv("SESSION_SIGNING_KEY", generated)

        settings = Settings()
        assert settings.session_signing_key == SecretStr(generated)


class TestPlaceholderShapeRule:
    """Unit tests for the shared shape rule itself."""

    @pytest.mark.parametrize(
        "value",
        [
            "your-session-signing-key-here",
            "your-api-key-here",
            "your-token-here",
            "insert-key-here",
            "api-key-here",
            "YOUR-SESSION-SIGNING-KEY-HERE",
            "changeme",
            "replace-me",
        ],
    )
    def test_placeholders_detected(self, value: str) -> None:
        """Enumerated strings and the `...-here` shape are both caught."""
        assert is_placeholder(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "9f2c1d7e4a8b60335f1e2d9c8b7a6543",
            "sk-prod-a1b2c3d4e5f6g7h8",
            "test-api-key-12345",
            "hereafter-is-not-a-placeholder",
        ],
    )
    def test_real_secrets_not_flagged(self, value: str) -> None:
        """Ordinary secret values are not misclassified."""
        assert is_placeholder(value) is False
