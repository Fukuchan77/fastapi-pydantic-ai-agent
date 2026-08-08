"""Tests for the closed `app_env` vocabulary (Req 1.1-1.5).

Every production guard in this repository compares `app_env` with equality
(`app/config/security.py` mock-tool and wildcard-host guards,
`app/agents/chat_agent.py`, `app/logging_config.py`), so a misspelled value such
as `prod` or `Production` would silently disable them. Closing the vocabulary at
settings-construction time makes those guards sound without touching any of
them.
"""

import pytest
from pydantic import ValidationError

from app.config import Settings
from tests.conftest import build_test_settings


PERMITTED_VALUES = ["development", "staging", "production"]

# `staging` and `production` both reject the default wildcard host allow-list, so
# name a concrete host to keep these tests focused on the environment vocabulary.
NON_WILDCARD_HOSTS = ["api.example.com"]

REJECTED_VALUES = [
    "Production",  # case variant (Req 1.2)
    "prod",  # abbreviation (Req 1.2)
    "PRODUCTION",
    "Development",
    "dev",
    "test",
    "",
]

# Values a normalizing implementation would coerce into "production" (Req 1.4).
NORMALIZATION_BAIT = ["Production", "PRODUCTION", " production", "production "]


@pytest.mark.parametrize("value", PERMITTED_VALUES)
def test_permitted_app_env_values_are_accepted(value: str) -> None:
    """Each of the three permitted environment values constructs successfully.

    Req 1.1.

    Args:
        value: A permitted `app_env` value.
    """
    settings = build_test_settings(app_env=value, allowed_hosts=NON_WILDCARD_HOSTS)

    assert settings.app_env == value


@pytest.mark.parametrize("value", REJECTED_VALUES)
def test_non_permitted_app_env_values_fail_construction(value: str) -> None:
    """Any value outside the vocabulary fails settings construction.

    Covers the case variant (`Production`) and the abbreviation (`prod`) that
    Req 1.2/1.5 name explicitly, plus neighbouring spellings.

    Args:
        value: A non-permitted `app_env` value.
    """
    with pytest.raises(ValidationError):
        build_test_settings(app_env=value, allowed_hosts=NON_WILDCARD_HOSTS)


@pytest.mark.parametrize("value", ["Production", "prod"])
def test_rejection_error_names_the_field_and_lists_permitted_values(value: str) -> None:
    """The validation error names `app_env` and enumerates the permitted values.

    Req 1.2, 1.5. The operator has to be able to fix the misspelling from the
    message alone, without reading `app/config/`.

    Args:
        value: A non-permitted `app_env` value.
    """
    with pytest.raises(ValidationError) as exc_info:
        build_test_settings(app_env=value, allowed_hosts=NON_WILDCARD_HOSTS)

    error_str = str(exc_info.value)
    assert "app_env" in error_str
    for permitted in PERMITTED_VALUES:
        assert repr(permitted) in error_str, f"{permitted!r} missing from {error_str!r}"


@pytest.mark.parametrize("value", NORMALIZATION_BAIT)
def test_app_env_is_not_normalized_into_a_permitted_value(value: str) -> None:
    """No lowercasing, stripping, or aliasing repairs a misspelled value.

    Req 1.4 — a misspelling is an operator error that stops startup rather than
    something the service quietly fixes.

    Args:
        value: A value that a normalizing implementation would coerce to
            `"production"`.
    """
    with pytest.raises(ValidationError):
        build_test_settings(app_env=value, allowed_hosts=NON_WILDCARD_HOSTS)


def test_app_env_rejected_from_the_environment_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vocabulary is closed for `APP_ENV` env-var input, not only kwargs.

    Req 1.1, 1.2. `test_env` already supplies the remaining required variables.
    `ALLOWED_HOSTS` is named explicitly so the failure comes from the vocabulary
    rather than from the wildcard-host guard, which also rejects a non-development
    environment and would otherwise make this test a false green.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    error_str = str(exc_info.value)
    assert "app_env" in error_str
    for permitted in PERMITTED_VALUES:
        assert repr(permitted) in error_str, f"{permitted!r} missing from {error_str!r}"


def test_invalid_app_env_aborts_before_any_store_model_or_agent_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid environment value stops `create_app` before resource construction.

    Req 1.3. `create_app` resolves `Settings` before the lifespan runs, so the
    store, model, and agent factories are never reached. `ALLOWED_HOSTS` is named
    explicitly so the abort is attributable to the vocabulary rather than to the
    wildcard-host guard.

    The sentinels are installed on `app.main`, which imports each factory by name
    (`from app.stores.factory import build_session_store`), so patching the
    factory module itself would intercept nothing. If major task 10 moves the
    lifespan into `app/lifespan.py`, these three targets move with the call site.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import app.main as main_module

    def fail(*args: object, **kwargs: object) -> object:
        """Fail the test if a resource factory is reached.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            AssertionError: Always.
        """
        raise AssertionError("resource factory reached despite invalid app_env")

    monkeypatch.setattr(main_module, "build_session_store", fail)
    monkeypatch.setattr(main_module, "build_vector_store", fail)
    monkeypatch.setattr(main_module, "build_chat_agent", fail)
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("ALLOWED_HOSTS", "api.example.com")

    with pytest.raises(ValidationError) as exc_info:
        main_module.create_app()

    assert "app_env" in str(exc_info.value)
    for permitted in PERMITTED_VALUES:
        assert repr(permitted) in str(exc_info.value)
