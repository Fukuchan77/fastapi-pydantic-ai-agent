"""Unit tests for Principal derivation (Req 11.1)."""

from app.security.principal import Principal
from app.security.principal import derive_principal_id


def test_derive_principal_id_is_deterministic() -> None:
    """The same API key must always derive the same principal id."""
    assert derive_principal_id("some-api-key") == derive_principal_id("some-api-key")


def test_derive_principal_id_differs_for_different_keys() -> None:
    """Different API keys must derive different principal ids."""
    assert derive_principal_id("key-one") != derive_principal_id("key-two")


def test_derive_principal_id_does_not_contain_raw_key() -> None:
    """The derived id must not leak the raw API key value."""
    api_key = "super-secret-api-key-value"
    assert api_key not in derive_principal_id(api_key)


def test_derive_principal_id_is_hex_and_fixed_length() -> None:
    """The derived id is a fixed-length hex digest (safe for session_id embedding)."""
    principal_id = derive_principal_id("any-api-key")
    assert len(principal_id) == 16
    assert all(c in "0123456789abcdef" for c in principal_id)


def test_principal_model_holds_id() -> None:
    """Principal is a simple id holder."""
    principal = Principal(id=derive_principal_id("some-api-key"))
    assert principal.id == derive_principal_id("some-api-key")
