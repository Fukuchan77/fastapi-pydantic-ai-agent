"""Unit tests for server-issued, principal-bound session ids (Req 11.1/11.2)."""

import pytest
from fastapi import HTTPException

from app.security.principal import Principal
from app.services.session_service import authorize_session
from app.services.session_service import start_session
from tests.conftest import build_test_settings


@pytest.fixture
def settings():
    """Provide test settings with a fixed session_signing_key."""
    return build_test_settings()


@pytest.fixture
def alice() -> Principal:
    """A principal."""
    return Principal(id="alice0000000000")


@pytest.fixture
def bob() -> Principal:
    """A different principal."""
    return Principal(id="bob0000000000000")


@pytest.mark.asyncio
async def test_start_session_mints_three_dot_separated_parts(alice, settings) -> None:
    """A minted session_id has the {principal.id}.{token}.{signature} shape."""
    session_id = await start_session(alice, settings)
    parts = session_id.split(".")
    assert len(parts) == 3
    assert parts[0] == alice.id


@pytest.mark.asyncio
async def test_start_session_mints_unique_ids(alice, settings) -> None:
    """Repeated calls mint different session ids (random token per call)."""
    first = await start_session(alice, settings)
    second = await start_session(alice, settings)
    assert first != second


@pytest.mark.asyncio
async def test_authorize_session_accepts_own_minted_session(alice, settings) -> None:
    """The principal that minted a session_id can be authorized for it."""
    session_id = await start_session(alice, settings)
    await authorize_session(alice, session_id, settings)  # must not raise


@pytest.mark.asyncio
async def test_authorize_session_rejects_cross_principal_access(alice, bob, settings) -> None:
    """A session_id minted for one principal is rejected (403) for another (IDOR, Req 11.2)."""
    session_id = await start_session(alice, settings)

    with pytest.raises(HTTPException) as exc_info:
        await authorize_session(bob, session_id, settings)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_authorize_session_rejects_malformed_id(alice, settings) -> None:
    """A session_id that isn't the signed 3-part shape is rejected (403)."""
    with pytest.raises(HTTPException) as exc_info:
        await authorize_session(alice, "not-a-signed-session-id", settings)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_authorize_session_rejects_tampered_signature(alice, settings) -> None:
    """A session_id with a valid shape but forged signature is rejected (403)."""
    session_id = await start_session(alice, settings)
    principal_id, token, _signature = session_id.split(".")
    forged = f"{principal_id}.{token}.deadbeefdeadbeef"

    with pytest.raises(HTTPException) as exc_info:
        await authorize_session(alice, forged, settings)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_authorize_session_rejects_forged_principal_segment(alice, bob, settings) -> None:
    """Swapping in another principal's id segment (without re-signing) is rejected (403)."""
    session_id = await start_session(alice, settings)
    _principal_id, token, signature = session_id.split(".")
    forged = f"{bob.id}.{token}.{signature}"

    with pytest.raises(HTTPException) as exc_info:
        await authorize_session(bob, forged, settings)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_authorize_session_rejects_non_ascii_id_with_403_not_500(alice, settings) -> None:
    """A non-ASCII session_id is rejected with 403, not an unhandled 500.

    `secrets.compare_digest()` raises `TypeError` on a non-ASCII `str` -
    without an explicit ASCII check first, a malformed non-ASCII session_id
    would hit that TypeError before either `compare_digest()` call below,
    turning into an unhandled 500 instead of the 403 this function's
    contract promises for any malformed id.
    """
    with pytest.raises(HTTPException) as exc_info:
        await authorize_session(alice, "a.b.ééé", settings)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_authorize_session_rejects_wrong_signing_key(alice, settings) -> None:
    """A session_id signed with a different signing key is rejected (403)."""
    session_id = await start_session(alice, settings)
    other_settings = build_test_settings(
        session_signing_key="a-completely-different-signing-key-value",
    )

    with pytest.raises(HTTPException) as exc_info:
        await authorize_session(alice, session_id, other_settings)

    assert exc_info.value.status_code == 403
