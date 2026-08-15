"""Integration tests: `RedisSessionStore` against a real Redis server (Req 6.1-6.13).

`RedisSessionStore`'s redis-py surface is only five calls (`from_url`, `get`,
`set(ex=)`, `delete`, `aclose`), so this module is the one place they run
against a real server rather than a mock - the whole point of this lane
(Req 6.4). Gated behind a live reachability probe rather than an opt-in
env var, since - unlike Chroma's one-time model download - Redis's
dependency is a reachable service (Req 6.2, 6.3). Placed in `tests/integration/`,
the same tier directory the existing opt-in Chroma lane uses, so one gating
convention covers both (Req 6.11). No in-process Redis substitute is used
anywhere in this module (Req 6.8).
"""

import asyncio
import uuid

import pytest
import redis.asyncio as redis
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import UserPromptPart

from app.stores.factory import dry_run_stores
from app.stores.session_store import RedisSessionStore
from tests.support.redis import REDIS_UNREACHABLE_SKIP_REASON
from tests.support.redis import REDIS_URL
from tests.support.redis import redis_reachable


pytestmark = pytest.mark.redis

_PRE_CUTOVER_PREFIX = "session:"
"""The key prefix in force before the Requirement 7 cutover (task 7.2).

Named as a literal rather than read off `RedisSessionStore.DEFAULT_KEY_PREFIX`,
so the isolation tests below keep asserting the same fixed pre-cutover value
once the class default moves to the v2 prefix - only then do the two
directions actually observe the separation they assert.
"""


@pytest.fixture(autouse=True)
async def _require_redis() -> None:
    """Skip every test in this module unless a real Redis server answers PING (Req 6.3)."""
    if not await redis_reachable():
        pytest.skip(REDIS_UNREACHABLE_SKIP_REASON)


def _unique_key_prefix() -> str:
    """Build a key prefix unique to this test invocation, avoiding cross-test collisions."""
    return f"test-redis-live-{uuid.uuid4().hex}:"


@pytest.mark.asyncio
async def test_save_and_retrieve_round_trip() -> None:
    """save_history persists messages that get_history retrieves unchanged."""
    store = RedisSessionStore(redis_url=REDIS_URL, key_prefix=_unique_key_prefix())
    session_id = store.generate_session_id()
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(parts=[TextPart(content="Hi there!")]),
    ]
    try:
        await store.save_history(session_id, messages)

        assert await store.get_history(session_id) == messages
    finally:
        await store.clear(session_id)
        await store.close()


@pytest.mark.asyncio
async def test_ttl_expiry_clears_history() -> None:
    """A session saved with a short TTL is gone from Redis once the TTL elapses."""
    store = RedisSessionStore(redis_url=REDIS_URL, session_ttl=1, key_prefix=_unique_key_prefix())
    session_id = store.generate_session_id()
    messages = [ModelRequest(parts=[UserPromptPart(content="expiring")])]
    try:
        await store.save_history(session_id, messages)
        assert await store.get_history(session_id) == messages

        await asyncio.sleep(1.5)

        assert await store.get_history(session_id) == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_history_trimming_on_save() -> None:
    """save_history trims to max_messages at a tool-pairing-safe boundary before persisting.

    With no tool calls in play, `trim_history` always keeps `messages[0]`
    plus the trailing `max_messages - 1` messages (`app/stores/session_store/_trim.py`).
    """
    store = RedisSessionStore(redis_url=REDIS_URL, max_messages=3, key_prefix=_unique_key_prefix())
    session_id = store.generate_session_id()
    messages = [
        ModelRequest(parts=[UserPromptPart(content="u1")]),
        ModelResponse(parts=[TextPart(content="r1")]),
        ModelRequest(parts=[UserPromptPart(content="u2")]),
        ModelResponse(parts=[TextPart(content="r2")]),
        ModelRequest(parts=[UserPromptPart(content="u3")]),
    ]
    try:
        await store.save_history(session_id, messages)
        retrieved = await store.get_history(session_id)

        assert retrieved == [messages[0], messages[3], messages[4]]
    finally:
        await store.clear(session_id)
        await store.close()


@pytest.mark.asyncio
async def test_close_is_safe_to_call_once_after_use() -> None:
    """close() completes without error after the store has been used."""
    store = RedisSessionStore(redis_url=REDIS_URL, key_prefix=_unique_key_prefix())
    session_id = store.generate_session_id()
    await store.save_history(session_id, [ModelRequest(parts=[UserPromptPart(content="x")])])

    await store.clear(session_id)
    await store.close()


@pytest.mark.asyncio
async def test_startup_dry_run_probe_succeeds_against_a_reachable_server() -> None:
    """The factory's startup connectivity probe succeeds against a real server.

    Neither `RedisSessionStore` nor the `SessionStore` Protocol exposes a
    `ping` operation, so the probe this lane verifies is the harmless
    `get_history` round-trip `dry_run_stores` already performs at startup
    (Req 6.13's rationale) - no separate probe method is added to the store.
    """
    store = RedisSessionStore(redis_url=REDIS_URL, key_prefix=_unique_key_prefix())
    try:
        await dry_run_stores(store)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pre_cutover_prefix_reader_does_not_observe_v2_written_history() -> None:
    """A pre-cutover-prefix reader must not see history a v2 instance wrote (Req 7.2, 7.3).

    Red ahead of task 7.2: `RedisSessionStore.DEFAULT_KEY_PREFIX` still equals
    `_PRE_CUTOVER_PREFIX` today, so the default-constructed "v2 instance" below
    and the explicit pre-cutover reader collide on the same Redis key and this
    assertion currently fails. It turns green once 7.2 moves the class default
    to a distinct value. Reachability is recorded by this module's autouse
    `_require_redis` fixture - a skipped run is not a demonstrated failure.
    """
    v2_store = RedisSessionStore(redis_url=REDIS_URL)
    pre_cutover_reader = RedisSessionStore(redis_url=REDIS_URL, key_prefix=_PRE_CUTOVER_PREFIX)
    session_id = v2_store.generate_session_id()
    messages = [ModelRequest(parts=[UserPromptPart(content="v2 history")])]
    try:
        await v2_store.save_history(session_id, messages)

        assert await pre_cutover_reader.get_history(session_id) == []
    finally:
        await v2_store.clear(session_id)
        await v2_store.close()
        await pre_cutover_reader.close()


@pytest.mark.asyncio
async def test_pre_cutover_key_is_not_observable_via_v2_store() -> None:
    """A key under the pre-cutover prefix must not surface via the v2 store (Req 7.2, 7.3).

    Red ahead of task 7.2 for the same reason as the sibling test above:
    `RedisSessionStore.DEFAULT_KEY_PREFIX` has not yet moved off
    `_PRE_CUTOVER_PREFIX`, so a key written directly under it - bypassing the
    store entirely, standing in for pre-existing v1 data - is currently visible
    through a default-constructed (v2) store too.
    """
    v2_store = RedisSessionStore(redis_url=REDIS_URL)
    session_id = v2_store.generate_session_id()
    messages = [ModelRequest(parts=[UserPromptPart(content="pre-cutover history")])]
    raw_client = redis.from_url(REDIS_URL, decode_responses=False)
    pre_cutover_key = f"{_PRE_CUTOVER_PREFIX}{session_id}"
    try:
        await raw_client.set(pre_cutover_key, ModelMessagesTypeAdapter.dump_json(messages), ex=60)

        assert await v2_store.get_history(session_id) == []
    finally:
        await raw_client.delete(pre_cutover_key)
        await raw_client.aclose()
        await v2_store.close()
