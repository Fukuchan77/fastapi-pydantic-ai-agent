"""Cross-backend proof that session-history capacity degrades by trimming.

Task 4.7 owns the full parametrized proof (in-memory + Redis, tool-call
pairing intact, session usable on the following turn — Req 3.5, 3.6, 3.7).
The minimal Redis-only test below was created ahead of that task, per the
4.1/4.2 precedent (`.sdd/specs/002-review-roadmap-remediation/tasks.md`), to
drive task 4.4's Redis-side implementation: proof that
`RedisSessionStore.save_history()` trims to its configured cap instead of
storing the full oversized history. It is widened here, not replaced.

The parametrized tests below use a small stateful fake for the Redis
client's `get`/`set` surface (a plain dict behind `AsyncMock(side_effect=...)`)
rather than pre-computing expected serialized bytes by hand, because Req 3.5
requires proving an actual save-then-read round trip across *two* successive
turns — call-argument inspection alone (the 4.4-era test's approach) cannot
show that a session stays readable and writable after trimming. This is
still the existing `mock_redis`/`patch("redis.asyncio.from_url", ...)`
fixture pattern (`tests/unit/stores/test_redis_session_store.py`): no real
Redis daemon, no new dependency such as `fakeredis`.
"""

from collections.abc import Sequence
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai.messages import TextPart
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.messages import UserPromptPart

from app.stores.session_store import InMemorySessionStore
from app.stores.session_store import RedisSessionStore
from app.stores.session_store import SessionStore
from app.stores.session_store._trim import trim_history


CAP = 5


@pytest.mark.asyncio
async def test_redis_store_save_history_trims_to_max_messages() -> None:
    """RedisSessionStore trims to max_messages on save instead of raising (Req 3.3, 3.6)."""
    mock_redis = AsyncMock()
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        store = RedisSessionStore(redis_url="redis://localhost:6379/0", max_messages=5)

    messages = [ModelRequest(parts=[UserPromptPart(content=f"Msg {i}")]) for i in range(6)]

    await store.save_history("session-1", messages)

    mock_redis.set.assert_called_once()
    serialized = mock_redis.set.call_args[0][1]
    saved = ModelMessagesTypeAdapter.validate_json(serialized)
    assert len(saved) == 5


def _make_stateful_mock_redis() -> AsyncMock:
    """A mock Redis client whose `get()` returns whatever `set()` last stored.

    Backed by a plain dict, not a real connection — same `AsyncMock` double
    as `test_redis_session_store.py`'s `mock_redis` fixture, just with
    `side_effect` callables instead of fixed `return_value`s so a save is
    actually observable on the next read, across as many turns as the test
    needs.
    """
    backing: dict[str, bytes] = {}
    mock = AsyncMock()

    async def _set(key: str, value: bytes, ex: int | None = None) -> None:
        backing[key] = value

    async def _get(key: str) -> bytes | None:
        return backing.get(key)

    async def _delete(key: str) -> None:
        backing.pop(key, None)

    mock.set = AsyncMock(side_effect=_set)
    mock.get = AsyncMock(side_effect=_get)
    mock.delete = AsyncMock(side_effect=_delete)
    return mock


@pytest.fixture(params=["in_memory", "redis"])
def capacity_session_store(request: pytest.FixtureRequest) -> SessionStore:
    """A `SessionStore` of the parametrized backend, capped at `CAP` messages."""
    if request.param == "in_memory":
        return InMemorySessionStore(max_messages=CAP)
    mock_redis = _make_stateful_mock_redis()
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        return RedisSessionStore(redis_url="redis://localhost:6379/0", max_messages=CAP)


def _build_history_with_pairing(n_turns: int) -> list[ModelMessage]:
    """Build a head-pinned history of `n_turns` user/tool-call/tool-return/reply cycles."""
    messages: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart(content="sys")])]
    for i in range(n_turns):
        call = ToolCallPart(tool_name="lookup", args={}, tool_call_id=f"c{i}")
        ret = ToolReturnPart(tool_name="lookup", content=f"r{i}", tool_call_id=f"c{i}")
        messages.extend(
            [
                ModelRequest(parts=[UserPromptPart(content=f"q{i}")]),
                ModelResponse(parts=[call]),
                ModelRequest(parts=[ret]),
                ModelResponse(parts=[TextPart(content=f"a{i}")]),
            ]
        )
    return messages


def _assert_pairing_intact(messages: Sequence[ModelMessage]) -> None:
    """Assert no retained `ToolReturnPart` lacks its originating `ToolCallPart`."""
    openers = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    closers = {
        part.tool_call_id
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert closers <= openers


@pytest.mark.asyncio
async def test_save_past_cap_persists_within_head_pin_bound_and_keeps_pairing(
    capacity_session_store: SessionStore,
) -> None:
    """A session driven past the cap persists what the shared trimmer computes.

    Req 3.1, 3.2, 3.6, 3.7.
    """
    session_id = "cap-session"
    history = _build_history_with_pairing(n_turns=10)  # far exceeds CAP

    await capacity_session_store.save_history(session_id, history)
    saved = await capacity_session_store.get_history(session_id)

    # Identical to what the shared trimmer would compute directly — the
    # store adds no backend-specific behaviour on top of it (Req 3.6).
    assert saved == trim_history(history, CAP)
    # 3.1/3.7 slack: the pinned head can push the result one message past
    # the cap (spec.md Recorded Deviations, "3.1 / 3.7 allow one message of
    # slack") — never more.
    assert len(saved) <= CAP + 1
    assert saved[0] == history[0]
    _assert_pairing_intact(saved)


@pytest.mark.asyncio
async def test_session_stays_readable_and_writable_after_hitting_cap(
    capacity_session_store: SessionStore,
) -> None:
    """A session at its cap completes, and stays usable, on the following turn.

    Req 3.3, 3.4, 3.5.
    """
    session_id = "cap-session-next-turn"
    history = _build_history_with_pairing(n_turns=10)  # far exceeds CAP

    await capacity_session_store.save_history(session_id, history)  # must not raise
    saved = await capacity_session_store.get_history(session_id)

    next_turn = [
        *saved,
        ModelRequest(parts=[UserPromptPart(content="one more question")]),
        ModelResponse(parts=[TextPart(content="one more answer")]),
    ]
    await capacity_session_store.save_history(session_id, next_turn)  # must not raise again
    after = await capacity_session_store.get_history(session_id)

    # The second turn's save is a fresh, ordinary trim of `next_turn` — not
    # a special case — so it is held to the same equivalence as the first
    # turn's, proving the session is genuinely writable again, not merely
    # non-raising.
    assert after == trim_history(next_turn, CAP)
    assert len(after) <= CAP + 1
    assert after[0] == history[0]
    _assert_pairing_intact(after)
    # The turn just appended survived the second trim (it is always more
    # recent than the ideal cut point) — the session is genuinely usable
    # afterwards, not merely non-raising.
    assert isinstance(after[-1], ModelResponse)
    assert after[-1].parts == [TextPart(content="one more answer")]


@pytest.mark.asyncio
async def test_trimmed_content_is_identical_across_backends() -> None:
    """The same over-cap history trims to identical content on both backends (Req 3.6)."""
    history = _build_history_with_pairing(n_turns=10)

    in_memory_store = InMemorySessionStore(max_messages=CAP)
    await in_memory_store.save_history("s", history)
    in_memory_result = await in_memory_store.get_history("s")

    mock_redis = _make_stateful_mock_redis()
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        redis_store = RedisSessionStore(redis_url="redis://localhost:6379/0", max_messages=CAP)
    await redis_store.save_history("s", history)
    redis_result = await redis_store.get_history("s")

    assert in_memory_result == redis_result
