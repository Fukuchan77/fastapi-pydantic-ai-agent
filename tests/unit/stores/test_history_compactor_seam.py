"""Tests for the Stage 1 `history_compactor` seam (X-7, `docs/context-budget.md`).

Two things must hold simultaneously:
1. **Stage 0 is unchanged.** With no compactor configured (the default), a
   save is byte-identical to what `trim_history()` alone would produce —
   proven directly against `tests/unit/stores/test_session_trim_across_backends.py`,
   which is left untouched by this PR precisely so it stays the Stage-0 proof.
2. **The compactor runs before trim, and trim always has final say.** A
   compactor that itself breaks `trim_history()`'s invariants (orphans a
   tool-call pair, over-produces past the cap) must still yield a valid,
   capped, pairing-safe result — because `trim_history()` runs on the
   compactor's *output*, not the compactor's absence.
"""

from collections.abc import Sequence
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from pydantic_ai.messages import ModelMessage
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


def _make_stateful_mock_redis() -> AsyncMock:
    """A mock Redis client whose `get()` returns whatever `set()` last stored.

    Same pattern as `test_session_trim_across_backends.py`'s helper of the
    same name (not imported from there — that module is deliberately left
    untouched by this PR as the Stage-0 proof, so this is a local copy).
    """
    backing: dict[str, bytes] = {}
    mock = AsyncMock()

    async def _set(key: str, value: bytes, ex: int | None = None) -> None:
        backing[key] = value

    async def _get(key: str) -> bytes | None:
        return backing.get(key)

    mock.set = AsyncMock(side_effect=_set)
    mock.get = AsyncMock(side_effect=_get)
    return mock


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


@pytest.fixture(params=["in_memory", "redis"])
def uncompacted_store(request: pytest.FixtureRequest) -> SessionStore:
    """A `SessionStore` with no `history_compactor` set — Stage 0 behavior."""
    if request.param == "in_memory":
        return InMemorySessionStore(max_messages=CAP)
    mock_redis = _make_stateful_mock_redis()
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        return RedisSessionStore(redis_url="redis://localhost:6379/0", max_messages=CAP)


class TestNoCompactorIsByteIdenticalToStage0:
    """`history_compactor=None` (the default) changes nothing about existing behavior."""

    @pytest.mark.asyncio
    async def test_saved_history_matches_trim_history_exactly(
        self, uncompacted_store: SessionStore
    ) -> None:
        """With no compactor, a save equals `trim_history()` alone — no seam involvement."""
        history = _build_history_with_pairing(n_turns=10)  # far exceeds CAP

        await uncompacted_store.save_history("s", history)
        saved = await uncompacted_store.get_history("s")

        assert saved == trim_history(history, CAP)

    @pytest.mark.asyncio
    async def test_under_cap_history_is_also_unaffected(
        self, uncompacted_store: SessionStore
    ) -> None:
        """A history already under the cap round-trips unchanged, same as before this seam."""
        history = _build_history_with_pairing(n_turns=1)

        await uncompacted_store.save_history("s", history)
        saved = await uncompacted_store.get_history("s")

        assert saved == history


class TestCompactorRunsBeforeTrim:
    """A configured compactor's output — not the raw input — is what gets trimmed."""

    @pytest.mark.asyncio
    async def test_in_memory_store_calls_compactor_with_the_saved_messages(self) -> None:
        """The compactor receives exactly what `save_history()` was called with."""
        received: list[Sequence[ModelMessage]] = []

        def spy_compactor(messages: Sequence[ModelMessage]) -> Sequence[ModelMessage]:
            received.append(messages)
            return messages

        store = InMemorySessionStore(max_messages=CAP, history_compactor=spy_compactor)
        history = _build_history_with_pairing(n_turns=2)

        await store.save_history("s", history)

        assert len(received) == 1
        assert list(received[0]) == history

    @pytest.mark.asyncio
    async def test_redis_store_calls_compactor_with_the_saved_messages(self) -> None:
        """The compactor receives exactly what `save_history()` was called with (Redis)."""
        received: list[Sequence[ModelMessage]] = []

        def spy_compactor(messages: Sequence[ModelMessage]) -> Sequence[ModelMessage]:
            received.append(messages)
            return messages

        mock_redis = _make_stateful_mock_redis()
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            store = RedisSessionStore(
                redis_url="redis://localhost:6379/0",
                max_messages=CAP,
                history_compactor=spy_compactor,
            )
        history = _build_history_with_pairing(n_turns=2)

        await store.save_history("s", history)

        assert len(received) == 1
        assert list(received[0]) == history

    @pytest.mark.asyncio
    async def test_trim_still_enforces_the_cap_when_compactor_over_produces(self) -> None:
        """A compactor that returns MORE than the cap is still trimmed down to it.

        Proves `trim_history()` runs on the compactor's output, not on the
        original input — a compactor is not a way to bypass the cap.
        """

        def noop_compactor(messages: Sequence[ModelMessage]) -> Sequence[ModelMessage]:
            return messages  # returns the full, still-over-cap history unchanged

        store = InMemorySessionStore(max_messages=CAP, history_compactor=noop_compactor)
        history = _build_history_with_pairing(n_turns=10)  # far exceeds CAP

        await store.save_history("s", history)
        saved = await store.get_history("s")

        assert len(saved) <= CAP + 1
        _assert_pairing_intact(saved)

    @pytest.mark.asyncio
    async def test_final_result_reflects_compacted_content_not_the_original(self) -> None:
        """`trim_history()` runs on the compactor's output, so the save reflects it.

        A compactor that drops all but the last 2 turns produces a
        different final save than trimming the original 10-turn history
        would — proving composition order (compact, then trim) rather than
        the compactor being a no-op wrapper around the original input.
        """
        last_n_turns = 2

        def keep_last_n_turns(messages: Sequence[ModelMessage]) -> Sequence[ModelMessage]:
            head, *rest = messages
            per_turn = 4  # user / tool-call / tool-return / reply
            return [head, *rest[-last_n_turns * per_turn :]]

        store = InMemorySessionStore(max_messages=100, history_compactor=keep_last_n_turns)
        history = _build_history_with_pairing(n_turns=10)  # well under CAP=100 uncompacted

        await store.save_history("s", history)
        saved = await store.get_history("s")

        # Trimming the compactor's (short) output at cap=100 is a no-op, so
        # the save equals the compacted content exactly - proving the save
        # reflects what the compactor produced, not the original 41-message
        # history (which trim_history(history, 100) would have passed
        # through unchanged had the compactor not run first).
        assert saved == keep_last_n_turns(history)
        assert len(saved) < len(history)
        _assert_pairing_intact(saved)


class TestCompactorOutputIsRevalidated:
    """A compactor returning something malformed fails loudly, not silently."""

    @pytest.mark.asyncio
    async def test_non_model_message_output_raises_type_error(self) -> None:
        """`InMemorySessionStore` re-validates the compactor's output, same as raw input."""

        def broken_compactor(messages: Sequence[ModelMessage]) -> Sequence[ModelMessage]:
            return cast("Sequence[ModelMessage]", ["not a ModelMessage"])

        store = InMemorySessionStore(max_messages=CAP, history_compactor=broken_compactor)
        history = _build_history_with_pairing(n_turns=1)

        with pytest.raises(TypeError):
            await store.save_history("s", history)
