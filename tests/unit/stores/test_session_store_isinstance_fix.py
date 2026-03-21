"""Unit tests for Task 3.14: Fix _validate_messages() isinstance check."""

import pytest

from app.stores.session_store import InMemorySessionStore


class TestValidateMessagesInstanceOfFix:
    """Test suite for Task 3.14: Fix _validate_messages() to use strict isinstance check."""

    @pytest.fixture
    def store(self) -> InMemorySessionStore:
        """Create an InMemorySessionStore instance for testing."""
        return InMemorySessionStore()

    @pytest.mark.asyncio
    async def test_duck_typed_object_with_parts_attribute_is_rejected(
        self, store: InMemorySessionStore
    ) -> None:
        """Objects with a 'parts' attribute but not ModelMessage instances should be rejected.

        This test verifies that the validation uses isinstance() instead of hasattr(),
        preventing duck-typed objects from bypassing validation.
        """

        # Create a duck-typed object that has a 'parts' attribute but is not a ModelMessage
        class FakeMessage:
            def __init__(self):
                self.parts = ["fake", "parts"]

        fake_message = FakeMessage()
        session_id = "test-session"

        # save_history should reject the fake message with TypeError
        with pytest.raises(TypeError, match="All messages must be ModelMessage instances"):
            await store.save_history(session_id, [fake_message])

    @pytest.mark.asyncio
    async def test_object_without_parts_attribute_is_rejected(
        self, store: InMemorySessionStore
    ) -> None:
        """Objects without a 'parts' attribute should be rejected.

        This test verifies that non-ModelMessage objects are rejected regardless
        of whether they have a 'parts' attribute.
        """

        # Create an object without a 'parts' attribute
        class NotAMessage:
            pass

        not_a_message = NotAMessage()
        session_id = "test-session"

        # save_history should reject with TypeError
        with pytest.raises(TypeError, match="All messages must be ModelMessage instances"):
            await store.save_history(session_id, [not_a_message])

    @pytest.mark.asyncio
    async def test_string_with_parts_attribute_is_rejected(
        self, store: InMemorySessionStore
    ) -> None:
        """String objects should be rejected even though strings have attributes.

        This test ensures that only actual ModelMessage instances are accepted.
        """
        session_id = "test-session"

        # save_history should reject string with TypeError
        with pytest.raises(TypeError, match="All messages must be ModelMessage instances"):
            await store.save_history(session_id, ["not a message"])

    @pytest.mark.asyncio
    async def test_dict_with_parts_key_is_rejected(self, store: InMemorySessionStore) -> None:
        """Dictionary objects with 'parts' key should be rejected.

        This test verifies that dict-like objects don't pass validation.
        """
        fake_message_dict = {"parts": ["fake", "parts"]}
        session_id = "test-session"

        # save_history should reject dict with TypeError
        with pytest.raises(TypeError, match="All messages must be ModelMessage instances"):
            await store.save_history(session_id, [fake_message_dict])
