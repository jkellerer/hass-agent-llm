"""Unit tests for conversation history management."""

import pytest

from custom_components.home_agent.conversation import ConversationHistoryManager
from custom_components.home_agent.const import (
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MIN_MESSAGES,
)


class TestConversationHistoryManager:
    """Test ConversationHistoryManager class."""

    def test_initialization_defaults(self):
        """Test manager initializes with default values."""
        manager = ConversationHistoryManager()
        assert manager._max_messages == DEFAULT_HISTORY_MAX_MESSAGES
        assert manager._min_messages == DEFAULT_HISTORY_MIN_MESSAGES
        assert manager._max_tokens is None
        assert len(manager._histories) == 0

    def test_initialization_custom_limits(self):
        """Test manager initializes with custom limits."""
        manager = ConversationHistoryManager(max_messages=5, max_tokens=1000)
        assert manager._max_messages == 5
        assert manager._max_tokens == 1000

    def test_add_message(self):
        """Test adding messages to conversation history."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")

        assert "conv_123" in manager._histories
        assert len(manager._histories["conv_123"]) == 1
        assert manager._histories["conv_123"][0]["role"] == "user"
        assert manager._histories["conv_123"][0]["content"] == "Hello"
        assert "timestamp" in manager._histories["conv_123"][0]

    def test_add_multiple_messages(self):
        """Test adding multiple messages to same conversation."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi there!")
        manager.add_message("conv_123", "user", "How are you?")

        assert len(manager._histories["conv_123"]) == 3
        assert manager._histories["conv_123"][1]["role"] == "assistant"
        assert manager._histories["conv_123"][2]["content"] == "How are you?"

    def test_add_message_empty_conversation_id(self):
        """Test adding message with empty conversation_id is ignored."""
        manager = ConversationHistoryManager()
        manager.add_message("", "user", "Hello")

        assert len(manager._histories) == 0

    def test_add_message_empty_content(self):
        """Test adding message with empty content is ignored."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "")

        assert len(manager._histories) == 0

    def test_add_messages_to_different_conversations(self):
        """Test messages are stored separately per conversation."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello from 123")
        manager.add_message("conv_456", "user", "Hello from 456")

        assert len(manager._histories) == 2
        assert len(manager._histories["conv_123"]) == 1
        assert len(manager._histories["conv_456"]) == 1
        assert manager._histories["conv_123"][0]["content"] == "Hello from 123"
        assert manager._histories["conv_456"][0]["content"] == "Hello from 456"

    def test_get_history_empty(self):
        """Test getting history for non-existent conversation."""
        manager = ConversationHistoryManager()
        history = manager.get_history("conv_123")

        assert history == []

    def test_get_history_basic(self):
        """Test retrieving conversation history."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi!")

        history = manager.get_history("conv_123")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_get_history_with_message_limit(self):
        """Test history respects max_messages limit."""
        manager = ConversationHistoryManager(max_messages=2)

        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_123", role, f"Message {i}")

        history = manager.get_history("conv_123")

        # Should only get the last 2 messages
        assert len(history) == 2
        assert history[0]["content"] == "Message 3"
        assert history[1]["content"] == "Message 4"

    def test_get_history_override_message_limit(self):
        """Test overriding message limit in get_history."""
        manager = ConversationHistoryManager(max_messages=10)

        # Add 5 messages
        for i in range(5):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_123", role, f"Message {i}")

        # Override to get only last 2
        history = manager.get_history("conv_123", max_messages=2)

        assert len(history) == 2
        assert history[0]["content"] == "Message 3"
        assert history[1]["content"] == "Message 4"

    def test_get_history_with_token_limit(self):
        """Test history respects max_tokens limit."""
        manager = ConversationHistoryManager(max_tokens=100)

        # Add messages with varying lengths that fit within limit
        manager.add_message("conv_123", "user", "Short")
        manager.add_message("conv_123", "assistant", "Medium length message")
        manager.add_message("conv_123", "user", "Another short one")

        history = manager.get_history("conv_123")

        # Should keep all messages as they fit within limit
        estimated_tokens = manager.estimate_tokens(history)
        assert estimated_tokens <= 100
        assert len(history) == 3

    def test_get_history_override_token_limit(self):
        """Test overriding token limit in get_history."""
        manager = ConversationHistoryManager()

        # Add long messages
        for i in range(3):
            manager.add_message("conv_123", "user", "A" * 200)

        # Set a tight token limit
        history = manager.get_history("conv_123", max_tokens=100)

        estimated_tokens = manager.estimate_tokens(history)
        assert estimated_tokens <= 100

    def test_clear_history(self):
        """Test clearing specific conversation history."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_456", "user", "Hi")

        manager.clear_history("conv_123")

        assert "conv_123" not in manager._histories
        assert "conv_456" in manager._histories
        assert len(manager._histories["conv_456"]) == 1

    def test_clear_nonexistent_history(self):
        """Test clearing non-existent conversation doesn't error."""
        manager = ConversationHistoryManager()
        # Should not raise an error
        manager.clear_history("conv_nonexistent")

    def test_clear_all(self):
        """Test clearing all conversation histories."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_456", "user", "Hi")
        manager.add_message("conv_789", "user", "Hey")

        manager.clear_all()

        assert len(manager._histories) == 0
        assert manager.get_all_conversation_ids() == []

    def test_get_all_conversation_ids(self):
        """Test retrieving all conversation IDs."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_456", "user", "Hi")

        ids = manager.get_all_conversation_ids()

        assert len(ids) == 2
        assert "conv_123" in ids
        assert "conv_456" in ids

    def test_get_all_conversation_ids_empty(self):
        """Test getting conversation IDs when none exist."""
        manager = ConversationHistoryManager()
        ids = manager.get_all_conversation_ids()

        assert ids == []

    def test_get_message_count(self):
        """Test getting message count for a conversation."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi")
        manager.add_message("conv_123", "user", "How are you?")

        count = manager.get_message_count("conv_123")

        assert count == 3

    def test_get_message_count_empty(self):
        """Test getting message count for non-existent conversation."""
        manager = ConversationHistoryManager()
        count = manager.get_message_count("conv_123")

        assert count == 0

    def test_estimate_tokens_single_message(self):
        """Test token estimation for single message."""
        manager = ConversationHistoryManager()
        messages = [{"role": "user", "content": "Hello"}]

        tokens = manager.estimate_tokens(messages)

        # "user" (4) + "Hello" (5) + overhead (20) = 29 chars / 4 = ~7 tokens
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_multiple_messages(self):
        """Test token estimation for multiple messages."""
        manager = ConversationHistoryManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        tokens = manager.estimate_tokens(messages)

        assert tokens > 0
        # Should be more than single message
        single_tokens = manager.estimate_tokens([messages[0]])
        assert tokens > single_tokens

    def test_estimate_tokens_empty_messages(self):
        """Test token estimation for empty message list."""
        manager = ConversationHistoryManager()
        tokens = manager.estimate_tokens([])

        assert tokens == 0

    def test_estimate_tokens_long_content(self):
        """Test token estimation scales with content length."""
        manager = ConversationHistoryManager()
        short_message = [{"role": "user", "content": "Hi"}]
        long_message = [{"role": "user", "content": "A" * 1000}]

        short_tokens = manager.estimate_tokens(short_message)
        long_tokens = manager.estimate_tokens(long_message)

        assert long_tokens > short_tokens

    def test_truncate_by_tokens_no_truncation_needed(self):
        """Test truncate by tokens when content fits."""
        manager = ConversationHistoryManager()
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]

        truncated = manager._truncate_by_tokens(history, 1000)

        assert len(truncated) == 2
        assert truncated == history

    def test_truncate_by_tokens_removes_oldest(self):
        """Test truncate by tokens removes oldest messages first."""
        manager = ConversationHistoryManager()
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]

        # Very tight limit - should only keep most recent
        truncated = manager._truncate_by_tokens(history, 50)

        # Should keep at least the most recent message
        assert len(truncated) >= 1
        assert truncated[-1]["content"] == "Response 2"

    def test_truncate_by_tokens_keeps_at_least_one(self):
        """Test truncate by tokens keeps at least one message."""
        manager = ConversationHistoryManager()
        history = [{"role": "user", "content": "A" * 1000}]

        # Even with very small limit, should keep at least one message
        truncated = manager._truncate_by_tokens(history, 10)

        assert len(truncated) == 1
        assert truncated[0]["content"] == "A" * 1000

    def test_truncate_by_tokens_empty_history(self):
        """Test truncate by tokens with empty history."""
        manager = ConversationHistoryManager()
        truncated = manager._truncate_by_tokens([], 100)

        assert truncated == []

    def test_update_limits_max_messages(self):
        """Test updating max_messages limit."""
        manager = ConversationHistoryManager(max_messages=5)

        manager.update_limits(max_messages=20)

        assert manager._max_messages == 20

    def test_update_limits_max_tokens(self):
        """Test updating max_tokens limit."""
        manager = ConversationHistoryManager(max_tokens=1000)

        manager.update_limits(max_tokens=2000)

        assert manager._max_tokens == 2000

    def test_update_limits_both(self):
        """Test updating both limits."""
        manager = ConversationHistoryManager(max_messages=5, max_tokens=1000)

        manager.update_limits(max_messages=10, max_tokens=2000)

        assert manager._max_messages == 10
        assert manager._max_tokens == 2000

    def test_update_limits_none_no_change(self):
        """Test update_limits with None doesn't change values."""
        manager = ConversationHistoryManager(max_messages=5, max_tokens=1000)

        manager.update_limits(max_messages=None, max_tokens=None)

        assert manager._max_messages == 5
        assert manager._max_tokens == 1000

    def test_history_order_preserved(self):
        """Test that message order is preserved chronologically."""
        manager = ConversationHistoryManager()

        messages = [
            ("user", "First"),
            ("assistant", "Second"),
            ("user", "Third"),
            ("assistant", "Fourth"),
        ]

        for role, content in messages:
            manager.add_message("conv_123", role, content)

        history = manager.get_history("conv_123")

        assert len(history) == 4
        for i, (expected_role, expected_content) in enumerate(messages):
            assert history[i]["role"] == expected_role
            assert history[i]["content"] == expected_content

    def test_message_format_openai_compatible(self):
        """Test that message format matches OpenAI format."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", "user", "Hello")

        history = manager.get_history("conv_123")

        # Should have exactly 'role' and 'content' keys
        assert len(history[0]) == 2
        assert "role" in history[0]
        assert "content" in history[0]
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_concurrent_conversations_isolated(self):
        """Test that multiple conversations remain isolated."""
        manager = ConversationHistoryManager()

        # Add messages to different conversations
        manager.add_message("conv_1", "user", "Hello from 1")
        manager.add_message("conv_2", "user", "Hello from 2")
        manager.add_message("conv_1", "assistant", "Response to 1")
        manager.add_message("conv_2", "assistant", "Response to 2")

        history_1 = manager.get_history("conv_1")
        history_2 = manager.get_history("conv_2")

        # Each should have their own messages
        assert len(history_1) == 2
        assert len(history_2) == 2
        assert history_1[0]["content"] == "Hello from 1"
        assert history_2[0]["content"] == "Hello from 2"

    def test_large_conversation_handling(self):
        """Test handling of large conversations with many messages."""
        manager = ConversationHistoryManager(max_messages=100)

        # Add 500 messages
        for i in range(500):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_large", role, f"Message {i}")

        history = manager.get_history("conv_large")

        # Should only return last 100 due to limit
        assert len(history) == 100
        assert history[-1]["content"] == "Message 499"
        assert history[0]["content"] == "Message 400"

    def test_special_characters_in_content(self):
        """Test handling of special characters in message content."""
        manager = ConversationHistoryManager()

        special_content = "Hello! 你好 مرحبا 🎉 \n\t\\\"'"
        manager.add_message("conv_123", "user", special_content)

        history = manager.get_history("conv_123")

        assert history[0]["content"] == special_content

    def test_very_long_single_message(self):
        """Test handling of very long single messages."""
        manager = ConversationHistoryManager()

        # Create a very long message (100KB)
        long_content = "A" * 100000
        manager.add_message("conv_123", "user", long_content)

        history = manager.get_history("conv_123")

        assert len(history) == 1
        assert history[0]["content"] == long_content

    @pytest.mark.parametrize(
        "max_messages,expected_count",
        [
            (1, 1),
            (5, 5),
            (10, 10),
            (100, 20),  # Added 20 messages, limit is 100, so all 20 returned
        ],
    )
    def test_various_message_limits(self, max_messages, expected_count):
        """Test various max_messages limits."""
        manager = ConversationHistoryManager(max_messages=max_messages)

        # Add 20 messages
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message("conv_123", role, f"Message {i}")

        history = manager.get_history("conv_123")

        assert len(history) == min(expected_count, 20)


class TestConversationPersistence:
    """Test conversation persistence functionality."""

    @pytest.mark.asyncio
    async def test_load_from_storage_success(self):
        """Test successfully loading conversation history from storage."""
        from unittest.mock import AsyncMock, MagicMock

        # Mock Home Assistant
        mock_hass = MagicMock()

        # Create manager with persistence enabled
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock storage data
        storage_data = {
            "version": 1,
            "conversations": {
                "conv_123": [
                    {"role": "user", "content": "Hello", "timestamp": 1234567890},
                    {"role": "assistant", "content": "Hi there!", "timestamp": 1234567891},
                ],
                "conv_456": [
                    {"role": "user", "content": "How are you?", "timestamp": 1234567892},
                ],
            },
        }

        # Mock the store's async_load method
        manager._store.async_load = AsyncMock(return_value=storage_data)

        # Load from storage
        await manager.load_from_storage()

        # Verify conversations were loaded
        assert len(manager._histories) == 2
        assert len(manager._histories["conv_123"]) == 2
        assert len(manager._histories["conv_456"]) == 1
        assert manager._histories["conv_123"][0]["content"] == "Hello"
        assert manager._histories["conv_456"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_load_from_storage_no_existing_data(self):
        """Test loading when no storage data exists."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock async_load returning None (no data)
        manager._store.async_load = AsyncMock(return_value=None)

        # Should not raise error
        await manager.load_from_storage()

        # Histories should be empty
        assert len(manager._histories) == 0

    @pytest.mark.asyncio
    async def test_load_from_storage_corrupted_data(self):
        """Test loading handles corrupted storage data gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock corrupted storage data
        corrupted_data = {
            "version": 1,
            "conversations": {
                "conv_123": [
                    {"role": "user", "content": "Valid message", "timestamp": 1234567890},
                    "invalid_message_string",  # Invalid: should be dict
                    {"role": "assistant"},  # Invalid: missing content
                    {"content": "No role"},  # Invalid: missing role
                ],
                "conv_456": "invalid_not_a_list",  # Invalid: should be list
            },
        }

        manager._store.async_load = AsyncMock(return_value=corrupted_data)

        # Should handle gracefully without crashing
        await manager.load_from_storage()

        # Should only load valid messages
        assert "conv_123" in manager._histories
        assert len(manager._histories["conv_123"]) == 1
        assert manager._histories["conv_123"][0]["content"] == "Valid message"

        # Invalid conversation should be skipped
        assert "conv_456" not in manager._histories

    @pytest.mark.asyncio
    async def test_load_from_storage_without_persistence(self):
        """Test that load_from_storage returns early when persistence disabled."""

        # Create manager without persistence
        manager = ConversationHistoryManager(max_messages=10, persist=False)

        # Should return early without error
        await manager.load_from_storage()

        # No histories should be loaded
        assert len(manager._histories) == 0

    @pytest.mark.asyncio
    async def test_load_from_storage_handles_exception(self):
        """Test that storage load errors are caught and logged."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock async_load raising an exception
        manager._store.async_load = AsyncMock(side_effect=Exception("Storage error"))

        # Should not raise exception (catches and logs)
        await manager.load_from_storage()

        # Histories should remain empty
        assert len(manager._histories) == 0

    @pytest.mark.asyncio
    async def test_save_to_storage_success(self):
        """Test successfully saving conversation history to storage."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        mock_hass.bus = MagicMock()
        mock_hass.bus.async_fire = MagicMock()

        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Add some messages
        manager.add_message("conv_123", "user", "Hello")
        manager.add_message("conv_123", "assistant", "Hi!")

        # Mock the store's async_save method
        manager._store.async_save = AsyncMock()

        # Save to storage
        await manager.save_to_storage()

        # Verify async_save was called
        manager._store.async_save.assert_called_once()

        # Verify the data structure passed to async_save
        call_args = manager._store.async_save.call_args[0][0]
        assert call_args["version"] == 2
        assert "conversations" in call_args
        assert "conv_123" in call_args["conversations"]
        assert len(call_args["conversations"]["conv_123"]) == 2

    @pytest.mark.asyncio
    async def test_save_to_storage_without_persistence(self):
        """Test that save_to_storage returns early when persistence disabled."""

        manager = ConversationHistoryManager(max_messages=10, persist=False)
        manager.add_message("conv_123", "user", "Hello")

        # Should return early without error (no _store to call)
        await manager.save_to_storage()

    @pytest.mark.asyncio
    async def test_save_to_storage_failure(self):
        """Test that save_to_storage handles errors gracefully."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        manager.add_message("conv_123", "user", "Hello")

        # Mock async_save raising an exception
        manager._store.async_save = AsyncMock(side_effect=Exception("Storage error"))

        # Should not raise exception (catches and logs)
        await manager.save_to_storage()

    @pytest.mark.asyncio
    async def test_save_to_storage_fires_event(self):
        """Test that save_to_storage fires history saved event."""
        from unittest.mock import AsyncMock, MagicMock

        from custom_components.home_agent.const import EVENT_HISTORY_SAVED

        mock_hass = MagicMock()
        mock_hass.bus = MagicMock()
        mock_hass.bus.async_fire = MagicMock()

        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        manager.add_message("conv_123", "user", "Hello")
        manager._store.async_save = AsyncMock()

        await manager.save_to_storage()

        # Verify event was fired
        mock_hass.bus.async_fire.assert_called()
        call_args = mock_hass.bus.async_fire.call_args[0]
        assert call_args[0] == EVENT_HISTORY_SAVED

        # Verify event data
        event_data = call_args[1]
        assert "conversation_count" in event_data
        assert "message_count" in event_data
        assert "size_bytes" in event_data
        assert "timestamp" in event_data

    @pytest.mark.asyncio
    async def test_save_to_storage_warns_on_large_size(self):
        """Test that save_to_storage warns when storage size exceeds limit."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10000, hass=mock_hass, persist=True)

        # Add many large messages to exceed size limit
        for i in range(1000):
            manager.add_message(f"conv_{i}", "user", "A" * 10000)

        manager._store.async_save = AsyncMock()

        # Should complete without crashing (just logs warning)
        await manager.save_to_storage()

        # Verify save was still called
        manager._store.async_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_debounced_save_coalesces_multiple_saves(self):
        """Test that debounced save coalesces multiple save requests."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(
            max_messages=10, hass=mock_hass, persist=True, save_delay=0.1
        )

        # Mock save_to_storage
        manager.save_to_storage = AsyncMock()

        # Trigger multiple debounced saves rapidly
        await manager._debounced_save()
        await manager._debounced_save()
        await manager._debounced_save()

        # Wait for debounce delay plus a bit
        await asyncio.sleep(0.2)

        # Should have only saved once (debounced)
        assert manager.save_to_storage.call_count == 1

    @pytest.mark.asyncio
    async def test_debounced_save_cancels_previous_task(self):
        """Test that debounced save cancels previous pending save."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(
            max_messages=10, hass=mock_hass, persist=True, save_delay=0.2
        )

        manager.save_to_storage = AsyncMock()

        # Start first debounced save
        await manager._debounced_save()
        first_task = manager._save_task

        # Start second debounced save before first completes
        await asyncio.sleep(0.05)
        await manager._debounced_save()
        second_task = manager._save_task

        # First task should be cancelled
        assert first_task.cancelled() or first_task != second_task

        # Wait for second task to complete
        await asyncio.sleep(0.25)

        # Should have only saved once
        assert manager.save_to_storage.call_count == 1

    @pytest.mark.asyncio
    async def test_debounced_save_handles_cancellation(self):
        """Test that debounced save handles cancellation gracefully."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(
            max_messages=10, hass=mock_hass, persist=True, save_delay=0.2
        )

        manager.save_to_storage = AsyncMock()

        # Start debounced save
        await manager._debounced_save()
        task = manager._save_task

        # Cancel the task
        task.cancel()

        # Wait a bit
        await asyncio.sleep(0.05)

        # Should handle cancellation without error
        assert task.cancelled()

    def test_enable_persistence_activates_storage(self):
        """Test that enable_persistence creates storage when enabled."""
        from unittest.mock import MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=False)

        # Initially no store
        assert manager._persist is False
        assert manager._store is None

        # Enable persistence
        manager.enable_persistence(True)

        # Should have store now
        assert manager._persist is True
        assert manager._store is not None

    def test_enable_persistence_without_hass(self):
        """Test that enable_persistence fails gracefully without hass."""
        manager = ConversationHistoryManager(max_messages=10, persist=False)

        # Try to enable persistence without hass
        manager.enable_persistence(True)

        # Should remain disabled
        assert manager._persist is False
        assert manager._store is None

    def test_enable_persistence_disable(self):
        """Test that enable_persistence can disable persistence."""
        from unittest.mock import MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        assert manager._persist is True

        # Disable persistence
        manager.enable_persistence(False)

        assert manager._persist is False
        # Store should still exist (data not deleted)
        assert manager._store is not None

    @pytest.mark.asyncio
    async def test_migrate_storage_v1_to_v2(self):
        """Test storage migration from version 1 to current version."""
        from unittest.mock import MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock v1 data (currently v1 is the only version)
        old_data = {
            "version": 1,
            "conversations": {
                "conv_123": [{"role": "user", "content": "Hello"}],
            },
        }

        # Migrate (currently should return unchanged)
        migrated_data = await manager._migrate_storage(1, old_data)

        # Should return same data for v1
        assert migrated_data == old_data

    @pytest.mark.asyncio
    async def test_migrate_storage_unknown_version(self):
        """Test migration handles unknown versions gracefully."""
        from unittest.mock import MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock future version data
        future_data = {
            "version": 99,
            "conversations": {},
        }

        # Should return data as-is with warning
        migrated_data = await manager._migrate_storage(99, future_data)

        assert migrated_data == future_data

    @pytest.mark.asyncio
    async def test_scheduled_cleanup_removes_old_conversations(self):
        """Test that scheduled cleanup removes conversations older than 24 hours."""
        import time
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock save_to_storage
        manager.save_to_storage = AsyncMock()

        # Add old conversation (>24 hours ago)
        old_timestamp = int(time.time()) - (25 * 60 * 60)  # 25 hours ago
        manager._histories["old_conv"] = [
            {"role": "user", "content": "Old message", "timestamp": old_timestamp}
        ]

        # Add recent conversation
        recent_timestamp = int(time.time()) - (1 * 60 * 60)  # 1 hour ago
        manager._histories["recent_conv"] = [
            {"role": "user", "content": "Recent message", "timestamp": recent_timestamp}
        ]

        # Add empty conversation
        manager._histories["empty_conv"] = []

        # Run cleanup
        await manager._async_cleanup_old_conversations()

        # Old and empty conversations should be deleted
        assert "old_conv" not in manager._histories
        assert "empty_conv" not in manager._histories

        # Recent conversation should remain
        assert "recent_conv" in manager._histories

        # Save should have been called
        manager.save_to_storage.assert_called_once()

    @pytest.mark.asyncio
    async def test_scheduled_cleanup_no_deletions(self):
        """Test that cleanup doesn't save when nothing to delete."""
        import time
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        manager.save_to_storage = AsyncMock()

        # Add only recent conversations
        recent_timestamp = int(time.time()) - (1 * 60 * 60)
        manager._histories["recent_conv"] = [
            {"role": "user", "content": "Recent", "timestamp": recent_timestamp}
        ]

        # Run cleanup
        await manager._async_cleanup_old_conversations()

        # Conversation should remain
        assert "recent_conv" in manager._histories

        # Save should not be called (nothing deleted)
        manager.save_to_storage.assert_not_called()

    def test_setup_scheduled_cleanup(self):
        """Test that setup_scheduled_cleanup schedules periodic cleanup."""
        from unittest.mock import MagicMock, patch

        mock_hass = MagicMock()

        with patch(
            "custom_components.home_agent.conversation.async_track_time_interval"
        ) as mock_track:
            manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

            # Setup scheduled cleanup
            manager.setup_scheduled_cleanup()

            # Should have called async_track_time_interval
            mock_track.assert_called_once()

            # Verify cleanup listener was set
            assert manager._cleanup_listener is not None

    def test_shutdown_scheduled_cleanup(self):
        """Test that shutdown_scheduled_cleanup stops the scheduler."""
        from unittest.mock import MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        # Mock cleanup listener
        mock_listener = MagicMock()
        manager._cleanup_listener = mock_listener

        # Shutdown cleanup
        manager.shutdown_scheduled_cleanup()

        # Listener should be called to cancel
        mock_listener.assert_called_once()

        # Listener should be cleared
        assert manager._cleanup_listener is None

    @pytest.mark.asyncio
    async def test_add_message_triggers_debounced_save(self):
        """Test that adding a message triggers debounced save when persistence enabled."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(
            max_messages=10, hass=mock_hass, persist=True, save_delay=0.1
        )

        manager.save_to_storage = AsyncMock()

        # Add message (should trigger debounced save)
        manager.add_message("conv_123", "user", "Hello")

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Save should have been called
        manager.save_to_storage.assert_called()

    @pytest.mark.asyncio
    async def test_clear_history_triggers_debounced_save(self):
        """Test that clearing history triggers debounced save when persistence enabled."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()
        manager = ConversationHistoryManager(
            max_messages=10, hass=mock_hass, persist=True, save_delay=0.1
        )

        manager.save_to_storage = AsyncMock()

        # Add and clear message
        manager.add_message("conv_123", "user", "Hello")
        await asyncio.sleep(0.15)  # Wait for first save
        manager.save_to_storage.reset_mock()

        manager.clear_history("conv_123")

        # Wait for debounce
        await asyncio.sleep(0.15)

        # Save should have been called again
        manager.save_to_storage.assert_called()

    @pytest.mark.asyncio
    async def test_persistence_integration_flow(self):
        """Test full persistence flow: save, load, verify."""
        from unittest.mock import AsyncMock, MagicMock

        mock_hass = MagicMock()

        # Create first manager and add data
        manager1 = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        manager1.add_message("conv_123", "user", "Hello")
        manager1.add_message("conv_123", "assistant", "Hi there!")

        # Capture saved data
        saved_data = None

        async def capture_save(data):
            nonlocal saved_data
            saved_data = data

        manager1._store.async_save = AsyncMock(side_effect=capture_save)

        # Save
        await manager1.save_to_storage()

        assert saved_data is not None

        # Create second manager and load data
        manager2 = ConversationHistoryManager(max_messages=10, hass=mock_hass, persist=True)

        manager2._store.async_load = AsyncMock(return_value=saved_data)

        await manager2.load_from_storage()

        # Verify data matches
        history = manager2.get_history("conv_123")
        assert len(history) == 2
        assert history[0]["content"] == "Hello"
        assert history[1]["content"] == "Hi there!"


class TestToolCallMessages:
    """Test tool call message support in ConversationHistoryManager."""

    def test_add_assistant_message_with_tool_calls_dict_format(self):
        """Test adding assistant message with tool_calls using dict format."""
        manager = ConversationHistoryManager()
        tool_calls_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_xxx",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"location": "NYC"}'},
                }
            ],
        }
        manager.add_message("conv_123", tool_calls_msg)

        assert len(manager._histories["conv_123"]) == 1
        msg = manager._histories["conv_123"][0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"] == tool_calls_msg["tool_calls"]
        assert "timestamp" in msg

    def test_add_tool_result_message(self):
        """Test adding tool result message with tool_call_id."""
        manager = ConversationHistoryManager()
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": "call_xxx",
            "content": '{"temperature": 25, "condition": "sunny"}',
        }
        manager.add_message("conv_123", tool_result_msg)

        assert len(manager._histories["conv_123"]) == 1
        msg = manager._histories["conv_123"][0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_xxx"
        assert msg["content"] == '{"temperature": 25, "condition": "sunny"}'

    def test_tool_message_with_empty_content_allowed(self):
        """Test that tool messages with empty content are accepted."""
        manager = ConversationHistoryManager()
        tool_result_msg = {
            "role": "tool",
            "tool_call_id": "call_xxx",
            "content": "",
        }
        manager.add_message("conv_123", tool_result_msg)

        # Tool messages with empty content should be accepted
        assert len(manager._histories["conv_123"]) == 1

    def test_is_tool_message_detects_tool_role(self):
        """Test _is_tool_message returns True for role=tool."""
        manager = ConversationHistoryManager()
        msg = {"role": "tool", "tool_call_id": "call_x", "content": "result"}
        assert manager._is_tool_message(msg) is True

    def test_is_tool_message_detects_assistant_with_tool_calls(self):
        """Test _is_tool_message returns True for assistant with tool_calls."""
        manager = ConversationHistoryManager()
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_x", "type": "function", "function": {}}],
        }
        assert manager._is_tool_message(msg) is True

    def test_is_tool_message_false_for_normal_messages(self):
        """Test _is_tool_message returns False for normal messages."""
        manager = ConversationHistoryManager()
        assert manager._is_tool_message({"role": "user", "content": "Hello"}) is False
        assert manager._is_tool_message({"role": "assistant", "content": "Hi"}) is False

    def test_is_tool_message_false_for_assistant_without_tool_calls(self):
        """Test _is_tool_message returns False for assistant without tool_calls."""
        manager = ConversationHistoryManager()
        msg = {"role": "assistant", "content": "Hello there!"}
        assert manager._is_tool_message(msg) is False

    def test_count_conversation_turns_excludes_tool_messages(self):
        """Test _count_conversation_turns excludes tool messages from count."""
        manager = ConversationHistoryManager()
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "assistant", "content": "It's sunny."},
        ]
        # 2 real turns: user question + assistant text response
        # Tool messages (assistant with tool_calls + tool result) are excluded
        turns = manager._count_conversation_turns(messages)
        assert turns == 2

    def test_count_conversation_turns_all_text(self):
        """Test _count_conversation_turns with only text messages."""
        manager = ConversationHistoryManager()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        turns = manager._count_conversation_turns(messages)
        assert turns == 3

    def test_count_conversation_turns_empty(self):
        """Test _count_conversation_turns with empty list."""
        manager = ConversationHistoryManager()
        turns = manager._count_conversation_turns([])
        assert turns == 0

    def test_trim_history_preserves_tool_chains(self):
        """Test that trimming preserves tool message chains and only removes complete turns."""
        manager = ConversationHistoryManager(max_messages=2)
        # Add 3 user turns with tool calls in the middle
        manager.add_message("conv_123", {"role": "user", "content": "First question"})
        manager.add_message("conv_123", {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {}}]})
        manager.add_message("conv_123", {"role": "tool", "tool_call_id": "call_1", "content": "result1"})
        manager.add_message("conv_123", {"role": "assistant", "content": "First answer"})

        manager.add_message("conv_123", {"role": "user", "content": "Second question"})
        manager.add_message("conv_123", {"role": "assistant", "content": "Second answer"})

        manager.add_message("conv_123", {"role": "user", "content": "Third question"})
        manager.add_message("conv_123", {"role": "assistant", "content": "Third answer"})

        history = manager.get_history("conv_123")
        # max_messages=2 means 2 conversation turns (4 non-tool messages max)
        # Tool messages are exempt from the count
        non_tool_msgs = [m for m in history if not manager._is_tool_message(m)]
        assert len(non_tool_msgs) <= 2

    def test_trim_history_tool_exempt_from_count(self):
        """Test that tool messages don't count toward max_messages limit."""
        manager = ConversationHistoryManager(max_messages=2)
        manager.add_message("conv_123", {"role": "user", "content": "Q1"})
        manager.add_message("conv_123", {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {}}]})
        manager.add_message("conv_123", {"role": "tool", "tool_call_id": "c1", "content": "r1"})
        manager.add_message("conv_123", {"role": "assistant", "content": "A1"})
        manager.add_message("conv_123", {"role": "user", "content": "Q2"})
        manager.add_message("conv_123", {"role": "assistant", "content": "A2"})

        history = manager.get_history("conv_123")
        non_tool_msgs = [m for m in history if not manager._is_tool_message(m)]
        # max_messages=2, tool messages exempt, so 2 non-tool messages remain
        assert len(non_tool_msgs) == 2

    def test_add_messages_batch(self):
        """Test add_messages adds multiple messages in one call."""
        manager = ConversationHistoryManager()
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 25C"},
            {"role": "assistant", "content": "It's sunny and 25C"},
        ]
        manager.add_messages("conv_123", messages)

        assert len(manager._histories["conv_123"]) == 4
        history = manager.get_history("conv_123")
        assert len(history) == 4
        assert history[0]["content"] == "What's the weather?"
        assert history[3]["content"] == "It's sunny and 25C"

    def test_add_messages_batch_empty_list(self):
        """Test add_messages with empty list does nothing."""
        manager = ConversationHistoryManager()
        manager.add_messages("conv_123", [])

        assert len(manager._histories) == 0

    def test_add_messages_batch_empty_conversation_id(self):
        """Test add_messages with empty conversation_id is ignored."""
        manager = ConversationHistoryManager()
        manager.add_messages("", [{"role": "user", "content": "Hello"}])

        assert len(manager._histories) == 0

    def test_get_history_preserves_tool_calls(self):
        """Test get_history returns tool_calls field intact."""
        manager = ConversationHistoryManager()
        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            }
        ]
        manager.add_message("conv_123", {"role": "assistant", "content": "", "tool_calls": tool_calls})
        manager.add_message("conv_123", {"role": "tool", "tool_call_id": "call_abc", "content": "Rainy"})

        history = manager.get_history("conv_123")
        assert len(history) == 2
        assert history[0]["tool_calls"] == tool_calls
        assert history[1]["tool_call_id"] == "call_abc"
        # timestamp should be stripped
        assert "timestamp" not in history[0]
        assert "timestamp" not in history[1]

    def test_get_history_preserves_name_field(self):
        """Test get_history preserves the name field when present."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", {
            "role": "user",
            "content": "Hello",
            "name": "alice",
        })

        history = manager.get_history("conv_123")
        assert history[0]["name"] == "alice"

    def test_strip_timestamp_preserves_tool_fields(self):
        """Test _strip_timestamp removes timestamp but keeps tool fields."""
        manager = ConversationHistoryManager()
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {}}],
            "timestamp": 12345,
        }
        stripped = manager._strip_timestamp(msg)

        assert "timestamp" not in stripped
        assert stripped["tool_calls"] == msg["tool_calls"]
        assert stripped["role"] == "assistant"

    def test_estimate_tokens_with_tool_calls(self):
        """Test token estimation accounts for tool_calls content."""
        manager = ConversationHistoryManager()
        msg_with_tools = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": '{"location": "NYC", "units": "celsius"}'},
                    }
                ],
            }
        ]
        msg_without_tools = [{"role": "assistant", "content": ""}]

        tokens_with = manager.estimate_tokens(msg_with_tools)
        tokens_without = manager.estimate_tokens(msg_without_tools)

        # Tool calls should add to token count
        assert tokens_with > tokens_without

    def test_full_tool_call_turn_workflow(self):
        """Test a complete tool call turn: user -> assistant(tool) -> tool -> assistant(text)."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", {"role": "user", "content": "Turn on the living room light"})
        manager.add_message("conv_123", {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_turn_on",
                    "type": "function",
                    "function": {
                        "name": "ha_control",
                        "arguments": '{"entity_id": "light.living_room", "state": "on"}',
                    },
                }
            ],
        })
        manager.add_message("conv_123", {
            "role": "tool",
            "tool_call_id": "call_turn_on",
            "content": '{"success": true}',
        })
        manager.add_message("conv_123", {"role": "assistant", "content": "I've turned on the living room light."})

        history = manager.get_history("conv_123")
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert "tool_calls" in history[1]
        assert history[2]["role"] == "tool"
        assert history[2]["tool_call_id"] == "call_turn_on"
        assert history[3]["content"] == "I've turned on the living room light."

        # Check turn count: 2 turns (user question + assistant text response)
        # Tool messages (assistant with tool_calls + tool result) are excluded
        turns = manager._count_conversation_turns(
            manager._histories["conv_123"]
        )
        assert turns == 2

    def test_multiple_tool_calls_in_one_message(self):
        """Test assistant message with multiple tool_calls in a single message."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        })

        history = manager.get_history("conv_123")
        assert len(history) == 1
        assert len(history[0]["tool_calls"]) == 2
        assert history[0]["tool_calls"][0]["id"] == "call_1"
        assert history[0]["tool_calls"][1]["id"] == "call_2"

    def test_add_message_dict_format_simple(self):
        """Test add_message with simple dict format (no tool calls)."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", {"role": "user", "content": "Hello"})

        assert len(manager._histories["conv_123"]) == 1
        assert manager._histories["conv_123"][0]["role"] == "user"
        assert manager._histories["conv_123"][0]["content"] == "Hello"

    def test_add_message_dict_format_missing_role(self):
        """Test add_message with dict missing 'role' is rejected."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", {"content": "Hello"})

        assert len(manager._histories) == 0

    def test_add_message_dict_format_missing_content_non_tool(self):
        """Test add_message with dict missing 'content' for non-tool is rejected."""
        manager = ConversationHistoryManager()
        manager.add_message("conv_123", {"role": "user"})

        assert len(manager._histories) == 0
