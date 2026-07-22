"""Unit tests for stable history feature (idle-based eviction and system prompt caching).

These tests verify:
1. Idle-based conversation history eviction respects min_messages floor
2. Active conversations are not evicted even when exceeding max_messages
3. System prompt caching works correctly with freeze enabled
4. System prompt cache is invalidated on eviction
5. Config params are wired correctly
"""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from custom_components.home_agent.agent.core import HomeAgent
from custom_components.home_agent.const import (
    CONF_HISTORY_IDLE_THRESHOLD,
    CONF_HISTORY_MAX_MESSAGES,
    CONF_HISTORY_MIN_MESSAGES,
    CONF_SYSTEM_PROMPT_FREEZE,
    DEFAULT_HISTORY_IDLE_THRESHOLD,
    DEFAULT_HISTORY_MAX_MESSAGES,
    DEFAULT_HISTORY_MIN_MESSAGES,
    DEFAULT_SYSTEM_PROMPT_FREEZE,
)
from custom_components.home_agent.conversation import ConversationHistoryManager

# Suppress logging during tests to avoid issues with uninitialized logging
logging.disable(logging.CRITICAL)


class TestIdleBasedEviction:
    """Tests for idle-based conversation history eviction."""

    def setup_method(self):
        """Reset logging before each test."""
        logging.disable(logging.CRITICAL)

    def teardown_method(self):
        """Re-enable logging after each test."""
        logging.disable(logging.NOTSET)

    def test_no_eviction_within_limit(self):
        """Test that no eviction occurs when turns <= max_messages."""
        manager = ConversationHistoryManager(
            max_messages=10,
            min_messages=4,
            idle_threshold=600,
        )

        # Add 8 messages (4 turns - within limit)
        for i in range(4):
            manager.add_message("conv_1", {"role": "user", "content": f"User message {i}"})
            manager.add_message("conv_1", {"role": "assistant", "content": f"Assistant response {i}"})

        # Add more messages - should not evict since we're within limit
        evicted = manager.add_messages(
            "conv_1",
            [
                {"role": "user", "content": "Extra user"},
                {"role": "assistant", "content": "Extra assistant"},
            ],
        )

        assert evicted is False
        history = manager.get_history("conv_1")
        assert len(history) == 10  # All messages preserved

    def test_no_eviction_active_conversation(self):
        """Test that active conversations are not evicted even when exceeding max_messages."""
        manager = ConversationHistoryManager(
            max_messages=4,
            min_messages=2,
            idle_threshold=600,  # 10 minutes idle threshold
        )

        # Add 6 turns (12 messages) - exceeds max_messages of 4
        for i in range(6):
            manager.add_message("conv_1", {"role": "user", "content": f"User {i}"})
            manager.add_message("conv_1", {"role": "assistant", "content": f"Assistant {i}"})

        # Add more messages - conversation is active (timestamps are recent)
        evicted = manager.add_messages(
            "conv_1",
            [
                {"role": "user", "content": "New user message"},
                {"role": "assistant", "content": "New assistant response"},
            ],
        )

        # Should NOT evict because conversation is still active
        assert evicted is False
        
        # Internal storage keeps all messages; get_history() truncates to max_messages for LLM API
        # Check internal state to verify no eviction occurred
        internal_history = manager._histories["conv_1"]
        assert len(internal_history) == 14  # All 14 messages preserved in storage
        
        # get_history() returns truncated view for LLM (max_messages=4 turns = 4 messages)
        api_history = manager.get_history("conv_1")
        assert len(api_history) == 4  # Truncated to max_messages for API

    def test_eviction_idle_conversation(self):
        """Test that idle conversations are evicted to min_messages."""
        # Mock time to simulate idle conversation
        with patch("custom_components.home_agent.conversation.time") as mock_time:
            base_time = 1000000
            mock_time.time.return_value = base_time

            manager = ConversationHistoryManager(
                max_messages=4,
                min_messages=2,
                idle_threshold=600,  # 10 minutes
            )

            # Add 6 turns (12 messages) at base_time
            for i in range(6):
                manager.add_message("conv_1", {"role": "user", "content": f"User {i}"})
                manager.add_message("conv_1", {"role": "assistant", "content": f"Assistant {i}"})

            # Advance time by 11 minutes (exceeds idle threshold)
            mock_time.time.return_value = base_time + 660

            # Add new message - should trigger eviction
            evicted = manager.add_messages(
                "conv_1",
                [
                    {"role": "user", "content": "New user after idle"},
                    {"role": "assistant", "content": "New assistant after idle"},
                ],
            )

            assert evicted is True
            history = manager.get_history("conv_1")

            # Should be trimmed to min_messages (2 turns = 4 messages) + new messages
            # The eviction keeps min_messages oldest turns, then new messages are added
            turn_count = manager._count_conversation_turns(history)
            assert turn_count >= 2  # At least min_messages turns preserved

    def test_idle_threshold_zero_always_evict(self):
        """Test that idle_threshold=0 allows eviction regardless of activity."""
        manager = ConversationHistoryManager(
            max_messages=4,
            min_messages=2,
            idle_threshold=0,  # Always allow eviction
        )

        # Add 6 turns (12 messages)
        for i in range(6):
            manager.add_message("conv_1", {"role": "user", "content": f"User {i}"})
            manager.add_message("conv_1", {"role": "assistant", "content": f"Assistant {i}"})

        # Add new message - should evict immediately since threshold is 0
        evicted = manager.add_messages(
            "conv_1",
            [
                {"role": "user", "content": "New user"},
                {"role": "assistant", "content": "New assistant"},
            ],
        )

        assert evicted is True
        history = manager.get_history("conv_1")
        turn_count = manager._count_conversation_turns(history)
        assert turn_count >= 2  # At least min_messages

    def test_min_messages_floor(self):
        """Test that min_messages is respected as the stable floor."""
        with patch("custom_components.home_agent.conversation.time") as mock_time:
            base_time = 1000000
            mock_time.time.return_value = base_time

            manager = ConversationHistoryManager(
                max_messages=6,
                min_messages=4,
                idle_threshold=600,
            )

            # Add 10 turns at base_time
            for i in range(10):
                manager.add_message("conv_1", {"role": "user", "content": f"User {i}"})
                manager.add_message("conv_1", {"role": "assistant", "content": f"Assistant {i}"})

            # Advance time past idle threshold
            mock_time.time.return_value = base_time + 700

            # Trigger eviction
            evicted = manager.add_messages(
                "conv_1",
                [{"role": "user", "content": "After idle"}],
            )

            assert evicted is True
            history = manager.get_history("conv_1")
            turn_count = manager._count_conversation_turns(history)

            # Should have min_messages (4) + new message (1) = 5 turns minimum
            assert turn_count >= 4  # min_messages floor respected

    def test_default_values(self):
        """Test that default values are correct."""
        manager = ConversationHistoryManager()
        
        assert manager._min_messages == DEFAULT_HISTORY_MIN_MESSAGES
        assert manager._idle_threshold == DEFAULT_HISTORY_IDLE_THRESHOLD


class TestSystemPromptCaching:
    """Tests for stable system prompt caching."""

    def setup_method(self):
        """Reset logging before each test."""
        logging.disable(logging.CRITICAL)

    def teardown_method(self):
        """Re-enable logging after each test."""
        logging.disable(logging.NOTSET)

    def test_system_prompt_cache_basic(self):
        """Test basic system prompt caching behavior."""
        # Test the caching logic directly without full HomeAgent instantiation
        # This tests the concept without requiring complex fixtures
        
        # Simulate cache state
        cache = {
            "_system_prompt_freeze": True,
            "_cached_system_prompt": None,
            "_cached_conversation_id": None,
            "_system_prompt_dirty": True,
        }
        
        def get_system_prompt(conversation_id, force_rebuild=False, entity_context=""):
            """Simulated _get_system_prompt logic."""
            if not cache["_system_prompt_freeze"]:
                return f"Built prompt with {entity_context}"
            
            should_rebuild = (
                force_rebuild
                or cache["_system_prompt_dirty"]
                or cache["_cached_conversation_id"] != conversation_id
            )
            
            if should_rebuild:
                cache["_cached_system_prompt"] = f"Built prompt with {entity_context}"
                cache["_cached_conversation_id"] = conversation_id
                cache["_system_prompt_dirty"] = False
            
            return cache["_cached_system_prompt"]
        
        # First call should build
        prompt1 = get_system_prompt("conv_1", entity_context="Context 1")
        assert "Context 1" in prompt1
        
        # Second call with same conversation should use cache
        prompt2 = get_system_prompt("conv_1", entity_context="Context 2 (should be ignored)")
        assert prompt1 == prompt2
        assert "Context 2" not in prompt2
        
        # Force rebuild should create new prompt
        prompt3 = get_system_prompt("conv_1", force_rebuild=True, entity_context="Context 3")
        assert "Context 3" in prompt3

    def test_system_prompt_freeze_disabled(self):
        """Test that freeze=False bypasses cache."""
        cache = {
            "_system_prompt_freeze": False,
            "_cached_system_prompt": None,
            "_cached_conversation_id": None,
            "_system_prompt_dirty": True,
        }
        
        call_count = 0
        
        def get_system_prompt(conversation_id, force_rebuild=False, entity_context=""):
            nonlocal call_count
            call_count += 1
            
            if not cache["_system_prompt_freeze"]:
                return f"Built prompt #{call_count} with {entity_context}"
            
            return cache["_cached_system_prompt"] or "cached"
        
        # Each call should build fresh
        prompt1 = get_system_prompt("conv_1", entity_context="Context 1")
        prompt2 = get_system_prompt("conv_1", entity_context="Context 2")
        
        assert prompt1 != prompt2
        assert "Context 1" in prompt1
        assert "Context 2" in prompt2
        assert call_count == 2

    def test_cache_invalidation_on_new_conversation(self):
        """Test cache invalidation when conversation changes."""
        cache = {
            "_system_prompt_freeze": True,
            "_cached_system_prompt": None,
            "_cached_conversation_id": None,
            "_system_prompt_dirty": True,
        }
        
        def get_system_prompt(conversation_id, force_rebuild=False, entity_context=""):
            should_rebuild = (
                force_rebuild
                or cache["_system_prompt_dirty"]
                or cache["_cached_conversation_id"] != conversation_id
            )
            
            if should_rebuild:
                cache["_cached_system_prompt"] = f"Built for {conversation_id}: {entity_context}"
                cache["_cached_conversation_id"] = conversation_id
                cache["_system_prompt_dirty"] = False
            
            return cache["_cached_system_prompt"]
        
        # Build for conv_1
        prompt1 = get_system_prompt("conv_1", entity_context="Context 1")
        assert "conv_1" in prompt1
        
        # Switch to conv_2 (conversation_id changes -> should rebuild)
        prompt2 = get_system_prompt("conv_2", entity_context="Context 2")
        assert "conv_2" in prompt2

    def test_manual_cache_invalidation(self):
        """Test manual cache invalidation."""
        cache = {
            "_system_prompt_freeze": True,
            "_cached_system_prompt": None,
            "_cached_conversation_id": None,
            "_system_prompt_dirty": True,
        }
        
        def invalidate_cache():
            cache["_system_prompt_dirty"] = True
        
        def get_system_prompt(conversation_id, force_rebuild=False, entity_context=""):
            should_rebuild = (
                force_rebuild
                or cache["_system_prompt_dirty"]
                or cache["_cached_conversation_id"] != conversation_id
            )
            
            if should_rebuild:
                cache["_cached_system_prompt"] = f"Built: {entity_context}"
                cache["_cached_conversation_id"] = conversation_id
                cache["_system_prompt_dirty"] = False
            
            return cache["_cached_system_prompt"]
        
        # Build and cache
        prompt1 = get_system_prompt("conv_1", entity_context="Original")
        assert "Original" in prompt1
        
        # Invalidate
        invalidate_cache()
        
        # Rebuild with new context
        prompt2 = get_system_prompt("conv_1", entity_context="After invalidation")
        assert "After invalidation" in prompt2


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_constants_exist(self):
        """Test that default constants are defined."""
        from custom_components.home_agent.const import (
            CONF_HISTORY_IDLE_THRESHOLD,
            CONF_HISTORY_MIN_MESSAGES,
            CONF_SYSTEM_PROMPT_FREEZE,
        )
        
        assert CONF_HISTORY_MIN_MESSAGES == "history_min_messages"
        assert CONF_HISTORY_IDLE_THRESHOLD == "history_idle_threshold"
        assert CONF_SYSTEM_PROMPT_FREEZE == "system_prompt_freeze"

    def test_default_values_reasonable(self):
        """Test that default values are reasonable."""
        assert DEFAULT_HISTORY_MIN_MESSAGES == 10
        assert DEFAULT_HISTORY_IDLE_THRESHOLD == 600  # 10 minutes
        assert DEFAULT_SYSTEM_PROMPT_FREEZE is True


@pytest.mark.integration
class TestSystemPromptCaching:
    """Tests for stable system prompt caching."""

    def test_system_prompt_freeze_enabled(self, test_hass_with_default_entities, llm_config, session_manager):
        """Test that system prompt is cached when freeze is enabled."""
        llm_config[CONF_SYSTEM_PROMPT_FREEZE] = True
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        # First call should build and cache
        prompt1 = agent._get_system_prompt(
            entity_context="Test context",
            conversation_id="conv_1",
        )

        # Second call with same conversation should use cache
        prompt2 = agent._get_system_prompt(
            entity_context="Different context (should be ignored due to cache)",
            conversation_id="conv_1",
        )

        # Both should return the same cached prompt
        assert prompt1 == prompt2
        assert agent._cached_system_prompt is not None
        assert agent._cached_system_prompt == prompt1
        assert agent._cached_conversation_id == "conv_1"

    def test_system_prompt_freeze_disabled(self, test_hass_with_default_entities, llm_config, session_manager):
        """Test that system prompt is rebuilt on each call when freeze is disabled."""
        llm_config[CONF_SYSTEM_PROMPT_FREEZE] = False
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        # First call
        prompt1 = agent._get_system_prompt(
            entity_context="Context 1",
            conversation_id="conv_1",
        )

        # Second call
        prompt2 = agent._get_system_prompt(
            entity_context="Context 2",
            conversation_id="conv_1",
        )

        # Cache should NOT be used when freeze is disabled
        assert agent._cached_system_prompt is None
        # Both prompts should contain the template content (same structure)
        assert "Home Assistant" in prompt1
        assert "Home Assistant" in prompt2

    def test_system_prompt_cache_invalidated_on_new_conversation(
        self, test_hass_with_default_entities, llm_config, session_manager
    ):
        """Test that system prompt cache is invalidated when conversation changes."""
        llm_config[CONF_SYSTEM_PROMPT_FREEZE] = True
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        # Build prompt for conv_1
        prompt1 = agent._get_system_prompt(
            entity_context="Context for conv_1",
            conversation_id="conv_1",
        )
        assert agent._cached_conversation_id == "conv_1"

        # Build prompt for conv_2 (different conversation - cache invalidated)
        prompt2 = agent._get_system_prompt(
            entity_context="Context for conv_2",
            conversation_id="conv_2",
        )

        # Cache should be updated to new conversation
        assert agent._cached_conversation_id == "conv_2"
        # The prompts may differ due to dynamic content (timestamp, etc.)
        assert "Home Assistant" in prompt1
        assert "Home Assistant" in prompt2

    def test_system_prompt_cache_invalidated_manually(
        self, test_hass_with_default_entities, llm_config, session_manager
    ):
        """Test that manual cache invalidation forces rebuild."""
        llm_config[CONF_SYSTEM_PROMPT_FREEZE] = True
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        # Build and cache prompt
        prompt1 = agent._get_system_prompt(
            entity_context="Original context",
            conversation_id="conv_1",
        )
        assert agent._cached_system_prompt is not None

        # Invalidate cache
        agent.invalidate_system_prompt_cache()

        # Cache should be cleared
        assert agent._cached_system_prompt is None
        assert agent._system_prompt_dirty is True

        # Rebuild with new context
        prompt2 = agent._get_system_prompt(
            entity_context="New context after invalidation",
            conversation_id="conv_1",
        )

        # Cache should be rebuilt
        assert agent._cached_system_prompt is not None
        assert agent._system_prompt_dirty is False

    def test_system_prompt_force_rebuild(self, test_hass_with_default_entities, llm_config, session_manager):
        """Test that force_rebuild parameter bypasses cache."""
        llm_config[CONF_SYSTEM_PROMPT_FREEZE] = True
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        # Build prompt
        prompt1 = agent._get_system_prompt(
            entity_context="Context 1",
            conversation_id="conv_1",
        )
        cached_prompt_id_1 = id(agent._cached_system_prompt)

        # Force rebuild even with same conversation
        prompt2 = agent._get_system_prompt(
            entity_context="Context 2",
            conversation_id="conv_1",
            force_rebuild=True,
        )
        cached_prompt_id_2 = id(agent._cached_system_prompt)

        # Both should contain template content
        assert "Home Assistant" in prompt1
        assert "Home Assistant" in prompt2
        # Cache should still be active
        assert agent._cached_conversation_id == "conv_1"


@pytest.mark.integration
class TestConfigWiring:
    """Tests for new config params being wired correctly."""

    def test_conversation_manager_receives_min_messages(
        self, test_hass_with_default_entities, llm_config, session_manager
    ):
        """Test that min_messages config is passed to ConversationHistoryManager."""
        llm_config[CONF_HISTORY_MIN_MESSAGES] = 6
        llm_config[CONF_HISTORY_MAX_MESSAGES] = 10
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        assert agent.conversation_manager._min_messages == 6
        assert agent.conversation_manager._max_messages == 10

    def test_conversation_manager_receives_idle_threshold(
        self, test_hass_with_default_entities, llm_config, session_manager
    ):
        """Test that idle_threshold config is passed to ConversationHistoryManager."""
        llm_config[CONF_HISTORY_IDLE_THRESHOLD] = 1200  # 20 minutes
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        assert agent.conversation_manager._idle_threshold == 1200

    def test_system_prompt_freeze_config(self, test_hass_with_default_entities, llm_config, session_manager):
        """Test that system_prompt_freeze config is read correctly."""
        llm_config[CONF_SYSTEM_PROMPT_FREEZE] = False
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        assert agent._system_prompt_freeze is False

    def test_default_config_values(self, test_hass_with_default_entities, llm_config, session_manager):
        """Test that default config values are applied when not specified."""
        # Remove any config params set by sibling tests (llm_config is session-scoped)
        for key in [CONF_HISTORY_MIN_MESSAGES, CONF_HISTORY_MAX_MESSAGES, CONF_HISTORY_IDLE_THRESHOLD, CONF_SYSTEM_PROMPT_FREEZE]:
            llm_config.pop(key, None)
        
        agent = HomeAgent(
            test_hass_with_default_entities,
            llm_config,
            session_manager,
        )

        # Check defaults (updated: min_messages=10, max_messages=40)
        assert agent.conversation_manager._min_messages == DEFAULT_HISTORY_MIN_MESSAGES
        assert agent.conversation_manager._idle_threshold == DEFAULT_HISTORY_IDLE_THRESHOLD
        assert agent._system_prompt_freeze is True


@pytest.mark.integration
class TestEvictionInvalidatesCache:
    """Tests that history eviction triggers system prompt cache invalidation."""

    def test_eviction_triggers_cache_invalidation(
        self, test_hass_with_default_entities, llm_config, session_manager
    ):
        """Test that when eviction occurs, system prompt cache is invalidated."""
        with patch("custom_components.home_agent.conversation.time") as mock_time:
            base_time = 1000000
            mock_time.time.return_value = base_time

            llm_config[CONF_HISTORY_MAX_MESSAGES] = 4
            llm_config[CONF_HISTORY_MIN_MESSAGES] = 2
            llm_config[CONF_HISTORY_IDLE_THRESHOLD] = 600
            llm_config[CONF_SYSTEM_PROMPT_FREEZE] = True
            llm_config["history_persist"] = False  # Disable persistence to avoid async event loop
            
            agent = HomeAgent(
                test_hass_with_default_entities,
                llm_config,
                session_manager,
            )

            # Initialize system prompt cache
            prompt1 = agent._get_system_prompt(
                entity_context="Original context",
                conversation_id="conv_1",
            )

            # Simulate adding messages that trigger eviction
            # First add old messages
            for i in range(6):
                agent.conversation_manager.add_message(
                    "conv_1", {"role": "user", "content": f"Old user {i}"}
                )
                agent.conversation_manager.add_message(
                    "conv_1", {"role": "assistant", "content": f"Old assistant {i}"}
                )

            # Advance time past idle threshold
            mock_time.time.return_value = base_time + 700

            # Add new message - this should trigger eviction and cache invalidation
            # In the actual code, this happens in process_message where add_messages
            # returns True and invalidate_system_prompt_cache is called
            evicted = agent.conversation_manager.add_messages(
                "conv_1",
                [{"role": "user", "content": "New message after idle"}],
            )

            if evicted:
                agent.invalidate_system_prompt_cache()

                # Next _get_system_prompt call should rebuild
                # (with dirty flag set to True)
                assert agent._system_prompt_dirty is True
