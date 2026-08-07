"""Unit tests for memory extraction optimization features.

Tests the new debounce timer, content heuristics, and conversation-end batching
functionality for memory extraction.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, Mock, patch

import pytest

from custom_components.home_agent.agent.memory_extraction import (
    MemoryExtractionMixin,
    MemoryExtractionState,
)
from custom_components.home_agent.const import (
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_DELAY,
    CONF_MEMORY_EXTRACTION_MODE,
    CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH,
    DEFAULT_MEMORY_EXTRACTION_DELAY,
    DEFAULT_MEMORY_EXTRACTION_MODE,
    DEFAULT_MEMORY_EXTRACTION_MIN_TURN_LENGTH,
    MEMORY_EXTRACTION_MODE_IMMEDIATE,
    MEMORY_EXTRACTION_MODE_IDLE,
)


@pytest.fixture
def mock_home_agent():
    """Create a mock HomeAgent instance with MemoryExtractionMixin."""
    agent = MagicMock(spec=MemoryExtractionMixin)
    agent.hass = MagicMock()
    agent.config = {
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IDLE,
        CONF_MEMORY_EXTRACTION_DELAY: 30,
        CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH: 4,
    }
    agent._memory_validator = MagicMock()
    agent._extraction_timer = None
    agent._pending_memories = []
    agent._conversation_active = False
    agent.memory_manager = MagicMock()
    
    # Bind mixin methods to the mock
    for method_name in dir(MemoryExtractionMixin):
        if not method_name.startswith('_') or method_name in [
            '_is_trivial_turn', '_get_extraction_mode', '_get_extraction_delay',
            '_get_min_turn_length', '_cancel_pending_extraction', '_schedule_extraction',
            '_execute_scheduled_extraction', '_extract_and_store_memories'
        ]:
            method = getattr(MemoryExtractionMixin, method_name)
            if callable(method):
                bound_method = MagicMock()
                bound_method.side_effect = lambda *args, method=method: method(agent, *args)
                setattr(agent, method_name, bound_method)
    
    return agent


class TestExtractionModeConfiguration:
    """Test extraction mode configuration retrieval."""

    def test_get_extraction_mode_default(self):
        """Test default extraction mode is idle."""
        mixin = MemoryExtractionMixin()
        mixin.config = {}
        
        mode = mixin._get_extraction_mode()
        assert mode == DEFAULT_MEMORY_EXTRACTION_MODE

    def test_get_extraction_mode_immediate(self):
        """Test immediate extraction mode."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IMMEDIATE}
        
        mode = mixin._get_extraction_mode()
        assert mode == MEMORY_EXTRACTION_MODE_IMMEDIATE

    def test_get_extraction_mode_idle(self):
        """Test idle extraction mode."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IDLE}
        
        mode = mixin._get_extraction_mode()
        assert mode == MEMORY_EXTRACTION_MODE_IDLE



class TestExtractionDelayConfiguration:
    """Test extraction delay configuration retrieval."""

    def test_get_extraction_delay_default(self):
        """Test default extraction delay."""
        mixin = MemoryExtractionMixin()
        mixin.config = {}
        
        delay = mixin._get_extraction_delay()
        assert delay == 30

    def test_get_extraction_delay_custom(self):
        """Test custom extraction delay."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: 60}
        
        delay = mixin._get_extraction_delay()
        assert delay == 60

    def test_get_extraction_delay_below_min_clamped(self):
        """Test that delay below minimum is clamped to 10."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: 5}
        
        delay = mixin._get_extraction_delay()
        assert delay == 10

    def test_get_extraction_delay_zero_clamped(self):
        """Test that zero delay is clamped to 10."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: 0}
        
        delay = mixin._get_extraction_delay()
        assert delay == 10

    def test_get_extraction_delay_negative_clamped(self):
        """Test that negative delay is clamped to 10."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: -5}
        
        delay = mixin._get_extraction_delay()
        assert delay == 10

    def test_get_extraction_delay_above_max_clamped(self):
        """Test that delay above maximum is clamped to 300."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: 500}
        
        delay = mixin._get_extraction_delay()
        assert delay == 300

    def test_get_extraction_delay_at_bounds(self):
        """Test that values at bounds are accepted."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: 10}
        assert mixin._get_extraction_delay() == 10

        mixin.config = {CONF_MEMORY_EXTRACTION_DELAY: 300}
        assert mixin._get_extraction_delay() == 300

class TestMinTurnLengthConfiguration:
    """Test minimum turn length configuration retrieval."""

    def test_get_min_turn_length_default(self):
        """Test default minimum turn length."""
        mixin = MemoryExtractionMixin()
        mixin.config = {}
        
        min_length = mixin._get_min_turn_length()
        assert min_length == DEFAULT_MEMORY_EXTRACTION_MIN_TURN_LENGTH

    def test_get_min_turn_length_custom(self):
        """Test custom minimum turn length."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH: 10}
        
        min_length = mixin._get_min_turn_length()
        assert min_length == 10

    def test_get_min_turn_length_disabled(self):
        """Test disabled minimum turn length (0)."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH: 0}
        
        min_length = mixin._get_min_turn_length()
        assert min_length == 0


class TestTrivialTurnDetection:
    """Test trivial turn detection (word count only, language-agnostic)."""

    def test_is_trivial_turn_length_filter(self):
        """Test turn length filtering."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH: 10}
        
        # Short turn should be trivial
        assert mixin._is_trivial_turn("Turn on lights", "OK") is True
        
        # Longer turn should not be trivial
        assert mixin._is_trivial_turn(
            "I prefer the bedroom temperature to be set to 68 degrees Fahrenheit for sleeping comfort",
            "I've noted your preference for 68°F in the bedroom for sleeping."
        ) is False

    def test_is_not_trivial_meaningful_conversation(self):
        """Test that meaningful conversations are not trivial."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH: 4}
        
        assert mixin._is_trivial_turn(
            "My birthday is March 15th and I work from home on Tuesdays",
            "I've saved that your birthday is March 15th and you work from home on Tuesdays."
        ) is False

        assert mixin._is_trivial_turn(
            "The kitchen has three ceiling lights and a pendant light over the island",
            "I've noted the kitchen lighting setup with three ceiling lights and a pendant."
        ) is False

    def test_is_trivial_disabled(self):
        """Test that trivial detection can be disabled with min_length=0."""
        mixin = MemoryExtractionMixin()
        mixin.config = {CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH: 0}
        
        # With min_length=0, nothing should be considered trivial
        assert mixin._is_trivial_turn("Hi", "Hi!") is False
        assert mixin._is_trivial_turn("OK", "OK") is False

    # CJK-aware token counting tests moved to test_helpers.py
    # (TestCountMeaningfulWords) — the shared implementation lives in helpers.py


class TestExtractionScheduling:
    """Test extraction scheduling with debounce."""

    def test_schedule_extraction_immediate_mode(self):
        """Test immediate extraction mode schedules with delay=1."""
        mixin = MemoryExtractionMixin()
        mixin.hass = MagicMock()
        mixin.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IMMEDIATE,
            CONF_MEMORY_EXTRACTION_DELAY: 30,
            CONF_MEMORY_ENABLED: True,
        }
        mixin._memory_manager = MagicMock()

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "This is a test message with enough words",
            "assistant_response": "This is a test response with enough words",
            "full_messages": [],
        }

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_timer = MagicMock()
            mock_loop.return_value.call_later.return_value = mock_timer

            mixin._schedule_extraction(kwargs)

            # Should schedule with delay=1 (not use async_create_task directly)
            mock_loop.return_value.call_later.assert_called_once()
            call_args = mock_loop.return_value.call_later.call_args
            assert call_args[0][0] == 1  # delay=1 for immediate mode
            state = mixin._extraction_states.get("conv_123")
            assert state is not None
            assert state.timer == mock_timer

    def test_schedule_extraction_idle_mode_with_delay(self):
        """Test idle mode schedules with configured delay per conversation."""
        mixin = MemoryExtractionMixin()
        mixin.hass = MagicMock()
        mixin.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IDLE,
            CONF_MEMORY_EXTRACTION_DELAY: 30,
            CONF_MEMORY_ENABLED: True,
        }
        mixin._memory_manager = MagicMock()

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "This is a test message with enough words",
            "assistant_response": "This is a test response with enough words",
            "full_messages": [],
        }

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_timer = MagicMock()
            mock_loop.return_value.call_later.return_value = mock_timer

            mixin._schedule_extraction(kwargs)

            # Should schedule with configured delay
            call_args = mock_loop.return_value.call_later.call_args
            assert call_args[0][0] == 30  # configured delay
            state = mixin._extraction_states.get("conv_123")
            assert state is not None
            assert state.timer == mock_timer

    def test_schedule_extraction_idle_mode_separate_conversations(self):
        """Test that different conversations have separate timers."""
        mixin = MemoryExtractionMixin()
        mixin.hass = MagicMock()
        mixin.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IDLE,
            CONF_MEMORY_EXTRACTION_DELAY: 30,
            CONF_MEMORY_ENABLED: True,
        }
        mixin._memory_manager = MagicMock()

        kwargs_a = {
            "conversation_id": "conv_a",
            "user_message": "This is a hello message with enough words",
            "assistant_response": "This is a hi response with enough words",
            "full_messages": [],
        }
        kwargs_b = {
            "conversation_id": "conv_b",
            "user_message": "This is a goodbye message with enough words",
            "assistant_response": "This is a bye response with enough words",
            "full_messages": [],
        }

        with patch('asyncio.get_event_loop') as mock_loop:
            timer_a = MagicMock()
            timer_b = MagicMock()
            mock_loop.return_value.call_later.side_effect = [timer_a, timer_b]

            mixin._schedule_extraction(kwargs_a)
            mixin._schedule_extraction(kwargs_b)

            # Both states should exist independently
            state_a = mixin._extraction_states.get("conv_a")
            state_b = mixin._extraction_states.get("conv_b")
            assert state_a is not None and state_a.timer == timer_a
            assert state_b is not None and state_b.timer == timer_b
            assert len(mixin._extraction_states) == 2

    def test_schedule_extraction_idle_mode_reschedules_same_conv(self):
        """Test that a new turn for the same conv cancels the old timer."""
        mixin = MemoryExtractionMixin()
        mixin.hass = MagicMock()
        mixin.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IDLE,
            CONF_MEMORY_EXTRACTION_DELAY: 30,
            CONF_MEMORY_ENABLED: True,
        }
        mixin._memory_manager = MagicMock()

        kwargs_v1 = {
            "conversation_id": "conv_123",
            "user_message": "This is the first turn message with enough words",
            "assistant_response": "This is the first response with enough words",
            "full_messages": [],
        }
        kwargs_v2 = {
            "conversation_id": "conv_123",
            "user_message": "This is the second turn message with enough words",
            "assistant_response": "This is the second response with enough words",
            "full_messages": [],
        }

        with patch('asyncio.get_event_loop') as mock_loop:
            timer_v1 = MagicMock()
            timer_v1.cancelled.return_value = False  # Not yet cancelled
            timer_v2 = MagicMock()
            mock_loop.return_value.call_later.side_effect = [timer_v1, timer_v2]

            mixin._schedule_extraction(kwargs_v1)
            state = mixin._extraction_states.get("conv_123")
            assert state is not None and state.timer == timer_v1

            # New turn for same conv should cancel old timer and schedule new
            mixin._schedule_extraction(kwargs_v2)
            timer_v1.cancel.assert_called_once()
            assert state.timer == timer_v2



class TestCancelPendingExtraction:
    """Test canceling pending extraction timers per conversation."""

    def test_cancel_pending_extraction(self):
        """Test canceling pending extraction timer for a conversation."""
        mixin = MemoryExtractionMixin()
        mock_timer = MagicMock()
        mock_timer.cancelled.return_value = False
        state = MemoryExtractionState()
        state.timer = mock_timer
        mixin._extraction_states = {"conv_123": state}
        
        mixin._cancel_pending_extraction("conv_123")
        
        mock_timer.cancel.assert_called_once()
        assert mixin._extraction_states["conv_123"].timer is None

    def test_cancel_pending_extraction_already_cancelled(self):
        """Test canceling already cancelled timer."""
        mixin = MemoryExtractionMixin()
        mock_timer = MagicMock()
        mock_timer.cancelled.return_value = True
        state = MemoryExtractionState()
        state.timer = mock_timer
        mixin._extraction_states = {"conv_123": state}
        
        mixin._cancel_pending_extraction("conv_123")
        
        # Should not cancel again
        mock_timer.cancel.assert_not_called()
        # Timer entry is still removed
        assert mixin._extraction_states["conv_123"].timer is None

    def test_cancel_pending_extraction_none(self):
        """Test canceling when no timer exists."""
        mixin = MemoryExtractionMixin()
        mixin._extraction_states = {}
        
        # Should not raise error
        mixin._cancel_pending_extraction("conv_123")

    def test_cancel_pending_extraction_other_conv_unchanged(self):
        """Test that canceling one conv doesn't affect others."""
        mixin = MemoryExtractionMixin()
        timer_a = MagicMock()
        timer_a.cancelled.return_value = False
        timer_b = MagicMock()
        timer_b.cancelled.return_value = False
        state_a = MemoryExtractionState()
        state_a.timer = timer_a
        state_b = MemoryExtractionState()
        state_b.timer = timer_b
        mixin._extraction_states = {"conv_a": state_a, "conv_b": state_b}

        mixin._cancel_pending_extraction("conv_a")

        timer_a.cancel.assert_called_once()
        timer_b.cancel.assert_not_called()
        assert mixin._extraction_states["conv_b"].timer == timer_b


class TestTurnDedup:
    """Test simplified turn-based dedup in _schedule_extraction."""

    def test_is_same_turn_matching(self):
        """Test that identical (user, assistant) turns are detected."""
        mixin = MemoryExtractionMixin()
        state = MemoryExtractionState()
        state.last_turn = ("What's the weather?", "It's sunny today.")
        mixin._extraction_states = {"conv_123": state}
        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "What's the weather?",
            "assistant_response": "It's sunny today.",
        }
        assert mixin._is_same_turn(kwargs, "conv_123") is True

    def test_is_same_turn_different_user(self):
        """Test that different user message is not the same turn."""
        mixin = MemoryExtractionMixin()
        state = MemoryExtractionState()
        state.last_turn = ("What's the weather?", "It's sunny today.")
        mixin._extraction_states = {"conv_123": state}
        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "What's the temperature?",
            "assistant_response": "It's sunny today.",
        }
        assert mixin._is_same_turn(kwargs, "conv_123") is False

    def test_is_same_turn_different_assistant(self):
        """Test that different assistant response is not the same turn."""
        mixin = MemoryExtractionMixin()
        state = MemoryExtractionState()
        state.last_turn = ("What's the weather?", "It's sunny today.")
        mixin._extraction_states = {"conv_123": state}
        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "What's the weather?",
            "assistant_response": "It's cloudy today.",
        }
        assert mixin._is_same_turn(kwargs, "conv_123") is False

    def test_is_same_turn_no_previous(self):
        """Test that first turn is never considered a duplicate."""
        mixin = MemoryExtractionMixin()
        mixin._extraction_states = {}
        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "Hello",
            "assistant_response": "Hi there!",
        }
        assert mixin._is_same_turn(kwargs, "conv_123") is False

    def test_schedule_extraction_dedup_skips_immediate(self):
        """Test that dedup in immediate mode skips scheduling entirely."""
        mock_agent = MagicMock()
        mock_agent.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IMMEDIATE,
            CONF_MEMORY_ENABLED: True,
        }
        mixin = MemoryExtractionMixin()
        mixin.hass = mock_agent
        mixin.config = mock_agent.config
        mixin._memory_manager = MagicMock()
        state = MemoryExtractionState()
        state.last_turn = ("What's the weather?", "It's sunny today.")
        mixin._extraction_states = {"conv_123": state}

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "What's the weather?",
            "assistant_response": "It's sunny today.",
        }
        mixin._schedule_extraction(kwargs)

        # Should NOT create a timer because it's the same turn
        mock_agent.async_create_task.assert_not_called()
        assert state.timer is None

    def test_schedule_extraction_dedup_skips_idle(self):
        """Test that dedup in idle mode skips timer creation."""
        mock_agent = MagicMock()
        mock_agent.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IDLE,
            CONF_MEMORY_EXTRACTION_DELAY: 30,
            CONF_MEMORY_ENABLED: True,
        }
        mixin = MemoryExtractionMixin()
        mixin.hass = mock_agent
        mixin.config = mock_agent.config
        mixin._memory_manager = MagicMock()
        state = MemoryExtractionState()
        state.last_turn = ("What's the weather?", "It's sunny today.")
        mixin._extraction_states = {"conv_123": state}

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "What's the weather?",
            "assistant_response": "It's sunny today.",
        }
        mixin._schedule_extraction(kwargs)

        # No timer should be created for a duplicate turn
        assert state.timer is None

    def test_schedule_extraction_new_turn_records_and_schedules(self):
        """Test that a new turn records the turn and schedules extraction."""
        mock_agent = MagicMock()
        mock_agent.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IMMEDIATE,
            CONF_MEMORY_ENABLED: True,
        }
        mixin = MemoryExtractionMixin()
        mixin.hass = mock_agent
        mixin.config = mock_agent.config
        mixin._memory_manager = MagicMock()
        mixin._extraction_states = {}

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "What's the weather?",
            "assistant_response": "It's sunny today.",
            "full_messages": [],
        }

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_timer = MagicMock()
            mock_loop.return_value.call_later.return_value = mock_timer

            mixin._schedule_extraction(kwargs)

            # Turn should be recorded
            state = mixin._extraction_states.get("conv_123")
            assert state is not None
            assert state.last_turn == (
                "What's the weather?",
                "It's sunny today.",
            )
            # Timer should be scheduled (not async_create_task directly)
            mock_loop.return_value.call_later.assert_called_once()
            assert state.timer == mock_timer


class TestPendingOverwrite:
    """Test that only the latest pending turn is kept (overwrite, not accumulate)."""

    def test_immediate_mode_overwrites_pending_when_lock_held(self):
        """Test that timer callback overwrites pending kwargs when lock is held.
        
        With unified scheduling, _schedule_extraction always creates a timer.
        The lock check and pending storage happens in _execute_scheduled_extraction
        (the timer callback). When lock is held, it stores/overwrites pending kwargs.
        """
        mock_agent = MagicMock()
        mock_agent.config = {
            CONF_MEMORY_EXTRACTION_MODE: MEMORY_EXTRACTION_MODE_IMMEDIATE,
            CONF_MEMORY_ENABLED: True,
        }
        mixin = MemoryExtractionMixin()
        mixin.hass = mock_agent
        mixin.config = mock_agent.config
        mixin._memory_manager = MagicMock()

        # Create a lock that is held
        loop = asyncio.new_event_loop()
        lock = asyncio.Lock()
        loop.run_until_complete(lock.acquire())
        state = MemoryExtractionState()
        state.lock = lock
        mixin._extraction_states = {"conv_123": state}

        kwargs1 = {
            "conversation_id": "conv_123",
            "user_message": "This is the first message with enough words",
            "assistant_response": "This is the first response with enough words",
            "full_messages": [],
        }

        # Simulate timer callback executing while lock is held
        mixin._execute_scheduled_extraction(kwargs1)

        # Lock is held, so pending is stored (overwrite)
        assert state.pending_kwargs is not None
        assert state.pending_kwargs["user_message"] == "This is the first message with enough words"

        # Second callback with different content — should overwrite
        kwargs2 = {
            "conversation_id": "conv_123",
            "user_message": "This is the second message with enough words",
            "assistant_response": "This is the second response with enough words",
            "full_messages": [],
        }
        mixin._execute_scheduled_extraction(kwargs2)

        # Only the latest should be queued (overwrite semantics)
        assert state.pending_kwargs["user_message"] == "This is the second message with enough words"

        lock.release()

    def test_execute_scheduled_overwrites_pending_when_lock_held(self):
        """Test that timer callback overwrites pending when lock is held."""
        mock_agent = MagicMock()
        mixin = MemoryExtractionMixin()
        mixin.hass = mock_agent

        # Create a lock that is held
        loop = asyncio.new_event_loop()
        lock = asyncio.Lock()
        loop.run_until_complete(lock.acquire())
        state = MemoryExtractionState()
        state.lock = lock
        mixin._extraction_states = {"conv_123": state}

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "Timer message",
            "assistant_response": "Timer response",
            "full_messages": [],
        }
        mixin._execute_scheduled_extraction(kwargs)

        # Lock is held, so pending is overwritten and no task is created
        assert state.pending_kwargs is not None
        assert state.pending_kwargs["user_message"] == "Timer message"
        mock_agent.async_create_task.assert_not_called()

        lock.release()

    def test_execute_scheduled_creates_task_when_lock_free(self):
        """Test that timer callback creates task when lock is free."""
        mock_agent = MagicMock()
        mixin = MemoryExtractionMixin()
        mixin.hass = mock_agent
        mixin._extraction_states = {}

        kwargs = {
            "conversation_id": "conv_123",
            "user_message": "Timer message",
            "assistant_response": "Timer response",
            "full_messages": [],
        }
        mixin._execute_scheduled_extraction(kwargs)

        # Lock is free, so task is created
        mock_agent.async_create_task.assert_called_once()
        state = mixin._extraction_states.get("conv_123")
        assert state is None or state.pending_kwargs is None



