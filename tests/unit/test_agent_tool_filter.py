"""Unit tests for HomeAgent tool message filtering."""

import pytest

from custom_components.home_agent.agent import HomeAgent
from custom_components.home_agent.const import (
    CONF_HISTORY_RECORD_TOOL_CALLS,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_MODEL,
    DEFAULT_HISTORY_RECORD_TOOL_CALLS,
)


class TestFilterToolMessages:
    """Test _filter_tool_messages static method."""

    def test_filter_tool_messages_removes_tool_role(self):
        """Test that tool role messages are removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Let me check", "tool_calls": [{"id": "call_1", "function": {"name": "test"}, "type": "function"}]},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Done"},
        ]
        filtered = HomeAgent._filter_tool_messages(messages)
        assert len(filtered) == 2
        assert filtered[0] == {"role": "user", "content": "Hello"}
        assert filtered[1] == {"role": "assistant", "content": "Done"}

    def test_filter_tool_messages_keeps_normal_messages(self):
        """Test that normal messages are kept."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good!"},
        ]
        filtered = HomeAgent._filter_tool_messages(messages)
        assert len(filtered) == 4
        assert filtered == messages

    def test_filter_tool_messages_empty_list(self):
        """Test filtering empty message list."""
        filtered = HomeAgent._filter_tool_messages([])
        assert filtered == []

    def test_filter_tool_messages_all_tools(self):
        """Test filtering when all messages are tool related."""
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "function": {"name": "test"}, "type": "function"}]},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]
        filtered = HomeAgent._filter_tool_messages(messages)
        assert filtered == []

    def test_filter_tool_messages_multiple_tool_calls(self):
        """Test filtering with multiple tool call sequences."""
        messages = [
            {"role": "user", "content": "Turn on lights and check weather"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "function": {"name": "light_on"}, "type": "function"},
                {"id": "call_2", "function": {"name": "get_weather"}, "type": "function"}
            ]},
            {"role": "tool", "content": "done", "tool_call_id": "call_1"},
            {"role": "tool", "content": "sunny", "tool_call_id": "call_2"},
            {"role": "assistant", "content": "Lights are on and it's sunny."},
        ]
        filtered = HomeAgent._filter_tool_messages(messages)
        assert len(filtered) == 2
        assert filtered[0]["role"] == "user"
        assert filtered[1]["role"] == "assistant"
        assert filtered[1]["content"] == "Lights are on and it's sunny."


class TestHistoryRecordToolCallsConfig:
    """Test CONF_HISTORY_RECORD_TOOL_CALLS configuration."""

    def test_default_value_is_true(self):
        """Test that the default is True (backward compatible)."""
        assert DEFAULT_HISTORY_RECORD_TOOL_CALLS is True

    def test_config_key_exists(self):
        """Test that the config key is defined."""
        assert CONF_HISTORY_RECORD_TOOL_CALLS == "history_record_tool_calls"
