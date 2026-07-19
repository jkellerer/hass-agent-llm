"""Conversation history management for Home Agent.

This module provides conversation history storage and retrieval for maintaining
context across multiple turns in a conversation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable

from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.storage import Store

from .const import EVENT_HISTORY_SAVED, HISTORY_STORAGE_KEY

_LOGGER = logging.getLogger(__name__)

# Token estimation: rough estimate of ~4 characters per token
# This is a conservative estimate that works across most models
CHARS_PER_TOKEN = 4

# Storage version for migrations
# v1: simple {"role", "content", "timestamp"} messages
# v2: full OpenAI format with tool_calls, tool_call_id, name support
STORAGE_VERSION_HISTORY = 2

# Default debounce delay for saving (seconds)
DEFAULT_SAVE_DELAY = 5

# Maximum storage size in bytes (10 MB)
MAX_STORAGE_SIZE = 10 * 1024 * 1024


class ConversationHistoryManager:
    """Manage conversation history with token and message limits.

    Stores conversation history per conversation_id and provides methods to:
    - Add messages to conversation history
    - Retrieve recent history with limits
    - Clear specific or all conversations
    - Estimate and manage token usage
    - Persist history across Home Assistant restarts (optional)

    History can be stored in memory only or persisted to disk using Home Assistant's
    Store helper. When persistence is enabled, conversations are automatically saved
    with debouncing to reduce I/O operations.
    """

    def __init__(
        self,
        max_messages: int = 10,
        max_tokens: int | None = None,
        hass: HomeAssistant | None = None,
        persist: bool = False,
        storage_key: str = HISTORY_STORAGE_KEY,
        save_delay: int = DEFAULT_SAVE_DELAY,
    ) -> None:
        """Initialize the conversation history manager.

        Args:
            max_messages: Maximum number of messages to retain per conversation
            max_tokens: Maximum token count for conversation history (None = no limit)
            hass: Home Assistant instance (required for persistence)
            persist: Enable persistent storage across restarts
            storage_key: Storage key for persistence (default: from const.py)
            save_delay: Debounce delay in seconds before saving (default: 5)
        """
        self._histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._hass = hass
        self._persist = persist
        self._storage_key = storage_key
        self._save_delay = save_delay

        # Storage and debouncing state
        self._store: Store | None = None
        self._save_task: asyncio.Task | None = None
        self._pending_save = False
        self._cleanup_listener: Callable[[], None] | None = None

        # Initialize storage if persistence is enabled
        if self._persist:
            if not self._hass:
                _LOGGER.warning(
                    "Persistence enabled but no Home Assistant instance provided. "
                    "History will be stored in memory only."
                )
                self._persist = False
            else:
                self._store = Store(
                    self._hass,
                    STORAGE_VERSION_HISTORY,
                    self._storage_key,
                )
                _LOGGER.debug(
                    "Initialized persistent storage at %s",
                    self._storage_key,
                )

        _LOGGER.debug(
            "Initialized ConversationHistoryManager with max_messages=%d, max_tokens=%s, "
            "persist=%s, save_delay=%ds",
            max_messages,
            max_tokens,
            persist,
            save_delay,
        )

    async def load_from_storage(self) -> None:
        """Load conversation history from persistent storage.

        Loads all conversations from storage and populates the in-memory history.
        Handles corrupted storage gracefully by logging errors and continuing.
        Supports storage version migrations.

        Raises:
            RuntimeError: If persistence is not enabled or Store is not initialized
        """
        if not self._persist or not self._store:
            _LOGGER.debug("Persistence not enabled, skipping load from storage")
            return

        try:
            data = await self._store.async_load()

            if data is None:
                _LOGGER.debug("No existing conversation history found in storage")
                return

            # Check storage version and migrate if needed
            storage_version = data.get("version", 1)
            if storage_version > STORAGE_VERSION_HISTORY:
                _LOGGER.warning(
                    "Storage version %d is newer than supported version %d. "
                    "Some data may not load correctly.",
                    storage_version,
                    STORAGE_VERSION_HISTORY,
                )
            elif storage_version < STORAGE_VERSION_HISTORY:
                _LOGGER.info(
                    "Migrating storage from version %d to %d",
                    storage_version,
                    STORAGE_VERSION_HISTORY,
                )
                data = await self._migrate_storage(storage_version, data)

            # Load conversations
            conversations = data.get("conversations", {})
            loaded_count = 0
            total_messages = 0

            for conv_id, messages in conversations.items():
                if not isinstance(messages, list):
                    _LOGGER.warning(
                        "Invalid message list for conversation %s, skipping",
                        conv_id,
                    )
                    continue

                # Validate and load messages
                valid_messages = []
                for msg in messages:
                    if not isinstance(msg, dict):
                        _LOGGER.warning("Invalid message format, skipping: %s", msg)
                        continue

                    if "role" not in msg or "content" not in msg:
                        _LOGGER.warning("Message missing role or content, skipping: %s", msg)
                        continue

                    valid_messages.append(msg)

                if valid_messages:
                    # Trim to max_messages to handle previously oversized histories
                    if self._max_messages is not None and len(valid_messages) > self._max_messages:
                        valid_messages = valid_messages[-self._max_messages :]
                    self._histories[conv_id] = valid_messages
                    loaded_count += 1
                    total_messages += len(valid_messages)

            _LOGGER.info(
                "Loaded %d conversations with %d total messages from storage",
                loaded_count,
                total_messages,
            )

        except Exception as error:  # pylint: disable=broad-except
            _LOGGER.error(
                "Failed to load conversation history from storage: %s",
                error,
                exc_info=True,
            )

    async def save_to_storage(self) -> None:
        """Save conversation history to persistent storage.

        Saves all conversations to storage in the format:
        {
            "version": 1,
            "conversations": {
                "conversation_id": [
                    {"role": "user", "content": "...", "timestamp": 1234567890},
                    ...
                ]
            }
        }

        Checks storage size and logs warnings if approaching limits.

        Raises:
            RuntimeError: If persistence is not enabled or Store is not initialized
        """
        if not self._persist or not self._store:
            _LOGGER.debug("Persistence not enabled, skipping save to storage")
            return

        try:
            # Prepare data for storage
            data = {
                "version": STORAGE_VERSION_HISTORY,
                "conversations": dict(self._histories),
            }

            # Check storage size (rough estimate)
            import json

            estimated_size = len(json.dumps(data))
            if estimated_size > MAX_STORAGE_SIZE:
                _LOGGER.warning(
                    "Conversation history size (%d bytes) exceeds recommended limit (%d bytes). "
                    "Consider reducing max_messages or clearing old conversations.",
                    estimated_size,
                    MAX_STORAGE_SIZE,
                )

            await self._store.async_save(data)

            # Count total messages
            total_messages = sum(len(msgs) for msgs in self._histories.values())

            # Fire history saved event
            if self._hass:
                try:
                    self._hass.bus.async_fire(
                        EVENT_HISTORY_SAVED,
                        {
                            "conversation_count": len(self._histories),
                            "message_count": total_messages,
                            "size_bytes": estimated_size,
                            "timestamp": int(time.time()),
                        },
                    )
                except Exception as event_err:
                    _LOGGER.warning("Failed to fire history saved event: %s", event_err)

            _LOGGER.debug(
                "Saved %d conversations to storage (~%d bytes)",
                len(self._histories),
                estimated_size,
            )

        except Exception as error:  # pylint: disable=broad-except
            _LOGGER.error(
                "Failed to save conversation history to storage: %s",
                error,
                exc_info=True,
            )

    async def _debounced_save(self) -> None:
        """Debounced save to reduce I/O operations.

        Waits for save_delay seconds before actually saving. If another save
        is requested during the delay, the timer resets. This prevents excessive
        writes when messages are added in quick succession.
        """
        self._pending_save = True

        # Cancel any existing save task
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()

        async def _delayed_save() -> None:
            """Perform the actual save after delay."""
            try:
                await asyncio.sleep(self._save_delay)
                if self._pending_save:
                    await self.save_to_storage()
                    self._pending_save = False
            except asyncio.CancelledError:
                _LOGGER.debug("Save task cancelled, will be rescheduled")
            except Exception as error:  # pylint: disable=broad-except
                _LOGGER.error("Error in debounced save: %s", error, exc_info=True)
                self._pending_save = False

        self._save_task = asyncio.create_task(_delayed_save())

    def enable_persistence(self, persist: bool) -> None:
        """Enable or disable persistence.

        Args:
            persist: True to enable persistence, False to disable

        Note:
            Enabling persistence requires a Home Assistant instance to have been
            provided during initialization. Disabling persistence will not delete
            existing stored data.
        """
        if persist and not self._hass:
            _LOGGER.warning(
                "Cannot enable persistence without Home Assistant instance. "
                "Initialize with hass parameter to enable persistence."
            )
            return

        old_persist = self._persist
        self._persist = persist

        if persist and not self._store and self._hass:
            self._store = Store(
                self._hass,
                STORAGE_VERSION_HISTORY,
                self._storage_key,
            )

        _LOGGER.info(
            "Persistence %s (was: %s)",
            "enabled" if persist else "disabled",
            "enabled" if old_persist else "disabled",
        )

    async def _migrate_storage(self, old_version: int, data: dict[str, Any]) -> dict[str, Any]:
        """Migrate storage from old version to current version.

        Args:
            old_version: The old storage version
            data: The data in old format

        Returns:
            Migrated data in current format

        Note:
            v1 → v2: Simple {"role", "content"} messages are forward-compatible.
            v2 adds support for tool_calls, tool_call_id, and name fields.
            Existing v1 messages remain valid as a subset of v2 format.
        """
        _LOGGER.info(
            "Migrating conversation history storage from version %d to %d",
            old_version,
            STORAGE_VERSION_HISTORY,
        )

        if old_version == 1:
            data = self._migrate_v1_to_v2(data)

        # Future migrations would go here
        # if old_version == 2:
        #     data = migrate_v2_to_v3(data)
        # if old_version == 3:
        #     data = migrate_v3_to_v4(data)

        if data["version"] < STORAGE_VERSION_HISTORY:
            _LOGGER.warning(
                "No migration path from version %d to %d, returning data as-is",
                old_version,
                STORAGE_VERSION_HISTORY,
            )
        return data
    
    def _migrate_v1_to_v2(self, data: dict[str, Any]) -> dict[str, Any]:
        # v1 messages only have {"role", "content", "timestamp"}
        # These are a valid subset of v2 format — no transformation needed.
        # Just update the version number.
        data["version"] = STORAGE_VERSION_HISTORY
        _LOGGER.debug(
            "Migrated %d conversations from v1 to v2 (no data transformation needed)",
            len(data.get("conversations", {})),
        )
        return data

    @staticmethod
    def _is_tool_message(msg: dict[str, Any]) -> bool:
        """Check if a message is a tool-related message (tool result or assistant with tool_calls).

        Tool messages are exempt from max_messages counting since they are part of
        the execution chain for a single conversation turn.

        Args:
            msg: Message dictionary

        Returns:
            True if the message is a tool result or an assistant message with tool_calls
        """
        if msg.get("role") == "tool":
            return True
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
        return False

    def _count_conversation_turns(self, messages: list[dict[str, Any]]) -> int:
        """Count the number of conversation turns (user/assistant text pairs), excluding tool messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Number of non-tool messages (user messages + assistant text responses)
        """
        return sum(1 for msg in messages if not self._is_tool_message(msg))

    def _trim_history(self, conversation_id: str) -> None:
        """Trim conversation history to stay within max_messages limit.

        Tool messages (role="tool" or assistant with tool_calls) are exempt from
        the count. Only user messages and assistant text responses count toward
        the max_messages limit. When trimming, complete turns are removed from
        the oldest end, preserving tool message chains intact.

        Args:
            conversation_id: Unique identifier for the conversation
        """
        if self._max_messages is None:
            return

        history = self._histories[conversation_id]
        turn_count = self._count_conversation_turns(history)

        if turn_count <= self._max_messages:
            return

        # Remove turns from the oldest end, preserving tool message chains.
        # We remove messages from the front until turn_count is within limits.
        messages_to_remove = 0
        for msg in history:
            if self._is_tool_message(msg):
                # Skip tool messages — they stay attached to their turn
                continue
            if turn_count - messages_to_remove > self._max_messages:
                messages_to_remove += 1
            else:
                break

        # Now find the actual index to cut at (skipping tool messages that
        # belong to turns we're keeping)
        non_tool_removed = 0
        cut_index = 0
        for i, msg in enumerate(history):
            if not self._is_tool_message(msg):
                non_tool_removed += 1
                if non_tool_removed > messages_to_remove:
                    cut_index = i
                    break
            cut_index = i + 1

        if cut_index > 0:
            del self._histories[conversation_id][:cut_index]
            _LOGGER.debug(
                "Trimmed conversation %s by %d messages (%d turns removed) to stay within limit",
                conversation_id,
                cut_index,
                messages_to_remove,
            )

    def add_message(
        self,
        conversation_id: str,
        message: dict[str, Any] | str,
        content: str | None = None,
    ) -> None:
        """Add a message to conversation history.

        Supports two calling conventions for backward compatibility:

        1. Full OpenAI dict format (new):
            >>> manager.add_message("conv_123", {"role": "user", "content": "Hello"})

        2. Simple role/content (legacy):
            >>> manager.add_message("conv_123", "user", "Hello")

        Args:
            conversation_id: Unique identifier for the conversation
            message: Either a message dict (with 'role' and 'content') or a role string
                     for legacy calls. Dict may include 'tool_calls', 'tool_call_id', 'name'.
            content: Content string (only used when message is a role string, for legacy calls)

        Example:
            Simple text message:
                >>> manager.add_message("conv_123", {"role": "user", "content": "Turn on the lights"})
                >>> manager.add_message("conv_123", "user", "Turn on the lights")  # legacy

            Assistant message with tool calls:
                >>> manager.add_message("conv_123", {
                ...     "role": "assistant",
                ...     "content": "",
                ...     "tool_calls": [{"id": "call_xxx", "type": "function", "function": {...}}]
                ... })

            Tool result message:
                >>> manager.add_message("conv_123", {
                ...     "role": "tool",
                ...     "tool_call_id": "call_xxx",
                ...     "content": '{"success": true}'
                ... })

        Note:
            If persistence is enabled, this will trigger a debounced save to storage.
        """
        if not conversation_id:
            _LOGGER.warning("Attempted to add message with empty conversation_id")
            return

        # Handle legacy (conversation_id, role, content) signature
        if isinstance(message, str):
            role = message
            msg_content = content
            if not msg_content:
                _LOGGER.warning("Attempted to add empty message to conversation %s", conversation_id)
                return
            message = {"role": role, "content": msg_content}
        elif not isinstance(message, dict):
            _LOGGER.warning("Message must be a dictionary or role string, got %s", type(message))
            return

        if "role" not in message:
            _LOGGER.warning("Message missing 'role' field: %s", message)
            return

        # Tool messages, tool results, and assistant messages with tool_calls may have empty content — that's valid
        has_content = message.get("content")
        has_tool_calls = message.get("tool_calls")
        is_tool = message.get("role") == "tool"
        if not has_content and not has_tool_calls and not is_tool:
            _LOGGER.warning("Attempted to add empty message to conversation %s", conversation_id)
            return

        msg_with_timestamp = dict(message)
        msg_with_timestamp["timestamp"] = int(time.time())

        self._histories[conversation_id].append(msg_with_timestamp)

        # Trim to max_messages to prevent unbounded growth (tool messages exempt)
        self._trim_history(conversation_id)

        _LOGGER.debug(
            "Added %s message to conversation %s (now %d messages, %d turns)",
            message.get("role"),
            conversation_id,
            len(self._histories[conversation_id]),
            self._count_conversation_turns(self._histories[conversation_id]),
        )

        # Trigger debounced save if persistence is enabled
        if self._persist and self._hass:
            asyncio.create_task(self._debounced_save())

    def add_messages(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        """Add multiple messages to conversation history in one call.

        Used to save a complete conversation turn including tool calls and results
        in the correct order.

        Args:
            conversation_id: Unique identifier for the conversation
            messages: List of message dictionaries (OpenAI format)

        Example:
            >>> manager.add_messages("conv_123", [
            ...     {"role": "user", "content": "What's the weather?"},
            ...     {"role": "assistant", "content": "", "tool_calls": [...]},
            ...     {"role": "tool", "tool_call_id": "call_x", "content": "Sunny, 25°C"},
            ...     {"role": "assistant", "content": "It's sunny and 25°C"},
            ... ])

        Note:
            If persistence is enabled, a single debounced save is triggered after
            all messages are added.
        """
        if not conversation_id:
            _LOGGER.warning("Attempted to add messages with empty conversation_id")
            return

        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                _LOGGER.warning("Skipping invalid message: %s", msg)
                continue

            if "role" not in msg:
                _LOGGER.warning("Skipping message missing 'role': %s", msg)
                continue

            msg_with_timestamp = dict(msg)
            msg_with_timestamp["timestamp"] = int(time.time())
            self._histories[conversation_id].append(msg_with_timestamp)

        # Trim once after all messages are added
        self._trim_history(conversation_id)

        _LOGGER.debug(
            "Added %d messages to conversation %s (now %d messages, %d turns)",
            len(messages),
            conversation_id,
            len(self._histories[conversation_id]),
            self._count_conversation_turns(self._histories[conversation_id]),
        )

        # Trigger debounced save if persistence is enabled
        if self._persist and self._hass:
            asyncio.create_task(self._debounced_save())

    def _strip_timestamp(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the message without the internal timestamp field.

        Preserves all OpenAI-compatible fields: role, content, tool_calls,
        tool_call_id, name — anything that the LLM API expects.

        Args:
            msg: Message dictionary (may contain 'timestamp')

        Returns:
            Message dict without 'timestamp'
        """
        result = dict(msg)
        result.pop("timestamp", None)
        return result

    def get_history(
        self,
        conversation_id: str,
        max_messages: int | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get conversation history with optional limits.

        Retrieves recent conversation history, applying message and token limits.
        If both limits are specified, the more restrictive one is applied.

        Returns full OpenAI-format messages including tool_calls, tool_call_id,
        and name fields (when present). Only the internal 'timestamp' field is
        stripped.

        Args:
            conversation_id: Unique identifier for the conversation
            max_messages: Override default max messages limit (None = use default)
            max_tokens: Override default max tokens limit (None = use default)

        Returns:
            List of message dictionaries in OpenAI format,
            in chronological order (oldest first)

        Example:
            >>> manager = ConversationHistoryManager(max_messages=10)
            >>> manager.add_message("conv_123", {"role": "user", "content": "Hello"})
            >>> manager.add_message("conv_123", {"role": "assistant", "content": "Hi!"})
            >>> history = manager.get_history("conv_123")
            >>> len(history)
            2
        """
        if conversation_id not in self._histories:
            _LOGGER.debug("No history found for conversation %s", conversation_id)
            return []

        history = self._histories[conversation_id]

        # Apply message limit (tool messages are exempt from the count)
        effective_max_messages = max_messages if max_messages is not None else self._max_messages
        if effective_max_messages is not None:
            turn_count = self._count_conversation_turns(history)
            if turn_count > effective_max_messages:
                # Trim from the front, preserving tool message chains
                turns_to_remove = turn_count - effective_max_messages
                non_tool_removed = 0
                cut_index = 0
                for i, msg in enumerate(history):
                    if not self._is_tool_message(msg):
                        non_tool_removed += 1
                        if non_tool_removed > turns_to_remove:
                            cut_index = i
                            break
                    cut_index = i + 1

                if cut_index > 0:
                    history = history[cut_index:]
                    _LOGGER.debug(
                        "Truncated conversation %s to %d turns (from %d, removed %d messages)",
                        conversation_id,
                        self._count_conversation_turns(history),
                        turn_count,
                        cut_index,
                    )

        # Apply token limit
        effective_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        if effective_max_tokens is not None:
            history = self._truncate_by_tokens(history, effective_max_tokens)

        # Strip internal timestamp to maintain OpenAI compatibility
        return [self._strip_timestamp(msg) for msg in history]

    def clear_history(self, conversation_id: str) -> None:
        """Clear history for a specific conversation.

        Args:
            conversation_id: Unique identifier for the conversation to clear

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.clear_history("conv_123")
            >>> manager.get_history("conv_123")
            []

        Note:
            If persistence is enabled, this will trigger a debounced save to storage.
        """
        if conversation_id in self._histories:
            message_count = len(self._histories[conversation_id])
            del self._histories[conversation_id]
            _LOGGER.info("Cleared conversation %s (%d messages)", conversation_id, message_count)

            # Trigger debounced save if persistence is enabled
            if self._persist and self._hass:
                asyncio.create_task(self._debounced_save())
        else:
            _LOGGER.debug("Attempted to clear non-existent conversation %s", conversation_id)

    def clear_all(self) -> None:
        """Clear all conversation histories.

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_456", "user", "Hi")
            >>> manager.clear_all()
            >>> len(manager.get_all_conversation_ids())
            0

        Note:
            If persistence is enabled, this will trigger a debounced save to storage.
        """
        conversation_count = len(self._histories)
        total_messages = sum(len(history) for history in self._histories.values())

        self._histories.clear()

        _LOGGER.info(
            "Cleared all conversation histories (%d conversations, %d total messages)",
            conversation_count,
            total_messages,
        )

        # Trigger debounced save if persistence is enabled
        if self._persist and self._hass:
            asyncio.create_task(self._debounced_save())

    def get_all_conversation_ids(self) -> list[str]:
        """Get list of all conversation IDs with stored history.

        Returns:
            List of conversation IDs

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_456", "user", "Hi")
            >>> sorted(manager.get_all_conversation_ids())
            ['conv_123', 'conv_456']
        """
        return list(self._histories.keys())

    def get_message_count(self, conversation_id: str) -> int:
        """Get the number of messages in a conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            Number of messages in the conversation

        Example:
            >>> manager = ConversationHistoryManager()
            >>> manager.add_message("conv_123", "user", "Hello")
            >>> manager.add_message("conv_123", "assistant", "Hi!")
            >>> manager.get_message_count("conv_123")
            2
        """
        return len(self._histories.get(conversation_id, []))

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count for a list of messages.

        Uses a conservative estimate of ~4 characters per token, which works
        across most models. For more accurate token counting, consider using
        a model-specific tokenizer (e.g., tiktoken for OpenAI models).

        Accounts for tool_calls, tool_call_id, and name fields in addition
        to role and content.

        Args:
            messages: List of message dictionaries (OpenAI format)

        Returns:
            Estimated token count

        Example:
            >>> manager = ConversationHistoryManager()
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> manager.estimate_tokens(messages) > 0
            True
        """
        import json

        total_chars = 0
        for message in messages:
            # Count role characters
            total_chars += len(message.get("role", ""))
            # Count content characters
            total_chars += len(str(message.get("content", "")))
            # Add overhead for message structure (role/content keys, etc.)
            total_chars += 20

            # Account for tool_calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                try:
                    total_chars += len(json.dumps(tool_calls))
                except (TypeError, ValueError):
                    total_chars += len(str(tool_calls))

            # Account for tool_call_id and name (tool result messages)
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                total_chars += len(tool_call_id) + 10  # field name overhead

            name = message.get("name")
            if name:
                total_chars += len(name) + 10  # field name overhead

        estimated_tokens = total_chars // CHARS_PER_TOKEN

        _LOGGER.debug(
            "Estimated %d tokens for %d messages (%d chars)",
            estimated_tokens,
            len(messages),
            total_chars,
        )

        return estimated_tokens

    def _truncate_by_tokens(
        self, history: list[dict[str, Any]], max_tokens: int
    ) -> list[dict[str, Any]]:
        """Truncate history to fit within token limit.

        Removes oldest messages until the history fits within max_tokens.
        Always keeps at least the most recent message pair (user + assistant).

        Args:
            history: Full conversation history
            max_tokens: Maximum token count

        Returns:
            Truncated history that fits within token limit
        """
        if not history:
            return []

        # Start from the end and work backwards
        truncated: list[dict[str, str]] = []
        current_tokens = 0

        for message in reversed(history):
            message_tokens = self.estimate_tokens([message])

            if current_tokens + message_tokens <= max_tokens:
                truncated.insert(0, message)
                current_tokens += message_tokens
            else:
                # Stop adding messages if we've exceeded the limit
                # but ensure we keep at least one message
                if not truncated:
                    truncated.insert(0, message)
                break

        if len(truncated) < len(history):
            _LOGGER.debug(
                "Truncated history from %d to %d messages to fit %d token limit "
                "(estimated %d tokens)",
                len(history),
                len(truncated),
                max_tokens,
                current_tokens,
            )

        return truncated

    def update_limits(self, max_messages: int | None = None, max_tokens: int | None = None) -> None:
        """Update the default limits for conversation history.

        Args:
            max_messages: New max messages limit (None = don't change)
            max_tokens: New max tokens limit (None = don't change)

        Example:
            >>> manager = ConversationHistoryManager(max_messages=5)
            >>> manager.update_limits(max_messages=10, max_tokens=2000)
        """
        if max_messages is not None:
            old_max = self._max_messages
            self._max_messages = max_messages
            _LOGGER.info("Updated max_messages from %s to %d", old_max, max_messages)
            # Trim existing conversations to new limit
            for conv_id in self._histories:
                if len(self._histories[conv_id]) > max_messages:
                    del self._histories[conv_id][:-max_messages]

        if max_tokens is not None:
            old_max_tokens = self._max_tokens
            self._max_tokens = max_tokens
            _LOGGER.info("Updated max_tokens from %s to %d", old_max_tokens, max_tokens)

    def setup_scheduled_cleanup(self) -> None:
        """Start periodic cleanup of old conversations.

        Schedules automatic cleanup of conversations older than 24 hours to run
        every hour. Only operates when persistence is enabled.
        """
        if self._persist and self._hass:
            self._cleanup_listener = async_track_time_interval(
                self._hass,
                self._async_cleanup_old_conversations,
                timedelta(hours=1),
            )
            _LOGGER.info("Scheduled conversation cleanup every hour")

    async def _async_cleanup_old_conversations(self, _now=None) -> None:
        """Remove conversations older than 24 hours.

        Args:
            _now: Unused parameter for compatibility with async_track_time_interval

        Note:
            Checks the timestamp of the most recent message in each conversation.
            If the most recent message is older than 24 hours, the entire
            conversation is removed.
        """
        cutoff = datetime.now() - timedelta(hours=24)
        cutoff_timestamp = int(cutoff.timestamp())
        to_delete = []

        for conv_id, messages in self._histories.items():
            if not messages:
                # Empty conversation, mark for deletion
                to_delete.append(conv_id)
                continue

            # Check the last message timestamp (most recent message)
            last_message = messages[-1]
            last_activity = last_message.get("timestamp", 0)

            if last_activity < cutoff_timestamp:
                to_delete.append(conv_id)

        # Delete old conversations
        for conv_id in to_delete:
            del self._histories[conv_id]

        if to_delete:
            await self.save_to_storage()
            _LOGGER.info("Cleaned up %d old conversations", len(to_delete))

    def shutdown_scheduled_cleanup(self) -> None:
        """Stop cleanup scheduler.

        Cancels the periodic cleanup task. Should be called during component shutdown
        to prevent cleanup from running after the component is unloaded.
        """
        if self._cleanup_listener:
            self._cleanup_listener()
            self._cleanup_listener = None
            _LOGGER.info("Stopped scheduled conversation cleanup")
