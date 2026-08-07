"""Memory extraction functionality mixin for HomeAgent.

This module provides the MemoryExtractionMixin class that automatically extracts
and stores long-term memories from conversations. It uses LLM analysis to identify
important information worth remembering, such as user preferences, facts about the
home, and contextual information that can improve future interactions.

Architecture:
    MemoryExtractionMixin is designed as a mixin class to be inherited by HomeAgent.
    It runs asynchronously after each conversation to extract memories without
    blocking the user's response. The extraction can use either the local LLM or
    an external LLM depending on configuration.

Key Classes:
    MemoryExtractionMixin: Mixin providing automatic memory extraction with:
        - LLM-powered analysis of conversation content
        - Structured extraction of facts, preferences, context, and events
        - Quality validation to filter low-value memories
        - Integration with MemoryManager for storage
        - Support for local or external LLM extraction
        - Event emission for observability

Core Responsibilities:
    - Analyze completed conversations for memorable information
    - Build extraction prompts with conversation context
    - Call LLM (local or external) to extract structured memories
    - Parse and validate extracted memory data
    - Filter out low-quality or transient information
    - Store validated memories via MemoryManager
    - Emit events for monitoring extraction success

Memory Types:
    The system extracts four types of memories:

    1. Facts: Concrete information about home, devices, or users
       Example: "User's birthday is September 28th, 1982"

    2. Preferences: User preferences for settings and behaviors
       Example: "User prefers bedroom temperature at 68°F for sleeping"

    3. Context: Background information for future interactions
       Example: "User works night shifts and sleeps during daytime"

    4. Events: Significant actions or occurrences
       Example: "Installed new smart lock on front door on 2024-01-15"

Extraction Flow:
    1. Conversation completes in HomeAgent
    2. _extract_and_store_memories() called asynchronously (fire and forget)
    3. Build extraction prompt with conversation history
    4. Call configured LLM (local or external) with extraction prompt
    5. LLM returns JSON array of memory objects
    6. Parse JSON and validate each memory against quality rules
    7. Store validated memories via MemoryManager
    8. Emit memory_extracted event with count

Quality Validation:
    Memories are validated against multiple quality criteria:

    - Minimum length: At least 10 meaningful words (>2 chars)
    - Importance threshold: Score >= 0.4
    - No transient states: Rejects "light is on", "temperature is 72°F"
    - No conversation metadata: Rejects "we discussed X", "user asked Y"
    - No negative statements: Rejects "there is no X", "does not have Y"
    - No timestamps: Rejects temporal references to conversation time
    - No generic statements: Rejects obvious capabilities without value

    See _build_extraction_prompt() for detailed rejection criteria.

Usage Example:
    The mixin is used through inheritance in HomeAgent:

        class HomeAgent(LLMMixin, StreamingMixin, MemoryExtractionMixin):
            async def process_message(self, text, conversation_id):
                # Process conversation normally
                response = await self._process_conversation(text, ...)

                # Extract memories asynchronously (fire and forget)
                if self.config.get(CONF_MEMORY_EXTRACTION_ENABLED, True):
                    self.hass.async_create_task(
                        self._extract_and_store_memories(
                            conversation_id=conversation_id,
                            user_message=text,
                            assistant_response=response,
                            full_messages=messages
                        )
                    )

                return response

Expected Host Class Attributes:
    The mixin expects the host class to provide:

    - hass: HomeAssistant
        Home Assistant instance for async task creation and event bus

    - config: dict[str, Any]
        Configuration dictionary containing:
        - CONF_MEMORY_ENABLED: Enable memory system (default: True)
        - CONF_MEMORY_EXTRACTION_ENABLED: Enable extraction (default: True)
        - CONF_MEMORY_EXTRACTION_LLM: "local" or "external" (default: "local")
        - CONF_EXTERNAL_LLM_ENABLED: Required if extraction_llm="external"
        - CONF_EMIT_EVENTS: Enable event emission (default: True)

    - tool_handler: ToolHandler
        Tool handler for executing query_external_llm tool

    - memory_manager: Any (property)
        MemoryManager instance for storing extracted memories

    - _call_llm(): Async method (from LLMMixin)
        Used for local LLM extraction

Integration Points:
    - MemoryManager: Stores validated memories with metadata
    - LLMMixin: Provides _call_llm() for local extraction
    - ToolHandler: Executes query_external_llm for external extraction
    - Home Assistant event bus: Emits memory_extracted events

Extraction Prompt Structure:
    The extraction prompt provides the LLM with:

    - Previous conversation history (context)
    - Current conversation turn (user + assistant messages)
    - Detailed instructions on memory types
    - Quality criteria and examples
    - Rejection patterns to avoid low-value memories
    - JSON schema for structured output

External vs Local LLM:
    Local LLM (default):
        - Uses the same LLM as conversations
        - Lower latency, no additional API calls
        - May have lower quality extraction with smaller models
        - Temperature set to 0.3 for consistency

    External LLM:
        - Uses specialized external LLM (e.g., GPT-4, Claude)
        - Higher quality extraction with larger models
        - Additional API costs and latency
        - Requires CONF_EXTERNAL_LLM_ENABLED: true

Memory Storage Format:
    Each memory is stored with:
        {
            "content": "User prefers bedroom at 68°F for sleeping",
            "type": "preference",  // fact, preference, context, event
            "importance": 0.8,  // 0.0 to 1.0
            "conversation_id": "conv_123",
            "metadata": {
                "entities_involved": ["climate.bedroom"],
                "topics": ["temperature", "bedroom", "sleep"],
                "extraction_method": "automatic"
            }
        }

Events:
    Emits memory_extracted event when successful:
        {
            "conversation_id": "conv_123",
            "memories_extracted": 3,
            "extraction_llm": "local",
            "timestamp": "2024-01-15T10:30:00"
        }

Configuration Example:
    # Use local LLM for extraction
    config = {
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_EXTRACTION_ENABLED: True,
        CONF_MEMORY_EXTRACTION_LLM: "local",
        CONF_EMIT_EVENTS: True
    }

    # Use external LLM for better quality
    config = {
        CONF_MEMORY_ENABLED: True,
        CONF_MEMORY_EXTRACTION_ENABLED: True,
        CONF_MEMORY_EXTRACTION_LLM: "external",
        CONF_EXTERNAL_LLM_ENABLED: True,
        CONF_EMIT_EVENTS: True
    }

Error Handling:
    - Extraction failures are logged but never block conversation responses
    - Invalid JSON responses are caught and logged
    - Individual memory storage failures don't stop other memories
    - Missing MemoryManager gracefully skips extraction
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..const import (
    CONF_EMIT_EVENTS,
    CONF_EXTERNAL_LLM_ENABLED,
    CONF_MEMORY_ENABLED,
    CONF_MEMORY_EXTRACTION_DELAY,
    CONF_MEMORY_EXTRACTION_LLM,
    CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH,
    CONF_MEMORY_EXTRACTION_MODE,
    CONF_MEMORY_MIN_IMPORTANCE,
    CONF_MEMORY_MIN_WORDS,
    DEFAULT_MEMORY_ENABLED,
    DEFAULT_MEMORY_EXTRACTION_DELAY,
    DEFAULT_MEMORY_EXTRACTION_LLM,
    DEFAULT_MEMORY_EXTRACTION_MIN_TURN_LENGTH,
    DEFAULT_MEMORY_EXTRACTION_MODE,
    DEFAULT_MEMORY_MIN_WORD_LENGTH,
    DEFAULT_MEMORY_MIN_WORDS,
    MEMORY_EXTRACTION_MODE_IMMEDIATE,
    MEMORY_EXTRACTION_MODE_IDLE,
    EVENT_MEMORY_EXTRACTED,
)
from ..helpers import count_meaningful_words, strip_thinking_blocks
from ..memory.validator import MemoryValidator

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from ..tool_handler import ToolHandler

_LOGGER = logging.getLogger(__name__)

# How many _get_extraction_state() calls to skip between eviction sweeps.
_EVICT_EVERY_N_CALLS = 20

# Entries not accessed within this many seconds are considered stale.
_EVICT_AFTER_SECONDS = 3600  # 1 hour


@dataclass
class MemoryExtractionState:
    """Per-conversation scheduling state for memory extraction.

    Holds everything that was previously spread across four separate dicts.
    The `last_used` timestamp enables time-based eviction of stale entries
    instead of insertion-order caps.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timer: asyncio.TimerHandle | None = None
    last_turn: tuple[str, str] | None = None
    pending_kwargs: dict[str, Any] | None = None
    last_used: float = field(default_factory=time.monotonic)

    def touch(self):
        """Update last_used to now."""
        self.last_used = time.monotonic()


class MemoryExtractionMixin:
    """Mixin providing memory extraction functionality.

    This mixin expects the following attributes from the host class:
    - hass: HomeAssistant - Home Assistant instance
    - config: dict[str, Any] - Configuration dictionary
    - tool_handler: ToolHandler - Tool handler instance
    - memory_manager: Any - Memory manager property
    """

    hass: "HomeAssistant"
    config: dict[str, Any]
    tool_handler: "ToolHandler"
    _memory_validator: MemoryValidator | None = None

    # Per-conversation scheduling state — single dict of state objects
    _extraction_states: dict[str, MemoryExtractionState]

    # Counter for periodic eviction (every _EVICT_EVERY_N_CALLS accesses)
    _extraction_state_access_count: int = 0

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize per-conversation state storage."""
        super().__init__(*args, **kwargs)
        self._init_memory_extraction()

    @property
    def memory_manager(self) -> Any:
        """Get memory manager (provided by host class).

        Returns _memory_manager if set, otherwise None.
        Host classes (e.g. HomeAgent) override this with proper initialization.
        """
        return getattr(self, "_memory_manager", None)

    def _init_memory_extraction(self):
        """Initialize per-conversation state storage for memory extraction.

        Uses hasattr guards so it's safe to call multiple times (lazy-init
        pattern for mixin scenarios where __init__ may not have run yet).
        """
        if not hasattr(self, "_extraction_states"):
            self._extraction_states = {}
        if not hasattr(self, "_extraction_state_access_count"):
            self._extraction_state_access_count = 0

    def _get_extraction_state(self, conversation_id: str) -> MemoryExtractionState:
        """Get or create the MemoryExtractionState for a conversation.

        Lazily creates a new state object if the conversation_id is unseen.
        Updates `last_used` on every access, and periodically evicts entries
        that have not been used for more than _EVICT_AFTER_SECONDS.

        Args:
            conversation_id: Conversation ID

        Returns:
            MemoryExtractionState for this conversation
        """
        self._extraction_state_access_count += 1

        if conversation_id not in self._extraction_states:
            self._extraction_states[conversation_id] = MemoryExtractionState()
        else:
            self._extraction_states[conversation_id].touch()

        # Periodic eviction of stale entries
        if self._extraction_state_access_count % _EVICT_EVERY_N_CALLS == 0:
            self._evict_stale_states()

        return self._extraction_states[conversation_id]

    def _evict_stale_states(self):
        """Evict conversation states not accessed for more than _EVICT_AFTER_SECONDS.

        Also cancels any pending timers for evicted conversations.
        """
        now = time.monotonic()
        cutoff = now - _EVICT_AFTER_SECONDS

        stale_keys = [
            cid for cid, state in self._extraction_states.items()
            if state.last_used < cutoff
        ]

        if not stale_keys:
            return

        for cid in stale_keys:
            state = self._extraction_states.pop(cid)
            if state.timer and not state.timer.cancelled():
                state.timer.cancel()

        _LOGGER.debug(
            "Evicted %d stale conversation states (cutoff: %ds)",
            len(stale_keys),
            _EVICT_AFTER_SECONDS,
        )

    def _get_extraction_mode(self) -> str:
        """Get the configured extraction mode.

        Returns:
            Extraction mode string (immediate or idle)
        """
        return self.config.get(
            CONF_MEMORY_EXTRACTION_MODE, DEFAULT_MEMORY_EXTRACTION_MODE
        )

    def _get_extraction_delay(self) -> int:
        """Get the configured extraction delay in seconds.

        Enforces bounds (10-300) at runtime for defense-in-depth, in case the
        config was loaded from storage bypassing schema validation.

        Returns:
            Delay in seconds (10-300), clamped to valid range
        """
        delay = self.config.get(
            CONF_MEMORY_EXTRACTION_DELAY, DEFAULT_MEMORY_EXTRACTION_DELAY
        )
        return max(10, min(300, delay))

    def _get_min_turn_length(self) -> int:
        """Get the configured minimum turn length.

        Returns:
            Minimum number of meaningful words (0-50)
        """
        return self.config.get(
            CONF_MEMORY_EXTRACTION_MIN_TURN_LENGTH,
            DEFAULT_MEMORY_EXTRACTION_MIN_TURN_LENGTH,
        )

    def _is_trivial_turn(self, user_message: str, assistant_response: str) -> bool:
        """Check if a conversation turn is trivial and should skip extraction.

        Uses a language-agnostic word count heuristic to filter very short turns.
        Counts user_message and assistant_response separately with early-exit
        once min_length is reached to avoid unnecessary CPU work.

        Args:
            user_message: User's message
            assistant_response: Assistant's response

        Returns:
            True if the turn is trivial, False if extraction should proceed
        """
        min_length = self._get_min_turn_length()

        if min_length <= 0:
            return False

        # Count tokens in user message first (early-exit if already sufficient)
        # min_word_length=2 allows 2+ char words for turn filtering
        word_count = count_meaningful_words(
            user_message, min_word_length=2, limit=min_length
        )

        if word_count >= min_length:
            return False

        # Add tokens from assistant response (only need remaining to reach threshold)
        remaining = min_length - word_count
        word_count += count_meaningful_words(
            assistant_response, min_word_length=2, limit=remaining
        )

        if word_count < min_length:
            _LOGGER.debug(
                "Skipping extraction: turn too short (%d words < %d minimum)",
                word_count,
                min_length,
            )
            return True

        return False

    def _is_same_turn(self, kwargs: dict[str, Any], conversation_id: str) -> bool:
        """Check if this is the same (user, assistant) turn as the last extraction.

        Uses a simple tuple of (user_message, assistant_response) for dedup
        instead of building the full prompt with history.  This catches the
        "same turn submitted twice" case without expensive prompt construction
        and avoids creating timer/task threads for duplicates entirely.

        Args:
            kwargs: Arguments to pass to _extract_and_store_memories
            conversation_id: Conversation ID

        Returns:
            True if this is the same turn as the last extraction attempt
        """
        user_message = kwargs.get("user_message", "")
        assistant_response = kwargs.get("assistant_response", "")
        current_turn = (user_message, assistant_response)
        state = self._extraction_states.get(conversation_id)
        last_turn = state.last_turn if state else None
        return last_turn is not None and last_turn == current_turn

    def _cancel_pending_extraction(self, conversation_id: str):
        """Cancel any pending extraction timer for a conversation.

        Args:
            conversation_id: Conversation ID to cancel timer for
        """
        state = self._extraction_states.get(conversation_id)
        if state is None:
            return
        timer = state.timer
        state.timer = None
        if timer and not timer.cancelled():
            timer.cancel()
            _LOGGER.debug("Cancelled pending memory extraction for %s", conversation_id)

    def _get_extraction_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get or create the asyncio.Lock for a conversation.

        Delegates to _get_extraction_state which handles lazy-init and
        periodic time-based eviction.

        Args:
            conversation_id: Conversation ID

        Returns:
            asyncio.Lock for this conversation
        """
        return self._get_extraction_state(conversation_id).lock

    def _execute_scheduled_extraction(self, kwargs: dict[str, Any]):
        """Execute scheduled memory extraction.

        Removes the timer entry and fires off the async extraction task.
        If extraction is already running under the lock, overwrite the pending
        kwargs so only the latest turn gets queued after the current one finishes.

        Args:
            kwargs: Arguments to pass to _extract_and_store_memories
        """
        conversation_id = kwargs.get("conversation_id", "unknown")
        state = self._extraction_states.get(conversation_id)
        if state:
            state.timer = None

        # If lock is held, overwrite pending kwargs (latest turn wins)
        if state and state.lock.locked():
            state.pending_kwargs = kwargs
            _LOGGER.debug(
                "Queued extraction (overwrite) for %s while lock held", conversation_id
            )
            return

        self.hass.async_create_task(self._extract_and_store_memories(**kwargs))

    def _schedule_extraction(self, kwargs: dict[str, Any]):
        """Schedule memory extraction with debounce, per conversation_id.

        Only one timer and one extraction run per conversation_id at a time.
        If extraction is already running, the new kwargs *overwrite* any
        previously pending kwargs — the later turn already sees the full
        message history, so keeping earlier queued turns is redundant.

        Both immediate and idle modes share the same code path — the only
        difference is the delay (1 second for immediate, configured value
        for idle).

        Args:
            kwargs: Arguments to pass to _extract_and_store_memories
        """
        conversation_id = kwargs.get("conversation_id", "unknown")

        # Lazy-init state dicts (mixin may be used before __init__ runs)
        self._init_memory_extraction()

        # Early exit: memory manager not available
        if not self.config.get(CONF_MEMORY_ENABLED, DEFAULT_MEMORY_ENABLED):
            return

        if self.memory_manager is None:
            _LOGGER.debug("Memory manager not available, skipping scheduling")
            return

        # Early exit: trivial turn (before creating any timer/task)
        user_message = kwargs.get("user_message", "")
        assistant_response = kwargs.get("assistant_response", "")
        if self._is_trivial_turn(user_message, assistant_response):
            return

        # Early dedup: skip if this is the same turn as the last attempt
        if self._is_same_turn(kwargs, conversation_id):
            _LOGGER.debug(
                "Skipping scheduling: same turn already extracted for %s", conversation_id
            )
            # Still overwrite pending in case extraction is running — the newer
            # call has the same content but may arrive while lock is held
            state = self._extraction_states.get(conversation_id)
            if state and state.lock.locked():
                state.pending_kwargs = kwargs
            return

        # Determine delay based on mode (immediate=1s, idle=configured delay)
        mode = self._get_extraction_mode()
        if mode == MEMORY_EXTRACTION_MODE_IMMEDIATE:
            delay = 1
        else:
            delay = self._get_extraction_delay()

        # Cancel any pending timer for this conversation
        self._cancel_pending_extraction(conversation_id)

        # Record the turn so the timer callback can still dedup
        state = self._get_extraction_state(conversation_id)
        state.last_turn = (user_message, assistant_response)

        # Schedule extraction with delay
        loop = asyncio.get_event_loop()
        state.timer = loop.call_later(
            delay,
            self._execute_scheduled_extraction,
            kwargs,
        )
        mode_label = "immediate" if mode == MEMORY_EXTRACTION_MODE_IMMEDIATE else "idle"
        _LOGGER.debug(
            "Scheduled memory extraction in %d seconds (%s mode) for %s",
            delay,
            mode_label,
            conversation_id,
        )

    @property
    def memory_validator(self) -> MemoryValidator:
        """Get or create the memory validator instance.

        Returns:
            MemoryValidator configured with current settings
        """
        if self._memory_validator is None:
            # Use config value if set, otherwise use 0.4 (historical default for extraction)
            # Note: DEFAULT_MEMORY_MIN_IMPORTANCE (0.3) is for storage, but extraction
            # has historically used 0.4 as a stricter threshold
            min_importance = self.config.get(CONF_MEMORY_MIN_IMPORTANCE, 0.4)
            min_word_count = self.config.get(CONF_MEMORY_MIN_WORDS, DEFAULT_MEMORY_MIN_WORDS)
            self._memory_validator = MemoryValidator(
                min_word_count=min_word_count,
                min_importance=min_importance,
                min_word_length=DEFAULT_MEMORY_MIN_WORD_LENGTH,
            )
        return self._memory_validator

    # This method is provided by LLMMixin
    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call the LLM API (provided by LLMMixin)."""
        ...

    def _format_conversation_for_extraction(self, messages: list[dict[str, Any]]) -> str:
        """Format conversation history for memory extraction.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted conversation text
        """
        formatted_parts = []

        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")

            # Skip system messages and empty messages
            if role.lower() == "system" or not content:
                continue

            # Skip tool messages
            if role.lower() == "tool":
                continue

            formatted_parts.append(f"{role}: {content}")

        return "\n".join(formatted_parts)

    def _build_extraction_prompt(
        self,
        user_message: str,
        assistant_response: str,
        full_messages: list[dict[str, Any]],
    ) -> str:
        """Build prompt for memory extraction.

        Args:
            user_message: Current user message
            assistant_response: Assistant's response
            full_messages: Complete conversation history

        Returns:
            Extraction prompt
        """
        # Format conversation history (exclude current turn)
        history_messages = [
            msg for msg in full_messages if msg.get("role") not in ["system", "tool"]
        ]

        # Get previous turns (exclude the current user message we just added)
        if history_messages and history_messages[-1].get("content") == user_message:
            previous_turns = history_messages[:-1]
        else:
            previous_turns = history_messages

        conversation_text = ""
        if previous_turns:
            conversation_text = self._format_conversation_for_extraction(previous_turns)

        prompt = f"""You are a memory extraction assistant. Analyze this conversation \
and extract important information that should be remembered for future conversations.

Extract the following types of information:
1. **Facts**: Concrete information about the home, devices, or user
2. **Preferences**: User preferences for temperature, lighting, routines, etc.
3. **Context**: Background information useful for future interactions
4. **Events**: Significant events or actions that occurred

## Instructions

Extract memories as a JSON array. Each memory should have:
- "type": One of "fact", "preference", "context", "event"
- "content": Clear, concise description (1-2 sentences)
- "importance": Score from 0.0 to 1.0 (1.0 = very important)
- "entities": List of Home Assistant entity IDs mentioned (if any)
- "topics": List of topic tags (e.g., ["temperature", "bedroom"])

**Importance Score Guidelines:**
- 0.9-1.0: Critical personal info (birthdays, allergies, security codes, medical needs)
- 0.7-0.8: Strong preferences with specific values (temperature 68°F, lights at 50%)
- 0.5-0.6: General preferences and useful context (prefers dim lights, works from home)
- 0.3-0.4: Minor details mentioned in passing (asked about a feature once)
- 0.1-0.2: Trivial or uncertain information (might want something someday)

**Critical Rules:**
- Only extract genuinely useful, long-term information
- Each memory must be at least 7 words long
- Be specific and concrete with actionable details
- If nothing worth remembering, return empty array: []

**NEVER extract (these will be automatically rejected):**
- ❌ Current device states: "light is on", "temperature is 72°F",
  "door is closed", "lights are currently on"
- ❌ Transient states: "is currently", "are now", "was on", "were off", "right now", "at the moment"
- ❌ Current time/clock: "the current time is 10:30 PM", "it is currently 8 AM", "the time is"
- ❌ Current weather: "weather is sunny", "it's raining",
  "forecast shows rain", "temperature outside is 65°F"
- ❌ Current date/day: "today is Monday", "this week", "this month", "it's Tuesday"
- ❌ Current location/presence: "user is home", "user is away", "just arrived", "nobody is home"
- ❌ Conversation meta-data: "conversation occurred at 3pm", "we discussed X", "user asked about Y"
- ❌ Negative statements: "there is no X", "no specific Y sensor", "does not have Z"
- ❌ Timestamps of conversation: "at 8:59 PM on November 4", "during the conversation"
- ❌ Generic statements: "the lights can be controlled", "temperature can be adjusted"
- ❌ Questions without answers: "user asked about temperature" (unless you provide the answer)
- ❌ Very brief statements under 7 words

**ALWAYS extract (these are valuable):**
- ✅ User preferences with values: "user prefers bedroom temperature at 68°F for sleeping"
- ✅ Permanent facts: "user's birthday is March 15th", "kitchen has 3 ceiling lights"
- ✅ Patterns and routines: "user wants kitchen lights at 50% brightness during daytime"
- ✅ Device capabilities (not states): "bedroom thermostat supports heating and cooling modes"
- ✅ Important context: "user works night shifts and sleeps during the day"
- ✅ Scheduled events: "user has a dentist appointment on January 15th at 2pm"

**Examples of REJECTED memories:**
- "The kitchen lights are currently on" → REJECT: transient device state
- "Conversation occurred at 20:59 PM" → REJECT: conversation metadata
- "There is no bed sensor in the guest room" → REJECT: negative statement
- "User asked about lights" → REJECT: question without answer, too brief
- "The current time is 10:29 PM" → REJECT: current time (changes constantly)
- "It's raining outside" → REJECT: current weather (ephemeral)
- "User is currently at home" → REJECT: current presence state
- "Today is Wednesday" → REJECT: current date (changes daily)

**Examples of GOOD memories:**
- "User prefers kitchen lights at 50% brightness during daytime hours" → GOOD: preference with value
- "User's birthday is March 15th" → GOOD: permanent fact
- "User works night shifts Mon-Fri and sleeps during daytime"
  → GOOD: important routine
- "User's anniversary is on June 20th" → GOOD: permanent date fact

Return ONLY valid JSON, no other text:

```json
[
  {{
    "type": "preference",
    "content": "User prefers bedroom temperature at 68°F for sleeping",
    "importance": 0.8,
    "entities": ["climate.bedroom"],
    "topics": ["temperature", "bedroom", "sleep"]
  }}
]

## Previous Conversation

{conversation_text if conversation_text else "(No previous conversation)"}

## Latest Turn

User: {user_message}
Assistant: {assistant_response}
```"""

        return prompt

    async def _call_primary_llm_for_extraction(self, extraction_prompt: str) -> dict[str, Any]:
        """Call primary/local LLM for memory extraction.

        Args:
            extraction_prompt: The extraction prompt

        Returns:
            Dictionary with success, result, and error fields
        """
        try:
            # Build simple message for extraction
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a memory extraction assistant. Extract important "
                        "information from conversations and return it as JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": extraction_prompt,
                },
            ]

            # Call LLM without tool definitions
            # Use lower temperature (0.3) for more consistent extraction
            response = await self._call_llm(
                messages,
                tools=None,
                temperature=0.3,
            )

            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

            return {
                "success": True,
                "result": content,
                "error": None,
            }

        except Exception as err:
            _LOGGER.error("Primary LLM extraction failed: %s", err)
            return {
                "success": False,
                "result": None,
                "error": str(err),
            }

    async def _parse_and_store_memories(
        self,
        extraction_result: str,
        conversation_id: str,
    ) -> int:
        """Parse LLM extraction result and store memories.

        Args:
            extraction_result: JSON string from LLM
            conversation_id: Conversation ID

        Returns:
            Number of memories stored
        """
        try:
            # Clean up the result - first strip thinking blocks from reasoning models
            result = strip_thinking_blocks(extraction_result) or ""

            # Then extract JSON if wrapped in markdown
            if "```json" in result:
                # Extract JSON from markdown code block
                start = result.find("```json") + 7
                end = result.find("```", start)
                result = result[start:end].strip()
            elif "```" in result:
                # Extract from generic code block
                start = result.find("```") + 3
                end = result.find("```", start)
                result = result[start:end].strip()

            # Parse JSON response
            memories = json.loads(result)

            if not isinstance(memories, list):
                _LOGGER.error(
                    "Expected JSON array from memory extraction, got: %s",
                    type(memories).__name__,
                )
                return 0

            if not memories:
                _LOGGER.debug("No memories extracted from conversation")
                return 0

            # Store each memory and track rejection reasons
            stored_count = 0
            rejection_counts: dict[str, int] = {}
            total_count = len(memories)

            for memory_data in memories:
                try:
                    # Validate memory using MemoryValidator
                    is_valid, rejection_reason = self.memory_validator.validate(memory_data)

                    if not is_valid:
                        content = (
                            memory_data.get("content", "")[:50]
                            if isinstance(memory_data, dict)
                            else str(memory_data)[:50]
                        )
                        _LOGGER.debug(
                            "Rejecting memory (%s): %s",
                            rejection_reason,
                            content,
                        )
                        # Track rejection reason
                        rejection_counts[rejection_reason] = (
                            rejection_counts.get(rejection_reason, 0) + 1
                        )
                        continue

                    content = memory_data["content"]
                    memory_type = memory_data.get("type", "fact")

                    memory_id = await self.memory_manager.add_memory(
                        content=content,
                        memory_type=memory_type,
                        conversation_id=conversation_id,
                        importance=memory_data.get("importance", 0.5),
                        metadata={
                            "entities_involved": memory_data.get("entities", []),
                            "topics": memory_data.get("topics", []),
                            "extraction_method": "automatic",
                        },
                    )
                    stored_count += 1
                    _LOGGER.debug("Stored memory %s: %s", memory_id, content[:50])

                except Exception as err:
                    _LOGGER.error("Failed to store memory: %s", err)
                    continue

            # Log summary statistics
            rejected_count = total_count - stored_count
            if rejected_count > 0:
                # Format rejection reasons as "reason1: count1, reason2: count2"
                rejection_summary = ", ".join(
                    f"{reason}: {count}" for reason, count in sorted(rejection_counts.items())
                )
                _LOGGER.info(
                    "Memory extraction: %d/%d stored, %d rejected (reasons: %s)",
                    stored_count,
                    total_count,
                    rejected_count,
                    rejection_summary,
                )
            elif stored_count > 0:
                _LOGGER.info(
                    "Memory extraction: %d/%d stored, 0 rejected",
                    stored_count,
                    total_count,
                )

            return stored_count

        except json.JSONDecodeError as err:
            _LOGGER.error("Failed to parse memory extraction JSON: %s", err)
            _LOGGER.debug("Raw extraction result: %s", extraction_result)
            return 0
        except Exception as err:
            _LOGGER.error("Error parsing and storing memories: %s", err)
            return 0

    async def _extract_and_store_memories(
        self,
        conversation_id: str,
        user_message: str,
        assistant_response: str,
        full_messages: list[dict[str, Any]],
    ) -> None:
        """Extract memories from completed conversation using configured LLM.

        This method:
        1. Acquires per-conversation lock (only one extraction at a time per conv)
        2. Determines which LLM to use (external or local)
        3. Builds extraction prompt
        4. Calls LLM to extract memories
        5. Parses JSON response
        6. Stores memories via MemoryManager
        7. After release, checks if there's a pending newer turn to process

        Prerequisites (memory_enabled, memory_manager, trivial turn, dedup)
        are checked upstream in _schedule_extraction before scheduling.

        Args:
            conversation_id: Conversation ID
            user_message: User's message
            assistant_response: Assistant's response
            full_messages: Complete conversation history
        """
        # Lazy-init per-conversation state (mixin may be used before __init__ runs)
        self._init_memory_extraction()

        lock = self._get_extraction_lock(conversation_id)

        async with lock:
            try:
                # Determine which LLM to use for extraction
                extraction_llm = self.config.get(
                    CONF_MEMORY_EXTRACTION_LLM, DEFAULT_MEMORY_EXTRACTION_LLM
                )

                # Build extraction prompt
                extraction_prompt = self._build_extraction_prompt(
                    user_message=user_message,
                    assistant_response=assistant_response,
                    full_messages=full_messages,
                )

                # Call appropriate LLM
                if extraction_llm == "external":
                    # Check if external LLM is enabled
                    if not self.config.get(CONF_EXTERNAL_LLM_ENABLED, False):
                        _LOGGER.warning(
                            "Memory extraction configured to use external LLM, "
                            "but external LLM is not enabled. Skipping extraction."
                        )
                        return

                    # Use external LLM tool
                    _LOGGER.debug("Using external LLM for memory extraction")
                    result = await self.tool_handler.execute_tool(
                        tool_name="query_external_llm",
                        parameters={"prompt": extraction_prompt},
                        conversation_id=conversation_id,
                    )

                    if not result.get("success"):
                        _LOGGER.error(
                            "External LLM memory extraction failed: %s",
                            result.get("error"),
                        )
                        return

                    extraction_result = result.get("result", "[]")

                else:
                    # Use local/primary LLM
                    _LOGGER.debug("Using local LLM for memory extraction")
                    result = await self._call_primary_llm_for_extraction(extraction_prompt)

                    if not result.get("success"):
                        _LOGGER.error(
                            "Local LLM memory extraction failed: %s",
                            result.get("error"),
                        )
                        return

                    extraction_result = result.get("result", "[]")

                # Parse and store memories
                stored_count = await self._parse_and_store_memories(
                    extraction_result=extraction_result,
                    conversation_id=conversation_id,
                )

                # Fire event if memories were extracted
                if stored_count > 0 and self.config.get(CONF_EMIT_EVENTS, True):
                    from datetime import datetime

                    self.hass.bus.async_fire(
                        EVENT_MEMORY_EXTRACTED,
                        {
                            "conversation_id": conversation_id,
                            "memories_extracted": stored_count,
                            "extraction_llm": extraction_llm,
                            "timestamp": datetime.now().isoformat(
                                timespec="seconds"
                            ),
                        },
                    )

            except Exception as err:
                _LOGGER.exception("Error during memory extraction: %s", err)
            finally:
                # After releasing lock, check if there's a pending newer turn.
                # Only one pending set of kwargs is kept (the most recent),
                # because the later turn already sees the full message history.
                #
                # NOTE: We intentionally do NOT remove the state object here —
                # the lock must persist to serialize concurrent extractions for
                # the same conversation, and last_turn is needed for dedup.
                # Stale entries are evicted by _evict_stale_states() based on
                # last_used age.
                state = self._extraction_states.get(conversation_id)
                if state:
                    pending_kwargs = state.pending_kwargs
                    state.pending_kwargs = None
                    if pending_kwargs is not None:
                        _LOGGER.debug(
                            "Processing deferred extraction for %s", conversation_id
                        )
                        self.hass.async_create_task(
                            self._extract_and_store_memories(**pending_kwargs)
                        )
