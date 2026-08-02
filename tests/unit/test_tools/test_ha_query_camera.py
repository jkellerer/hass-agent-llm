"""Unit tests for camera image support in HomeAssistantQueryTool."""

import base64
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import State

from custom_components.home_agent.tools.ha_query import (
    HomeAssistantQueryTool,
    IMAGE_CAPABLE_DOMAINS,
)


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = MagicMock()
    hass.states = MagicMock()
    hass.states.async_entity_ids = MagicMock(return_value=[])
    hass.states.get = MagicMock(return_value=None)
    hass.config = MagicMock()
    hass.config.internal_url = "http://localhost:8123"
    hass.config.external_url = "https://myhome.duckdns.org"
    return hass


@pytest.fixture
def camera_state():
    """Create a sample camera entity state with entity_picture."""
    return State(
        entity_id="camera.front_door",
        state="streaming",
        attributes={
            "friendly_name": "Front Door Camera",
            "brand": "Ring",
            "model": "Doorbell 3",
            "duration": 15,
            "entity_picture": (
                "/api/camera_proxy/camera.front_door"
                "?token=abc123def456"
            ),
        },
        last_changed=datetime.now(),
        last_updated=datetime.now(),
    )


@pytest.fixture
def camera_state_no_picture():
    """Create a sample camera entity state without entity_picture (fallback)."""
    return State(
        entity_id="camera.old_camera",
        state="streaming",
        attributes={
            "friendly_name": "Old Camera",
            "brand": "Generic",
        },
        last_changed=datetime.now(),
        last_updated=datetime.now(),
    )


@pytest.fixture
def image_state():
    """Create a sample image.* entity state (e.g. Roborock vacuum photo)."""
    return State(
        entity_id="image.vacuum_map",
        state="idle",
        attributes={
            "friendly_name": "Roborock QE JK JK Wohnung",
            "entity_picture": (
                "/api/image_proxy/image.vacuum_map"
                "?token=94d691783b3a8be109ed17073184f5bb4dff56445e10a8348289b68da63d52be"
            ),
        },
        last_changed=datetime.now(),
        last_updated=datetime.now(),
    )


@pytest.fixture
def media_player_state():
    """Create a sample media player entity state."""
    return State(
        entity_id="media_player.living_room_tv",
        state="idle",
        attributes={
            "friendly_name": "Living Room TV",
            "source": "HDMI 1",
        },
        last_changed=datetime.now(),
        last_updated=datetime.now(),
    )


@pytest.fixture
def light_state():
    """Create a sample light entity state (non-image entity)."""
    return State(
        entity_id="light.living_room",
        state="on",
        attributes={
            "friendly_name": "Living Room Light",
            "brightness": 200,
        },
        last_changed=datetime.now(),
        last_updated=datetime.now(),
    )


class TestImageCapableDomains:
    """Test the IMAGE_CAPABLE_DOMAINS constant."""

    def test_camera_domain_included(self):
        """Test that camera domain is included."""
        assert "camera" in IMAGE_CAPABLE_DOMAINS

    def test_media_player_domain_included(self):
        """Test that media_player domain is included."""
        assert "media_player" in IMAGE_CAPABLE_DOMAINS

    def test_image_domain_included(self):
        """Test that image domain is included."""
        assert "image" in IMAGE_CAPABLE_DOMAINS

    def test_light_domain_excluded(self):
        """Test that light domain is not included."""
        assert "light" not in IMAGE_CAPABLE_DOMAINS

    def test_sensor_domain_excluded(self):
        """Test that sensor domain is not included."""
        assert "sensor" not in IMAGE_CAPABLE_DOMAINS


class TestIsImageEntity:
    """Test the _is_image_entity helper method."""

    def test_camera_entity_detected(self, mock_hass):
        """Test that camera entities are detected as image-capable."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool._is_image_entity("camera.front_door") is True

    def test_media_player_entity_detected(self, mock_hass):
        """Test that media_player entities are detected as image-capable."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool._is_image_entity("media_player.tv") is True

    def test_light_entity_not_image(self, mock_hass):
        """Test that light entities are not image-capable."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool._is_image_entity("light.living_room") is False

    def test_sensor_entity_not_image(self, mock_hass):
        """Test that sensor entities are not image-capable."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool._is_image_entity("sensor.temperature") is False

    def test_switch_entity_not_image(self, mock_hass):
        """Test that switch entities are not image-capable."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool._is_image_entity("switch.power_outlet") is False

    def test_image_entity_detected(self, mock_hass):
        """Test that image.* entities are detected as image-capable."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert tool._is_image_entity("image.vacuum_map") is True


class TestGetImageUrl:
    """Test the unified _get_image_url method."""

    @pytest.mark.asyncio
    async def test_camera_with_entity_picture(self, mock_hass, camera_state):
        """Test camera URL uses entity_picture when available."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=camera_state)

        url = await tool._get_image_url("camera.front_door")

        assert url is not None
        assert "myhome.duckdns.org" in url
        assert "/api/camera_proxy/camera.front_door" in url
        assert "token=abc123def456" in url

    @pytest.mark.asyncio
    async def test_camera_without_entity_picture_fallback(self, mock_hass, camera_state_no_picture):
        """Test camera falls back to camera_proxy when no entity_picture."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=camera_state_no_picture)

        with patch("homeassistant.util.dt.utcnow") as mock_now:
            mock_now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            url = await tool._get_image_url("camera.old_camera")

        assert url is not None
        assert "myhome.duckdns.org" in url
        assert "/api/camera_proxy/camera.old_camera" in url

    @pytest.mark.asyncio
    async def test_image_with_entity_picture(self, mock_hass, image_state):
        """Test image.* URL uses entity_picture attribute."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=image_state)

        url = await tool._get_image_url("image.vacuum_map")

        assert url is not None
        assert "myhome.duckdns.org" in url
        assert "/api/image_proxy/image.vacuum_map" in url
        assert "token=" in url

    @pytest.mark.asyncio
    async def test_image_without_entity_picture_fallback(self, mock_hass):
        """Test image.* falls back to camera_proxy when no entity_picture."""
        state_no_pic = State(
            entity_id="image.no_pic",
            state="idle",
            attributes={"friendly_name": "No Pic"},
            last_changed=datetime.now(),
            last_updated=datetime.now(),
        )
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=state_no_pic)

        with patch("homeassistant.util.dt.utcnow") as mock_now:
            mock_now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            url = await tool._get_image_url("image.no_pic")

        assert url is not None
        assert "/api/camera_proxy/image.no_pic" in url

    @pytest.mark.asyncio
    async def test_unavailable_entity(self, mock_hass):
        """Test URL returns None for unavailable entities."""
        unavailable_state = State(
            entity_id="camera.broken",
            state="unavailable",
            attributes={"entity_picture": "/api/camera_proxy/camera.broken"},
            last_changed=datetime.now(),
            last_updated=datetime.now(),
        )
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=unavailable_state)

        url = await tool._get_image_url("camera.broken")

        assert url is None

    @pytest.mark.asyncio
    async def test_none_state(self, mock_hass):
        """Test URL returns None when state is None."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=None)

        url = await tool._get_image_url("camera.offline")

        assert url is None

    @pytest.mark.asyncio
    async def test_internal_url_fallback(self, mock_hass, camera_state):
        """Test URL falls back to internal_url when external_url is None."""
        mock_hass.config.external_url = None
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=camera_state)

        url = await tool._get_image_url("camera.front_door")

        assert url is not None
        assert "localhost:8123" in url

    @pytest.mark.asyncio
    async def test_no_base_url(self, mock_hass, camera_state):
        """Test URL returns None when no base URL is configured."""
        mock_hass.config.external_url = None
        mock_hass.config.internal_url = None
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=camera_state)

        url = await tool._get_image_url("camera.front_door")

        assert url is None

    @pytest.mark.asyncio
    async def test_absolute_entity_picture_url(self, mock_hass):
        """Test entity_picture with absolute URL is used as-is."""
        absolute_url_state = State(
            entity_id="camera.cloud",
            state="streaming",
            attributes={
                "friendly_name": "Cloud Camera",
                "entity_picture": "https://cloud.example.com/camera.jpg?token=xyz",
            },
            last_changed=datetime.now(),
            last_updated=datetime.now(),
        )
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.get = MagicMock(return_value=absolute_url_state)

        url = await tool._get_image_url("camera.cloud")

        assert url == "https://cloud.example.com/camera.jpg?token=xyz"


class TestQueryWithIncludeImage:
    """Test ha_query with include_image parameter."""

    @pytest.mark.asyncio
    async def test_query_camera_without_image_flag(self, mock_hass, camera_state):
        """Test camera query without include_image returns no images."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        result = await tool.execute(entity_id="camera.front_door")

        assert result["success"] is True
        assert "images" not in result or result["images"] == []

    @pytest.mark.asyncio
    async def test_query_camera_with_image_flag(self, mock_hass, camera_state):
        """Test camera query with include_image=true returns image URL from entity_picture."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        result = await tool.execute(
            entity_id="camera.front_door",
            include_image=True,
        )

        assert result["success"] is True
        assert "images" in result
        assert len(result["images"]) == 1
        assert result["images"][0]["entity_id"] == "camera.front_door"
        assert "url" in result["images"][0]
        assert "token=abc123def456" in result["images"][0]["url"]

    @pytest.mark.asyncio
    async def test_query_non_image_entity_with_image_flag(self, mock_hass, light_state):
        """Test non-image entity with include_image=true ignores flag."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get = MagicMock(return_value=light_state)

        result = await tool.execute(
            entity_id="light.living_room",
            include_image=True,
        )

        assert result["success"] is True
        assert "images" not in result or result["images"] == []

    @pytest.mark.asyncio
    async def test_query_image_entity_with_image_flag(self, mock_hass, image_state):
        """Test image.* query with include_image=true returns image URL."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = [
            "image.vacuum_map"
        ]
        mock_hass.states.get = MagicMock(return_value=image_state)

        result = await tool.execute(
            entity_id="image.vacuum_map",
            include_image=True,
        )

        assert result["success"] is True
        assert "images" in result
        assert len(result["images"]) == 1
        assert result["images"][0]["entity_id"] == "image.vacuum_map"
        assert "url" in result["images"][0]
        assert "/api/image_proxy/" in result["images"][0]["url"]

    @pytest.mark.asyncio
    async def test_query_multiple_cameras_all_returned(self, mock_hass, camera_state):
        """Test that all matching camera images are returned (limiting happens in core.py)."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = [
            "camera.front_door",
            "camera.back_door",
            "camera.garage",
            "camera.driveway",
            "camera.porch",
        ]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        result = await tool.execute(
            entity_id="camera.*",
            include_image=True,
        )

        assert result["success"] is True
        # ha_query returns all images; core.py enforces DEFAULT_MAX_IMAGES when building
        # the multimodal message for the LLM
        assert len(result["images"]) == 5


class TestToolSchema:
    """Test the tool parameter schema includes include_image."""

    def test_include_image_parameter_in_schema(self, mock_hass):
        """Test that include_image parameter is in the schema."""
        tool = HomeAssistantQueryTool(mock_hass)
        params = tool.parameters

        assert "include_image" in params["properties"]
        assert params["properties"]["include_image"]["type"] == "boolean"
        assert params["properties"]["include_image"]["default"] is False

    def test_description_mentions_camera(self, mock_hass):
        """Test that tool description mentions camera/image capability."""
        tool = HomeAssistantQueryTool(mock_hass)
        assert "camera" in tool.description.lower() or "image" in tool.description.lower()


class TestDetectImageMimeType:
    """Test the _detect_image_mime_type static method."""

    def test_png_magic_bytes(self):
        """Test PNG detection via magic bytes."""
        # PNG magic: \x89PNG\r\n\x1a\n
        png_bytes = b"\x89\x50\x4e\x47\r\n\x1a\n" + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(png_bytes)
        assert mime == "image/png"

    def test_jpeg_magic_bytes(self):
        """Test JPEG detection via magic bytes."""
        # JPEG magic: \xff\xd8\xff
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(jpeg_bytes)
        assert mime == "image/jpeg"

    def test_gif87a_magic_bytes(self):
        """Test GIF87a detection via magic bytes."""
        gif_bytes = b"GIF87a" + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(gif_bytes)
        assert mime == "image/gif"

    def test_gif89a_magic_bytes(self):
        """Test GIF89a detection via magic bytes."""
        gif_bytes = b"GIF89a" + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(gif_bytes)
        assert mime == "image/gif"

    def test_webp_magic_bytes(self):
        """Test WebP detection via RIFF + WEBP magic bytes."""
        # WebP: RIFF....WEBP....
        webp_bytes = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(webp_bytes)
        assert mime == "image/webp"

    def test_unknown_format_fallback(self):
        """Test fallback to image/unknown for unrecognized format."""
        unknown_bytes = b"\x00\x01\x02\x03" + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(unknown_bytes)
        assert mime == "image/unknown"

    def test_empty_bytes_fallback(self):
        """Test fallback for empty bytes."""
        mime = HomeAssistantQueryTool._detect_image_mime_type(b"")
        assert mime == "image/unknown"

    def test_riff_without_webp_fallback(self):
        """Test RIFF without WEBP at offset 8 falls back to unknown."""
        riff_bytes = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 100
        mime = HomeAssistantQueryTool._detect_image_mime_type(riff_bytes)
        assert mime == "image/unknown"


class MockAsyncContextManager:
    """Helper class for mocking async context managers in tests."""

    def __init__(self, enter_result, exit_result=None):
        self.enter_result = enter_result
        self.exit_result = exit_result

    async def __aenter__(self):
        return self.enter_result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.exit_result


class TestFetchAndEncodeImage:
    """Test the _fetch_and_encode_image async method."""

    @pytest.mark.asyncio
    async def test_successful_png_encoding(self, mock_hass):
        """Test successful PNG image fetch and base64 encoding via chunked read."""
        tool = HomeAssistantQueryTool(mock_hass)

        png_bytes = b"\x89\x50\x4e\x47\r\n\x1a\n" + b"\x00" * 100

        async def iter_chunked(chunk_size):
            yield png_bytes

        mock_content = MagicMock()
        mock_content.iter_chunked = iter_chunked

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_content

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MockAsyncContextManager(mock_response)
        )

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool._fetch_and_encode_image("http://example.com/image.png")

        assert result is not None
        assert result.startswith("data:image/png;base64,")
        b64_data = result.split("base64,")[1]
        decoded = base64.b64decode(b64_data)
        assert decoded == png_bytes

    @pytest.mark.asyncio
    async def test_successful_jpeg_encoding(self, mock_hass):
        """Test successful JPEG image fetch and base64 encoding via chunked read."""
        tool = HomeAssistantQueryTool(mock_hass)

        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        async def iter_chunked(chunk_size):
            yield jpeg_bytes

        mock_content = MagicMock()
        mock_content.iter_chunked = iter_chunked

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_content

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MockAsyncContextManager(mock_response)
        )

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool._fetch_and_encode_image("http://example.com/image.jpg")

        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_unknown_mime_fallback(self, mock_hass):
        """Test fallback to image/unknown for unrecognized format via chunked read."""
        tool = HomeAssistantQueryTool(mock_hass)

        unknown_bytes = b"\x00\x01\x02\x03" + b"\x00" * 100

        async def iter_chunked(chunk_size):
            yield unknown_bytes

        mock_content = MagicMock()
        mock_content.iter_chunked = iter_chunked

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_content

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MockAsyncContextManager(mock_response)
        )

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool._fetch_and_encode_image("http://example.com/image.bin")

        assert result is not None
        assert result.startswith("data:image/unknown;base64,")

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self, mock_hass):
        """Test that HTTP errors return None."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_response = MagicMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MockAsyncContextManager(mock_response)
        )

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool._fetch_and_encode_image("http://example.com/missing.png")

        assert result is None

    @pytest.mark.asyncio
    async def test_image_too_large_aborts_early(self, mock_hass):
        """Test that images exceeding size limit abort download early via chunked read."""
        tool = HomeAssistantQueryTool(mock_hass)

        # Send chunks that exceed the 5MB limit
        chunk1 = b"\x89\x50\x4e\x47" + b"\x00" * (3 * 1024 * 1024)  # 3MB
        chunk2 = b"\x00" * (3 * 1024 * 1024)  # Another 3MB (total 6MB > 5MB limit)

        async def iter_chunked(chunk_size):
            yield chunk1
            yield chunk2  # Should trigger early abort

        mock_content = MagicMock()
        mock_content.iter_chunked = iter_chunked

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_content

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MockAsyncContextManager(mock_response)
        )

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool._fetch_and_encode_image("http://example.com/large.png")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_exception_returns_none(self, mock_hass):
        """Test that fetch exceptions are handled gracefully."""
        tool = HomeAssistantQueryTool(mock_hass)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool._fetch_and_encode_image("http://example.com/error.png")

        assert result is None


class TestImageFormatParameter:
    """Test the image_format parameter in ha_query tool."""

    def test_image_format_in_schema(self, mock_hass):
        """Test that image_format parameter is in the schema."""
        tool = HomeAssistantQueryTool(mock_hass)
        params = tool.parameters

        assert "image_format" in params["properties"]
        assert params["properties"]["image_format"]["type"] == "string"
        assert params["properties"]["image_format"]["enum"] == ["url", "inline"]
        assert params["properties"]["image_format"]["default"] == "url"

    @pytest.mark.asyncio
    async def test_default_image_format_is_url(self, mock_hass, camera_state):
        """Test that default image_format is 'url' (no inline encoding)."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        result = await tool.execute(
            entity_id="camera.front_door",
            include_image=True,
        )

        assert result["success"] is True
        assert "images" in result
        # Default format is URL, so image URL should be a regular URL
        assert result["images"][0]["url"].startswith("https://")
        assert not result["images"][0]["url"].startswith("data:")

    @pytest.mark.asyncio
    async def test_image_format_inline_calls_fetch(self, mock_hass, camera_state):
        """Test that image_format='inline' triggers base64 encoding via chunked read."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        png_bytes = b"\x89\x50\x4e\x47\r\n\x1a\n" + b"\x00" * 100

        async def iter_chunked(chunk_size):
            yield png_bytes

        mock_content = MagicMock()
        mock_content.iter_chunked = iter_chunked

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = mock_content

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MockAsyncContextManager(mock_response)
        )

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool.execute(
                entity_id="camera.front_door",
                include_image=True,
                image_format="inline",
            )

        assert result["success"] is True
        assert "images" in result
        assert len(result["images"]) == 1
        # Inline format should produce a data URI
        assert result["images"][0]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_image_format_inline_fetch_failure_returns_no_images(self, mock_hass, camera_state):
        """Test that inline fetch failure results in empty images."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        # Simulate fetch failure
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("Network error"))

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session,
        ):
            result = await tool.execute(
                entity_id="camera.front_door",
                include_image=True,
                image_format="inline",
            )

        assert result["success"] is True
        # Images should be empty when fetch fails
        assert result.get("images", []) == []

    @pytest.mark.asyncio
    async def test_image_format_url_ignores_inline(self, mock_hass, camera_state):
        """Test that image_format='url' does not trigger inline encoding."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        result = await tool.execute(
            entity_id="camera.front_door",
            include_image=True,
            image_format="url",
        )

        assert result["success"] is True
        assert "images" in result
        # URL format should return a regular URL, not a data URI
        assert not result["images"][0]["url"].startswith("data:")

    @pytest.mark.asyncio
    async def test_image_format_with_non_image_entity(self, mock_hass, light_state):
        """Test that image_format is ignored for non-image entities."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["light.living_room"]
        mock_hass.states.get = MagicMock(return_value=light_state)

        result = await tool.execute(
            entity_id="light.living_room",
            include_image=True,
            image_format="inline",
        )

        assert result["success"] is True
        # Non-image entities should have no images regardless of format
        assert result.get("images", []) == []

    @pytest.mark.asyncio
    async def test_image_format_without_include_image(self, mock_hass, camera_state):
        """Test that image_format is ignored if include_image is false."""
        tool = HomeAssistantQueryTool(mock_hass)
        mock_hass.states.async_entity_ids.return_value = ["camera.front_door"]
        mock_hass.states.get = MagicMock(return_value=camera_state)

        result = await tool.execute(
            entity_id="camera.front_door",
            include_image=False,
            image_format="inline",
        )

        assert result["success"] is True
        # No images should be returned if include_image is false
        assert result.get("images", []) == []


class TestImageCapableDomainsConstant:
    """Test the IMAGE_CAPABLE_DOMAINS constant directly."""

    def test_domains_are_correct(self):
        """Test that IMAGE_CAPABLE_DOMAINS contains expected domains."""
        assert IMAGE_CAPABLE_DOMAINS == {"camera", "media_player", "image"}
