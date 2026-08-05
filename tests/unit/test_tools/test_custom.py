"""Unit tests for CustomToolHandler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.home_agent.const import (
    CUSTOM_TOOL_HANDLER_MCP,
    CUSTOM_TOOL_HANDLER_SERVICE,
)
from custom_components.home_agent.exceptions import ToolExecutionError, ValidationError
from custom_components.home_agent.tools.custom import (
    CustomToolHandler,
    RestCustomTool,
    ServiceCustomTool,
)


class TestCustomToolHandler:
    """Test the CustomToolHandler factory class."""

    @pytest.mark.asyncio
    async def test_create_rest_tool_success(self, mock_hass):
        """Test successful creation of REST custom tool."""
        config = {
            "name": "check_weather",
            "description": "Get weather forecast",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
            "handler": {
                "type": "rest",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET",
            },
        }

        tools = await CustomToolHandler.create_tools_from_config(mock_hass, config)

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, RestCustomTool)
        assert tool.name == "check_weather"
        assert tool.description == "Get weather forecast"

    @pytest.mark.asyncio
    async def test_create_tool_missing_required_keys(self, mock_hass):
        """Test that creation fails when required keys are missing."""
        # Missing 'description'
        config = {"name": "test_tool", "parameters": {}, "handler": {"type": "rest"}}

        with pytest.raises(ValidationError) as exc_info:
            await CustomToolHandler.create_tools_from_config(mock_hass, config)

        assert "missing required keys" in str(exc_info.value).lower()
        assert "description" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_create_tool_missing_handler_type(self, mock_hass):
        """Test that creation fails when handler type is missing."""
        config = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {},
            "handler": {},  # Missing 'type'
        }

        with pytest.raises(ValidationError) as exc_info:
            await CustomToolHandler.create_tools_from_config(mock_hass, config)

        assert "type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_create_tool_unsupported_handler_type(self, mock_hass):
        """Test that creation fails for unsupported handler types."""
        config = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {},
            "handler": {"type": "unknown_handler"},
        }

        with pytest.raises(ValidationError) as exc_info:
            await CustomToolHandler.create_tools_from_config(mock_hass, config)

        assert "unknown handler type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_create_tool_service_handler(self, mock_hass):
        """Test that service handler creates ServiceCustomTool."""
        # Mock has_service to return True
        mock_hass.services.has_service = MagicMock(return_value=True)

        config = {
            "name": "test_tool",
            "description": "Test tool",
            "parameters": {},
            "handler": {
                "type": CUSTOM_TOOL_HANDLER_SERVICE,
                "service": "light.turn_on",
            },
        }

        tools = await CustomToolHandler.create_tools_from_config(mock_hass, config)

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, ServiceCustomTool)
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"

    @pytest.mark.asyncio
    async def test_create_mcp_tools_success(self, mock_hass):
        """Test that MCP handler creates MCPToolWrapper instances."""
        from contextlib import asynccontextmanager
        from unittest.mock import MagicMock, patch
        from custom_components.home_agent.tools.mcp_proxy import MCPToolWrapper

        # Create proper mock session with async context manager
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "remote_weather",
                        "description": "Get remote weather",
                        "inputSchema": {"type": "object", "properties": {"location": {"type": "string"}}}
                    }
                ]
            }
        })

        @asynccontextmanager
        async def post_context(*args, **kwargs):
            yield mock_response

        mock_session.post = post_context

        config = {
            "name": "weather_mcp",
            "description": "MCP weather proxy",
            "parameters": {},
            "handler": {
                "type": CUSTOM_TOOL_HANDLER_MCP,
                "server_url": "https://mcp-server.example.com/mcp",
                "timeout": 30,
            },
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            tools = await CustomToolHandler.create_tools_from_config(mock_hass, config)

        assert len(tools) == 1
        assert isinstance(tools[0], MCPToolWrapper)
        assert "remote_weather" in tools[0].name

    @pytest.mark.asyncio
    async def test_create_mcp_tools_missing_server_url(self, mock_hass):
        """Test MCP tool creation fails without server_url."""
        config = {
            "name": "bad_mcp",
            "description": "Missing URL",
            "parameters": {},
            "handler": {
                "type": CUSTOM_TOOL_HANDLER_MCP,
            },
        }

        with pytest.raises(ToolExecutionError):
            await CustomToolHandler.create_tools_from_config(mock_hass, config)

    @pytest.mark.asyncio
    async def test_create_mcp_tools_server_unreachable(self, mock_hass):
        """Test MCP tool creation fails when server is unreachable."""
        from unittest.mock import patch
        import aiohttp
        mock_session = AsyncMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        config = {
            "name": "bad_mcp",
            "description": "Unreachable server",
            "parameters": {},
            "handler": {
                "type": CUSTOM_TOOL_HANDLER_MCP,
                "server_url": "https://broken.example.com/mcp",
            },
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            with pytest.raises(ToolExecutionError):
                await CustomToolHandler.create_tools_from_config(mock_hass, config)


class TestRestCustomToolInitialization:
    """Test RestCustomTool initialization."""

    def test_initialization_success(self, mock_hass):
        """Test successful initialization of REST custom tool."""
        config = {
            "name": "weather_api",
            "description": "Get weather data",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "handler": {"type": "rest", "url": "https://api.example.com/weather", "method": "GET"},
        }

        tool = RestCustomTool(mock_hass, config)

        assert tool.hass == mock_hass
        assert tool._config == config
        assert tool._handler_config == config["handler"]

    def test_initialization_missing_url(self, mock_hass):
        """Test initialization fails when URL is missing."""
        config = {
            "name": "test_tool",
            "description": "Test",
            "parameters": {},
            "handler": {
                "type": "rest",
                "method": "GET",
                # Missing 'url'
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            RestCustomTool(mock_hass, config)

        assert "missing required keys" in str(exc_info.value).lower()
        assert "url" in str(exc_info.value).lower()

    def test_initialization_missing_method(self, mock_hass):
        """Test initialization fails when HTTP method is missing."""
        config = {
            "name": "test_tool",
            "description": "Test",
            "parameters": {},
            "handler": {
                "type": "rest",
                "url": "https://api.example.com",
                # Missing 'method'
            },
        }

        with pytest.raises(ValidationError) as exc_info:
            RestCustomTool(mock_hass, config)

        assert "missing required keys" in str(exc_info.value).lower()
        assert "method" in str(exc_info.value).lower()

    def test_initialization_invalid_method(self, mock_hass):
        """Test initialization fails with invalid HTTP method."""
        config = {
            "name": "test_tool",
            "description": "Test",
            "parameters": {},
            "handler": {"type": "rest", "url": "https://api.example.com", "method": "INVALID"},
        }

        with pytest.raises(ValidationError) as exc_info:
            RestCustomTool(mock_hass, config)

        assert "invalid http method" in str(exc_info.value).lower()


class TestRestCustomToolProperties:
    """Test RestCustomTool properties."""

    @pytest.fixture
    def rest_tool(self, mock_hass):
        """Create a REST custom tool for testing."""
        config = {
            "name": "weather_api",
            "description": "Get weather forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
            "handler": {
                "type": "rest",
                "url": "https://api.weather.com/v1/forecast",
                "method": "GET",
            },
        }
        return RestCustomTool(mock_hass, config)

    def test_tool_name(self, rest_tool):
        """Test that tool name is correct."""
        assert rest_tool.name == "weather_api"

    def test_tool_description(self, rest_tool):
        """Test that tool description is correct."""
        assert rest_tool.description == "Get weather forecast for a location"

    def test_tool_parameters(self, rest_tool):
        """Test that tool parameters are correct."""
        params = rest_tool.parameters

        assert params["type"] == "object"
        assert "location" in params["properties"]
        assert "location" in params["required"]

    def test_get_definition(self, rest_tool):
        """Test get_definition returns correct format."""
        definition = rest_tool.get_definition()

        assert definition["name"] == "weather_api"
        assert definition["description"] == "Get weather forecast for a location"
        assert "parameters" in definition

    def test_to_openai_format(self, rest_tool):
        """Test conversion to OpenAI format."""
        openai_format = rest_tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert "function" in openai_format
        assert openai_format["function"]["name"] == "weather_api"
