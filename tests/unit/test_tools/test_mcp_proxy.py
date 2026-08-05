"""Unit tests for MCP proxy tools."""
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from custom_components.home_agent.tools.mcp_proxy import (
    MCPHttpClient,
    MCPProxyFactory,
    MCPToolWrapper,
)
from custom_components.home_agent.exceptions import ToolExecutionError


def make_mock_session(json_response: dict, status: int = 200):
    """Create a properly mocked aiohttp session with async context manager support."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_response)
    
    @asynccontextmanager
    async def post_context(*args, **kwargs):
        yield mock_response
    
    mock_session.post = post_context
    return mock_session


class TestMCPHttpClient:
    """Test MCPHttpClient JSON-RPC over HTTP."""

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful MCP initialization."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": "2024-11-05", "capabilities": {}}
        })

        client = MCPHttpClient(mock_hass, "https://example.com/mcp")
        client._session = mock_session
        await client.initialize()
        assert client._initialized

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test MCP initialization failure."""
        mock_hass = MagicMock()
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))

        client = MCPHttpClient(mock_hass, "https://broken.com/mcp")
        client._session = mock_session
        with pytest.raises(ToolExecutionError):
            await client.initialize()
        assert not client._initialized

    @pytest.mark.asyncio
    async def test_list_tools_success(self):
        """Test listing tools from MCP server."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "tools": [
                    {"name": "tool1", "description": "Tool 1", "inputSchema": {"type": "object"}},
                    {"name": "tool2", "description": "Tool 2", "inputSchema": {"type": "object"}}
                ]
            }
        })

        client = MCPHttpClient(mock_hass, "https://example.com/mcp")
        client._session = mock_session
        client._initialized = True
        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "tool1"

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test calling a remote tool."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "OK"}]}
        })

        client = MCPHttpClient(mock_hass, "https://example.com/mcp")
        client._session = mock_session
        client._initialized = True
        result = await client.call_tool("test_tool", {"arg": "value"})
        assert result["success"]
        assert result["content"][0] == "OK"

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        """Test calling a tool that returns an error."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32603, "message": "Internal error"}
        })

        client = MCPHttpClient(mock_hass, "https://example.com/mcp")
        client._session = mock_session
        client._initialized = True
        with pytest.raises(ToolExecutionError):
            await client.call_tool("bad_tool", {})


class TestMCPToolWrapper:
    """Test MCPToolWrapper."""

    def test_initialization(self):
        """Test MCPToolWrapper initialization."""
        mock_hass = MagicMock()
        mock_client = AsyncMock(spec=MCPHttpClient)
        mock_client.server_url = "https://example.com/mcp"
        config = {
            "name": "remote_tool",
            "description": "Remote tool description",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}
        }
        wrapper = MCPToolWrapper(mock_hass, mock_client, config)

        assert "remote_tool" in wrapper.name
        assert "Remote tool description" in wrapper.description

    @pytest.mark.asyncio
    async def test_execute_delegates_to_client(self):
        """Test execute delegates to client.call_tool."""
        mock_hass = MagicMock()
        mock_client = AsyncMock(spec=MCPHttpClient)
        mock_client.server_url = "https://example.com/mcp"
        mock_client.call_tool = AsyncMock(return_value={
            "success": True,
            "content": ["result"],
            "raw_result": {"content": [{"type": "text", "text": "result"}]}
        })
        config = {
            "name": "remote_tool",
            "description": "Remote tool",
            "inputSchema": {"type": "object"}
        }
        wrapper = MCPToolWrapper(mock_hass, mock_client, config)

        result = await wrapper.execute(x="test")
        mock_client.call_tool.assert_awaited_once_with("remote_tool", {"x": "test"})
        assert result["success"]

    @pytest.mark.asyncio
    async def test_execute_propagates_error(self):
        """Test execute propagates ToolExecutionError."""
        mock_hass = MagicMock()
        mock_client = AsyncMock(spec=MCPHttpClient)
        mock_client.server_url = "https://example.com/mcp"
        mock_client.call_tool = AsyncMock(side_effect=ToolExecutionError("Remote error"))
        config = {
            "name": "remote_tool",
            "description": "Remote tool",
            "inputSchema": {"type": "object"}
        }
        wrapper = MCPToolWrapper(mock_hass, mock_client, config)

        with pytest.raises(ToolExecutionError, match="Remote error"):
            await wrapper.execute(x="test")


class TestMCPProxyFactory:
    """Test MCPProxyFactory static factory."""

class TestMCPProxyFactory:
    """Test MCPProxyFactory static factory."""

    @pytest.mark.asyncio
    async def test_create_tools_from_config_success(self):
        """Test creating MCP tools from config."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "weather",
                        "description": "Get weather",
                        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}}
                    }
                ]
            }
        })

        config = {
            "server_url": "https://mcp.example.com/mcp",
            "timeout": 30,
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            tools = await MCPProxyFactory.create_tools_from_config(mock_hass, config, "weather_mcp")
        assert len(tools) == 1
        assert isinstance(tools[0], MCPToolWrapper)
        assert tools[0].name == "weather_mcp_weather"

    @pytest.mark.asyncio
    async def test_create_tools_missing_server_url(self):
        """Test creation fails without server_url."""
        mock_hass = MagicMock()
        config = {
            "timeout": 30,
        }

        with pytest.raises(ToolExecutionError):
            await MCPProxyFactory.create_tools_from_config(mock_hass, config, "bad_mcp")

    @pytest.mark.asyncio
    async def test_create_tools_server_unreachable(self):
        """Test creation fails when server is unreachable."""
        mock_hass = MagicMock()
        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=aiohttp.ClientError("Connection failed"))

        config = {
            "server_url": "https://broken.com/mcp",
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            with pytest.raises(ToolExecutionError):
                await MCPProxyFactory.create_tools_from_config(mock_hass, config, "bad_mcp")

    def test_filter_tools_include(self):
        """Test filtering tools with include patterns."""
        tools = [
            {"name": "weather_forecast"},
            {"name": "weather_alerts"},
            {"name": "news_headlines"},
            {"name": "sports_scores"},
        ]

        # Include only weather tools
        filtered = MCPProxyFactory._filter_tools(tools, include_patterns=["weather"])
        assert len(filtered) == 2
        assert filtered[0]["name"] == "weather_forecast"
        assert filtered[1]["name"] == "weather_alerts"

    def test_filter_tools_exclude(self):
        """Test filtering tools with exclude patterns."""
        tools = [
            {"name": "weather_forecast"},
            {"name": "weather_alerts"},
            {"name": "news_headlines"},
            {"name": "sports_scores"},
        ]

        # Exclude news and sports
        filtered = MCPProxyFactory._filter_tools(tools, exclude_patterns=["news|sports"])
        assert len(filtered) == 2
        assert filtered[0]["name"] == "weather_forecast"
        assert filtered[1]["name"] == "weather_alerts"

    def test_filter_tools_exclude_takes_precedence(self):
        """Test that exclude patterns take precedence over include patterns."""
        tools = [
            {"name": "weather_forecast"},
            {"name": "weather_alerts"},
            {"name": "weather_internal"},
            {"name": "news_headlines"},
        ]

        # Include weather but exclude internal
        filtered = MCPProxyFactory._filter_tools(
            tools,
            include_patterns=["weather"],
            exclude_patterns=["internal"]
        )
        assert len(filtered) == 2
        assert filtered[0]["name"] == "weather_forecast"
        assert filtered[1]["name"] == "weather_alerts"

    def test_filter_tools_no_filters(self):
        """Test that all tools pass when no filters are specified."""
        tools = [
            {"name": "weather_forecast"},
            {"name": "news_headlines"},
            {"name": "sports_scores"},
        ]

        filtered = MCPProxyFactory._filter_tools(tools)
        assert len(filtered) == 3

    def test_filter_tools_include_no_match(self):
        """Test that no tools pass when include pattern matches nothing."""
        tools = [
            {"name": "weather_forecast"},
            {"name": "news_headlines"},
        ]

        filtered = MCPProxyFactory._filter_tools(tools, include_patterns=["^nonexistent"])
        assert len(filtered) == 0

    def test_filter_tools_regex_patterns(self):
        """Test regex pattern matching for filters."""
        tools = [
            {"name": "get_weather"},
            {"name": "set_thermostat"},
            {"name": "get_news"},
            {"name": "get_sports"},
            {"name": "internal_debug"},
        ]

        # Include only tools starting with "get_"
        filtered = MCPProxyFactory._filter_tools(tools, include_patterns=["^get_"])
        assert len(filtered) == 3

        # Exclude internal tools
        filtered = MCPProxyFactory._filter_tools(tools, exclude_patterns=["internal"])
        assert len(filtered) == 5 - 1  # All except internal_debug

    def test_filter_tools_empty_list(self):
        """Test filtering empty tool list."""
        filtered = MCPProxyFactory._filter_tools([], include_patterns=["weather"])
        assert len(filtered) == 0

    def test_should_include_tool_basic(self):
        """Test _should_include_tool with basic patterns."""
        assert MCPProxyFactory._should_include_tool("weather_forecast", ["weather"])
        assert MCPProxyFactory._should_include_tool("weather_forecast", None, None)
        assert not MCPProxyFactory._should_include_tool("news_headlines", ["weather"])

    def test_should_include_tool_exclude_precedence(self):
        """Test that exclude takes precedence over include."""
        # Tool matches include but also matches exclude
        assert not MCPProxyFactory._should_include_tool(
            "weather_internal", ["weather"], ["internal"]
        )
        # Tool matches include but not exclude
        assert MCPProxyFactory._should_include_tool(
            "weather_forecast", ["weather"], ["internal"]
        )

    def test_filter_tools_single_string_pattern(self):
        """Test that single string patterns work (not list)."""
        tools = [
            {"name": "weather_forecast"},
            {"name": "news_headlines"},
        ]

        # Single string should work
        filtered = MCPProxyFactory._filter_tools(tools, include_patterns=["weather"])
        assert len(filtered) == 1

    @pytest.mark.asyncio
    async def test_create_tools_with_include_filter(self):
        """Test creating MCP tools with include filter."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "weather",
                        "description": "Get weather",
                        "inputSchema": {"type": "object"}
                    },
                    {
                        "name": "news",
                        "description": "Get news",
                        "inputSchema": {"type": "object"}
                    },
                ]
            }
        })

        config = {
            "server_url": "https://mcp.example.com/mcp",
            "tool_include": ["weather"],
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            tools = await MCPProxyFactory.create_tools_from_config(mock_hass, config, "filtered_mcp")
        assert len(tools) == 1
        assert tools[0].name == "filtered_mcp_weather"

    @pytest.mark.asyncio
    async def test_create_tools_with_exclude_filter(self):
        """Test creating MCP tools with exclude filter."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "weather",
                        "description": "Get weather",
                        "inputSchema": {"type": "object"}
                    },
                    {
                        "name": "internal_debug",
                        "description": "Debug tool",
                        "inputSchema": {"type": "object"}
                    },
                ]
            }
        })

        config = {
            "server_url": "https://mcp.example.com/mcp",
            "tool_exclude": ["internal"],
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            tools = await MCPProxyFactory.create_tools_from_config(mock_hass, config, "filtered_mcp")
        assert len(tools) == 1
        assert tools[0].name == "filtered_mcp_weather"

    @pytest.mark.asyncio
    async def test_create_tools_with_both_filters(self):
        """Test creating MCP tools with both include and exclude filters."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "weather_forecast",
                        "description": "Get weather forecast",
                        "inputSchema": {"type": "object"}
                    },
                    {
                        "name": "weather_alerts",
                        "description": "Get weather alerts",
                        "inputSchema": {"type": "object"}
                    },
                    {
                        "name": "weather_internal",
                        "description": "Internal weather tool",
                        "inputSchema": {"type": "object"}
                    },
                    {
                        "name": "news_headlines",
                        "description": "Get news",
                        "inputSchema": {"type": "object"}
                    },
                ]
            }
        })

        config = {
            "server_url": "https://mcp.example.com/mcp",
            "tool_include": ["weather"],
            "tool_exclude": ["internal"],
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            tools = await MCPProxyFactory.create_tools_from_config(mock_hass, config, "filtered_mcp")
        assert len(tools) == 2
        assert tools[0].name == "filtered_mcp_weather_forecast"
        assert tools[1].name == "filtered_mcp_weather_alerts"

    @pytest.mark.asyncio
    async def test_create_tools_all_filtered_out(self):
        """Test when all tools are filtered out."""
        mock_hass = MagicMock()
        mock_session = make_mock_session({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "tools": [
                    {
                        "name": "weather",
                        "description": "Get weather",
                        "inputSchema": {"type": "object"}
                    }
                ]
            }
        })

        config = {
            "server_url": "https://mcp.example.com/mcp",
            "tool_include": ["^nonexistent"],  # Nothing matches
        }

        with patch(
            "homeassistant.helpers.aiohttp_client.async_get_clientsession",
            return_value=mock_session
        ):
            tools = await MCPProxyFactory.create_tools_from_config(mock_hass, config, "filtered_mcp")
        assert len(tools) == 0
