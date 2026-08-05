"""MCP (Model Context Protocol) proxy tools for Home Agent.

This module provides MCP proxy functionality that allows Home Agent to connect
to remote MCP servers over HTTP and expose their tools as local Home Agent tools.

Architecture:
- MCPHttpClient: Handles JSON-RPC 2.0 communication over HTTP
- MCPToolWrapper: Wraps a single remote MCP tool as a local BaseTool
- MCPProxyFactory: Factory for discovering and creating MCP tools from config
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any, Final

from homeassistant.core import HomeAssistant, HomeAssistantError

from ..const import (
    CONF_MCP_HEADERS,
    CONF_MCP_SERVER_URL,
    CONF_MCP_TIMEOUT,
    CONF_MCP_TRANSPORT,
    CUSTOM_TOOL_HANDLER_MCP,
    DEFAULT_MCP_TIMEOUT,
    DEFAULT_MCP_TRANSPORT,
    MCP_TRANSPORT_STREAMABLE_HTTP,
)
from ..exceptions import ToolExecutionError
from .registry import BaseTool

import aiohttp

if TYPE_CHECKING:
    from aiohttp import ClientSession

_LOGGER: Final = logging.getLogger(__name__)

# JSON-RPC 2.0 method names per MCP spec
MCP_METHOD_INITIALIZE: Final = "initialize"
MCP_METHOD_TOOLS_LIST: Final = "tools/list"
MCP_METHOD_TOOLS_CALL: Final = "tools/call"

# MCP initialize request payload (minimal compatible client)
MCP_INITIALIZE_PAYLOAD: Final = {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "hass-agent-llm", "version": "1.0.0"},
}


class MCPHttpClient:
    """HTTP client for MCP (Model Context Protocol) servers.

    Handles JSON-RPC 2.0 communication over Streamable HTTP transport.
    Each tool call is a separate HTTP POST - no persistent connections needed.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        server_url: str,
        headers: dict[str, str] | None = None,
        timeout: int = DEFAULT_MCP_TIMEOUT,
    ) -> None:
        """Initialize the MCP HTTP client.

        Args:
            hass: Home Assistant instance
            server_url: URL of the MCP server endpoint
            headers: Optional HTTP headers to send with requests (e.g. auth)
            timeout: Request timeout in seconds
        """
        self.hass = hass
        self.server_url = server_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout
        self._session: ClientSession | None = None
        self._initialized = False
        self._server_capabilities: dict[str, Any] = {}

    async def _get_session(self) -> ClientSession:
        """Get or create the aiohttp client session.

        Uses Home Assistant's shared session for connection pooling.

        Returns:
            aiohttp client session
        """
        if self._session is None:
            from homeassistant.helpers.aiohttp_client import async_get_clientsession

            self._session = async_get_clientsession(self.hass)
        return self._session

    async def initialize(self) -> dict[str, Any]:
        """Initialize the MCP connection and discover server capabilities.

        Returns:
            Server info and capabilities

        Raises:
            ToolExecutionError: If initialization fails
        """
        if self._initialized:
            return self._server_capabilities

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": MCP_METHOD_INITIALIZE,
            "params": MCP_INITIALIZE_PAYLOAD,
        }

        try:
            response = await self._send_request(payload)
        except ToolExecutionError as e:
            raise ToolExecutionError(
                f"MCP initialization failed for {self.server_url}: {e}"
            ) from e

        # Send initialized notification
        await self._send_notification("notifications/initialized", {})

        self._initialized = True
        self._server_capabilities = response.get("result", {}).get("capabilities", {})
        _LOGGER.debug(
            "MCP server %s initialized with capabilities: %s",
            self.server_url,
            self._server_capabilities,
        )
        return self._server_capabilities

    async def list_tools(self) -> list[dict[str, Any]]:
        """List all tools available on the remote MCP server.

        Returns:
            List of tool definitions from the server

        Raises:
            ToolExecutionError: If listing tools fails
        """
        await self.initialize()

        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": MCP_METHOD_TOOLS_LIST,
        }

        try:
            response = await self._send_request(payload)
        except ToolExecutionError as e:
            raise ToolExecutionError(
                f"MCP tools/list failed for {self.server_url}: {e}"
            ) from e

        tools = response.get("result", {}).get("tools", [])
        _LOGGER.debug("MCP server %s has %d tools", self.server_url, len(tools))
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on the remote MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If the tool call fails
        """
        await self.initialize()

        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": MCP_METHOD_TOOLS_CALL,
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        try:
            response = await self._send_request(payload)
        except ToolExecutionError as e:
            raise ToolExecutionError(
                f"MCP tools/call failed for {tool_name} on {self.server_url}: {e}"
            ) from e

        result = response.get("result", {})
        content = result.get("content", [])

        # Extract text content from the response
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return {
            "success": True,
            "content": text_parts,
            "raw_result": result,
        }

    async def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request to the MCP server.

        Args:
            payload: JSON-RPC payload to send

        Returns:
            JSON-RPC response

        Raises:
            ToolExecutionError: If the request fails
        """
        session = await self._get_session()

        try:
            async with session.post(
                self.server_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ToolExecutionError(
                        f"MCP server returned HTTP {response.status}: {error_body}"
                    )

                data = await response.json()

                if "error" in data:
                    error_msg = data["error"].get("message", "Unknown error")
                    raise ToolExecutionError(f"MCP error: {error_msg}")

                return data

        except asyncio.TimeoutError:
            raise ToolExecutionError(
                f"MCP request to {self.server_url} timed out after {self.timeout}s"
            )
        except Exception as e:
            if isinstance(e, ToolExecutionError):
                raise
            raise ToolExecutionError(f"MCP request failed: {e}") from e

    async def _send_notification(
        self, method: str, params: dict[str, Any]
    ) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: Notification method name
            params: Notification parameters
        """
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        session = await self._get_session()

        try:
            await session.post(
                self.server_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        except Exception:
            # Notifications are best-effort
            _LOGGER.debug("Failed to send MCP notification %s", method)


class MCPToolWrapper(BaseTool):
    """Wraps a remote MCP tool as a local Home Agent tool.

    Each remote tool discovered on the MCP server gets wrapped in this class
    and registered as a local tool. When executed, it proxies the call to
    the remote server.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        client: MCPHttpClient,
        tool_definition: dict[str, Any],
        server_name: str = "mcp",
    ) -> None:
        """Initialize the MCP tool wrapper.

        Args:
            hass: Home Assistant instance
            client: MCP HTTP client for communication
            tool_definition: Tool definition from the remote server
            server_name: Human-readable server name for naming/descriptions
        """
        super().__init__(hass)
        self._client = client
        self._tool_definition = tool_definition
        self._server_name = server_name

        # Extract tool info from definition
        self._remote_tool_name = tool_definition.get("name", "unknown")
        self._remote_tool_description = tool_definition.get(
            "description", "No description available"
        )

    @property
    def name(self) -> str:
        """Return the tool name with server prefix to avoid collisions."""
        return f"{self._server_name}_{self._remote_tool_name}"

    @property
    def description(self) -> str:
        """Return the tool description indicating it's a remote MCP tool."""
        return (
            f"[MCP: {self._server_name}] {self._remote_tool_description}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool parameter schema from the remote server."""
        input_schema = self._tool_definition.get("inputSchema", {})

        # MCP uses JSON Schema, convert to OpenAI format if needed
        if "type" not in input_schema:
            input_schema["type"] = "object"

        return input_schema

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool on the remote MCP server.

        Args:
            **kwargs: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If execution fails
        """
        _LOGGER.debug(
            "Executing MCP tool %s on %s with args: %s",
            self._remote_tool_name,
            self._client.server_url,
            kwargs,
        )

        try:
            result = await self._client.call_tool(self._remote_tool_name, kwargs)

            text_output = "\n".join(result.get("content", []))
            return {
                "success": True,
                "tool": self.name,
                "output": text_output,
                "raw": result.get("raw_result", {}),
            }

        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to execute MCP tool {self._remote_tool_name}: {e}"
            ) from e

    def get_definition(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MCPProxyFactory:
    """Factory for creating MCP tools from configuration.

    Discovers tools on the remote MCP server and creates local wrappers.
    Supports regex-based tool filtering via tool_include and tool_exclude.
    """

    @staticmethod
    def _filter_tools(
        tools: list[dict[str, Any]],
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter tools based on include/exclude regex patterns.

        Rules:
        - Excluded tools cannot be included (exclude takes precedence)
        - If neither include nor exclude is specified, all tools pass
        - If only include is specified, only matching tools pass
        - If only exclude is specified, non-matching tools pass
        - Patterns are regex, matched against tool name

        Args:
            tools: List of tool definitions from MCP server
            include_patterns: List of regex patterns to include
            exclude_patterns: List of regex patterns to exclude

        Returns:
            Filtered list of tool definitions
        """
        if not tools:
            return tools

        filtered: list[dict[str, Any]] = []
        for tool_def in tools:
            tool_name = tool_def.get("name", "unknown")
            if not MCPProxyFactory._should_include_tool(
                tool_name, include_patterns, exclude_patterns
            ):
                _LOGGER.debug(
                    "Filtering out MCP tool '%s' based on include/exclude patterns",
                    tool_name,
                )
                continue
            filtered.append(tool_def)

        return filtered

    @staticmethod
    def _should_include_tool(
        tool_name: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        """Determine if a tool should be included based on filter patterns.

        Args:
            tool_name: Name of the tool to check
            include_patterns: List of regex patterns to include
            exclude_patterns: List of regex patterns to exclude

        Returns:
            True if the tool should be included
        """
        # Exclude takes precedence - if excluded, never include
        if exclude_patterns:
            for pattern in exclude_patterns:
                if re.search(pattern, tool_name):
                    return False

        # If include patterns are specified, tool must match at least one
        if include_patterns:
            for pattern in include_patterns:
                if re.search(pattern, tool_name):
                    return True
            return False

        # No include patterns means all non-excluded tools pass
        return True

    @staticmethod
    async def create_tools_from_config(
        hass: HomeAssistant, config: dict[str, Any], name: str = "mcp"
    ) -> list[BaseTool]:
        """Create MCP tool wrappers from configuration.

        Connects to the remote MCP server, discovers available tools,
        and creates local wrappers for each.

        Args:
            hass: Home Assistant instance
            config: MCP configuration dict
            name: Name prefix for the server (used in tool names)

        Returns:
            List of MCP tool wrappers (one per remote tool)

        Raises:
            ToolExecutionError: If connection or discovery fails
        """
        server_url = config.get(CONF_MCP_SERVER_URL)
        if not server_url:
            raise ToolExecutionError(
                "MCP configuration missing required 'server_url' parameter"
            )

        headers = config.get(CONF_MCP_HEADERS, {})
        timeout = config.get(CONF_MCP_TIMEOUT, DEFAULT_MCP_TIMEOUT)
        transport = config.get(CONF_MCP_TRANSPORT, DEFAULT_MCP_TRANSPORT)

        if transport != MCP_TRANSPORT_STREAMABLE_HTTP:
            raise ToolExecutionError(
                f"Unsupported MCP transport: {transport}. "
                f"Only '{MCP_TRANSPORT_STREAMABLE_HTTP}' is supported."
            )

        # Get tool filter patterns
        include_patterns = config.get("tool_include")
        exclude_patterns = config.get("tool_exclude")

        if include_patterns and isinstance(include_patterns, str):
            include_patterns = [include_patterns]
        if exclude_patterns and isinstance(exclude_patterns, str):
            exclude_patterns = [exclude_patterns]

        if include_patterns:
            _LOGGER.debug(
                "MCP tool include patterns: %s", include_patterns
            )
        if exclude_patterns:
            _LOGGER.debug(
                "MCP tool exclude patterns: %s", exclude_patterns
            )

        _LOGGER.info(
            "Creating MCP proxy for server %s with name '%s'", server_url, name
        )

        # Create HTTP client
        client = MCPHttpClient(
            hass=hass,
            server_url=server_url,
            headers=headers,
            timeout=timeout,
        )

        # Discover remote tools
        remote_tools = await client.list_tools()

        if not remote_tools:
            _LOGGER.warning(
                "MCP server %s returned no tools", server_url
            )
            return []

        # Apply include/exclude filters
        filtered_tools = MCPProxyFactory._filter_tools(
            remote_tools, include_patterns, exclude_patterns
        )

        if not filtered_tools:
            _LOGGER.warning(
                "MCP server %s returned %d tools but all were filtered out by include/exclude patterns",
                server_url,
                len(remote_tools),
            )
            return []

        _LOGGER.info(
            "MCP server %s: %d tools discovered, %d after filtering",
            server_url,
            len(remote_tools),
            len(filtered_tools),
        )

        # Create wrappers for each tool
        tools: list[BaseTool] = []
        for tool_def in filtered_tools:
            tool_name = tool_def.get("name", "unknown")
            _LOGGER.debug(
                "Creating MCP tool wrapper: %s_%s", name, tool_name
            )
            tools.append(
                MCPToolWrapper(
                    hass=hass,
                    client=client,
                    tool_definition=tool_def,
                    server_name=name,
                )
            )

        _LOGGER.info(
            "Created %d MCP tool wrappers from server %s", len(tools), server_url
        )
        return tools
