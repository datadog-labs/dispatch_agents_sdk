"""Claude Agent SDK integration for Dispatch Agents.

This module provides helpers for configuring MCP servers with the Claude Agent SDK.

Usage Example::

    from dispatch_agents.contrib.claude import get_mcp_servers

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
    import dispatch_agents

    my_options: ClaudeAgentOptions  # Module-level, initialized by @init

    @dispatch_agents.init
    async def setup():
        global my_options
        mcp_servers = await get_mcp_servers()
        my_options = ClaudeAgentOptions(
            mcp_servers=mcp_servers,
            allowed_tools=["mcp__datadog__*"],
            permission_mode="bypassPermissions",
        )

    @dispatch_agents.on(topic="query")
    async def handle_query(payload: QueryRequest) -> QueryResponse:
        async for message in query(prompt=payload.prompt, options=my_options):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                return QueryResponse(result=message.result)

Type Compatibility:
    The :func:`get_mcp_servers` function returns a ``dict[str, McpSdkServerConfig]``
    which is directly compatible with ``ClaudeAgentOptions.mcp_servers``.

Trace Context:
    Trace context (trace_id, parent_id) is automatically injected into each MCP
    tool call for distributed tracing. This enables correlation of tool calls
    with the parent agent invocation in the Dispatch dashboard.
"""

import logging
from datetime import timedelta
from typing import Any

from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, create_sdk_mcp_server
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

from dispatch_agents.mcp import _load_mcp_config, get_mcp_client

# Default timeout for MCP tool calls (5 minutes)
DEFAULT_READ_TIMEOUT_SECONDS = 300.0

logger = logging.getLogger(__name__)

# Singleton storage for MCP server configs
_mcp_servers: dict[str, McpSdkServerConfig] | None = None


async def _get_server_info_and_tools(
    server_name: str,
    url: str,
    headers: dict[str, str],
) -> tuple[str | None, list[Tool]]:
    """Connect to MCP server and retrieve server info and tools.

    Args:
        server_name: Name of the server (for logging).
        url: The MCP server endpoint URL.
        headers: HTTP headers for authentication.

    Returns:
        Tuple of (server_version, list of tools). Version may be None if not provided.
    """
    server_version: str | None = None
    tools: list[Tool] = []

    try:
        async with streamablehttp_client(url=url, headers=headers) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=DEFAULT_READ_TIMEOUT_SECONDS),
            ) as session:
                # Initialize and get server info
                init_result = await session.initialize()
                if init_result.serverInfo and init_result.serverInfo.version:
                    server_version = init_result.serverInfo.version

                # Fetch tools list
                tools_result = await session.list_tools()
                tools = list(tools_result.tools)

    except Exception as e:
        logger.warning(
            f"Failed to connect to MCP server '{server_name}' at {url}: {e}. "
            "The server may not be available yet."
        )

    return server_version, tools


def _create_proxy_tool(
    server_name: str,
    tool: Tool,
) -> SdkMcpTool[Any]:
    """Create a proxy tool that forwards calls to an upstream MCP server.

    Args:
        server_name: The MCP server name (as configured in .mcp.json).
        tool: Tool definition from the upstream server.

    Returns:
        An SdkMcpTool that proxies to the upstream server with trace context.
    """
    tool_name = tool.name
    description = tool.description or ""
    input_schema = tool.inputSchema if tool.inputSchema else {"type": "object"}

    async def proxy_handler(args: dict[str, Any]) -> dict[str, Any]:
        """Proxy handler that forwards to upstream with trace context."""
        try:
            async with get_mcp_client(server_name) as client:
                result = await client.call_tool(tool_name, args)

            # Convert MCP CallToolResult to Claude SDK format
            content = [
                {"type": c.type, "text": getattr(c, "text", str(c))}
                for c in result.content
            ]
            return {"content": content, "is_error": result.isError or False}

        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on '{server_name}': {e}")
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "is_error": True,
            }

    return SdkMcpTool(
        name=tool_name,
        description=description,
        input_schema=input_schema,
        handler=proxy_handler,
    )


async def _create_proxy_server(
    server_name: str,
    url: str,
    headers: dict[str, str],
) -> McpSdkServerConfig:
    """Create an SDK server that proxies to an HTTP MCP server with trace context.

    Args:
        server_name: Name for the proxy server.
        url: The upstream MCP server URL.
        headers: HTTP headers for the upstream server.

    Returns:
        McpSdkServerConfig for the proxy server.
    """
    # Get server info and tools using MCP SDK
    server_version, tools = await _get_server_info_and_tools(server_name, url, headers)

    # Create proxy tools
    proxy_tools = [_create_proxy_tool(server_name, tool) for tool in tools]

    # Create SDK server with proxy tools
    return create_sdk_mcp_server(
        name=server_name,
        version=server_version,
        tools=proxy_tools,
    )


async def get_mcp_servers() -> dict[str, McpSdkServerConfig]:
    """Get MCP servers for Claude Agent SDK.

    Returns a singleton dict of SDK server configurations. On first call, loads
    configuration from ``.mcp.json`` and creates server configurations.
    Subsequent calls return the cached servers.

    Trace context (trace_id, parent_id) is automatically injected into each MCP
    tool call for distributed tracing.

    Returns:
        A dict mapping server names to their SDK server configuration.
        This can be passed directly to ``ClaudeAgentOptions(mcp_servers=...)``.

    Raises:
        FileNotFoundError: If ``.mcp.json`` config file is not found.
            Ensure ``mcp_servers`` is declared in ``.dispatch.yaml``
            and the agent is deployed.

    Example::

        from dispatch_agents.contrib.claude import get_mcp_servers
        from claude_agent_sdk import ClaudeAgentOptions, query
        import dispatch_agents

        my_options: ClaudeAgentOptions  # Initialized by @init

        @dispatch_agents.init
        async def setup():
            global my_options
            mcp_servers = await get_mcp_servers()
            my_options = ClaudeAgentOptions(
                mcp_servers=mcp_servers,
                allowed_tools=["mcp__datadog__*"],
                permission_mode="bypassPermissions",
            )

        @dispatch_agents.on(topic="query")
        async def handle(payload: QueryRequest) -> QueryResponse:
            async for message in query(prompt=payload.prompt, options=my_options):
                if isinstance(message, ResultMessage) and message.subtype == "success":
                    return QueryResponse(result=message.result)

    See Also:
        - Claude Agent SDK documentation for ``ClaudeAgentOptions``.
    """
    global _mcp_servers

    if _mcp_servers is not None:
        return _mcp_servers

    config = _load_mcp_config()
    servers: dict[str, McpSdkServerConfig] = {}

    for server_name, server_config in config.get("mcpServers", {}).items():
        url = server_config.get("url", "")
        headers = dict(server_config.get("headers", {}))

        proxy_server = await _create_proxy_server(server_name, url, headers)
        servers[server_name] = proxy_server

    _mcp_servers = servers
    return _mcp_servers


__all__ = ["McpSdkServerConfig", "get_mcp_servers"]
