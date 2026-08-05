"""Claude Agent SDK integration for Dispatch Agents.

Use :func:`get_mcp_servers` to create MCP server configurations compatible with
``ClaudeAgentOptions(mcp_servers=...)``. The helper builds server configs from
``dispatch.yaml`` (plus any user-provided ``.mcp.json``) and proxies MCP tool
calls with Dispatch trace context.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import httpx
from claude_agent_sdk import McpSdkServerConfig, SdkMcpTool, create_sdk_mcp_server
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import Tool

from dispatch_agents.mcp import get_mcp_client, get_mcp_servers_config
from dispatch_agents.models import JsonObject, JsonValue

__all__ = ["get_mcp_servers"]

_DEFAULT_READ_TIMEOUT_SECONDS = 300.0

_logger = logging.getLogger(__name__)

_mcp_servers: dict[str, McpSdkServerConfig] | None = None


async def _get_server_info_and_tools(
    server_name: str,
    url: str,
    headers: dict[str, str],
) -> tuple[str | None, list[Tool]]:
    """Connect to an MCP server and retrieve server info and tools."""
    server_version: str | None = None
    tools: list[Tool] = []

    try:
        async with (
            httpx.AsyncClient(headers=headers) as http_client,
            streamable_http_client(url=url, http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ),
        ):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=_DEFAULT_READ_TIMEOUT_SECONDS),
            ) as session:
                init_result = await session.initialize()
                if init_result.serverInfo and init_result.serverInfo.version:
                    server_version = init_result.serverInfo.version

                tools_result = await session.list_tools()
                tools = list(tools_result.tools)

    except Exception as e:
        _logger.warning(
            f"Failed to connect to MCP server '{server_name}' at {url}: {e}. "
            "The server may not be available yet."
        )

    return server_version, tools


def _create_proxy_tool(
    server_name: str,
    tool: Tool,
) -> SdkMcpTool[JsonObject]:
    """Create a proxy tool that forwards calls to an upstream MCP server."""
    tool_name = tool.name
    description = tool.description or ""
    input_schema = tool.inputSchema if tool.inputSchema else {"type": "object"}

    async def proxy_handler(args: JsonObject) -> JsonObject:
        """Proxy handler that forwards to upstream with trace context."""
        try:
            async with get_mcp_client(server_name) as client:
                result = await client.call_tool(tool_name, args)

            content: list[JsonValue] = []
            content.extend(result.content)
            return {"content": content, "is_error": result.is_error}

        except Exception as e:
            _logger.error(f"Error calling tool '{tool_name}' on '{server_name}': {e}")
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
    """Create an SDK server that proxies to an HTTP MCP server."""
    server_version, tools = await _get_server_info_and_tools(server_name, url, headers)
    proxy_tools = [_create_proxy_tool(server_name, tool) for tool in tools]
    return create_sdk_mcp_server(
        name=server_name,
        version=server_version,
        tools=proxy_tools,
    )


async def get_mcp_servers() -> dict[str, McpSdkServerConfig]:
    """Return MCP server configs for the Claude Agent SDK.

    On first call, loads the merged MCP server config (dispatch-managed servers
    from ``dispatch.yaml`` plus any user-provided ``.mcp.json``) and creates SDK
    server configurations. Subsequent calls return the cached server mapping.

    Trace context (trace_id, parent_id) is automatically injected into each MCP
    tool call for distributed tracing.

    Returns:
        Mapping of server name to Claude SDK server configuration.

    Example::

        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
        from dispatch_agents import init, on
        from dispatch_agents.contrib.claude import get_mcp_servers

        options: ClaudeAgentOptions

        @init
        async def setup() -> None:
            global options
            options = ClaudeAgentOptions(
                mcp_servers=await get_mcp_servers(),
                allowed_tools=["mcp__datadog__*"],
                permission_mode="bypassPermissions",
            )

        @on(topic="query")
        async def handle_query(payload: QueryRequest) -> QueryResponse:
            async for message in query(prompt=payload.prompt, options=options):
                if isinstance(message, ResultMessage) and message.subtype == "success":
                    return QueryResponse(result=message.result)
            return QueryResponse(result="")
    """
    global _mcp_servers

    if _mcp_servers is not None:
        return _mcp_servers

    servers: dict[str, McpSdkServerConfig] = {}

    for server_name, server_config in get_mcp_servers_config().items():
        proxy_server = await _create_proxy_server(
            server_name,
            server_config.url,
            server_config.headers,
        )
        servers[server_name] = proxy_server

    _mcp_servers = servers
    return _mcp_servers
