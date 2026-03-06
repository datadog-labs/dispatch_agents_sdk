"""MCP client utilities for Dispatch agents."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Literal, NotRequired, TypedDict

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.session import ProgressFnT
from mcp.types import (
    AnyUrl,
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
)

from dispatch_agents.events import (
    get_current_invocation_id,
    get_current_trace_id,
    get_invocation_id_for_trace,
)

MCP_CONFIG_PATH = os.environ.get("MCP_CONFIG_PATH", ".mcp.json")


class _TraceMeta(TypedDict, total=False):
    """Trace context metadata for MCP _meta field."""

    dispatch_trace_id: str
    dispatch_invocation_id: str


def _build_trace_meta() -> _TraceMeta | None:
    """Build trace context metadata for MCP _meta field.

    Uses a fallback mechanism when Python context variables aren't properly
    propagated (e.g., when external SDKs use task pools or worker threads).

    Returns:
        A _TraceMeta with trace context fields, or None if no trace context is available.
    """
    meta: _TraceMeta = {}

    trace_id = get_current_trace_id()
    invocation_id = get_current_invocation_id()

    # If invocation_id isn't available from context variables, try fallback lookup.
    # This handles cases where external SDKs (Claude, OpenAI) don't properly
    # propagate Python context (e.g., task pools, worker threads).
    if not invocation_id and trace_id:
        invocation_id = get_invocation_id_for_trace(trace_id)

    if trace_id:
        meta["dispatch_trace_id"] = trace_id
    if invocation_id:
        meta["dispatch_invocation_id"] = invocation_id

    return meta if meta else None


class TracingClientSession:
    """Wrapper around MCP ClientSession that injects trace context into tool calls.

    This wrapper intercepts call_tool calls and automatically injects trace context
    into the _meta field for distributed tracing.
    """

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: timedelta | None = None,
        progress_callback: ProgressFnT | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call a tool with automatic trace context injection.

        Trace context is automatically injected into the meta field for
        distributed tracing. Any user-provided meta is merged with trace context.
        """
        # Merge trace context with user-provided meta
        trace_meta = _build_trace_meta()
        merged_meta: dict[str, Any] | None
        if trace_meta:
            if meta:
                merged_meta = {**trace_meta, **meta}
            else:
                merged_meta = dict(trace_meta)
        else:
            merged_meta = meta

        return await self._session.call_tool(
            name=name,
            arguments=arguments,
            read_timeout_seconds=read_timeout_seconds,
            progress_callback=progress_callback,
            meta=merged_meta,
        )

    # Delegate other methods to the underlying session
    async def list_tools(self) -> ListToolsResult:
        """List available tools from the MCP server."""
        return await self._session.list_tools()

    async def list_resources(self) -> ListResourcesResult:
        """List available resources from the MCP server."""
        return await self._session.list_resources()

    async def read_resource(self, uri: AnyUrl) -> ReadResourceResult:
        """Read a resource from the MCP server."""
        return await self._session.read_resource(uri)

    async def list_prompts(self) -> ListPromptsResult:
        """List available prompts from the MCP server."""
        return await self._session.list_prompts()

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> GetPromptResult:
        """Get a prompt from the MCP server."""
        return await self._session.get_prompt(name, arguments)


class McpHttpServerConfig(TypedDict):
    """MCP HTTP server configuration compatible with Claude Agent SDK.

    This matches the McpHttpServerConfig type from claude_agent_sdk.types.
    """

    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]


def _load_mcp_config() -> dict[str, Any]:
    """Load MCP configuration from .mcp.json file."""
    if not os.path.exists(MCP_CONFIG_PATH):
        raise FileNotFoundError(
            f"MCP configuration file not found: {MCP_CONFIG_PATH}. "
            "Ensure mcp_servers is declared in dispatch.yaml and the agent is deployed."
        )
    with open(MCP_CONFIG_PATH) as f:
        return json.load(f)


def _get_server_config(server_name: str) -> dict[str, Any]:
    """Get configuration for a specific MCP server."""
    config = _load_mcp_config()
    servers = config.get("mcpServers", {})
    if server_name not in servers:
        available = list(servers.keys())
        raise ValueError(
            f"MCP server '{server_name}' not found in config. "
            f"Available servers: {available}"
        )
    return servers[server_name]


def get_mcp_servers_config() -> dict[str, McpHttpServerConfig]:
    """Get MCP servers configuration as HTTP transport configs.

    Loads MCP server configuration from .mcp.json and converts it to
    a dict of HTTP transport configurations.

    This format is compatible with Claude Agent SDK's mcp_servers option
    and other MCP clients that support HTTP transport.

    Note:
        This function returns raw configurations without automatic trace context
        injection. For automatic tracing, use :func:`get_mcp_client` or the
        SDK-specific helpers in ``dispatch_agents.contrib.openai`` and
        ``dispatch_agents.contrib.claude``.

    Returns:
        Dict mapping server names to their HTTP transport configuration:
        {
            "server_name": {
                "type": "http",
                "url": "https://...",
                "headers": {"Authorization": "Bearer ...", ...}
            }
        }

    Raises:
        FileNotFoundError: If .mcp.json config file not found

    Example:
        from dispatch_agents import get_mcp_servers_config

        mcp_servers = get_mcp_servers_config()
        # Use with Claude Agent SDK or other MCP clients
    """
    config = _load_mcp_config()
    mcp_servers: dict[str, McpHttpServerConfig] = {}

    for server_name, server_config in config.get("mcpServers", {}).items():
        headers = dict(server_config.get("headers", {}))
        mcp_servers[server_name] = {
            "type": "http",
            "url": server_config.get("url", ""),
            "headers": headers,
        }

    return mcp_servers


@asynccontextmanager
async def get_mcp_client(
    server_name: str,
    *,
    timeout: float = 30.0,
    read_timeout: float = 300.0,
) -> AsyncIterator[TracingClientSession]:
    """Get an MCP client session for the specified server.

    Args:
        server_name: Server name as declared in dispatch.yaml mcp_servers
        timeout: HTTP connection timeout in seconds (default: 30)
        read_timeout: Timeout for waiting for responses in seconds (default: 300).
            This is the maximum time to wait for a tool call response.

    Yields:
        Initialized TracingClientSession ready for MCP operations. Trace context
        is automatically injected into tool calls via the ``_meta`` field.

    Raises:
        FileNotFoundError: If .mcp.json config file not found
        ValueError: If server_name not found in config

    Example:
        async with get_mcp_client("com.datadoghq.mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("tool_name", {"arg": "value"})
    """
    server_config = _get_server_config(server_name)

    url = server_config.get("url")
    if not url:
        raise ValueError(f"MCP server '{server_name}' missing 'url' in config")

    # Use config headers for authentication
    headers = server_config.get("headers", {})

    # Connect to MCP server with auth headers
    async with streamablehttp_client(url=url, headers=headers, timeout=timeout) as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=read_timeout),
        ) as session:
            await session.initialize()
            yield TracingClientSession(session)
