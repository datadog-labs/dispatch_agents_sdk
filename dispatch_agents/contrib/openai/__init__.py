"""OpenAI Agents SDK integration for Dispatch Agents.

This module provides helpers for configuring MCP servers with the OpenAI Agents SDK.

Usage Example::

    from agents import Agent, Runner
    from dispatch_agents.contrib.openai import get_mcp_servers
    import dispatch_agents

    my_agent: Agent  # Module-level, initialized by @init

    @dispatch_agents.init
    async def setup():
        global my_agent
        mcp_servers = await get_mcp_servers()
        my_agent = Agent(
            name="MyAssistant",
            instructions="Use MCP tools to answer questions.",
            mcp_servers=mcp_servers,
        )

    @dispatch_agents.on(topic="query")
    async def handle_query(payload: QueryRequest) -> QueryResponse:
        result = await Runner.run(my_agent, payload.prompt)
        return QueryResponse(result=result.final_output)

Type Compatibility:
    The :func:`get_mcp_servers` function returns a ``list[MCPServerStreamableHttp]``
    which is directly compatible with ``Agent.mcp_servers``.

Trace Context:
    Trace context (trace_id, parent_id) is automatically injected into each MCP
    tool call via the ``_meta`` field in the MCP protocol. This enables distributed
    tracing across agent invocations.
"""

from typing import Any

from agents.mcp import MCPServerStreamableHttp
from agents.mcp.util import MCPToolMetaContext

from dispatch_agents.events import (
    get_current_invocation_id,
    get_current_trace_id,
    get_invocation_id_for_trace,
)
from dispatch_agents.mcp import _load_mcp_config

# Singleton storage for MCP server connections
_mcp_servers: list[MCPServerStreamableHttp] | None = None


def _trace_meta_resolver(context: MCPToolMetaContext) -> dict[str, Any] | None:
    """Inject trace context into every MCP tool call via _meta.

    This resolver is called by the OpenAI Agents SDK before each tool invocation.
    The returned dict is passed to the MCP server in the ``_meta`` field of the
    JSON-RPC request, enabling distributed tracing.

    Uses a fallback mechanism when Python context variables aren't properly
    propagated (e.g., when the SDK uses task pools created at init time).

    Args:
        context: Context information about the tool invocation (unused but required
            by the MCPToolMetaResolver protocol).

    Returns:
        A dict with trace context fields, or None if no trace context is available.
    """
    meta: dict[str, Any] = {}

    trace_id = get_current_trace_id()
    invocation_id = get_current_invocation_id()

    # If invocation_id isn't available from context variables, try fallback lookup.
    # This handles cases where the SDK doesn't properly propagate Python context
    # (e.g., task pools, worker threads, or contexts created during init).
    if not invocation_id and trace_id:
        invocation_id = get_invocation_id_for_trace(trace_id)

    if trace_id:
        meta["dispatch_trace_id"] = trace_id
    if invocation_id:
        meta["dispatch_invocation_id"] = invocation_id

    return meta if meta else None


async def get_mcp_servers() -> list[MCPServerStreamableHttp]:
    """Get MCP servers for OpenAI Agents SDK.

    Returns a singleton list of connected MCP servers. On first call, loads
    configuration from ``.mcp.json``, creates ``MCPServerStreamableHttp``
    instances, and connects to each server. Subsequent calls return the
    same connected servers.

    Trace context is automatically injected into each MCP tool call via the
    ``tool_meta_resolver`` callback, which populates the ``_meta`` field in
    the MCP protocol. This enables distributed tracing without requiring
    per-request connection setup.

    Returns:
        A list of connected ``MCPServerStreamableHttp`` instances ready for use
        with ``Agent(mcp_servers=...)``.

    Raises:
        FileNotFoundError: If ``.mcp.json`` config file is not found.
            Ensure ``mcp_servers`` is declared in ``.dispatch.yaml``
            and the agent is deployed.

    Example::

        from agents import Agent, Runner
        from dispatch_agents.contrib.openai import get_mcp_servers
        import dispatch_agents

        my_agent: Agent  # Initialized by @init

        @dispatch_agents.init
        async def setup():
            global my_agent
            mcp_servers = await get_mcp_servers()
            my_agent = Agent(
                name="MyAgent",
                instructions="Use MCP tools to help the user.",
                mcp_servers=mcp_servers,
            )

        @dispatch_agents.on(topic="query")
        async def handle(payload: QueryRequest) -> QueryResponse:
            result = await Runner.run(my_agent, payload.prompt)
            return QueryResponse(result=result.final_output)

    See Also:
        - OpenAI Agents SDK documentation for ``Agent`` and ``Runner``.
    """
    global _mcp_servers

    if _mcp_servers is not None:
        return _mcp_servers

    config = _load_mcp_config()
    servers: list[MCPServerStreamableHttp] = []

    for server_name, server_config in config.get("mcpServers", {}).items():
        headers = dict(server_config.get("headers", {}))

        server = MCPServerStreamableHttp(
            name=server_name,
            params={
                "url": server_config.get("url", ""),
                "headers": headers,
            },
            cache_tools_list=True,
            tool_meta_resolver=_trace_meta_resolver,
            # Override the default 5-second timeout which is too short for LLM tool calls
            client_session_timeout_seconds=300.0,
        )
        await server.connect()
        servers.append(server)

    _mcp_servers = servers
    return _mcp_servers


__all__ = ["MCPServerStreamableHttp", "get_mcp_servers"]
