"""OpenAI Agents SDK integration for Dispatch Agents.

Use :func:`get_mcp_servers` to create MCP server connections compatible with
``agents.Agent(mcp_servers=...)``. The helper builds server configs from
``dispatch.yaml`` (plus any user-provided ``.mcp.json``) and injects Dispatch
trace context into MCP tool calls.
"""

from __future__ import annotations

from agents.mcp import MCPServerStreamableHttp
from agents.mcp.util import MCPToolMetaContext

from dispatch_agents._internal.dispatch import (
    get_current_invocation_id,
    get_current_trace_id,
    get_invocation_id_for_trace,
)
from dispatch_agents.mcp import get_mcp_servers_config
from dispatch_agents.models import JsonObject

__all__ = ["get_mcp_servers"]

_mcp_servers: list[MCPServerStreamableHttp] | None = None


def _trace_meta_resolver(context: MCPToolMetaContext) -> JsonObject | None:
    """Inject trace context into every MCP tool call via _meta."""
    meta: JsonObject = {}

    trace_id = get_current_trace_id()
    invocation_id = get_current_invocation_id()

    if not invocation_id and trace_id:
        invocation_id = get_invocation_id_for_trace(trace_id)

    if trace_id:
        meta["dispatch_trace_id"] = trace_id
    if invocation_id:
        meta["dispatch_invocation_id"] = invocation_id

    return meta if meta else None


async def get_mcp_servers() -> list[MCPServerStreamableHttp]:
    """Return connected MCP servers for the OpenAI Agents SDK.

    On first call, loads the merged MCP server config (dispatch-managed servers
    from ``dispatch.yaml`` plus any user-provided ``.mcp.json``), creates
    ``MCPServerStreamableHttp`` instances, and connects each server. Subsequent
    calls return the same connected server list.

    Trace context is automatically injected into each MCP tool call for
    distributed tracing.

    Returns:
        Connected servers ready to pass to ``Agent(mcp_servers=...)``.

    Example::

        from agents import Agent, Runner
        from dispatch_agents import init, on
        from dispatch_agents.contrib.openai import get_mcp_servers

        agent: Agent

        @init
        async def setup() -> None:
            global agent
            agent = Agent(
                name="assistant",
                instructions="Use MCP tools to answer questions.",
                mcp_servers=await get_mcp_servers(),
            )

        @on(topic="query")
        async def handle_query(payload: QueryRequest) -> QueryResponse:
            result = await Runner.run(agent, payload.prompt)
            return QueryResponse(result=result.final_output)
    """
    global _mcp_servers

    if _mcp_servers is not None:
        return _mcp_servers

    servers: list[MCPServerStreamableHttp] = []

    for server_name, server_config in get_mcp_servers_config().items():
        server = MCPServerStreamableHttp(
            name=server_name,
            params={
                "url": server_config.url,
                "headers": server_config.headers,
            },
            cache_tools_list=True,
            tool_meta_resolver=_trace_meta_resolver,
            client_session_timeout_seconds=300.0,
        )
        await server.connect()
        servers.append(server)

    _mcp_servers = servers
    return _mcp_servers
