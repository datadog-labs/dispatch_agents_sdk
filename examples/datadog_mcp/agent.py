"""
Datadog MCP Agent - Uses MCP Registry proxy to access Datadog tools.

This agent demonstrates three methods of connecting to MCP servers:
1. Dispatch SDK MCP client (handle_query) - recommended approach
2. Claude Agent SDK (handle_claude_query)
3. OpenAI Agents SDK (handle_openai_query)

When deployed, .mcp.json is automatically mounted at /app/.mcp.json with
HTTP transport configuration. All methods use this configuration.
"""

import os

# Disable OpenAI Agents SDK tracing - our OpenAI org has zero data retention (ZDR)
# which blocks trace ingestion and causes noisy ERROR logs
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "true")

import dispatch_agents

# OpenAI Agents SDK imports
from agents import Agent as OpenAIAgent
from agents import Runner as OpenAIRunner

# Claude Agent SDK imports
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk import ResultMessage as ClaudeResultMessage
from claude_agent_sdk import query as claude_query

# Dispatch SDK contrib packages for MCP server configuration
from dispatch_agents.contrib.claude import get_mcp_servers as get_claude_mcp_servers
from dispatch_agents.contrib.openai import get_mcp_servers as get_openai_mcp_servers

# MCP server installation name as declared in .dispatch.yaml
MCP_SERVER_NAME = "datadog"

# Module-level agents and config, initialized by @init
openai_datadog_agent: OpenAIAgent
claude_options: ClaudeAgentOptions


@dispatch_agents.init
async def setup():
    """Initialize MCP servers for both OpenAI and Claude agents."""
    global openai_datadog_agent, claude_options

    # Initialize OpenAI agent with MCP servers
    openai_mcp_servers = await get_openai_mcp_servers()
    openai_datadog_agent = OpenAIAgent(
        name="DatadogAssistant",
        instructions="Use the MCP tools to answer questions about Datadog.",
        mcp_servers=openai_mcp_servers,
    )

    # Initialize Claude options with MCP servers
    claude_mcp_servers = await get_claude_mcp_servers()
    claude_options = ClaudeAgentOptions(
        mcp_servers=claude_mcp_servers,
        allowed_tools=[f"mcp__{MCP_SERVER_NAME}__*"],
        permission_mode="bypassPermissions",
    )


class DatadogQuery(dispatch_agents.BasePayload):
    """Query payload for the Datadog MCP agent."""

    query: str
    tool_name: str | None = None


class DatadogQueryResponse(dispatch_agents.BasePayload):
    """Response from a Datadog query."""

    tool_names: list[str] | None = None
    result: dict | None = None


@dispatch_agents.fn()
async def query_datadog(request: DatadogQuery) -> DatadogQueryResponse:
    """Query Datadog using MCP tools via Dispatch SDK."""
    async with dispatch_agents.get_mcp_client(MCP_SERVER_NAME) as client:
        if request.tool_name:
            # Call specific tool
            result = await client.call_tool(request.tool_name, {"query": request.query})
            content = [c.model_dump() for c in result.content] if result.content else []
            return DatadogQueryResponse(
                result={"content": content, "isError": result.isError}
            )
        else:
            # List available tools
            tools = await client.list_tools()
            return DatadogQueryResponse(tool_names=[t.name for t in tools.tools])


# --- Shared payload types for Agent SDK integrations ---


class QueryRequest(dispatch_agents.BasePayload):
    """Query payload for Agent SDK handlers."""

    prompt: str


class QueryResponse(dispatch_agents.BasePayload):
    """Response from Agent SDK query."""

    result: str


# --- Claude Agent SDK Integration ---


@dispatch_agents.on(topic="datadog_mcp_query")
async def handle_claude_query(payload: QueryRequest) -> QueryResponse:
    """Handle a query using Claude Agent SDK with MCP tools."""
    result_text: str | None = None
    async for message in claude_query(prompt=payload.prompt, options=claude_options):
        if isinstance(message, ClaudeResultMessage) and message.subtype == "success":
            result_text = message.result

    if result_text is None:
        raise RuntimeError("No result returned from Claude query")

    return QueryResponse(result=result_text)


# --- OpenAI Agents SDK Integration ---


@dispatch_agents.on(topic="datadog_mcp_query")
async def handle_openai_query(payload: QueryRequest) -> QueryResponse:
    """Handle a query using OpenAI Agents SDK with MCP tools."""
    result = await OpenAIRunner.run(openai_datadog_agent, payload.prompt)
    return QueryResponse(result=result.final_output)
