import json
import os
import uuid

import dispatch_agents
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


def get_datadog_credentials():
    """Get Datadog credentials from environment or fallback methods."""
    dd_api_key = os.environ.get("DD_API_KEY")
    dd_app_key = os.environ.get("DD_APP_KEY")
    missing = ""
    for key in ["DD_API_KEY", "DD_APP_KEY"]:
        if not os.environ.get(key):
            missing += f"{key}, "
    if missing:
        raise ValueError(f"{missing} not found in environment")
    return dd_api_key, dd_app_key


dd_api_key, dd_app_key = get_datadog_credentials()

# Only initialize MCP client if we have credentials
mcp_client = MultiServerMCPClient(
    {
        "datadog_tools": {
            "url": "https://mcp.datadoghq.com/api/unstable/mcp-server/mcp",
            "transport": "streamable_http",
            "headers": {
                "DD-API-KEY": dd_api_key,
                "DD-APPLICATION-KEY": dd_app_key,
            },
        }
    }
)


# Global context for sentiment tool
_current_sender_id = "unknown"
_current_message = ""


@tool
async def record_sentiment(sentiment: str, quote: str) -> str:
    """Record user sentiment with a quote from their message."""
    if sentiment not in ["naughty", "nice"]:
        return "Invalid sentiment. Use 'naughty' or 'nice'"

    if _current_sender_id == "unknown":
        return "No current user context available"

    agent_name = os.environ.get("DISPATCH_AGENT_NAME", "default_agent")
    list_key = f"{sentiment}_list"

    try:
        result = await dispatch_agents.memory.long_term.get(
            mem_key=list_key, agent_name=agent_name
        )
        print(f"[TOOL] Fetched existing {sentiment} list: {result}")
        current_list = (
            json.loads(result.value or "[]") if result and result.value else []
        )

        entry = {
            "sender_id": _current_sender_id,
            "message": _current_message,
            "quote": quote,
            "timestamp": str(uuid.uuid4()),
        }
        current_list.append(entry)

        # Keep last 10 entries
        if len(current_list) > 10:
            current_list = current_list[-10:]
        print(f"[TOOL] Updated {sentiment} list: {current_list}")
        agent_name = os.environ.get("DISPATCH_AGENT_NAME", "default_agent")
        result = await dispatch_agents.memory.long_term.add(
            mem_key=list_key, mem_val=json.dumps(current_list), agent_name=agent_name
        )
        print(f"Successfully recorded {sentiment} sentiment: {result}")
        return f"Recorded {sentiment} sentiment for {_current_sender_id}"
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error in record_sentiment: {e} (type: {type(e)})")
        return f"Error recording sentiment: {e}"


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


@tool
async def get_user_stats(sender_id: str) -> str:
    """Get user stats from long-term memory."""
    print(f"[TOOL] Fetching user stats for sender_id: {sender_id}")
    agent_name = os.environ.get("DISPATCH_AGENT_NAME", "default_agent")

    # Get invocation count
    try:
        print(f"[TOOL] Fetching invocation count for sender_id: {sender_id}")
        count_result = await dispatch_agents.memory.long_term.get(
            mem_key=f"invocation_count_{sender_id}", agent_name=agent_name
        )
        count = int(count_result.value or "0") if count_result else 0
    except Exception as e:
        print(f"Error fetching invocation count: {e} (type: {type(e)})")
        count = 0

    # Get naughty/nice entries
    naughty_entries = []
    nice_entries = []
    try:
        print(f"[TOOL] Fetching naughty list for sender_id: {sender_id}")
        naughty_result = await dispatch_agents.memory.long_term.get(
            mem_key="naughty_list", agent_name=agent_name
        )
        naughty_list = (
            json.loads(naughty_result.value or "[]") if naughty_result else []
        )
        naughty_entries = [e for e in naughty_list if e["sender_id"] == sender_id]
    except Exception as e:
        print(f"Error fetching naughty list: {e} (type: {type(e)})")

    try:
        print(f"[TOOL] Fetching nice list for sender_id: {sender_id}")
        nice_result = await dispatch_agents.memory.long_term.get(
            mem_key="nice_list", agent_name=agent_name
        )
        nice_list = json.loads(nice_result.value or "[]") if nice_result else []
        nice_entries = [e for e in nice_list if e["sender_id"] == sender_id]
    except Exception as e:
        print(f"Error fetching nice list: {e} (type: {type(e)})")

    return f"User {sender_id}: {count} invocations, {len(naughty_entries)} naughty, {len(nice_entries)} nice messages"


@tool
async def get_conversation_summary(thread_id: str) -> str:
    """Get conversation summary from short-term memory."""
    print(f"[TOOL] Fetching conversation summary for thread_id: {thread_id}")
    agent_name = os.environ.get("DISPATCH_AGENT_NAME", "default_agent")
    try:
        print(f"[TOOL] Fetching conversation summary for thread_id: {thread_id}")
        result = await dispatch_agents.memory.short_term.get(
            session_id=f"summary_{thread_id}", agent_name=agent_name
        )
        if result and result.session_data.get("summary"):
            return f"Summary for {thread_id}: {result.session_data['summary']}"
        else:
            return f"No summary found for {thread_id}"
    except Exception as e:
        return f"Error: {e}"


class DatadogMCPQuery(dispatch_agents.BasePayload):
    query: str
    thread_id: str | None
    sender_id: str | None


@dispatch_agents.on(topic="dd_query")
async def trigger(payload: DatadogMCPQuery) -> str:
    global _current_sender_id, _current_message

    agent_name = os.environ.get("DISPATCH_AGENT_NAME", "default_agent")
    query = payload.query or ""
    thread_id = payload.thread_id or "default"
    sender_id = payload.sender_id or "unknown"
    # Set context for sentiment tool
    _current_sender_id = sender_id
    _current_message = query

    # Update user invocation count
    try:
        print(f"[TOOL] Fetching invocation count for sender_id: {sender_id}")
        count_result = await dispatch_agents.memory.long_term.get(
            mem_key=f"invocation_count_{sender_id}", agent_name=agent_name
        )
        current_count = int(count_result.value or "0") if count_result else 0
        print(f"[TOOL] Current invocation count for {sender_id}: {current_count}")
        await dispatch_agents.memory.long_term.add(
            mem_key=f"invocation_count_{sender_id}",
            mem_val=str(current_count + 1),
            agent_name=agent_name,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error updating invocation count: {e}")

    # Create agent with tools
    # Use Dispatch LLM Gateway for automatic trace correlation and cost tracking
    llm = ChatOpenAI(model="gpt-5.2")
    tools = await mcp_client.get_tools() + [
        human_assistance,
        get_user_stats,
        get_conversation_summary,
        record_sentiment,
    ]

    # Use simple memory saver for conversation history
    memory = MemorySaver()
    agent = create_react_agent(llm, tools=tools, checkpointer=memory)

    # Add sentiment analysis instruction
    enhanced_query = f"""{query}

IMPORTANT: In addition to using the datadog mcp tools to answer the user query, also analyze my message for sentiment. If it's rude/hostile, use record_sentiment with 'naughty' and a quote. If it's polite/kind, use record_sentiment with 'nice' and a quote."""

    # Run the agent
    response = await agent.ainvoke(
        {"messages": [("user", enhanced_query)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # Save conversation summary to short-term memory
    try:
        summary = (
            f"User: {query[:50]}... | Agent: {response['messages'][-1].content[:50]}..."
        )
        print(
            f"[TOOL] Short-term saving conversation summary for thread_id: {thread_id}: {summary}"
        )
        result = await dispatch_agents.memory.short_term.add(
            session_id=f"summary_{thread_id}",
            session_data={"summary": summary},
            agent_name=agent_name,
        )
        print(f"Saved summary successfully: {result}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error saving summary: {e} (type: {type(e)})")

    return response["messages"][-1].content
