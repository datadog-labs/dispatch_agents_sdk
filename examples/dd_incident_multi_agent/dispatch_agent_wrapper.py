"""
Simple Dispatch Agent Wrapper that implements ainvoke() interface.
This allows using dispatch agents as if they were regular LLM agents in LangGraph.
"""

import os
import uuid
from typing import Any

import dispatch_agents
from dispatch_agents import BasePayload
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class DatadogMCPQuery(BasePayload):
    query: str
    thread_id: str
    sender_id: str


class DispatchAgentWrapper:
    """
    A simple wrapper that makes dispatch agents look like regular LangChain agents.
    Implements the ainvoke() interface using dispatch_agents.invoke() internally.
    """

    def __init__(
        self,
        agent: str = "dd-logs-agentic-search",
        function: str = "trigger",
        timeout: int = 60,
    ):
        self.agent = "dd-logs-agentic-search"
        self.function = function
        self.timeout = timeout

    def _messages_to_query(self, messages: list[BaseMessage]) -> str:
        """
        Convert LangChain messages to a single query string for the dispatch agent.

        Args:
            messages: List of LangChain messages

        Returns:
            str: Formatted query combining system prompt and user message
        """
        system_parts = []
        user_parts = []

        for message in messages:
            content = message.content if hasattr(message, "content") else str(message)

            if isinstance(message, SystemMessage):
                system_parts.append(content)
            elif isinstance(message, HumanMessage | tuple) or (
                hasattr(message, "type") and message.type == "human"
            ):
                if isinstance(message, tuple):
                    user_parts.append(message[1])  # Extract content from tuple format
                else:
                    user_parts.append(content)
            elif hasattr(message, "type") and message.type == "user":
                user_parts.append(content)

        # Combine system and user parts
        query_parts = []
        if system_parts:
            query_parts.append("SYSTEM INSTRUCTIONS:\n" + "\n".join(system_parts))
        if user_parts:
            query_parts.append("USER REQUEST:\n" + "\n".join(user_parts))

        return "\n\n".join(query_parts)

    async def ainvoke(
        self, messages: list[BaseMessage], config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Invoke the dispatch agent with messages, similar to regular LLM ainvoke.

        Args:
            messages: List of messages (SystemMessage, HumanMessage, etc.)
            config: Optional config with thread_id and other parameters

        Returns:
            Dict with 'messages' containing the agent response
        """
        # Extract thread_id from config if available
        thread_id = None
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")

        # Generate unique thread_id if not provided
        if not thread_id:
            thread_id = f"dispatch_agent_{uuid.uuid4().hex[:8]}"

        # Convert messages to query format
        query = self._messages_to_query(messages)

        # Prepare payload for dispatch agent
        payload = DatadogMCPQuery(
            query=query,
            thread_id=thread_id,
            sender_id=os.getenv("DISPATCH_AGENT_NAME", "unknown-agent"),
        ).model_dump()

        try:
            # Call the dispatch agent
            response = await dispatch_agents.invoke(
                agent_name=self.agent,
                function_name=self.function,
                payload=payload,
                timeout=self.timeout,
            )

            # Return in expected LangChain format
            return {"messages": [AIMessage(content=str(response))]}

        except Exception as e:
            error_response = f"Error calling dispatch agent: {str(e)}"
            return {"messages": [AIMessage(content=error_response)]}
