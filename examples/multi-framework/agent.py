"""Multi-Framework Agent — rotate between AI frameworks with state tracking.

Demonstrates:
- Long-term Memory — persist rotation state across invocations
- Multiple LLM frameworks — Claude Agent SDK, OpenAI Agents SDK, LangGraph
- @fn() callable function — designed for cron or manual invocation

Each invocation rotates to the next AI framework and asks it to answer a
user-provided question (or a default one). The rotation index is persisted
in Dispatch's long-term memory so the agent picks up where it left off.

This is useful for:
- Comparing framework outputs side by side
- Testing framework interoperability on Dispatch
- Demonstrating that Dispatch is framework-agnostic
"""

import logging
from typing import TypedDict

from dispatch_agents import BasePayload, fn, memory
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

MEMORY_KEY = "last_framework_index"

FRAMEWORKS = [
    "claude_agent_sdk",
    "openai_agents_sdk",
    "langgraph_openai",
]

FRAMEWORK_DISPLAY = {
    "claude_agent_sdk": "Claude Agent SDK",
    "openai_agents_sdk": "OpenAI Agents SDK",
    "langgraph_openai": "LangGraph + OpenAI",
}


class AskRequest(BasePayload):
    """Input payload."""

    question: str = "What is the most important thing to know about building AI agents?"


class AskResponse(BasePayload):
    """Output payload."""

    answer: str
    framework: str


# ---------------------------------------------------------------------------
# Framework callers
# ---------------------------------------------------------------------------


async def _call_claude_sdk(question: str) -> str:
    """Answer using the Claude Agent SDK."""
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    options = ClaudeAgentOptions(
        allowed_tools=[],
        permission_mode="bypassPermissions",
    )

    result = ""
    async for message in query(prompt=question, options=options):
        if isinstance(message, ResultMessage) and hasattr(message, "result"):
            result = message.result

    return result or "Claude had no response."


async def _call_openai_agents(question: str) -> str:
    """Answer using the OpenAI Agents SDK."""
    from agents import Agent, Runner

    agent = Agent(
        name="Assistant",
        instructions="You are a helpful, concise assistant.",
    )
    run_result = await Runner.run(agent, question)
    return run_result.final_output or "OpenAI agent had no response."


class _LangGraphState(TypedDict):
    question: str
    answer: str


async def _call_langgraph(question: str) -> str:
    """Answer using LangGraph with OpenAI."""

    async def answer_question(state: _LangGraphState) -> _LangGraphState:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        response = await model.ainvoke(state["question"])
        return {"question": state["question"], "answer": response.content}

    workflow = StateGraph(_LangGraphState)
    workflow.add_node("answer", answer_question)
    workflow.set_entry_point("answer")
    workflow.add_edge("answer", END)
    app = workflow.compile()

    result = await app.ainvoke({"question": question, "answer": ""})
    return result["answer"] or "LangGraph had no response."


CALLERS = {
    "claude_agent_sdk": _call_claude_sdk,
    "openai_agents_sdk": _call_openai_agents,
    "langgraph_openai": _call_langgraph,
}


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------


async def _get_last_index() -> int:
    """Get the last framework index from memory. Returns -1 on first run."""
    result = await memory.long_term.get(mem_key=MEMORY_KEY)
    if result and result.value is not None:
        return int(result.value)
    return -1


async def _save_index(index: int) -> None:
    await memory.long_term.add(mem_key=MEMORY_KEY, mem_val=str(index))


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------


@fn()
async def ask(payload: AskRequest) -> AskResponse:
    """Ask a question using the next AI framework in the rotation.

    Each call rotates through Claude Agent SDK → OpenAI Agents SDK →
    LangGraph + OpenAI, persisting the rotation state in long-term memory.

    Designed for cron schedules or manual invocation.

    Example::

        {"question": "Explain event-driven architecture in 3 sentences."}
    """
    last_index = await _get_last_index()
    next_index = (last_index + 1) % len(FRAMEWORKS)
    framework_key = FRAMEWORKS[next_index]
    display_name = FRAMEWORK_DISPLAY[framework_key]

    logger.info(f"Using framework: {display_name} (index {next_index})")

    caller = CALLERS[framework_key]
    answer = await caller(payload.question)

    await _save_index(next_index)

    logger.info(f"Response from {display_name}: {answer[:100]}...")

    return AskResponse(answer=answer, framework=display_name)
