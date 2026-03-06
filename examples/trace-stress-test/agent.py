"""Trace Stress Test Agent — exercises the trace builder with complex, nested traces.

This agent spawns parallel Claude Agent SDK subagents to produce traces with
multiple concurrent subprocess_id values.  The goal is to:

  A) Verify that subprocess_id propagates end-to-end (SDK → proxy → ClickHouse → trace UI).
  B) Stress-test the trace builder / tree view with deep, wide, and interleaved LLM calls.

Functions (ordered by complexity):

  1. simple_llm_call        — single Claude query, no tools. Baseline trace.
  2. tool_using_query        — single Claude query with tool use (Bash).
  3. parallel_subagents      — spawns N subagents that all run concurrently.
  4. nested_subagents        — subagent that itself spawns a deeper subagent (2 levels).
  5. stress_test             — combines all of the above in one invocation.
"""

import asyncio

from claude_agent_sdk import (
    AgentDefinition,
    ClaudeAgentOptions,
    ResultMessage,
)
from claude_agent_sdk import (
    query as claude_query,
)
from dispatch_agents import BasePayload, fn

# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


class SimpleRequest(BasePayload):
    """Minimal request for a single LLM call."""

    prompt: str


class SimpleResponse(BasePayload):
    """Response wrapper carrying the LLM result text."""

    result: str


class ParallelRequest(BasePayload):
    """Request to spawn N parallel subagents."""

    prompts: list[str]


class ParallelResponse(BasePayload):
    """Aggregated results from parallel subagent runs."""

    results: list[str]


class StressTestRequest(BasePayload):
    """Full stress-test configuration."""

    parallel_count: int = 3
    enable_tool_use: bool = True
    enable_nesting: bool = True


class StressTestResponse(BasePayload):
    """Outcome summary from the stress test."""

    simple_result: str
    tool_result: str | None = None
    parallel_results: list[str]
    nested_result: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_claude(
    prompt: str,
    *,
    options: ClaudeAgentOptions | None = None,
) -> str:
    """Run a single Claude Agent SDK query and return the result text."""
    opts = options or ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model="haiku",
    )
    result_text: str | None = None
    async for message in claude_query(prompt=prompt, options=opts):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            result_text = message.result
    if result_text is None:
        raise RuntimeError("No result returned from Claude query")
    return result_text


# ---------------------------------------------------------------------------
# 1. simple_llm_call — baseline single LLM call
# ---------------------------------------------------------------------------


@fn()
async def simple_llm_call(payload: SimpleRequest) -> SimpleResponse:
    """Single Claude query with no tools.  Produces a minimal 1-LLM-call trace."""
    result = await _run_claude(
        payload.prompt,
        options=ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            model="claude-haiku-4-5",
            allowed_tools=[],  # No tools — pure LLM call
        ),
    )
    return SimpleResponse(result=result)


# ---------------------------------------------------------------------------
# 2. tool_using_query — Claude with tool use (generates tool_calls in trace)
# ---------------------------------------------------------------------------


@fn()
async def tool_using_query(payload: SimpleRequest) -> SimpleResponse:
    """Claude query that uses Bash tool.  Adds tool_calls nodes to the trace."""
    result = await _run_claude(
        payload.prompt,
        options=ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            model="claude-haiku-4-5",
            allowed_tools=["Bash"],
        ),
    )
    return SimpleResponse(result=result)


# ---------------------------------------------------------------------------
# 3. parallel_subagents — spawn N Claude subagents concurrently
# ---------------------------------------------------------------------------


@fn()
async def parallel_subagents(payload: ParallelRequest) -> ParallelResponse:
    """Spawn multiple Claude subagents in parallel via asyncio.gather.

    Each subagent gets its own subprocess_id, so the trace should show
    N parallel branches of LLM calls under one invocation.
    """
    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        model="claude-haiku-4-5",
        allowed_tools=["Bash"],
        agents={
            "researcher": AgentDefinition(
                description="Researches a topic and returns a concise summary.",
                prompt=(
                    "You are a research assistant. Answer the user's question "
                    "concisely in 2-3 sentences. If you need to check something, "
                    "use the Bash tool to run a quick command."
                ),
                tools=["Bash"],
                model="claude-haiku-4-5",
            ),
        },
    )

    async def _run_one(prompt: str) -> str:
        return await _run_claude(
            f"Use the researcher agent to answer: {prompt}",
            options=ClaudeAgentOptions(
                permission_mode="bypassPermissions",
                model="claude-haiku-4-5",
                allowed_tools=["Bash", "Task"],
                agents=options.agents,
            ),
        )

    results = await asyncio.gather(*[_run_one(p) for p in payload.prompts])
    return ParallelResponse(results=list(results))


# ---------------------------------------------------------------------------
# 4. nested_subagents — 2 levels of subagent depth
# ---------------------------------------------------------------------------


@fn()
async def nested_subagents(payload: SimpleRequest) -> SimpleResponse:
    """Spawn a subagent that itself spawns a deeper subagent.

    Produces a trace tree with 2 levels of subprocess nesting:
      invocation → subagent-orchestrator → subagent-worker
    """
    result = await _run_claude(
        (
            f"You need to answer this question: '{payload.prompt}'. "
            "First, use the planner agent to break the question into sub-tasks. "
            "Then, use the worker agent to execute each sub-task. "
            "Combine the results into a final answer."
        ),
        options=ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            model="claude-haiku-4-5",
            allowed_tools=["Task"],
            agents={
                "planner": AgentDefinition(
                    description="Breaks a complex question into 2-3 simpler sub-tasks.",
                    prompt=(
                        "You are a planning assistant. Given a question, break it "
                        "into 2-3 concrete sub-tasks that can each be answered independently. "
                        "Return a JSON array of task strings."
                    ),
                    tools=[],
                    model="claude-haiku-4-5",
                ),
                "worker": AgentDefinition(
                    description="Executes a specific sub-task and returns a concise answer.",
                    prompt=(
                        "You are a task execution assistant. Answer the given "
                        "sub-task concisely in 1-2 sentences."
                    ),
                    tools=[],
                    model="claude-haiku-4-5",
                ),
            },
        ),
    )
    return SimpleResponse(result=result)


# ---------------------------------------------------------------------------
# 5. stress_test — combines everything into one mega-trace
# ---------------------------------------------------------------------------


@fn()
async def stress_test(payload: StressTestRequest) -> StressTestResponse:
    """Run all trace patterns in a single invocation to produce a complex trace.

    This creates a trace with:
    - A simple baseline LLM call
    - A tool-using LLM call (if enabled)
    - N parallel subagent calls
    - A nested subagent call (if enabled)

    All running concurrently where possible, maximizing trace complexity.
    """
    # Build coroutines for concurrent execution
    tasks: dict[str, asyncio.Task] = {}

    async with asyncio.TaskGroup() as tg:
        # Always: simple baseline
        tasks["simple"] = tg.create_task(
            _run_claude(
                "What is 2+2? Answer in one word.",
                options=ClaudeAgentOptions(
                    permission_mode="bypassPermissions",
                    model="claude-haiku-4-5",
                    allowed_tools=[],
                ),
            )
        )

        # Optional: tool-using call
        if payload.enable_tool_use:
            tasks["tool"] = tg.create_task(
                _run_claude(
                    "Use Bash to run 'echo hello-from-stress-test' and tell me the output.",
                    options=ClaudeAgentOptions(
                        permission_mode="bypassPermissions",
                        model="claude-haiku-4-5",
                        allowed_tools=["Bash"],
                    ),
                )
            )

        # Parallel subagent calls
        prompts = [
            f"What is {i} * {i + 1}? Answer briefly."
            for i in range(1, payload.parallel_count + 1)
        ]
        parallel_tasks = []
        for prompt in prompts:
            parallel_tasks.append(
                tg.create_task(
                    _run_claude(
                        prompt,
                        options=ClaudeAgentOptions(
                            permission_mode="bypassPermissions",
                            model="claude-haiku-4-5",
                            allowed_tools=[],
                        ),
                    )
                )
            )

        # Optional: nested subagent
        if payload.enable_nesting:
            tasks["nested"] = tg.create_task(
                _run_claude(
                    (
                        "Use the planner agent to break this into sub-tasks, "
                        "then use the worker agent to solve each: "
                        "'What are the first 3 prime numbers and their sum?'"
                    ),
                    options=ClaudeAgentOptions(
                        permission_mode="bypassPermissions",
                        model="claude-haiku-4-5",
                        allowed_tools=["Task"],
                        agents={
                            "planner": AgentDefinition(
                                description="Breaks questions into sub-tasks.",
                                prompt="Break the question into 2-3 sub-tasks. Return a JSON array.",
                                tools=[],
                                model="claude-haiku-4-5",
                            ),
                            "worker": AgentDefinition(
                                description="Answers a single sub-task concisely.",
                                prompt="Answer the sub-task in 1 sentence.",
                                tools=[],
                                model="claude-haiku-4-5",
                            ),
                        },
                    ),
                )
            )

    return StressTestResponse(
        simple_result=tasks["simple"].result(),
        tool_result=tasks["tool"].result() if "tool" in tasks else None,
        parallel_results=[t.result() for t in parallel_tasks],
        nested_result=tasks["nested"].result() if "nested" in tasks else None,
    )
