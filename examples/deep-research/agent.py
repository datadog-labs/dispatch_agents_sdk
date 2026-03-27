"""Deep Research Agent — comprehensive research using parallel sub-agents.

Demonstrates:
- OpenAI SDK tool-calling loop (supervisor + researcher pattern)
- Parallel web search via Tavily API
- Multi-step orchestration: clarify → brief → research → report
- Structured Pydantic payloads

The supervisor breaks a research question into sub-topics, delegates each to
a researcher sub-agent that searches the web, then synthesises all findings
into a single comprehensive report.

Requires OPENAI_API_KEY and TAVILY_API_KEY environment variables.
"""

import logging

from configuration import Configuration, SearchAPI
from deep_researcher import DeepResearcher
from dispatch_agents import BasePayload, fn

logger = logging.getLogger(__name__)


class DeepResearchRequest(BasePayload):
    """Input payload for a deep research request."""

    query: str


class DeepResearchResponse(BasePayload):
    """Output payload containing the final research report."""

    report: str


@fn()
async def deep_research(payload: DeepResearchRequest) -> DeepResearchResponse:
    """Run deep research on a given query.

    A supervisor coordinates multiple parallel researcher sub-agents that
    search the web via Tavily, then compresses and synthesises the findings
    into a single comprehensive report.

    Example invocation::

        {"query": "What are the latest breakthroughs in fusion energy?"}
    """
    logger.info(f"Starting deep research: {payload.query[:100]}...")

    config = Configuration(
        search_api=SearchAPI.TAVILY,
        allow_clarification=False,
        max_researcher_iterations=6,
        max_react_tool_calls=10,
        max_concurrent_research_units=5,
    )

    researcher = DeepResearcher(config)
    result = await researcher.run(payload.query)

    logger.info(
        f"Research completed after {researcher.context.research_iterations} iterations"
    )

    return DeepResearchResponse(report=result)
