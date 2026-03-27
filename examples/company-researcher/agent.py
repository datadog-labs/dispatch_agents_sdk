"""Company Researcher Agent.

Generates a quick research report on a company using web search. Uses the
Claude Agent SDK with WebSearch and WebFetch tools to gather information,
then synthesizes it into a structured company profile.
"""

import json
import logging
import re

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from dispatch_agents import BasePayload, fn

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a company research analyst. Your job is to quickly gather key
information about a company using web search and produce a concise, factual report.

## Research Process

1. Search for the company to understand what they do
2. Look for recent news, funding, and key milestones
3. Find leadership team information
4. Identify their main products/services and target market
5. Look for competitive positioning and market context

## Output Format

Return a JSON object with this structure:

```json
{
  "company_name": "Official company name",
  "website": "https://...",
  "summary": "1-2 sentence description of what the company does",
  "founded": "Year or 'Unknown'",
  "headquarters": "City, State/Country or 'Unknown'",
  "employee_count": "Approximate headcount or range, or 'Unknown'",
  "funding": "Total funding and last round details, or 'Unknown' / 'Public company'",
  "leadership": [
    {"name": "...", "title": "CEO"},
    {"name": "...", "title": "CTO"}
  ],
  "products_and_services": ["Product/service 1", "Product/service 2"],
  "target_market": "Who their customers are",
  "competitors": ["Competitor 1", "Competitor 2"],
  "recent_news": [
    "Brief headline or development 1",
    "Brief headline or development 2"
  ],
  "notes": "Any caveats about the information gathered"
}
```

## Guidelines
- Be factual - only include information you found via search
- If you can't find a piece of information, use "Unknown" rather than guessing
- Keep the summary concise (1-2 sentences)
- List 2-5 competitors if identifiable
- Include 2-4 recent news items if available
- Leadership should include at least CEO/founder if findable
- Do NOT fabricate any information
"""


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------
class CompanyResearchRequest(BasePayload):
    """Input payload for company research."""

    company: str  # Company name or domain to research


class LeadershipEntry(BasePayload):
    """A member of the leadership team."""

    name: str = ""
    title: str = ""


class CompanyReport(BasePayload):
    """Structured company research report."""

    company_name: str = ""
    website: str = ""
    summary: str = ""
    founded: str = "Unknown"
    headquarters: str = "Unknown"
    employee_count: str = "Unknown"
    funding: str = "Unknown"
    leadership: list[LeadershipEntry] = []
    products_and_services: list[str] = []
    target_market: str = ""
    competitors: list[str] = []
    recent_news: list[str] = []
    notes: str = ""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
def parse_report(response_text: str) -> CompanyReport:
    """Parse the LLM's JSON response into a CompanyReport."""
    # Try to extract JSON from markdown code block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find a raw JSON object with "company_name" key
        json_match = re.search(r'\{[^{}]*"company_name".*\}', response_text, re.DOTALL)
        json_str = json_match.group(0) if json_match else response_text.strip()

    try:
        data = json.loads(json_str)
        return CompanyReport.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse company report: {e}")
        return CompanyReport(
            summary=f"Error parsing response: {e}\n\n{response_text[:500]}",
            notes="Failed to parse structured response",
        )


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------
@fn()
async def research_company(payload: CompanyResearchRequest) -> CompanyReport:
    """Research a company using web search and return a structured report.

    Uses the Claude Agent SDK with WebSearch and WebFetch to gather publicly
    available information about a company, then synthesizes it into a
    structured profile.
    """
    logger.info("Researching company", extra={"company": payload.company})

    prompt = f"""Research the company "{payload.company}" and produce a structured report.

Search the web to find:
- What the company does (core products/services)
- When and where it was founded
- Key leadership (CEO, founders, CTO)
- Funding history (if a startup) or market cap (if public)
- Approximate employee count
- Main competitors
- Recent news or developments

Return your findings as a JSON object in the specified format.
"""

    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch", "WebFetch"],
        system_prompt=SYSTEM_PROMPT,
        sandbox={"enabled": True},
    )

    final_result = ""
    last_text = ""

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            final_result = message.result
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "text") and block.text:
                    last_text = block.text

    return parse_report(final_result or last_text)
