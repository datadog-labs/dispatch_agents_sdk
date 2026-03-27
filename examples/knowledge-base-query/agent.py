"""Knowledge Base Query Agent.

Answers questions by searching a Confluence knowledge base. Builds a table of
contents from hub pages and their children, then uses an LLM to fetch and
synthesize relevant pages into a structured answer.
"""

import json
import logging
import os
import re

import httpx
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from dispatch_agents import BasePayload, fn
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Enable sandbox mode for claude_agent_sdk
os.environ["IS_SANDBOX"] = "True"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Knowledge base hub pages - index pages whose children contain the real content.
# Replace these with your own Confluence hub page IDs and spaces.
KNOWLEDGE_BASE_PAGES: list[dict[str, str]] = [
    # Example entries - replace with your own:
    # {"title": "Engineering Docs", "page_id": "123456789", "space": "ENG"},
    # {"title": "Product FAQ",      "page_id": "987654321", "space": "PROD"},
]

# Confluence spaces to include in CQL searches
SEARCHABLE_SPACES: list[str] = [
    # Example: "ENG", "PROD"
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledge-base research assistant. Your job is to find accurate
answers by searching Confluence documentation.

You have access to Confluence via the REST API. Use curl to fetch pages.

## Authentication
The following environment variables are available:
- CONFLUENCE_URL: The base URL (e.g., https://your-org.atlassian.net)
- CONFLUENCE_USERNAME: Your Atlassian account email
- CONFLUENCE_API_TOKEN: Your API token

For curl requests, use basic auth:
```bash
curl -u "$CONFLUENCE_USERNAME:$CONFLUENCE_API_TOKEN" "$CONFLUENCE_URL/wiki/api/v2/..."
```

## API Endpoints

### Get a page by ID (with content):
```bash
curl -u "$CONFLUENCE_USERNAME:$CONFLUENCE_API_TOKEN" \
  "$CONFLUENCE_URL/wiki/api/v2/pages/{page_id}?body-format=storage" | jq '.body.storage.value'
```

### Search within Confluence (CQL):
Use this when the table of contents doesn't have an obviously relevant page.

```bash
curl -u "$CONFLUENCE_USERNAME:$CONFLUENCE_API_TOKEN" \
  "$CONFLUENCE_URL/wiki/rest/api/content/search?cql=<URL-encoded CQL>&limit=10&expand=body.storage"
```

**CQL tips:**
- Search across spaces: `space in (SPACE1,SPACE2) and text ~ "your query"`
- Search by title: `space in (SPACE1,SPACE2) and title ~ "keyword"`
- URL-encode the CQL query parameter

### Get child pages (drill deeper into any page):
```bash
curl -u "$CONFLUENCE_USERNAME:$CONFLUENCE_API_TOKEN" \
  "$CONFLUENCE_URL/wiki/rest/api/content/{page_id}/child/page?limit=50&expand=body.storage"
```

## Your Task

You will receive:
1. **Table of contents** - Hub pages and their child page titles with page IDs.
2. **Query** - A question that needs answering from the knowledge base.

## Instructions

1. Carefully read the query to understand what information is being asked for
2. Review the table of contents to identify pages most likely to contain the answer
3. Fetch the SPECIFIC pages that are most relevant using their page IDs
   - Start with the most obviously relevant 2-4 pages
   - Fetch more if needed after reading the first batch
4. If the TOC doesn't have an obviously relevant page, use CQL search
5. If a child page seems like another hub, fetch ITS children too
6. Once you have enough context, synthesize a clear answer
7. Ensure all claims are grounded in the Confluence documentation

## Output Format

Return a JSON object:

```json
{
  "answer": "A clear, well-structured answer to the question",
  "sources": ["Page Title 1", "Page Title 2"],
  "confidence": "high | medium | low",
  "notes": "Any caveats, assumptions, or information that couldn't be verified"
}
```

## Guidelines
- Be concise but thorough
- Ground every claim in the Confluence documentation
- If you can't find enough information, say so honestly
- Use bullet points where appropriate for clarity
- Include ALL page titles you referenced in sources
- Do NOT fabricate information - if you can't find it in Confluence, say so
- When confidence is low, note what specific information is missing
"""


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------
class QueryRequest(BasePayload):
    """Input payload for a knowledge base query."""

    query: str


class QueryResponse(BasePayload):
    """Structured answer from the knowledge base."""

    answer: str = ""
    sources: list[str] = []
    confidence: str = "low"  # "high", "medium", or "low"
    notes: str = ""


# ---------------------------------------------------------------------------
# Confluence helpers
# ---------------------------------------------------------------------------


class ChildPage(BaseModel):
    """A child page in the Confluence hierarchy."""

    title: str
    page_id: str


def _confluence_auth() -> tuple[str, str, str]:
    """Return (base_url, username, api_token) from env."""
    return (
        os.environ.get("CONFLUENCE_URL", ""),
        os.environ.get("CONFLUENCE_USERNAME", ""),
        os.environ.get("CONFLUENCE_API_TOKEN", ""),
    )


async def fetch_child_pages(client: httpx.AsyncClient, page_id: str) -> list[ChildPage]:
    """Fetch child page titles and IDs for a given parent page."""
    base_url, username, api_token = _confluence_auth()
    if not all([base_url, username, api_token]):
        return []

    url = f"{base_url}/wiki/rest/api/content/{page_id}/child/page?limit=100"
    try:
        response = await client.get(url, auth=(username, api_token), timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return [
            ChildPage(title=child["title"], page_id=child["id"])
            for child in data.get("results", [])
        ]
    except Exception as e:
        logger.warning(
            "Failed to fetch children of page",
            extra={"page_id": page_id, "error": str(e)},
        )
        return []


async def build_table_of_contents() -> str:
    """Build a table of contents from all knowledge base hub pages and their children."""
    toc_parts = []

    async with httpx.AsyncClient() as client:
        for page in KNOWLEDGE_BASE_PAGES:
            children = await fetch_child_pages(client, page["page_id"])
            section = f"## {page['title']} (Space: {page['space']}, Page ID: {page['page_id']})\n"
            if children:
                for child in children:
                    section += f"  - {child.title} (Page ID: {child.page_id})\n"
            else:
                section += "  (no child pages - fetch this page directly for content)\n"
            toc_parts.append(section)
            logger.info(
                "Built TOC section",
                extra={"page": page["title"], "children": len(children)},
            )

    return "\n".join(toc_parts)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
def parse_response(response_text: str) -> QueryResponse:
    """Parse the LLM's JSON response into a QueryResponse."""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', response_text, re.DOTALL)
        json_str = json_match.group(0) if json_match else response_text.strip()

    try:
        data = json.loads(json_str)
        return QueryResponse.model_validate(data)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse knowledge base response: {e}")
        return QueryResponse(
            answer=f"Error parsing response: {e}\n\nRaw response:\n{response_text[:500]}",
            notes="Failed to parse structured response",
        )


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------
@fn()
async def query_knowledge_base(payload: QueryRequest) -> QueryResponse:
    """Answer a question by researching the Confluence knowledge base.

    This function:
    1. Builds a table of contents from knowledge base hub pages + their children
    2. Passes the TOC to an LLM so it knows what pages are available
    3. The LLM fetches only the specific pages relevant to the query
    4. The LLM can also search Confluence for pages not in the TOC
    5. Returns a structured answer grounded in documentation
    """
    logger.info("Received knowledge base query", extra={"query": payload.query})

    # Build table of contents from hub pages
    logger.info("Building knowledge base table of contents...")
    toc = await build_table_of_contents()

    searchable = (
        ", ".join(SEARCHABLE_SPACES) if SEARCHABLE_SPACES else "(none configured)"
    )

    prompt = f"""I need you to answer a question using the Confluence knowledge base.

## Knowledge Base - Table of Contents
Below is an index of all available knowledge base pages and their children.
Use the page IDs to fetch the full content of pages relevant to the question.

{toc}

## Searchable Spaces
{searchable}

## Question
{payload.query}

## Steps
1. Review the table of contents above
2. Identify which pages are most relevant to answering this question
3. Fetch those pages by their page IDs using the Confluence API
4. If you need more detail, search Confluence with CQL or drill into child pages
5. Synthesize a clear, well-structured answer based on what you found
6. Return your response as a JSON object in the specified format
"""

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
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

    return parse_response(final_result or last_text)
