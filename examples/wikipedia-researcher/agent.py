"""Wikipedia-backed research agent.

Demonstrates a non-trivial agent for end-to-end eval testing:
- Takes a natural-language question
- Hits the Wikipedia REST API for the most relevant article summary
- Feeds the article + question into the LLM gateway to produce a concise
  answer
- Returns {answer, source_title, source_url}

Used as the agent-under-test in the experiments-with-scorers e2e.
"""

import logging
from urllib.parse import quote

import aiohttp
from dispatch_agents import BasePayload, fn, llm

logger = logging.getLogger(__name__)

WIKIPEDIA_SEARCH = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Wikipedia requires a meaningful User-Agent; the default aiohttp header
# gets 403'd. See https://meta.wikimedia.org/wiki/User-Agent_policy.
_USER_AGENT = (
    "dispatch-agents-wikipedia-researcher/0.1 "
    "(https://github.com/DataDog/dispatch_agents)"
)
_HEADERS = {"User-Agent": _USER_AGENT, "Accept": "application/json"}


class ResearchRequest(BasePayload):
    """Input payload for a research question."""

    question: str


class ResearchResponse(BasePayload):
    """Output payload with the agent's answer and source attribution."""

    answer: str
    source_title: str
    source_url: str


async def _search_wikipedia(session: aiohttp.ClientSession, query: str) -> str | None:
    """Return the title of the best-matching Wikipedia article, or None."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1,
    }
    async with session.get(
        WIKIPEDIA_SEARCH, params=params, headers=_HEADERS, timeout=10
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0]["title"]


async def _fetch_summary(session: aiohttp.ClientSession, title: str) -> dict:
    url = f"{WIKIPEDIA_SUMMARY}/{quote(title)}"
    async with session.get(url, headers=_HEADERS, timeout=10) as resp:
        resp.raise_for_status()
        return await resp.json()


@fn()
async def research(payload: ResearchRequest) -> ResearchResponse:
    """Answer a factual question using Wikipedia as the knowledge source.

    Example::

        {"question": "In what year was the Python programming language first released?"}
    """
    logger.info("Researching question: %s", payload.question)

    async with aiohttp.ClientSession() as session:
        title = await _search_wikipedia(session, payload.question)
        if not title:
            return ResearchResponse(
                answer="No Wikipedia article matched the question.",
                source_title="",
                source_url="",
            )
        summary = await _fetch_summary(session, title)

    extract = summary.get("extract", "")
    source_url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")

    # Use the LLM gateway to synthesize a focused answer rather than dump
    # the whole article summary.
    messages = [
        {
            "role": "system",
            "content": (
                "You answer factual questions using a single Wikipedia article"
                " summary. Be concise (one or two sentences). If the article"
                " does not contain the answer, say so plainly. Do not"
                " hallucinate facts not in the summary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {payload.question}\n\n"
                f"Article ({title}):\n{extract}\n\n"
                "Answer:"
            ),
        },
    ]
    response = await llm.inference(messages)
    answer = (response.content or "").strip()

    return ResearchResponse(
        answer=answer,
        source_title=title,
        source_url=source_url,
    )
