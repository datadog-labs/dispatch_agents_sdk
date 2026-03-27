"""Daily Digest Agent — scheduled news summariser with memory.

Demonstrates:
- Schedules (cron) — designed to run on a recurring schedule
- Dispatch LLM Gateway — summarise content with any configured LLM
- Long-term Memory — track previously reported stories to avoid duplicates
- External API — fetches top stories from the Hacker News API (no key required)

When triggered (manually or by a cron schedule), the agent:
1. Fetches the current top stories from Hacker News
2. Checks long-term memory to skip stories already reported
3. Uses the LLM to generate a concise digest of new stories
4. Saves reported story IDs to memory for next run
"""

import json
import logging
from datetime import datetime

import httpx
from dispatch_agents import BasePayload, fn, llm, memory
from pydantic import BaseModel

logger = logging.getLogger(__name__)

HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{id}.json"

MAX_STORIES = 10
MEMORY_KEY = "reported_story_ids"


class DigestRequest(BasePayload):
    """Input payload. No fields required — designed for cron triggers."""

    max_stories: int = MAX_STORIES


class DigestResponse(BasePayload):
    """Output payload with the generated digest."""

    digest: str
    new_story_count: int
    skipped_count: int


class Story(BaseModel):
    """A Hacker News story."""

    id: int
    title: str
    url: str = ""
    score: int = 0
    by: str = "unknown"
    descendants: int = 0


async def _fetch_top_stories(max_stories: int) -> list[Story]:
    """Fetch top stories from Hacker News."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(HN_TOP_STORIES_URL)
        resp.raise_for_status()
        story_ids = resp.json()[: max_stories * 2]  # fetch extra in case some are dupes

        stories: list[Story] = []
        for story_id in story_ids:
            item_resp = await client.get(HN_ITEM_URL.format(id=story_id))
            if item_resp.status_code == 200:
                item = item_resp.json()
                if item and item.get("title"):
                    stories.append(
                        Story(
                            id=item["id"],
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            score=item.get("score", 0),
                            by=item.get("by", "unknown"),
                            descendants=item.get("descendants", 0),
                        )
                    )
            if len(stories) >= max_stories * 2:
                break

        return stories


async def _get_reported_ids() -> set[int]:
    """Load previously reported story IDs from long-term memory."""
    result = await memory.long_term.get(mem_key=MEMORY_KEY)
    if result and result.value:
        return set(json.loads(result.value))
    logger.debug("No previous reported IDs found — first run")
    return set()


async def _save_reported_ids(ids: set[int]) -> None:
    """Save reported story IDs to long-term memory (keep last 200)."""
    trimmed = sorted(ids)[-200:]
    await memory.long_term.add(mem_key=MEMORY_KEY, mem_val=json.dumps(trimmed))


@fn()
async def generate_digest(payload: DigestRequest) -> DigestResponse:
    """Generate a digest of top Hacker News stories.

    Designed to be triggered by a cron schedule (e.g. daily at 9am) or
    invoked manually. Skips stories that were already reported in previous
    runs using long-term memory.

    Example schedule::

        dispatch schedule create daily-digest generate_digest \\
            --cron "0 9 * * MON-FRI" \\
            --timezone "America/New_York" \\
            --payload '{}'

    Example manual invocation::

        dispatch agent invoke generate_digest '{"max_stories": 5}'
    """
    logger.info("Generating daily digest...")

    # 1. Fetch stories and filter out already-reported ones
    all_stories = await _fetch_top_stories(payload.max_stories * 2)
    reported_ids = await _get_reported_ids()

    new_stories = [s for s in all_stories if s.id not in reported_ids]
    new_stories = sorted(new_stories, key=lambda s: s.score, reverse=True)[
        : payload.max_stories
    ]

    skipped = len(all_stories) - len(new_stories)
    logger.info(f"Found {len(new_stories)} new stories ({skipped} already reported)")

    if not new_stories:
        return DigestResponse(
            digest="No new stories to report since the last digest.",
            new_story_count=0,
            skipped_count=skipped,
        )

    # 2. Build a prompt for the LLM
    today = datetime.now().strftime("%A, %B %d, %Y")
    stories_text = "\n".join(
        f"- [{s.title}]({s.url or 'no link'}) — {s.score} points, "
        f"{s.descendants} comments, by {s.by}"
        for s in new_stories
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You write concise, engaging tech news digests. "
                "Summarise the stories below into a short newsletter-style digest. "
                "Group related stories if possible. Use markdown formatting. "
                "Include the original links."
            ),
        },
        {
            "role": "user",
            "content": f"Here are today's top Hacker News stories ({today}):\n\n{stories_text}\n\n"
            "Write a brief digest covering the key themes and highlights.",
        },
    ]

    response = await llm.inference(messages)

    # 3. Remember which stories we reported
    new_ids = reported_ids | {s.id for s in new_stories}
    await _save_reported_ids(new_ids)

    return DigestResponse(
        digest=response.content or "",
        new_story_count=len(new_stories),
        skipped_count=skipped,
    )
