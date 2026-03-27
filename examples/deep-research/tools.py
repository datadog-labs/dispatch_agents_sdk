"""Tool functions and utilities for the Deep Research agent."""

import asyncio
import json
import logging
from datetime import datetime

from configuration import Configuration
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from prompts import SUMMARIZE_WEBPAGE_PROMPT
from state import Summary

logger = logging.getLogger(__name__)


def get_today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def is_token_limit_exceeded(exception: Exception) -> bool:
    """Check if the error is a context/token limit exceeded error."""
    if isinstance(exception, BadRequestError):
        if exception.code == "context_length_exceeded":
            return True
        error_str = str(exception).lower()
        if any(
            k in error_str for k in ["token", "context", "length", "maximum context"]
        ):
            return True
    return False


def is_rate_limit_error(exception: Exception) -> bool:
    """Check if the error is a rate limit error."""
    return isinstance(exception, RateLimitError)


def is_retryable_error(exception: Exception) -> bool:
    """Check if the error is a transient/retryable error."""
    return isinstance(exception, (APIConnectionError, APITimeoutError))


MODEL_TOKEN_LIMITS = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
}


def get_model_token_limit(model_string: str) -> int | None:
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string.lower():
            return token_limit
    return None


def filter_recent_messages(messages: list[dict], max_messages: int = 10) -> list[dict]:
    if len(messages) <= max_messages:
        return messages
    system_messages = [m for m in messages if m.get("role") == "system"]
    other_messages = [m for m in messages if m.get("role") != "system"]
    return system_messages + other_messages[-max_messages:]


def get_token_param(model: str, max_tokens: int) -> dict:
    """Return the correct max-tokens kwarg for the model."""
    if model.startswith("gpt-5") or model.startswith("gpt-6"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def format_messages_as_string(messages: list[dict]) -> str:
    return "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


async def tavily_search(
    queries: list[str],
    config: Configuration,
    max_results: int = 5,
    topic: str = "general",
) -> str:
    """Search the web via Tavily and return formatted results."""
    if not config.tavily_api_key:
        return "Error: Tavily API key not configured"

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=config.tavily_api_key)
        search_results = await asyncio.gather(
            *[
                client.search(
                    q, max_results=max_results, include_raw_content=True, topic=topic
                )
                for q in queries
            ]
        )

        # Deduplicate by URL
        unique: dict = {}
        for response in search_results:
            for result in response.get("results", []):
                url = result.get("url")
                if url and url not in unique:
                    unique[url] = result

        if not unique:
            return "No search results found. Try different queries."

        # Summarise long pages in parallel
        openai_client = AsyncOpenAI(api_key=config.openai_api_key)
        summarized: dict = {}
        tasks = []

        for url, result in unique.items():
            raw = result.get("raw_content", "")
            if raw and len(raw) > config.max_content_length:
                tasks.append(
                    (
                        url,
                        result,
                        summarize_webpage(
                            openai_client, raw[: config.max_content_length], config
                        ),
                    )
                )
            else:
                content = (
                    raw[: config.max_content_length]
                    if raw
                    else result.get("content", "")
                )
                summarized[url] = {
                    "title": result.get("title", "Untitled"),
                    "content": content,
                }

        if tasks:
            summaries = await asyncio.gather(*[t for _, _, t in tasks])
            for (url, result, _), summary in zip(tasks, summaries):
                summarized[url] = {
                    "title": result.get("title", "Untitled"),
                    "content": summary,
                }

        output = "Search results:\n\n"
        for i, (url, r) in enumerate(summarized.items()):
            output += f"\n--- SOURCE {i + 1}: {r['title']} ---\nURL: {url}\n\n{r['content']}\n\n{'—' * 40}\n"
        return output

    except Exception as e:
        return f"Error conducting search: {e}"


async def summarize_webpage(
    client: AsyncOpenAI, content: str, config: Configuration
) -> str:
    """Summarise long webpage content with the LLM."""
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=config.summarization_model,
                messages=[
                    {
                        "role": "system",
                        "content": SUMMARIZE_WEBPAGE_PROMPT.format(
                            date=get_today_str()
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                **get_token_param(
                    config.summarization_model, config.summarization_model_max_tokens
                ),
            ),
            timeout=config.summarization_timeout,
        )
        text = response.choices[0].message.content
        try:
            data = json.loads(text)
            s = Summary(**data)
            return f"<summary>\n{s.summary}\n</summary>\n<key_excerpts>\n{s.key_excerpts}\n</key_excerpts>"
        except (json.JSONDecodeError, TypeError, ValueError):
            return text
    except Exception:
        return content


# ---------------------------------------------------------------------------
# Compression & report generation
# ---------------------------------------------------------------------------


async def compress_research(messages: list[dict], config: Configuration) -> str:
    """Compress research findings. Handles token limits by trimming context."""
    from prompts import COMPRESS_RESEARCH_PROMPT, COMPRESS_RESEARCH_SIMPLE_MESSAGE

    client = AsyncOpenAI(api_key=config.openai_api_key)
    system = COMPRESS_RESEARCH_PROMPT.format(date=get_today_str())
    compression_msgs = messages + [
        {"role": "user", "content": COMPRESS_RESEARCH_SIMPLE_MESSAGE}
    ]

    try:
        response = await client.chat.completions.create(
            model=config.compression_model,
            messages=[{"role": "system", "content": system}] + compression_msgs,
            **get_token_param(
                config.compression_model, config.compression_model_max_tokens
            ),
        )
        return response.choices[0].message.content
    except BadRequestError as e:
        if is_token_limit_exceeded(e):
            # Trim context and retry once
            trimmed = filter_recent_messages(
                compression_msgs, max_messages=max(5, len(compression_msgs) // 2)
            )
            response = await client.chat.completions.create(
                model=config.compression_model,
                messages=[{"role": "system", "content": system}] + trimmed,
                **get_token_param(
                    config.compression_model, config.compression_model_max_tokens
                ),
            )
            return response.choices[0].message.content
        raise


async def generate_final_report(
    research_brief: str,
    messages: list[dict],
    findings: list[str],
    config: Configuration,
) -> str:
    """Generate the final research report. Trims findings on token overflow."""
    from prompts import FINAL_REPORT_GENERATION_PROMPT

    client = AsyncOpenAI(api_key=config.openai_api_key)
    messages_str = "\n".join(
        f"{m.get('role')}: {m.get('content', '')}" for m in messages
    )
    findings_str = "\n\n".join(findings)

    prompt = FINAL_REPORT_GENERATION_PROMPT.format(
        research_brief=research_brief,
        messages=messages_str,
        findings=findings_str,
        date=get_today_str(),
    )

    try:
        response = await client.chat.completions.create(
            model=config.final_report_model,
            messages=[{"role": "user", "content": prompt}],
            **get_token_param(
                config.final_report_model, config.final_report_model_max_tokens
            ),
        )
        return response.choices[0].message.content
    except BadRequestError as e:
        if is_token_limit_exceeded(e):
            # Trim findings and retry once
            findings_str = findings_str[: int(len(findings_str) * 0.7)]
            prompt = FINAL_REPORT_GENERATION_PROMPT.format(
                research_brief=research_brief,
                messages=messages_str,
                findings=findings_str,
                date=get_today_str(),
            )
            response = await client.chat.completions.create(
                model=config.final_report_model,
                messages=[{"role": "user", "content": prompt}],
                **get_token_param(
                    config.final_report_model, config.final_report_model_max_tokens
                ),
            )
            return response.choices[0].message.content
        raise
