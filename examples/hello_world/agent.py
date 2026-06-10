"""Generated agent entrypoint."""

import asyncio
import logging
from pathlib import Path

import aiohttp
import dispatch_agents
from dispatch_agents import BasePayload, fn, get_data_dir, on
from dispatch_agents.integrations.github.events import (
    CheckSuiteCompleted,
    PullRequestReviewCommentCreated,
)
from pydantic import Field, PositiveInt

logger = logging.getLogger(__name__)


class GreetingPayload(BasePayload):
    """Input payload for greeting requests."""

    subject: str = Field(default="World", description="The name or subject to greet")


class GreetingResponse(BasePayload):
    """Output payload for greeting responses."""

    greeting: str = Field(description="The greeting message")


@dispatch_agents.on(topic="test")
async def greet(payload: GreetingPayload) -> GreetingResponse:
    """Handle greeting requests with typed payloads.

    This handler demonstrates:
    - Typed input validation (payload is a validated GreetingPayload)
    - Typed output serialization (returns GreetingResponse)
    - Automatic schema extraction for API documentation
    """
    logger.info("Handling greet request for: %s", payload.subject)

    # Validate that subject field exists - ValueError is non-retryable
    if not payload.subject:
        raise ValueError("Missing required field 'subject' in message payload")

    # Demonstrate retryable error - an OSError will cause automatic retry
    if payload.subject == "oops":
        raise OSError(
            f"The value '{payload.subject}' raises a runtime error - will be retried"
        )

    return GreetingResponse(greeting=f"Hello {payload.subject}")


class SleepRequest(BasePayload):
    """Input payload for sleep requests."""

    duration_seconds: PositiveInt = Field(description="Duration to sleep in seconds")


class SleepResponse(BasePayload):
    """Output payload for sleep responses."""

    seconds_slept: int = Field(description="The number of seconds slept")


@dispatch_agents.on(topic="sleep")
async def sleep(payload: SleepRequest) -> SleepResponse:
    """Sleep for the specified duration, logging countdown progress."""
    logger.info("Starting sleep for %s seconds", payload.duration_seconds)

    for remaining in range(payload.duration_seconds, 0, -1):
        logger.info("Countdown: %s seconds remaining", remaining)
        await asyncio.sleep(1)

    logger.info("Sleep completed")
    return SleepResponse(seconds_slept=payload.duration_seconds)


class PRReviewCommentResponse(BasePayload):
    """Response for PR review comment events."""

    repo: str = Field(description="Repository full name (owner/repo)")
    user: str = Field(description="Username who made the comment")
    comment: str = Field(description="Comment text")


@on(github_event=PullRequestReviewCommentCreated)
async def on_pr_review_comment(
    event: PullRequestReviewCommentCreated,
) -> PRReviewCommentResponse:
    """Handle GitHub PR review comment created events."""
    logger.info("Received PR review comment from %s", event.comment.user.login)
    logger.info("Comment body: %.100s...", event.comment.body)
    logger.info("PR: %s", event.pull_request.title)

    return PRReviewCommentResponse(
        repo=event.repository.full_name,
        user=event.comment.user.login,
        comment=event.comment.body,
    )


class CheckSuiteCompletedResponse(BasePayload):
    """Response for check_suite.completed events."""

    repo: str | None = Field(description="Repository full name (owner/repo)")
    head_sha: str = Field(description="Head commit SHA of the suite")
    conclusion: str | None = Field(
        description="Suite conclusion (success, failure, ...)"
    )


@on(github_event=CheckSuiteCompleted)
async def on_check_suite_completed(
    event: CheckSuiteCompleted,
) -> CheckSuiteCompletedResponse:
    """Handle GitHub check_suite.completed events."""
    logger.info(
        "Check suite completed: repo=%s sha=%s conclusion=%s",
        event.repository.full_name if event.repository else None,
        event.check_suite.head_sha,
        event.check_suite.conclusion,
    )
    return CheckSuiteCompletedResponse(
        repo=event.repository.full_name if event.repository else None,
        head_sha=event.check_suite.head_sha,
        conclusion=event.check_suite.conclusion,
    )


class ReverseRequest(BasePayload):
    """Input for the reverse function."""

    text: str = Field(description="Text to reverse")


class ReverseResponse(BasePayload):
    """Output of the reverse function."""

    reversed_text: str = Field(description="The reversed text")


@fn()
async def reverse(payload: ReverseRequest) -> ReverseResponse:
    """Reverse the provided text string."""
    logger.info("Reversing: %r", payload.text)
    return ReverseResponse(reversed_text=payload.text[::-1])


class StorageWriteRequest(BasePayload):
    """Input for writing to persistent storage."""

    key: str = Field(description="Filename to write")
    value: str = Field(description="Content to write")


class StorageWriteResponse(BasePayload):
    """Output of storage write."""

    path: str = Field(description="Full path of the written file")


def _safe_data_path(key: str) -> Path:
    """Resolve a key under the persistent data directory."""
    data_dir = get_data_dir().resolve()
    path = (data_dir / key).resolve()
    if not path.is_relative_to(data_dir):
        raise ValueError("Invalid key: must resolve within the data directory")
    return path


@fn()
async def storage_write(payload: StorageWriteRequest) -> StorageWriteResponse:
    """Write a value to persistent storage."""
    path = _safe_data_path(payload.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(payload.value)
    logger.info("Wrote %s bytes to %s", len(payload.value), path)
    return StorageWriteResponse(path=str(path))


class StorageReadRequest(BasePayload):
    """Input for reading from persistent storage."""

    key: str = Field(description="Filename to read")


class StorageReadResponse(BasePayload):
    """Output of storage read."""

    value: str | None = Field(description="File content, or null if not found")
    exists: bool = Field(description="Whether the file exists")


@fn()
async def storage_read(payload: StorageReadRequest) -> StorageReadResponse:
    """Read a value from persistent storage."""
    path = _safe_data_path(payload.key)
    if path.exists():
        with path.open() as f:
            value = f.read()
        logger.info("Read %s bytes from %s", len(value), path)
        return StorageReadResponse(value=value, exists=True)
    logger.info("File not found: %s", path)
    return StorageReadResponse(value=None, exists=False)


class EgressTestRequest(BasePayload):
    """Input for the egress test function."""

    url: str = Field(
        default="https://jsonplaceholder.typicode.com/todos/1",
        description="URL to attempt to fetch",
    )


class EgressTestResponse(BasePayload):
    """Output of the egress test function."""

    success: bool = Field(description="Whether the request succeeded")
    status_code: int | None = Field(
        default=None, description="HTTP status code if successful"
    )
    body: str = Field(default="", description="Response body or error message")


@fn()
async def test_egress(payload: EgressTestRequest) -> EgressTestResponse:
    """Test outbound HTTP connectivity by fetching a URL.

    Useful for verifying network egress controls. When network.egress is
    configured, this request will be blocked unless the target domain is
    in allow_domains.
    """
    logger.info("Testing egress to: %s", payload.url)
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.get(payload.url) as resp:
                body = await resp.text()
                logger.info("Response: %s (%s bytes)", resp.status, len(body))
                return EgressTestResponse(
                    success=True,
                    status_code=resp.status,
                    body=body[:1000],
                )
    except Exception as e:
        logger.info("Request failed: %s: %s", type(e).__name__, e)
        return EgressTestResponse(
            success=False,
            body=f"{type(e).__name__}: {e}",
        )
