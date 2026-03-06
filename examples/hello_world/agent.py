"""Generated agent entrypoint."""

import asyncio

import dispatch_agents
from dispatch_agents import BasePayload, on
from dispatch_agents.integrations.github import PullRequestReviewCommentCreated
from pydantic import Field, PositiveInt


class GreetingPayload(BasePayload):
    """Input payload for greeting requests."""

    subject: str = Field(description="The name or subject to greet")


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
    print(f"Handling greet request for: {payload.subject}")

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
    print(f"Starting sleep for {payload.duration_seconds} seconds")

    for remaining in range(payload.duration_seconds, 0, -1):
        print(f"Countdown: {remaining} seconds remaining")
        await asyncio.sleep(1)

    print("Sleep completed")
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
    print(f"Received PR review comment from {event.comment.user.login}")
    print(f"Comment body: {event.comment.body[:100]}...")
    print(f"PR: {event.pull_request.title}")

    return PRReviewCommentResponse(
        repo=event.repository.full_name,
        user=event.comment.user.login,
        comment=event.comment.body,
    )
