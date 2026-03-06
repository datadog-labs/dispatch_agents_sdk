"""GitHub PR responder agent - demonstrates GitHub event handling."""

from dispatch_agents import BasePayload, on
from dispatch_agents.integrations.github import (
    GitHubEventPayload,
    IssueCommentCreated,
    PullRequestReviewCommentBase,
    PullRequestReviewCommentCreated,
    PullRequestReviewSubmitted,
)
from pydantic import Field

# =============================================================================
# Response Payloads
# =============================================================================


class PRCommentResponse(BasePayload):
    """Response for PR comment handling."""

    pr_number: int = Field(description="Pull request number")
    author: str = Field(description="Comment author username")
    comment: str = Field(description="Comment body")
    skipped: bool = Field(default=False, description="Whether the comment was skipped")


class PRReviewActivityResponse(BasePayload):
    """Response for PR review activity (reviews and review comments)."""

    event_type: str = Field(description="Type of event: 'review' or 'review_comment'")
    pr_number: int = Field(description="Pull request number")
    author: str = Field(description="Author username")
    body: str = Field(description="Review or comment body")
    # Optional fields for review comments
    file_path: str | None = Field(
        default=None, description="File path (for review comments)"
    )
    line: int | None = Field(
        default=None, description="Line number (for review comments)"
    )
    # Optional fields for reviews
    state: str | None = Field(default=None, description="Review state (for reviews)")


# =============================================================================
# GitHub Event Handlers
# =============================================================================


@on(github_event=IssueCommentCreated)
async def handle_pr_comment(payload: IssueCommentCreated) -> PRCommentResponse:
    """Handle comments on pull requests."""
    comment = payload.comment
    issue = payload.issue

    # issue_comment events fire for both issues and PRs
    if not issue.pull_request:
        print(f"Skipping issue comment (not a PR): #{issue.number}")
        return PRCommentResponse(
            pr_number=issue.number,
            author=comment.user.login,
            comment=comment.body,
            skipped=True,
        )

    print("=" * 60)
    print("Received PR comment!")
    print(f"  PR #{issue.number}: {issue.title}")
    print(f"  Repository: {payload.repository.full_name}")
    print(f"  Comment by: {comment.user.login}")
    print(f"  Comment: {comment.body}")
    print("=" * 60)

    return PRCommentResponse(
        pr_number=issue.number,
        author=comment.user.login,
        comment=comment.body,
    )


@on(github_event=[PullRequestReviewSubmitted, PullRequestReviewCommentCreated])
async def handle_pr_review_activity(
    payload: GitHubEventPayload,
) -> PRReviewActivityResponse:
    """Handle both PR review submissions and review comments with a single handler.

    This demonstrates how to subscribe to multiple event types using the common
    base class (GitHubEventPayload) and handle them with runtime type checking.
    """
    pr = payload.pull_request

    print("=" * 60)

    # Handle PR review submissions
    if isinstance(payload, PullRequestReviewSubmitted):
        review = payload.review
        reviewer = review.get("user", {}).get("login", "unknown")
        state = review.get("state", "unknown")
        body = review.get("body", "") or ""

        print("Received PR review!")
        print(f"  PR #{pr.number}: {pr.title}")
        print(f"  Repository: {payload.repository.full_name}")
        print(f"  Review by: {reviewer}")
        print(f"  Review state: {state}")
        if body:
            print(f"  Review body: {body}")
        print("=" * 60)

        return PRReviewActivityResponse(
            event_type="review",
            pr_number=pr.number,
            author=reviewer,
            body=body,
            state=state,
        )

    # Handle PR review comments (inline code comments)
    if isinstance(payload, PullRequestReviewCommentBase):
        comment = payload.comment
        line = comment.line or comment.original_line

        print("Received PR review comment!")
        print(f"  PR #{pr.number}: {pr.title}")
        print(f"  Repository: {payload.repository.full_name}")
        print(f"  Comment by: {comment.user.login}")
        print(f"  File: {comment.path}:{line}")
        print(f"  Comment: {comment.body}")
        print("=" * 60)

        return PRReviewActivityResponse(
            event_type="review_comment",
            pr_number=pr.number,
            author=comment.user.login,
            body=comment.body,
            file_path=comment.path,
            line=line,
        )

    # Fallback (shouldn't happen with proper subscription)
    raise ValueError(f"Unexpected payload type: {type(payload).__name__}")
