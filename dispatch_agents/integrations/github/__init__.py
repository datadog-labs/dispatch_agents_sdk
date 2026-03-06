"""GitHub integration for Dispatch agents.

This module provides typed payloads for GitHub webhook events that can be used
with the @on(github_event=...) decorator.

Quick Start:
    from dispatch_agents import on
    from dispatch_agents.integrations.github import PullRequestOpened

    @on(github_event=PullRequestOpened)
    async def handle_pr(payload: PullRequestOpened) -> None:
        print(f"New PR from @{payload.sender.login}: {payload.pull_request.title}")

    # Subscribe to multiple events using base class
    from dispatch_agents.integrations.github import (
        PullRequestOpened,
        PullRequestSynchronize,
        PullRequestBase,
    )

    @on(github_event=[PullRequestOpened, PullRequestSynchronize])
    async def handle_pr_changes(payload: PullRequestBase) -> None:
        ...

Action-Specific Event Classes:
    Each GitHub event has its own class that can be used with @on(github_event=...):
    - PullRequestOpened, PullRequestClosed, PullRequestSynchronize, ...
    - IssueOpened, IssueClosed, IssueLabeled, ...
    - IssueCommentCreated, IssueCommentEdited, ...
    - Push (no action)
    - CheckRunCreated, CheckRunCompleted, ...
    - WorkflowRunCompleted, ...
    - ReleasePublished, ...
    - Create, Delete, Fork (no action)
    - StarCreated, StarDeleted
    - InstallationCreated, ...

Base Classes (for subscribing to multiple events):
    - PullRequestBase: All pull_request.* events
    - IssueBase: All issues.* events
    - IssueCommentBase: All issue_comment.* events
    - CheckRunBase: All check_run.* events
    - WorkflowRunBase: All workflow_run.* events
    - ReleaseBase: All release.* events
    - StarBase: All star.* events
    - InstallationBase: All installation.* events

GitHub Topics:
    Events are routed to topics with the pattern "github.{event}.{action}":
    - github.pull_request.opened
    - github.pull_request.synchronize
    - github.issue_comment.created
    - github.push (no action for push events)
    - github.check_run.completed
    - etc.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict, Field

from dispatch_agents.events import BasePayload
from dispatch_agents.models import StrictBaseModel

# =============================================================================
# Constants
# =============================================================================

# Topic prefix for all GitHub events
GITHUB_TOPIC_PREFIX = "github."


# =============================================================================
# GitHubEvent Enum
# =============================================================================


# =============================================================================
# Base GitHub Model
# =============================================================================


class GitHubModel(StrictBaseModel):
    """Base model for GitHub types with permissive validation.

    GitHub webhooks include many fields that may not be relevant to most
    use cases. We use 'ignore' for extra fields to allow forward compatibility
    as GitHub adds new fields to their API.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# =============================================================================
# Core GitHub Types
# =============================================================================


class GitHubUser(GitHubModel):
    """GitHub user or organization account."""

    id: int = Field(description="Unique numeric ID")
    login: str = Field(description="Username/handle")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    avatar_url: str | None = Field(default=None, description="Avatar image URL")
    html_url: str | None = Field(default=None, description="Profile URL")
    type: str = Field(
        default="User", description="Account type: User, Bot, or Organization"
    )
    site_admin: bool = Field(
        default=False, description="Whether user is a GitHub site admin"
    )


class GitHubLabel(GitHubModel):
    """Label attached to issues or pull requests."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    url: str | None = Field(default=None, description="API URL")
    name: str = Field(description="Label name")
    description: str | None = Field(default=None, description="Label description")
    color: str = Field(description="Hex color code (without #)")
    default: bool = Field(default=False, description="Whether this is a default label")


class GitHubMilestone(GitHubModel):
    """Milestone for tracking progress on issues/PRs."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    number: int = Field(description="Milestone number")
    title: str = Field(description="Milestone title")
    description: str | None = Field(default=None, description="Milestone description")
    state: str = Field(description="State: open or closed")
    creator: GitHubUser | None = Field(
        default=None, description="User who created the milestone"
    )
    open_issues: int = Field(default=0, description="Number of open issues")
    closed_issues: int = Field(default=0, description="Number of closed issues")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    due_on: str | None = Field(default=None, description="ISO8601 due date")
    closed_at: str | None = Field(default=None, description="ISO8601 close timestamp")


class GitHubLicense(GitHubModel):
    """Repository license information."""

    key: str = Field(description="License identifier (e.g., 'mit')")
    name: str = Field(description="License name")
    spdx_id: str | None = Field(default=None, description="SPDX license identifier")
    url: str | None = Field(default=None, description="URL to license text")
    node_id: str | None = Field(default=None, description="GraphQL node ID")


class GitHubRepository(GitHubModel):
    """GitHub repository."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    name: str = Field(description="Repository name (without owner)")
    full_name: str = Field(description="Full repository name (owner/repo)")
    owner: GitHubUser = Field(description="Repository owner")
    private: bool = Field(default=False, description="Whether repository is private")
    html_url: str | None = Field(default=None, description="Web URL")
    description: str | None = Field(default=None, description="Repository description")
    fork: bool = Field(default=False, description="Whether this is a fork")
    url: str | None = Field(default=None, description="API URL")
    created_at: str | None = Field(
        default=None, description="ISO8601 creation timestamp"
    )
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    pushed_at: str | None = Field(
        default=None, description="ISO8601 last push timestamp"
    )
    homepage: str | None = Field(default=None, description="Homepage URL")
    size: int = Field(default=0, description="Repository size in KB")
    stargazers_count: int = Field(default=0, description="Number of stars")
    watchers_count: int = Field(default=0, description="Number of watchers")
    language: str | None = Field(default=None, description="Primary language")
    forks_count: int = Field(default=0, description="Number of forks")
    open_issues_count: int = Field(default=0, description="Number of open issues")
    default_branch: str = Field(default="main", description="Default branch name")
    license: GitHubLicense | None = Field(
        default=None, description="License information"
    )
    visibility: str = Field(default="public", description="Repository visibility")
    topics: list[str] = Field(
        default_factory=list, description="Repository topics/tags"
    )
    archived: bool = Field(default=False, description="Whether repository is archived")
    disabled: bool = Field(default=False, description="Whether repository is disabled")


class GitHubBranch(GitHubModel):
    """Branch reference in a pull request."""

    label: str | None = Field(default=None, description="Branch label (owner:branch)")
    ref: str = Field(description="Branch name")
    sha: str = Field(description="Commit SHA")
    user: GitHubUser | None = Field(
        default=None, description="User who owns the branch"
    )
    repo: GitHubRepository | None = Field(
        default=None, description="Repository containing the branch"
    )


class GitHubInstallation(GitHubModel):
    """GitHub App installation information.

    Present in webhook payloads when the event is delivered to a GitHub App.
    """

    id: int = Field(description="Installation ID (used for authentication)")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    account: GitHubUser | None = Field(
        default=None, description="Account where App is installed"
    )


# =============================================================================
# Pull Request Types
# =============================================================================


class GitHubLink(GitHubModel):
    """A hypermedia link (HAL-style {"href": "..."})."""

    href: str = Field(description="URL of the linked resource")


class GitHubPullRequestLinks(GitHubModel):
    """Hypermedia links for a pull request."""

    html: GitHubLink | None = Field(default=None, description="Link to HTML view")
    diff: GitHubLink | None = Field(default=None, description="Link to diff")
    patch: GitHubLink | None = Field(default=None, description="Link to patch")
    self_link: GitHubLink | None = Field(
        default=None, alias="self", description="Link to this resource"
    )
    comments: GitHubLink | None = Field(default=None, description="Link to comments")
    review_comments: GitHubLink | None = Field(
        default=None, description="Link to review comments"
    )
    review_comment: GitHubLink | None = Field(
        default=None, description="Link to review comment template"
    )
    commits: GitHubLink | None = Field(default=None, description="Link to commits")
    statuses: GitHubLink | None = Field(default=None, description="Link to statuses")


class GitHubAutoMerge(GitHubModel):
    """Auto-merge configuration for a pull request."""

    enabled_by: GitHubUser = Field(description="User who enabled auto-merge")
    merge_method: str = Field(description="Merge method: merge, squash, or rebase")
    commit_title: str | None = Field(
        default=None, description="Commit title for squash/merge"
    )
    commit_message: str | None = Field(
        default=None, description="Commit message for squash/merge"
    )


class GitHubPullRequest(GitHubModel):
    """Pull request data."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    number: int = Field(description="PR number within the repository")
    state: str = Field(description="State: open, closed")
    locked: bool = Field(default=False, description="Whether PR is locked")
    title: str = Field(description="PR title")
    body: str | None = Field(default=None, description="PR description body")
    user: GitHubUser = Field(description="User who opened the PR")
    labels: list[GitHubLabel] = Field(
        default_factory=list, description="Labels attached to PR"
    )
    milestone: GitHubMilestone | None = Field(
        default=None, description="Associated milestone"
    )
    assignee: GitHubUser | None = Field(default=None, description="Primary assignee")
    assignees: list[GitHubUser] = Field(
        default_factory=list, description="All assignees"
    )
    requested_reviewers: list[GitHubUser] = Field(
        default_factory=list, description="Requested reviewers"
    )
    head: GitHubBranch = Field(description="Source branch")
    base: GitHubBranch = Field(description="Target branch")
    html_url: str | None = Field(default=None, description="Web URL")
    diff_url: str | None = Field(default=None, description="Diff URL")
    patch_url: str | None = Field(default=None, description="Patch URL")
    created_at: str | None = Field(
        default=None, description="ISO8601 creation timestamp"
    )
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    closed_at: str | None = Field(default=None, description="ISO8601 close timestamp")
    merged_at: str | None = Field(default=None, description="ISO8601 merge timestamp")
    merge_commit_sha: str | None = Field(default=None, description="Merge commit SHA")
    merged: bool = Field(default=False, description="Whether PR is merged")
    mergeable: bool | None = Field(default=None, description="Whether PR is mergeable")
    mergeable_state: str | None = Field(default=None, description="Mergeable state")
    merged_by: GitHubUser | None = Field(default=None, description="User who merged")
    comments: int = Field(default=0, description="Number of comments")
    review_comments: int = Field(default=0, description="Number of review comments")
    commits: int = Field(default=0, description="Number of commits")
    additions: int = Field(default=0, description="Lines added")
    deletions: int = Field(default=0, description="Lines deleted")
    changed_files: int = Field(default=0, description="Number of files changed")
    draft: bool = Field(default=False, description="Whether PR is a draft")
    auto_merge: GitHubAutoMerge | None = Field(
        default=None, description="Auto-merge settings"
    )
    links: GitHubPullRequestLinks | None = Field(
        default=None, alias="_links", description="Hypermedia links"
    )


# =============================================================================
# Issue Types
# =============================================================================


class GitHubIssuePullRequest(GitHubModel):
    """Pull request info embedded in an issue (present when the issue is a PR)."""

    url: str | None = Field(default=None, description="API URL")
    html_url: str | None = Field(default=None, description="Web URL")
    diff_url: str | None = Field(default=None, description="Diff URL")
    patch_url: str | None = Field(default=None, description="Patch URL")
    merged_at: str | None = Field(default=None, description="ISO8601 merge timestamp")


class GitHubIssue(GitHubModel):
    """GitHub issue."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    number: int = Field(description="Issue number within the repository")
    title: str = Field(description="Issue title")
    body: str | None = Field(default=None, description="Issue description body")
    user: GitHubUser = Field(description="User who opened the issue")
    labels: list[GitHubLabel] = Field(default_factory=list, description="Labels")
    state: str = Field(description="State: open or closed")
    state_reason: str | None = Field(
        default=None, description="Reason for state (completed, not_planned, reopened)"
    )
    locked: bool = Field(default=False, description="Whether issue is locked")
    assignee: GitHubUser | None = Field(default=None, description="Primary assignee")
    assignees: list[GitHubUser] = Field(
        default_factory=list, description="All assignees"
    )
    milestone: GitHubMilestone | None = Field(
        default=None, description="Associated milestone"
    )
    comments: int = Field(default=0, description="Number of comments")
    created_at: str | None = Field(
        default=None, description="ISO8601 creation timestamp"
    )
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    closed_at: str | None = Field(default=None, description="ISO8601 close timestamp")
    html_url: str | None = Field(default=None, description="Web URL")
    # Pull request info is present when the issue is actually a PR
    pull_request: GitHubIssuePullRequest | None = Field(
        default=None, description="PR info if this issue is a PR"
    )


class GitHubComment(GitHubModel):
    """Comment on an issue or pull request."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    html_url: str = Field(description="Web URL")
    body: str = Field(description="Comment body (markdown)")
    user: GitHubUser = Field(description="User who created the comment")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    author_association: str = Field(
        default="NONE",
        description="Author's association with the repository: OWNER, MEMBER, COLLABORATOR, CONTRIBUTOR, etc.",
    )


class GitHubReviewComment(GitHubModel):
    """Review comment on a pull request diff."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    pull_request_review_id: int | None = Field(
        default=None, description="Associated review ID"
    )
    diff_hunk: str = Field(description="Diff context")
    path: str = Field(description="File path")
    position: int | None = Field(default=None, description="Position in diff")
    original_position: int | None = Field(default=None, description="Original position")
    commit_id: str = Field(description="Commit SHA")
    original_commit_id: str | None = Field(
        default=None, description="Original commit SHA"
    )
    user: GitHubUser = Field(description="Comment author")
    body: str = Field(description="Comment body")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    html_url: str = Field(description="Web URL")
    line: int | None = Field(default=None, description="Line number")
    original_line: int | None = Field(default=None, description="Original line number")
    start_line: int | None = Field(
        default=None, description="Start line for multi-line comments"
    )
    side: str | None = Field(
        default=None, description="Side of the diff: LEFT or RIGHT"
    )
    author_association: str = Field(
        default="NONE", description="Author's repository association"
    )


class GitHubReview(GitHubModel):
    """Pull request review."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    user: GitHubUser = Field(description="Reviewer")
    body: str | None = Field(default=None, description="Review body")
    state: str = Field(
        description="Review state: APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED, PENDING"
    )
    html_url: str = Field(description="Web URL")
    submitted_at: str | None = Field(
        default=None, description="ISO8601 submission timestamp"
    )
    commit_id: str = Field(description="Commit SHA being reviewed")
    author_association: str = Field(
        default="NONE", description="Reviewer's repository association"
    )


# =============================================================================
# Commit Types
# =============================================================================


class GitHubCommitUser(GitHubModel):
    """Git user information (author/committer)."""

    name: str = Field(description="Git user name")
    email: str = Field(description="Git user email")
    username: str | None = Field(default=None, description="GitHub username if linked")
    date: str | None = Field(default=None, description="ISO8601 timestamp")


class GitHubCommit(GitHubModel):
    """Git commit in a push event."""

    id: str = Field(description="Commit SHA")
    tree_id: str = Field(description="Tree SHA")
    message: str = Field(description="Commit message")
    timestamp: str = Field(description="ISO8601 timestamp")
    author: GitHubCommitUser = Field(description="Commit author")
    committer: GitHubCommitUser = Field(description="Commit committer")
    url: str = Field(description="API URL")
    distinct: bool = Field(
        default=True, description="Whether commit is distinct from previous pushes"
    )
    added: list[str] = Field(default_factory=list, description="Files added")
    removed: list[str] = Field(default_factory=list, description="Files removed")
    modified: list[str] = Field(default_factory=list, description="Files modified")


# =============================================================================
# Check Run / Workflow Types
# =============================================================================


class GitHubCheckRunOutput(GitHubModel):
    """Output of a check run."""

    title: str | None = Field(default=None, description="Check run title")
    summary: str | None = Field(default=None, description="Check run summary")
    text: str | None = Field(default=None, description="Check run details")
    annotations_count: int = Field(default=0, description="Number of annotations")
    annotations_url: str | None = Field(default=None, description="Annotations API URL")


class GitHubCheckSuite(GitHubModel):
    """Check suite containing one or more check runs."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    head_branch: str | None = Field(default=None, description="Branch name")
    head_sha: str = Field(description="Head commit SHA")
    status: str | None = Field(
        default=None, description="Status: queued, in_progress, completed"
    )
    conclusion: str | None = Field(
        default=None, description="Conclusion: success, failure, etc."
    )
    url: str | None = Field(default=None, description="API URL")
    created_at: str | None = Field(
        default=None, description="ISO8601 creation timestamp"
    )
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")


class GitHubCheckRun(GitHubModel):
    """Individual check run (CI job)."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    head_sha: str = Field(description="Commit SHA")
    external_id: str | None = Field(default=None, description="External reference ID")
    url: str | None = Field(default=None, description="API URL")
    html_url: str | None = Field(default=None, description="Web URL")
    details_url: str | None = Field(default=None, description="Details URL")
    status: str = Field(description="Status: queued, in_progress, completed")
    conclusion: str | None = Field(
        default=None, description="Conclusion: success, failure, neutral, etc."
    )
    started_at: str | None = Field(default=None, description="ISO8601 start timestamp")
    completed_at: str | None = Field(
        default=None, description="ISO8601 completion timestamp"
    )
    output: GitHubCheckRunOutput | None = Field(
        default=None, description="Check run output"
    )
    name: str = Field(description="Check run name")
    check_suite: GitHubCheckSuite | None = Field(
        default=None, description="Parent check suite"
    )


class GitHubWorkflowRun(GitHubModel):
    """GitHub Actions workflow run."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    name: str | None = Field(default=None, description="Workflow name")
    head_branch: str | None = Field(default=None, description="Branch name")
    head_sha: str = Field(description="Head commit SHA")
    run_number: int = Field(description="Run number within workflow")
    event: str = Field(description="Event that triggered the run")
    status: str | None = Field(
        default=None, description="Status: queued, in_progress, completed"
    )
    conclusion: str | None = Field(
        default=None, description="Conclusion: success, failure, etc."
    )
    workflow_id: int = Field(description="Workflow ID")
    url: str | None = Field(default=None, description="API URL")
    html_url: str | None = Field(default=None, description="Web URL")
    created_at: str | None = Field(
        default=None, description="ISO8601 creation timestamp"
    )
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    run_started_at: str | None = Field(
        default=None, description="ISO8601 start timestamp"
    )
    actor: GitHubUser | None = Field(
        default=None, description="User who triggered the run"
    )


# =============================================================================
# Release Types
# =============================================================================


class GitHubReleaseAsset(GitHubModel):
    """Asset attached to a release."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    name: str = Field(description="Asset filename")
    label: str | None = Field(default=None, description="Asset label")
    content_type: str = Field(description="MIME type")
    state: str = Field(description="State: uploaded")
    size: int = Field(description="File size in bytes")
    download_count: int = Field(default=0, description="Number of downloads")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    browser_download_url: str = Field(description="Download URL")
    uploader: GitHubUser | None = Field(
        default=None, description="User who uploaded the asset"
    )


class GitHubRelease(GitHubModel):
    """GitHub release."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    tag_name: str = Field(description="Git tag name")
    target_commitish: str | None = Field(
        default=None, description="Target commit/branch"
    )
    name: str | None = Field(default=None, description="Release name")
    body: str | None = Field(default=None, description="Release notes")
    draft: bool = Field(default=False, description="Whether release is a draft")
    prerelease: bool = Field(
        default=False, description="Whether release is a prerelease"
    )
    created_at: str = Field(description="ISO8601 creation timestamp")
    published_at: str | None = Field(
        default=None, description="ISO8601 publish timestamp"
    )
    author: GitHubUser = Field(description="Release author")
    html_url: str = Field(description="Web URL")
    tarball_url: str | None = Field(default=None, description="Tarball URL")
    zipball_url: str | None = Field(default=None, description="Zipball URL")
    assets: list[GitHubReleaseAsset] = Field(
        default_factory=list, description="Release assets"
    )


# =============================================================================
# Requested Action / Workflow / Review Thread Types
# =============================================================================


class GitHubRequestedAction(GitHubModel):
    """Action requested by a user on a check run."""

    identifier: str = Field(description="Identifier of the requested action")


class GitHubWorkflow(GitHubModel):
    """GitHub Actions workflow definition."""

    id: int = Field(description="Unique numeric ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    name: str = Field(description="Workflow name")
    path: str = Field(description="Path to workflow file")
    state: str = Field(description="Workflow state: active, disabled_manually, etc.")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str = Field(description="ISO8601 update timestamp")
    url: str = Field(description="API URL")
    html_url: str = Field(description="Web URL")
    badge_url: str | None = Field(default=None, description="Badge URL")


class GitHubWorkflowStep(GitHubModel):
    """A step in a GitHub Actions workflow job."""

    name: str = Field(description="Step name")
    status: str = Field(description="Status: queued, in_progress, completed")
    conclusion: str | None = Field(
        default=None, description="Conclusion: success, failure, skipped, etc."
    )
    number: int = Field(description="Step number")
    started_at: str | None = Field(default=None, description="ISO8601 start timestamp")
    completed_at: str | None = Field(
        default=None, description="ISO8601 completion timestamp"
    )


class GitHubWorkflowJob(GitHubModel):
    """A job in a GitHub Actions workflow run."""

    id: int = Field(description="Unique numeric ID")
    run_id: int = Field(description="Parent workflow run ID")
    node_id: str | None = Field(default=None, description="GraphQL node ID")
    head_sha: str = Field(description="Head commit SHA")
    url: str | None = Field(default=None, description="API URL")
    html_url: str | None = Field(default=None, description="Web URL")
    status: str = Field(description="Status: queued, in_progress, completed")
    conclusion: str | None = Field(
        default=None, description="Conclusion: success, failure, etc."
    )
    started_at: str | None = Field(default=None, description="ISO8601 start timestamp")
    completed_at: str | None = Field(
        default=None, description="ISO8601 completion timestamp"
    )
    name: str = Field(description="Job name")
    steps: list[GitHubWorkflowStep] = Field(
        default_factory=list, description="Job steps"
    )
    runner_id: int | None = Field(default=None, description="Runner ID")
    runner_name: str | None = Field(default=None, description="Runner name")
    runner_group_id: int | None = Field(default=None, description="Runner group ID")
    runner_group_name: str | None = Field(default=None, description="Runner group name")
    workflow_name: str | None = Field(default=None, description="Parent workflow name")


class GitHubReviewThread(GitHubModel):
    """A pull request review thread."""

    node_id: str = Field(description="GraphQL node ID")
    comments: list[GitHubReviewComment] = Field(
        default_factory=list, description="Comments in the thread"
    )


# =============================================================================
# Changes Type (for edit events)
# =============================================================================


class GitHubChangeValue(GitHubModel):
    """A changed value in an edit event (contains the previous value)."""

    # GitHub sends {"from": "old value"} for changed fields
    from_: str = Field(alias="from", description="Previous value before the change")


class GitHubChanges(GitHubModel):
    """Changes made in an edit event."""

    title: GitHubChangeValue | None = Field(
        default=None, description="Title change with previous value"
    )
    body: GitHubChangeValue | None = Field(
        default=None, description="Body change with previous value"
    )


# =============================================================================
# Base GitHub Event Payload
# =============================================================================


class GitHubEventPayload(BasePayload, GitHubModel):
    """Base class for all GitHub webhook event payloads.

    All GitHub event payloads share common fields: sender, repository,
    organization (optional), and installation (for GitHub Apps).

    Note: Inherits from both BasePayload (for dispatch compatibility)
    and GitHubModel (for permissive extra field handling).

    Subclasses should define `_dispatch_topic` as a ClassVar to specify
    which topic they subscribe to.
    """

    _dispatch_topic: ClassVar[str] = ""

    sender: GitHubUser = Field(description="User who triggered the event")
    repository: GitHubRepository = Field(description="Repository where event occurred")
    organization: GitHubUser | None = Field(
        default=None, description="Organization (only present for org-owned repos)"
    )

    @classmethod
    def dispatch_topic(cls) -> str:
        """Return the topic string for this event type.

        Raises:
            NotImplementedError: If the class doesn't define _dispatch_topic.
        """
        if not cls._dispatch_topic:
            raise NotImplementedError(
                f"{cls.__name__} must define _dispatch_topic to use dispatch_topic()"
            )
        return cls._dispatch_topic


# =============================================================================
# Pull Request Event Payloads
# =============================================================================


class PullRequestBase(GitHubEventPayload):
    """Base class for pull_request.* events.

    Triggered when a pull request is:
    - opened, closed, reopened
    - edited (title/body changed)
    - assigned, unassigned
    - review_requested, review_request_removed
    - labeled, unlabeled
    - synchronized (new commits pushed)
    - converted_to_draft, ready_for_review
    - locked, unlocked

    Use action-specific classes like PullRequestOpened, PullRequestClosed, etc.
    for type-safe event handling with the @on decorator.

    Example:
        from dispatch_agents import on
        from dispatch_agents.integrations.github import PullRequestOpened

        @on(github_event=PullRequestOpened)
        async def handle_pr_opened(payload: PullRequestOpened) -> None:
            print(f"PR #{payload.number} opened by {payload.sender.login}")
    """

    action: str = Field(description="Event action: opened, closed, synchronize, etc.")
    number: int = Field(description="PR number")
    pull_request: GitHubPullRequest = Field(description="Pull request data")


# =============================================================================
# Pull Request Action-Specific Classes
# =============================================================================


class PullRequestOpened(PullRequestBase):
    """Payload for github.pull_request.opened events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.opened"


class PullRequestClosed(PullRequestBase):
    """Payload for github.pull_request.closed events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.closed"


class PullRequestReopened(PullRequestBase):
    """Payload for github.pull_request.reopened events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.reopened"


class PullRequestSynchronize(PullRequestBase):
    """Payload for github.pull_request.synchronize events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.synchronize"
    before: str = Field(description="Previous head SHA before the push")
    after: str = Field(description="New head SHA after the push")


class PullRequestEdited(PullRequestBase):
    """Payload for github.pull_request.edited events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.edited"
    changes: GitHubChanges = Field(description="Changes made to the PR")


class PullRequestLabeled(PullRequestBase):
    """Payload for github.pull_request.labeled events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.labeled"
    label: GitHubLabel = Field(description="Label that was added")


class PullRequestUnlabeled(PullRequestBase):
    """Payload for github.pull_request.unlabeled events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.unlabeled"
    label: GitHubLabel = Field(description="Label that was removed")


class PullRequestAssigned(PullRequestBase):
    """Payload for github.pull_request.assigned events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.assigned"
    assignee: GitHubUser = Field(description="User who was assigned")


class PullRequestUnassigned(PullRequestBase):
    """Payload for github.pull_request.unassigned events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.unassigned"
    assignee: GitHubUser = Field(description="User who was unassigned")


class PullRequestReviewRequested(PullRequestBase):
    """Payload for github.pull_request.review_requested events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.review_requested"
    requested_reviewer: GitHubUser = Field(description="User requested for review")


class PullRequestReviewRequestRemoved(PullRequestBase):
    """Payload for github.pull_request.review_request_removed events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.review_request_removed"
    requested_reviewer: GitHubUser = Field(
        description="User whose review request was removed"
    )


class PullRequestReadyForReview(PullRequestBase):
    """Payload for github.pull_request.ready_for_review events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.ready_for_review"


class PullRequestConvertedToDraft(PullRequestBase):
    """Payload for github.pull_request.converted_to_draft events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.converted_to_draft"


class PullRequestLocked(PullRequestBase):
    """Payload for github.pull_request.locked events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.locked"


class PullRequestUnlocked(PullRequestBase):
    """Payload for github.pull_request.unlocked events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.unlocked"


class PullRequestReviewBase(GitHubEventPayload):
    """Base class for pull_request_review.* events.

    Triggered when a pull request review is:
    - submitted (includes approval, changes requested, or comment)
    - edited
    - dismissed

    Use action-specific classes like PullRequestReviewSubmitted, etc.
    """

    action: str = Field(description="Event action: submitted, edited, dismissed")
    pull_request: GitHubPullRequest = Field(description="Pull request being reviewed")
    review: GitHubReview = Field(description="Review data")


class PullRequestReviewSubmitted(PullRequestReviewBase):
    """Payload for github.pull_request_review.submitted events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review.submitted"


class PullRequestReviewEdited(PullRequestReviewBase):
    """Payload for github.pull_request_review.edited events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review.edited"
    changes: GitHubChanges = Field(description="Changes made to the review")


class PullRequestReviewDismissed(PullRequestReviewBase):
    """Payload for github.pull_request_review.dismissed events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review.dismissed"


class PullRequestReviewCommentBase(GitHubEventPayload):
    """Base class for pull_request_review_comment.* events.

    Triggered when a comment on a pull request diff is:
    - created
    - edited
    - deleted

    Use action-specific classes like PullRequestReviewCommentCreated, etc.
    """

    action: str = Field(description="Event action: created, edited, deleted")
    pull_request: GitHubPullRequest = Field(description="Pull request")
    comment: GitHubReviewComment = Field(description="Review comment data")


class PullRequestReviewCommentCreated(PullRequestReviewCommentBase):
    """Payload for github.pull_request_review_comment.created events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review_comment.created"


class PullRequestReviewCommentEdited(PullRequestReviewCommentBase):
    """Payload for github.pull_request_review_comment.edited events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review_comment.edited"
    changes: GitHubChanges = Field(description="Changes made to the comment")


class PullRequestReviewCommentDeleted(PullRequestReviewCommentBase):
    """Payload for github.pull_request_review_comment.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review_comment.deleted"


class PullRequestReviewThreadBase(GitHubEventPayload):
    """Base class for pull_request_review_thread.* events.

    Triggered when a pull request review thread is:
    - resolved
    - unresolved

    Use action-specific classes like PullRequestReviewThreadResolved, etc.
    """

    action: str = Field(description="Event action: resolved, unresolved")
    pull_request: GitHubPullRequest = Field(description="Pull request")
    thread: GitHubReviewThread = Field(description="Thread data")


class PullRequestReviewThreadResolved(PullRequestReviewThreadBase):
    """Payload for github.pull_request_review_thread.resolved events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review_thread.resolved"


class PullRequestReviewThreadUnresolved(PullRequestReviewThreadBase):
    """Payload for github.pull_request_review_thread.unresolved events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request_review_thread.unresolved"


# =============================================================================
# Issue Event Payloads
# =============================================================================


class IssueBase(GitHubEventPayload):
    """Base class for issues.* events.

    Triggered when an issue is:
    - opened, closed, reopened, deleted
    - edited (title/body changed)
    - assigned, unassigned
    - labeled, unlabeled
    - pinned, unpinned
    - locked, unlocked
    - transferred
    - milestoned, demilestoned

    Use action-specific classes like IssueOpened, IssueClosed, etc.
    """

    action: str = Field(description="Event action: opened, closed, edited, etc.")
    issue: GitHubIssue = Field(description="Issue data")


class IssueOpened(IssueBase):
    """Payload for github.issues.opened events."""

    _dispatch_topic: ClassVar[str] = "github.issues.opened"
    changes: GitHubChanges | None = Field(
        default=None, description="Changes made when issue was opened"
    )


class IssueClosed(IssueBase):
    """Payload for github.issues.closed events."""

    _dispatch_topic: ClassVar[str] = "github.issues.closed"


class IssueReopened(IssueBase):
    """Payload for github.issues.reopened events."""

    _dispatch_topic: ClassVar[str] = "github.issues.reopened"


class IssueEdited(IssueBase):
    """Payload for github.issues.edited events."""

    _dispatch_topic: ClassVar[str] = "github.issues.edited"
    changes: GitHubChanges = Field(description="Changes made to the issue")
    label: GitHubLabel | None = Field(
        default=None, description="Label associated with the edit"
    )


class IssueLabeled(IssueBase):
    """Payload for github.issues.labeled events."""

    _dispatch_topic: ClassVar[str] = "github.issues.labeled"
    label: GitHubLabel = Field(description="Label that was added")


class IssueUnlabeled(IssueBase):
    """Payload for github.issues.unlabeled events."""

    _dispatch_topic: ClassVar[str] = "github.issues.unlabeled"
    label: GitHubLabel = Field(description="Label that was removed")


class IssueAssigned(IssueBase):
    """Payload for github.issues.assigned events."""

    _dispatch_topic: ClassVar[str] = "github.issues.assigned"
    assignee: GitHubUser | None = Field(
        default=None, description="User who was assigned"
    )


class IssueUnassigned(IssueBase):
    """Payload for github.issues.unassigned events."""

    _dispatch_topic: ClassVar[str] = "github.issues.unassigned"
    assignee: GitHubUser | None = Field(
        default=None, description="User who was unassigned"
    )


class IssueMilestoned(IssueBase):
    """Payload for github.issues.milestoned events."""

    _dispatch_topic: ClassVar[str] = "github.issues.milestoned"
    milestone: GitHubMilestone = Field(description="Milestone added to the issue")


class IssueDemilestoned(IssueBase):
    """Payload for github.issues.demilestoned events."""

    _dispatch_topic: ClassVar[str] = "github.issues.demilestoned"
    milestone: GitHubMilestone = Field(description="Milestone removed from the issue")


class IssueLocked(IssueBase):
    """Payload for github.issues.locked events."""

    _dispatch_topic: ClassVar[str] = "github.issues.locked"


class IssueUnlocked(IssueBase):
    """Payload for github.issues.unlocked events."""

    _dispatch_topic: ClassVar[str] = "github.issues.unlocked"


class IssueTransferred(IssueBase):
    """Payload for github.issues.transferred events."""

    _dispatch_topic: ClassVar[str] = "github.issues.transferred"
    changes: GitHubChanges = Field(description="Changes from the transfer")


class IssuePinned(IssueBase):
    """Payload for github.issues.pinned events."""

    _dispatch_topic: ClassVar[str] = "github.issues.pinned"


class IssueUnpinned(IssueBase):
    """Payload for github.issues.unpinned events."""

    _dispatch_topic: ClassVar[str] = "github.issues.unpinned"


class IssueCommentBase(GitHubEventPayload):
    """Base class for issue_comment.* events.

    Triggered when a comment on an issue or pull request is:
    - created
    - edited
    - deleted

    Note: GitHub sends issue_comment events for both issues AND pull requests.
    Check issue.pull_request to determine if the comment is on a PR.

    Use action-specific classes like IssueCommentCreated, etc.
    """

    action: str = Field(description="Event action: created, edited, deleted")
    issue: GitHubIssue = Field(description="Issue (or PR) being commented on")
    comment: GitHubComment = Field(description="Comment data")

    @property
    def is_pull_request_comment(self) -> bool:
        """Return True if this comment is on a pull request, not an issue."""
        return self.issue.pull_request is not None


class IssueCommentCreated(IssueCommentBase):
    """Payload for github.issue_comment.created events."""

    _dispatch_topic: ClassVar[str] = "github.issue_comment.created"


class IssueCommentEdited(IssueCommentBase):
    """Payload for github.issue_comment.edited events."""

    _dispatch_topic: ClassVar[str] = "github.issue_comment.edited"
    changes: GitHubChanges = Field(description="Changes made to the comment")


class IssueCommentDeleted(IssueCommentBase):
    """Payload for github.issue_comment.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.issue_comment.deleted"


# =============================================================================
# Push Event Payloads
# =============================================================================


class Push(GitHubEventPayload):
    """Payload for github.push events.

    Triggered when commits are pushed to a repository branch or tag.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.push"

    ref: str = Field(description="Git ref (e.g., refs/heads/main)")
    before: str = Field(description="SHA before the push")
    after: str = Field(description="SHA after the push")
    created: bool = Field(default=False, description="Whether the ref was created")
    deleted: bool = Field(default=False, description="Whether the ref was deleted")
    forced: bool = Field(default=False, description="Whether the push was a force push")
    base_ref: str | None = Field(
        default=None, description="Base ref for branch creation"
    )
    compare: str = Field(description="URL comparing before/after")
    commits: list[GitHubCommit] = Field(
        default_factory=list, description="Commits pushed"
    )
    head_commit: GitHubCommit | None = Field(default=None, description="Head commit")
    pusher: GitHubCommitUser = Field(description="User who pushed")

    @property
    def branch(self) -> str | None:
        """Extract branch name from ref (e.g., 'main' from 'refs/heads/main')."""
        if self.ref.startswith("refs/heads/"):
            return self.ref[11:]
        return None

    @property
    def tag(self) -> str | None:
        """Extract tag name from ref (e.g., 'v1.0.0' from 'refs/tags/v1.0.0')."""
        if self.ref.startswith("refs/tags/"):
            return self.ref[10:]
        return None


# =============================================================================
# Check Run / Workflow Event Payloads
# =============================================================================


class CheckRunBase(GitHubEventPayload):
    """Base class for check_run.* events.

    Triggered when a check run is:
    - created
    - completed
    - rerequested
    - requested_action

    Use action-specific classes like CheckRunCreated, CheckRunCompleted, etc.
    """

    action: str = Field(
        description="Event action: created, completed, rerequested, requested_action"
    )
    check_run: GitHubCheckRun = Field(description="Check run data")
    requested_action: GitHubRequestedAction | None = Field(
        default=None, description="Requested action if action is 'requested_action'"
    )


class CheckRunCreated(CheckRunBase):
    """Payload for github.check_run.created events."""

    _dispatch_topic: ClassVar[str] = "github.check_run.created"


class CheckRunCompleted(CheckRunBase):
    """Payload for github.check_run.completed events."""

    _dispatch_topic: ClassVar[str] = "github.check_run.completed"


class CheckRunRerequested(CheckRunBase):
    """Payload for github.check_run.rerequested events."""

    _dispatch_topic: ClassVar[str] = "github.check_run.rerequested"


class CheckRunRequestedAction(CheckRunBase):
    """Payload for github.check_run.requested_action events."""

    _dispatch_topic: ClassVar[str] = "github.check_run.requested_action"
    requested_action: GitHubRequestedAction = Field(
        description="The action requested by the user"
    )


class CheckSuiteBase(GitHubEventPayload):
    """Base class for check_suite.* events.

    Triggered when a check suite is:
    - completed
    - requested
    - rerequested

    Use action-specific classes like CheckSuiteCompleted, etc.
    """

    action: str = Field(description="Event action: completed, requested, rerequested")
    check_suite: GitHubCheckSuite = Field(description="Check suite data")


class CheckSuiteCompleted(CheckSuiteBase):
    """Payload for github.check_suite.completed events."""

    _dispatch_topic: ClassVar[str] = "github.check_suite.completed"


class CheckSuiteRequested(CheckSuiteBase):
    """Payload for github.check_suite.requested events."""

    _dispatch_topic: ClassVar[str] = "github.check_suite.requested"


class CheckSuiteRerequested(CheckSuiteBase):
    """Payload for github.check_suite.rerequested events."""

    _dispatch_topic: ClassVar[str] = "github.check_suite.rerequested"


class WorkflowRunBase(GitHubEventPayload):
    """Base class for workflow_run.* events.

    Triggered when a GitHub Actions workflow run is:
    - requested (queued)
    - in_progress
    - completed

    Use action-specific classes like WorkflowRunCompleted, etc.
    """

    action: str = Field(description="Event action: requested, in_progress, completed")
    workflow_run: GitHubWorkflowRun = Field(description="Workflow run data")
    workflow: GitHubWorkflow = Field(description="Workflow definition")


class WorkflowRunRequested(WorkflowRunBase):
    """Payload for github.workflow_run.requested events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_run.requested"


class WorkflowRunInProgress(WorkflowRunBase):
    """Payload for github.workflow_run.in_progress events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_run.in_progress"


class WorkflowRunCompleted(WorkflowRunBase):
    """Payload for github.workflow_run.completed events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_run.completed"


class WorkflowJobBase(GitHubEventPayload):
    """Base class for workflow_job.* events.

    Triggered when a GitHub Actions workflow job is:
    - queued
    - in_progress
    - completed
    - waiting

    Use action-specific classes like WorkflowJobCompleted, etc.
    """

    action: str = Field(
        description="Event action: queued, in_progress, completed, waiting"
    )
    workflow_job: GitHubWorkflowJob = Field(description="Workflow job data")


class WorkflowJobQueued(WorkflowJobBase):
    """Payload for github.workflow_job.queued events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_job.queued"


class WorkflowJobInProgress(WorkflowJobBase):
    """Payload for github.workflow_job.in_progress events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_job.in_progress"


class WorkflowJobCompleted(WorkflowJobBase):
    """Payload for github.workflow_job.completed events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_job.completed"


class WorkflowJobWaiting(WorkflowJobBase):
    """Payload for github.workflow_job.waiting events."""

    _dispatch_topic: ClassVar[str] = "github.workflow_job.waiting"


# =============================================================================
# Release Event Payloads
# =============================================================================


class ReleaseBase(GitHubEventPayload):
    """Base class for release.* events.

    Triggered when a release is:
    - created (including draft releases)
    - published (draft -> public)
    - unpublished
    - edited
    - deleted
    - prereleased
    - released

    Use action-specific classes like ReleasePublished, etc.
    """

    action: str = Field(
        description="Event action: created, published, edited, deleted, etc."
    )
    release: GitHubRelease = Field(description="Release data")


class ReleaseCreated(ReleaseBase):
    """Payload for github.release.created events."""

    _dispatch_topic: ClassVar[str] = "github.release.created"


class ReleasePublished(ReleaseBase):
    """Payload for github.release.published events."""

    _dispatch_topic: ClassVar[str] = "github.release.published"


class ReleaseUnpublished(ReleaseBase):
    """Payload for github.release.unpublished events."""

    _dispatch_topic: ClassVar[str] = "github.release.unpublished"


class ReleaseEdited(ReleaseBase):
    """Payload for github.release.edited events."""

    _dispatch_topic: ClassVar[str] = "github.release.edited"
    changes: GitHubChanges = Field(description="Changes made to the release")


class ReleaseDeleted(ReleaseBase):
    """Payload for github.release.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.release.deleted"


class ReleasePrereleased(ReleaseBase):
    """Payload for github.release.prereleased events."""

    _dispatch_topic: ClassVar[str] = "github.release.prereleased"


class ReleaseReleased(ReleaseBase):
    """Payload for github.release.released events."""

    _dispatch_topic: ClassVar[str] = "github.release.released"


# =============================================================================
# Repository Event Payloads
# =============================================================================


class Create(GitHubEventPayload):
    """Payload for github.create events.

    Triggered when a branch or tag is created.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.create"

    ref: str = Field(description="Git ref name (branch or tag name)")
    ref_type: Literal["branch", "tag"] = Field(description="Type of ref: branch or tag")
    master_branch: str = Field(description="Default branch of the repository")
    description: str | None = Field(default=None, description="Repository description")
    pusher_type: str = Field(default="user", description="Pusher type")


class Delete(GitHubEventPayload):
    """Payload for github.delete events.

    Triggered when a branch or tag is deleted.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.delete"

    ref: str = Field(description="Git ref name (branch or tag name)")
    ref_type: Literal["branch", "tag"] = Field(description="Type of ref: branch or tag")
    pusher_type: str = Field(default="user", description="Pusher type")


class Fork(GitHubEventPayload):
    """Payload for github.fork events.

    Triggered when a repository is forked.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.fork"

    forkee: GitHubRepository = Field(description="Newly created fork repository")


class StarBase(GitHubEventPayload):
    """Base class for star.* events.

    Triggered when a repository is starred or unstarred.

    Use action-specific classes like StarCreated, StarDeleted.
    """

    action: str = Field(description="Event action: created, deleted")


class StarCreated(StarBase):
    """Payload for github.star.created events."""

    _dispatch_topic: ClassVar[str] = "github.star.created"
    starred_at: str = Field(description="ISO8601 timestamp when starred")


class StarDeleted(StarBase):
    """Payload for github.star.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.star.deleted"
    starred_at: None = Field(default=None, description="Always null for deleted events")


# =============================================================================
# Installation Event Payloads (GitHub App specific)
# =============================================================================


class InstallationBase(BasePayload, GitHubModel):
    """Base class for installation.* events.

    Triggered when a GitHub App installation is:
    - created (App installed)
    - deleted (App uninstalled)
    - new_permissions_accepted
    - suspend
    - unsuspend

    Note: Does not have repository field since it's about the App installation.

    Use action-specific classes like InstallationCreated, etc.
    """

    _dispatch_topic: ClassVar[str] = ""

    action: str = Field(
        description="Event action: created, deleted, suspend, unsuspend, etc."
    )
    installation: GitHubInstallation = Field(description="Installation data")
    sender: GitHubUser = Field(description="User who triggered the event")
    repositories: list[GitHubRepository] = Field(
        default_factory=list, description="Repositories with access"
    )

    @classmethod
    def dispatch_topic(cls) -> str:
        """Return the topic string for this event type."""
        if not cls._dispatch_topic:
            raise NotImplementedError(
                f"{cls.__name__} must define _dispatch_topic to use dispatch_topic()"
            )
        return cls._dispatch_topic


class InstallationCreated(InstallationBase):
    """Payload for github.installation.created events."""

    _dispatch_topic: ClassVar[str] = "github.installation.created"


class InstallationDeleted(InstallationBase):
    """Payload for github.installation.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.installation.deleted"


class InstallationSuspend(InstallationBase):
    """Payload for github.installation.suspend events."""

    _dispatch_topic: ClassVar[str] = "github.installation.suspend"


class InstallationUnsuspend(InstallationBase):
    """Payload for github.installation.unsuspend events."""

    _dispatch_topic: ClassVar[str] = "github.installation.unsuspend"


class InstallationNewPermissionsAccepted(InstallationBase):
    """Payload for github.installation.new_permissions_accepted events."""

    _dispatch_topic: ClassVar[str] = "github.installation.new_permissions_accepted"


class InstallationRepositoriesBase(BasePayload, GitHubModel):
    """Base class for installation_repositories.* events.

    Triggered when repositories are added or removed from an installation.

    Use action-specific classes like InstallationRepositoriesAdded, etc.
    """

    _dispatch_topic: ClassVar[str] = ""

    action: str = Field(description="Event action: added, removed")
    installation: GitHubInstallation = Field(description="Installation data")
    sender: GitHubUser = Field(description="User who triggered the event")
    requester: GitHubUser | None = Field(
        default=None, description="User who requested the change (may be null)"
    )
    repositories_added: list[GitHubRepository] = Field(
        default_factory=list, description="Repositories added"
    )
    repositories_removed: list[GitHubRepository] = Field(
        default_factory=list, description="Repositories removed"
    )
    repository_selection: str = Field(
        default="all", description="Repository selection: all or selected"
    )

    @classmethod
    def dispatch_topic(cls) -> str:
        """Return the topic string for this event type."""
        if not cls._dispatch_topic:
            raise NotImplementedError(
                f"{cls.__name__} must define _dispatch_topic to use dispatch_topic()"
            )
        return cls._dispatch_topic


class InstallationRepositoriesAdded(InstallationRepositoriesBase):
    """Payload for github.installation_repositories.added events."""

    _dispatch_topic: ClassVar[str] = "github.installation_repositories.added"


class InstallationRepositoriesRemoved(InstallationRepositoriesBase):
    """Payload for github.installation_repositories.removed events."""

    _dispatch_topic: ClassVar[str] = "github.installation_repositories.removed"


# =============================================================================
# Type alias for all GitHub event payloads
# =============================================================================

GitHubPayload = (
    PullRequestBase
    | PullRequestReviewBase
    | PullRequestReviewCommentBase
    | PullRequestReviewThreadBase
    | IssueBase
    | IssueCommentBase
    | Push
    | CheckRunBase
    | CheckSuiteBase
    | WorkflowRunBase
    | WorkflowJobBase
    | ReleaseBase
    | Create
    | Delete
    | Fork
    | StarBase
    | InstallationBase
    | InstallationRepositoriesBase
)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base event payload class
    "GitHubEventPayload",
    "GitHubPayload",
    # Pull Request events
    "PullRequestBase",
    "PullRequestOpened",
    "PullRequestClosed",
    "PullRequestReopened",
    "PullRequestSynchronize",
    "PullRequestEdited",
    "PullRequestLabeled",
    "PullRequestUnlabeled",
    "PullRequestAssigned",
    "PullRequestUnassigned",
    "PullRequestReviewRequested",
    "PullRequestReviewRequestRemoved",
    "PullRequestReadyForReview",
    "PullRequestConvertedToDraft",
    "PullRequestLocked",
    "PullRequestUnlocked",
    # Pull Request Review events
    "PullRequestReviewBase",
    "PullRequestReviewSubmitted",
    "PullRequestReviewEdited",
    "PullRequestReviewDismissed",
    # Pull Request Review Comment events
    "PullRequestReviewCommentBase",
    "PullRequestReviewCommentCreated",
    "PullRequestReviewCommentEdited",
    "PullRequestReviewCommentDeleted",
    # Pull Request Review Thread events
    "PullRequestReviewThreadBase",
    "PullRequestReviewThreadResolved",
    "PullRequestReviewThreadUnresolved",
    # Issue events
    "IssueBase",
    "IssueOpened",
    "IssueClosed",
    "IssueReopened",
    "IssueEdited",
    "IssueLabeled",
    "IssueUnlabeled",
    "IssueAssigned",
    "IssueUnassigned",
    "IssueMilestoned",
    "IssueDemilestoned",
    "IssueLocked",
    "IssueUnlocked",
    "IssueTransferred",
    "IssuePinned",
    "IssueUnpinned",
    # Issue Comment events
    "IssueCommentBase",
    "IssueCommentCreated",
    "IssueCommentEdited",
    "IssueCommentDeleted",
    # Push event (no action)
    "Push",
    # Check Run events
    "CheckRunBase",
    "CheckRunCreated",
    "CheckRunCompleted",
    "CheckRunRerequested",
    "CheckRunRequestedAction",
    # Check Suite events
    "CheckSuiteBase",
    "CheckSuiteCompleted",
    "CheckSuiteRequested",
    "CheckSuiteRerequested",
    # Workflow Run events
    "WorkflowRunBase",
    "WorkflowRunRequested",
    "WorkflowRunInProgress",
    "WorkflowRunCompleted",
    # Workflow Job events
    "WorkflowJobBase",
    "WorkflowJobQueued",
    "WorkflowJobInProgress",
    "WorkflowJobCompleted",
    "WorkflowJobWaiting",
    # Release events
    "ReleaseBase",
    "ReleaseCreated",
    "ReleasePublished",
    "ReleaseUnpublished",
    "ReleaseEdited",
    "ReleaseDeleted",
    "ReleasePrereleased",
    "ReleaseReleased",
    # Repository events (no action)
    "Create",
    "Delete",
    "Fork",
    # Star events
    "StarBase",
    "StarCreated",
    "StarDeleted",
    # Installation events
    "InstallationBase",
    "InstallationCreated",
    "InstallationDeleted",
    "InstallationSuspend",
    "InstallationUnsuspend",
    "InstallationNewPermissionsAccepted",
    # Installation Repositories events
    "InstallationRepositoriesBase",
    "InstallationRepositoriesAdded",
    "InstallationRepositoriesRemoved",
    # Core GitHub types
    "GitHubModel",
    "GitHubUser",
    "GitHubRepository",
    "GitHubInstallation",
    "GitHubBranch",
    "GitHubPullRequest",
    "GitHubPullRequestLinks",
    "GitHubAutoMerge",
    "GitHubIssue",
    "GitHubComment",
    "GitHubReviewComment",
    "GitHubReview",
    "GitHubLabel",
    "GitHubMilestone",
    "GitHubLicense",
    "GitHubCommit",
    "GitHubCommitUser",
    "GitHubCheckRun",
    "GitHubCheckRunOutput",
    "GitHubCheckSuite",
    "GitHubWorkflowRun",
    "GitHubWorkflow",
    "GitHubWorkflowStep",
    "GitHubWorkflowJob",
    "GitHubRelease",
    "GitHubReleaseAsset",
    "GitHubRequestedAction",
    "GitHubReviewThread",
    "GitHubLink",
    "GitHubIssuePullRequest",
    "GitHubChangeValue",
    "GitHubChanges",
]
