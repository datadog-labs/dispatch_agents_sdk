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
    - DeploymentCreated
    - DeploymentStatusCreated
    - DeploymentReviewApproved, DeploymentReviewRejected, DeploymentReviewRequested
    - DependabotAlertCreated, DependabotAlertFixed, DependabotAlertDismissed,
      DependabotAlertReintroduced, DependabotAlertAutoDismissed,
      DependabotAlertAutoReopened, DependabotAlertReopened
    - LabelCreated, LabelEdited, LabelDeleted

No-Action Events:
    Events with no action field (subscribe directly to the event class):
    - CommitStatus
    - WorkflowDispatch

Base Classes (for subscribing to multiple events):
    - PullRequestBase: All pull_request.* events
    - IssueBase: All issues.* events
    - IssueCommentBase: All issue_comment.* events
    - CheckRunBase: All check_run.* events
    - WorkflowRunBase: All workflow_run.* events
    - ReleaseBase: All release.* events
    - StarBase: All star.* events
    - InstallationBase: All installation.* events
    - DeploymentBase: All deployment.* events
    - DeploymentStatusBase: All deployment_status.* events
    - DeploymentReviewBase: All deployment_review.* events
    - DependabotAlertBase: All dependabot_alert.* events
    - LabelBase: All label.* events

GitHub Topics:
    Events are routed to topics with the pattern "github.{event}.{action}":
    - github.pull_request.opened
    - github.pull_request.synchronize
    - github.issue_comment.created
    - github.push (no action for push events)
    - github.check_run.completed
    - github.deployment.created
    - github.deployment_status.created
    - github.deployment_review.approved
    - github.deployment_review.rejected
    - github.deployment_review.requested
    - github.status (no action for commit status events)
    - github.workflow_dispatch (no action for workflow dispatch events)
    - github.dependabot_alert.created
    - github.label.created
    - etc.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

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

    sender: GitHubUser | None = Field(
        default=None,
        description="User who triggered the event (absent for some automated events)",
    )
    repository: GitHubRepository | None = Field(
        default=None,
        description="Repository where event occurred (absent for org-level events)",
    )
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


class PullRequestMilestoned(PullRequestBase):
    """Payload for github.pull_request.milestoned events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.milestoned"
    milestone: GitHubMilestone = Field(description="Milestone added to the PR")


class PullRequestDemilestoned(PullRequestBase):
    """Payload for github.pull_request.demilestoned events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.demilestoned"
    milestone: GitHubMilestone = Field(description="Milestone removed from the PR")


class PullRequestAutoMergeEnabled(PullRequestBase):
    """Payload for github.pull_request.auto_merge_enabled events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.auto_merge_enabled"
    reason: str = Field(description="Reason auto-merge was enabled")


class PullRequestAutoMergeDisabled(PullRequestBase):
    """Payload for github.pull_request.auto_merge_disabled events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.auto_merge_disabled"
    reason: str = Field(description="Reason auto-merge was disabled")


class PullRequestEnqueued(PullRequestBase):
    """Payload for github.pull_request.enqueued events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.enqueued"


class PullRequestDequeued(PullRequestBase):
    """Payload for github.pull_request.dequeued events."""

    _dispatch_topic: ClassVar[str] = "github.pull_request.dequeued"
    reason: str = Field(description="Reason the pull request was dequeued")


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


class IssueDeleted(IssueBase):
    """Payload for github.issues.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.issues.deleted"


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
# Deployment Event Payloads
# =============================================================================


class GitHubDeployment(GitHubModel):
    """GitHub deployment object."""

    id: int = Field(description="Unique numeric ID of the deployment")
    sha: str = Field(description="SHA of the commit that is being deployed")
    ref: str = Field(description="Ref (branch or tag) that was deployed")
    task: str = Field(description="Task name, e.g. 'deploy' or 'deploy:migrations'")
    environment: str = Field(description="Name of the target deployment environment")
    description: str | None = Field(
        default=None, description="Optional description of the deployment"
    )
    creator: GitHubUser | None = Field(
        default=None, description="User who created the deployment"
    )
    created_at: str = Field(
        description="ISO8601 timestamp when the deployment was created"
    )
    updated_at: str = Field(
        description="ISO8601 timestamp when the deployment was last updated"
    )
    statuses_url: str = Field(
        description="API URL to list statuses for this deployment"
    )
    repository_url: str = Field(description="API URL of the repository")
    url: str = Field(description="API URL of the deployment")
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Extra information sent to the deployment"
    )


class DeploymentBase(GitHubEventPayload):
    """Base class for deployment.* events.

    Triggered when a deployment is created.
    Use action-specific classes like DeploymentCreated.
    """

    action: str = Field(description="Event action, e.g. 'created'")
    deployment: GitHubDeployment = Field(description="Deployment object")
    workflow: GitHubWorkflow | None = Field(
        default=None, description="Workflow that triggered the deployment, if any"
    )
    workflow_run: GitHubWorkflowRun | None = Field(
        default=None, description="Workflow run that triggered the deployment, if any"
    )


class DeploymentCreated(DeploymentBase):
    """Payload for github.deployment.created events."""

    _dispatch_topic: ClassVar[str] = "github.deployment.created"


# =============================================================================
# Deployment Status Event Payloads
# =============================================================================


class GitHubDeploymentStatus(GitHubModel):
    """GitHub deployment status object."""

    id: int = Field(description="Unique numeric ID of the deployment status")
    state: str = Field(
        description="State of the deployment status, e.g. 'pending', 'success', 'failure', 'error', 'inactive', 'in_progress', 'queued', 'waiting'"
    )
    description: str | None = Field(
        default=None, description="Optional short description of the status"
    )
    environment: str = Field(
        description="Name of the target deployment environment for the status"
    )
    target_url: str | None = Field(
        default=None,
        description="URL associated with the status, e.g. a CI build URL",
    )
    created_at: str = Field(
        description="ISO8601 timestamp when the deployment status was created"
    )
    updated_at: str = Field(
        description="ISO8601 timestamp when the deployment status was last updated"
    )
    deployment_url: str = Field(description="API URL of the associated deployment")
    repository_url: str = Field(description="API URL of the repository")


class DeploymentStatusBase(GitHubEventPayload):
    """Base class for deployment_status.* events.

    Triggered when a deployment status is created.
    Use action-specific classes like DeploymentStatusCreated.
    """

    action: str = Field(description="Event action, e.g. 'created'")
    deployment_status: GitHubDeploymentStatus = Field(
        description="Deployment status object"
    )
    deployment: GitHubDeployment = Field(
        description="Deployment associated with the status"
    )


class DeploymentStatusCreated(DeploymentStatusBase):
    """Payload for github.deployment_status.created events."""

    _dispatch_topic: ClassVar[str] = "github.deployment_status.created"


# =============================================================================
# Deployment Review Event Payloads
# =============================================================================


class DeploymentReviewBase(GitHubEventPayload):
    """Base class for deployment_review.* events.

    Triggered when a deployment review is approved, rejected, or requested.
    This is a GitHub Apps/Environments-specific event.
    Use action-specific classes like DeploymentReviewApproved, etc.
    """

    action: str = Field(
        description="Event action, e.g. 'approved', 'rejected', 'requested'"
    )
    # workflow_run and reviewers use dict[str, Any] rather than typed models because
    # the deployment_review webhook delivers a stripped-down workflow run shape that
    # differs from GitHubWorkflowRun (missing many standard fields).
    workflow_run: dict[str, Any] | None = Field(
        description="Workflow run associated with the deployment review"
    )
    since: str = Field(
        description="ISO8601 timestamp indicating the start of the review period"
    )
    environment: str | None = Field(
        default=None, description="Name of the environment being reviewed"
    )
    workflow_job_run: dict[str, Any] | None = Field(
        default=None,
        description="Workflow job run associated with the deployment review",
    )
    reviewers: list[dict[str, Any]] | None = Field(
        default=None,
        description="List of reviewers (users or teams) for the deployment",
    )
    requester: GitHubUser | None = Field(
        default=None,
        alias="requestor",
        description="User who requested the deployment review",
    )
    reviewer: GitHubUser | None = Field(
        default=None, description="The user who reviewed the deployment"
    )
    comment: str | None = Field(
        default=None, description="Comment left by the reviewer"
    )


class DeploymentReviewApproved(DeploymentReviewBase):
    """Payload for github.deployment_review.approved events."""

    _dispatch_topic: ClassVar[str] = "github.deployment_review.approved"


class DeploymentReviewRejected(DeploymentReviewBase):
    """Payload for github.deployment_review.rejected events."""

    _dispatch_topic: ClassVar[str] = "github.deployment_review.rejected"


class DeploymentReviewRequested(DeploymentReviewBase):
    """Payload for github.deployment_review.requested events."""

    _dispatch_topic: ClassVar[str] = "github.deployment_review.requested"


# =============================================================================
# Commit Status Event Payload
# =============================================================================


class GitHubStatusCommit(GitHubModel):
    """Commit object embedded in a commit status event."""

    sha: str = Field(description="The commit SHA")
    url: str = Field(description="API URL for the commit")
    html_url: str = Field(description="Web URL for the commit")
    commit: dict[str, Any] = Field(
        description="Commit data including message and author"
    )


class CommitStatus(GitHubEventPayload):
    """Payload for github.status events (commit status updates).

    Triggered when the status of a Git commit changes. This event has no
    action field — it fires directly when a commit status is created or updated.
    """

    _dispatch_topic: ClassVar[str] = "github.status"

    id: int = Field(description="Unique numeric ID of the status event")
    sha: str = Field(description="The commit SHA the status applies to")
    name: str = Field(description="Repository name")
    target_url: str | None = Field(
        default=None, description="URL associated with the status (e.g. CI build URL)"
    )
    context: str = Field(
        description="Identifier for the status check (e.g. 'ci/circleci')"
    )
    description: str | None = Field(
        default=None, description="Short human-readable description of the status"
    )
    state: str = Field(
        description="State of the status: error, failure, pending, or success"
    )
    commit: GitHubStatusCommit = Field(
        description="The commit object associated with this status"
    )
    branches: list[dict[str, Any]] = Field(
        description="List of branches containing the commit SHA"
    )
    created_at: str = Field(description="ISO8601 timestamp when the status was created")
    updated_at: str = Field(
        description="ISO8601 timestamp when the status was last updated"
    )


# =============================================================================
# WorkflowDispatch Event Payload
# =============================================================================


class WorkflowDispatch(GitHubEventPayload):
    """Payload for github.workflow_dispatch events.

    Triggered when a workflow is manually dispatched via the GitHub UI or
    the API. This event has no action field — it fires directly when the
    workflow dispatch occurs.
    """

    _dispatch_topic: ClassVar[str] = "github.workflow_dispatch"

    workflow: str = Field(description="Path to the workflow file that was dispatched")
    ref: str = Field(
        description="The branch or tag ref that the workflow was dispatched on"
    )
    inputs: dict[str, Any] | None = Field(
        default=None,
        description="Input parameters provided when the workflow was dispatched",
    )


# =============================================================================
# Dependabot Alert Event Payloads
# =============================================================================


class GitHubDependabotDependency(GitHubModel):
    """Dependency information for a Dependabot alert."""

    package: dict[str, Any] = Field(
        description="Package details including ecosystem and name sub-keys"
    )
    manifest_path: str = Field(
        description="Path to the manifest file where the dependency is declared"
    )
    scope: str | None = Field(
        default=None, description="Dependency scope: runtime, development, or null"
    )


class GitHubDependabotAdvisory(GitHubModel):
    """Security advisory associated with a Dependabot alert."""

    ghsa_id: str = Field(description="GitHub Security Advisory identifier")
    cve_id: str | None = Field(
        default=None, description="CVE identifier, if applicable"
    )
    summary: str = Field(description="Short summary of the advisory")
    description: str = Field(description="Full description of the advisory")
    severity: str = Field(
        description="Severity level: low, moderate, high, or critical"
    )
    cvss: dict[str, Any] = Field(description="CVSS score and vector string details")
    cwes: list[dict[str, Any]] = Field(description="CWE weakness identifiers")
    identifiers: list[dict[str, Any]] = Field(
        description="External identifiers for the advisory"
    )
    references: list[dict[str, Any]] = Field(
        description="External references for the advisory"
    )
    published_at: str = Field(
        description="ISO8601 timestamp when the advisory was published"
    )
    updated_at: str = Field(
        description="ISO8601 timestamp when the advisory was last updated"
    )
    withdrawn_at: str | None = Field(
        default=None,
        description="ISO8601 timestamp when the advisory was withdrawn, if applicable",
    )


class GitHubDependabotVulnerability(GitHubModel):
    """Vulnerability details within a security advisory for a specific package."""

    package: dict[str, Any] = Field(
        description="Package details including ecosystem and name"
    )
    severity: str = Field(
        description="Severity level: low, moderate, high, or critical"
    )
    vulnerable_version_range: str = Field(
        description="Version range that is affected by the vulnerability"
    )
    first_patched_version: dict[str, Any] | None = Field(
        default=None,
        description="First version that includes a fix for the vulnerability, or null if unpatched",
    )


class GitHubDependabotAlert(GitHubModel):
    """A Dependabot security alert for a repository."""

    number: int = Field(description="Alert number within the repository")
    state: str = Field(
        description="Alert state: open, dismissed, fixed, or auto_dismissed"
    )
    dependency: GitHubDependabotDependency = Field(
        description="The dependency associated with the alert"
    )
    security_advisory: GitHubDependabotAdvisory = Field(
        description="The security advisory that triggered the alert"
    )
    security_vulnerability: GitHubDependabotVulnerability = Field(
        description="The specific vulnerability within the advisory for this dependency"
    )
    url: str = Field(description="REST API URL for this alert")
    html_url: str = Field(description="Web URL for this alert")
    created_at: str = Field(description="ISO8601 timestamp when the alert was created")
    updated_at: str = Field(
        description="ISO8601 timestamp when the alert was last updated"
    )
    dismissed_at: str | None = Field(
        default=None,
        description="ISO8601 timestamp when the alert was dismissed, if applicable",
    )
    dismissed_by: GitHubUser | None = Field(
        default=None, description="User who dismissed the alert, if applicable"
    )
    dismissed_reason: str | None = Field(
        default=None,
        description="Reason for dismissal: tolerable_risk, no_bandwidth, inaccurate, not_used, or null",
    )
    dismissed_comment: str | None = Field(
        default=None, description="Comment provided when dismissing the alert"
    )
    fixed_at: str | None = Field(
        default=None,
        description="ISO8601 timestamp when the alert was fixed, if applicable",
    )
    auto_dismissed_at: str | None = Field(
        default=None,
        description="ISO8601 timestamp when the alert was auto-dismissed, if applicable",
    )


class DependabotAlertBase(GitHubEventPayload):
    """Base class for dependabot_alert.* events.

    Triggered when a Dependabot alert is:
    - created
    - fixed
    - dismissed
    - reintroduced
    - auto_dismissed
    - auto_reopened
    - reopened

    Use action-specific classes like DependabotAlertCreated, etc.
    """

    action: str = Field(
        description="Event action: created, fixed, dismissed, reintroduced, auto_dismissed, auto_reopened, or reopened"
    )
    alert: GitHubDependabotAlert = Field(
        description="The Dependabot alert that triggered the event"
    )


class DependabotAlertCreated(DependabotAlertBase):
    """Payload for github.dependabot_alert.created events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.created"


class DependabotAlertFixed(DependabotAlertBase):
    """Payload for github.dependabot_alert.fixed events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.fixed"


class DependabotAlertDismissed(DependabotAlertBase):
    """Payload for github.dependabot_alert.dismissed events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.dismissed"


class DependabotAlertReintroduced(DependabotAlertBase):
    """Payload for github.dependabot_alert.reintroduced events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.reintroduced"


class DependabotAlertAutoDismissed(DependabotAlertBase):
    """Payload for github.dependabot_alert.auto_dismissed events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.auto_dismissed"


class DependabotAlertAutoReopened(DependabotAlertBase):
    """Payload for github.dependabot_alert.auto_reopened events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.auto_reopened"


class DependabotAlertReopened(DependabotAlertBase):
    """Payload for github.dependabot_alert.reopened events."""

    _dispatch_topic: ClassVar[str] = "github.dependabot_alert.reopened"


# =============================================================================
# Label Event Payloads
# =============================================================================


class LabelBase(GitHubEventPayload):
    """Base class for label.* events.

    Triggered when a label is:
    - created
    - edited
    - deleted

    Use action-specific classes like LabelCreated, etc.
    """

    action: str = Field(description="Event action: created, edited, or deleted")
    label: GitHubLabel = Field(description="The label that triggered the event")


class LabelCreated(LabelBase):
    """Payload for github.label.created events."""

    _dispatch_topic: ClassVar[str] = "github.label.created"


class LabelEdited(LabelBase):
    """Payload for github.label.edited events."""

    _dispatch_topic: ClassVar[str] = "github.label.edited"
    changes: GitHubChanges | None = Field(
        default=None, description="Changes made to the label"
    )


class LabelDeleted(LabelBase):
    """Payload for github.label.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.label.deleted"


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
# Watch Event Payloads
# =============================================================================


class WatchBase(GitHubEventPayload):
    """Base class for watch.* events.

    Triggered when a repository is starred (watch).
    GitHub only fires watch.started — watch.deleted is not sent.
    """

    action: str = Field(description="Event action: started")


class WatchStarted(WatchBase):
    """Payload for github.watch.started events."""

    _dispatch_topic: ClassVar[str] = "github.watch.started"


# =============================================================================
# Public Event Payload (no action)
# =============================================================================


class Public(GitHubEventPayload):
    """Payload for github.public events.

    Triggered when a private repository is made public.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.public"


# =============================================================================
# Gollum Event Payload (Wiki, no action)
# =============================================================================


class Gollum(GitHubEventPayload):
    """Payload for github.gollum events (wiki page changes).

    Triggered when a wiki page is created or updated.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.gollum"

    pages: list[dict[str, Any]] = Field(
        description="Wiki pages that were created or updated"
    )


# =============================================================================
# Ping Event Payload (no action)
# =============================================================================


class Ping(GitHubEventPayload):
    """Payload for github.ping events.

    Sent by GitHub when a webhook is first created or re-delivered manually.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.ping"

    zen: str = Field(description="A random zen message from GitHub")
    hook_id: int = Field(description="ID of the webhook that triggered the ping")
    hook: dict[str, Any] = Field(description="Webhook configuration object")


# =============================================================================
# Repository Import Event Payload (no action)
# =============================================================================


class RepositoryImport(GitHubEventPayload):
    """Payload for github.repository_import events.

    Triggered when a repository import is started, cancelled, or completed.
    This event has no action field — the status field indicates state.
    """

    _dispatch_topic: ClassVar[str] = "github.repository_import"

    status: str = Field(description="Import status: success, cancelled, or failure")


# =============================================================================
# Page Build Event Payload (no action)
# =============================================================================


class PageBuild(GitHubEventPayload):
    """Payload for github.page_build events.

    Triggered on every push to a GitHub Pages enabled branch.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.page_build"

    id: int = Field(description="Unique identifier of the page build")
    build: dict[str, Any] = Field(description="Page build details")


# =============================================================================
# GitHub App Authorization Event Payload
# =============================================================================


class GitHubAppAuthorizationBase(GitHubEventPayload):
    """Base class for github_app_authorization.* events.

    Triggered when a user revokes authorization of a GitHub App.
    """

    action: str = Field(description="Event action: revoked")


class GitHubAppAuthorizationRevoked(GitHubAppAuthorizationBase):
    """Payload for github.github_app_authorization.revoked events."""

    _dispatch_topic: ClassVar[str] = "github.github_app_authorization.revoked"


# =============================================================================
# Team Add Event Payload (no action)
# =============================================================================


class TeamAdd(GitHubEventPayload):
    """Payload for github.team_add events.

    Triggered when a repository is added to a team.
    This event has no action field.
    """

    _dispatch_topic: ClassVar[str] = "github.team_add"

    team: dict[str, Any] = Field(description="Team that was given access")


# =============================================================================
# Branch Protection Configuration Event Payloads
# =============================================================================


class BranchProtectionConfigurationBase(GitHubEventPayload):
    """Base class for branch_protection_configuration.* events.

    Triggered when branch protection is enabled or disabled for a repository.
    """

    action: str = Field(description="Event action: enabled or disabled")


class BranchProtectionConfigurationEnabled(BranchProtectionConfigurationBase):
    """Payload for github.branch_protection_configuration.enabled events."""

    _dispatch_topic: ClassVar[str] = "github.branch_protection_configuration.enabled"


class BranchProtectionConfigurationDisabled(BranchProtectionConfigurationBase):
    """Payload for github.branch_protection_configuration.disabled events."""

    _dispatch_topic: ClassVar[str] = "github.branch_protection_configuration.disabled"


# =============================================================================
# Branch Protection Rule Event Payloads
# =============================================================================


class BranchProtectionRuleBase(GitHubEventPayload):
    """Base class for branch_protection_rule.* events.

    Triggered when a branch protection rule is created, deleted, or edited.
    """

    action: str = Field(description="Event action: created, deleted, or edited")
    rule: dict[str, Any] = Field(description="Branch protection rule data")


class BranchProtectionRuleCreated(BranchProtectionRuleBase):
    """Payload for github.branch_protection_rule.created events."""

    _dispatch_topic: ClassVar[str] = "github.branch_protection_rule.created"


class BranchProtectionRuleDeleted(BranchProtectionRuleBase):
    """Payload for github.branch_protection_rule.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.branch_protection_rule.deleted"


class BranchProtectionRuleEdited(BranchProtectionRuleBase):
    """Payload for github.branch_protection_rule.edited events."""

    _dispatch_topic: ClassVar[str] = "github.branch_protection_rule.edited"
    changes: GitHubChanges | None = Field(
        default=None, description="Changes made to the rule"
    )


# =============================================================================
# Commit Comment Event Payloads
# =============================================================================


class CommitCommentBase(GitHubEventPayload):
    """Base class for commit_comment.* events.

    Triggered when a comment is created on a commit.
    """

    action: str = Field(description="Event action: created")
    comment: dict[str, Any] = Field(description="Comment on the commit")


class CommitCommentCreated(CommitCommentBase):
    """Payload for github.commit_comment.created events."""

    _dispatch_topic: ClassVar[str] = "github.commit_comment.created"


# =============================================================================
# Deploy Key Event Payloads
# =============================================================================


class DeployKeyBase(GitHubEventPayload):
    """Base class for deploy_key.* events.

    Triggered when a deploy key is created or deleted.
    """

    action: str = Field(description="Event action: created or deleted")
    key: dict[str, Any] = Field(description="Deploy key data")


class DeployKeyCreated(DeployKeyBase):
    """Payload for github.deploy_key.created events."""

    _dispatch_topic: ClassVar[str] = "github.deploy_key.created"


class DeployKeyDeleted(DeployKeyBase):
    """Payload for github.deploy_key.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.deploy_key.deleted"


# =============================================================================
# Member Event Payloads
# =============================================================================


class MemberBase(GitHubEventPayload):
    """Base class for member.* events.

    Triggered when a collaborator is added, removed, or when their
    permissions are changed on a repository.
    """

    action: str = Field(description="Event action: added, removed, or edited")
    member: GitHubUser = Field(description="User whose membership changed")


class MemberAdded(MemberBase):
    """Payload for github.member.added events."""

    _dispatch_topic: ClassVar[str] = "github.member.added"
    changes: GitHubChanges | None = Field(
        default=None, description="Changes to the member's permissions"
    )


class MemberEdited(MemberBase):
    """Payload for github.member.edited events."""

    _dispatch_topic: ClassVar[str] = "github.member.edited"
    changes: GitHubChanges = Field(description="Changes to the member's permissions")


class MemberRemoved(MemberBase):
    """Payload for github.member.removed events."""

    _dispatch_topic: ClassVar[str] = "github.member.removed"


# =============================================================================
# Membership Event Payloads
# =============================================================================


class MembershipBase(GitHubEventPayload):
    """Base class for membership.* events.

    Triggered when a user is added or removed from a team.
    """

    action: str = Field(description="Event action: added or removed")
    scope: str = Field(description="Scope of the membership: team")
    member: GitHubUser = Field(description="User whose team membership changed")
    team: dict[str, Any] = Field(
        description="Team the user was added to or removed from"
    )


class MembershipAdded(MembershipBase):
    """Payload for github.membership.added events."""

    _dispatch_topic: ClassVar[str] = "github.membership.added"


class MembershipRemoved(MembershipBase):
    """Payload for github.membership.removed events."""

    _dispatch_topic: ClassVar[str] = "github.membership.removed"


# =============================================================================
# Merge Group Event Payloads
# =============================================================================


class MergeGroupBase(GitHubEventPayload):
    """Base class for merge_group.* events.

    Triggered when a merge group is created or destroyed, or when its
    checks are requested.
    """

    action: str = Field(description="Event action: checks_requested or destroyed")
    merge_group: dict[str, Any] = Field(description="Merge group data")


class MergeGroupChecksRequested(MergeGroupBase):
    """Payload for github.merge_group.checks_requested events."""

    _dispatch_topic: ClassVar[str] = "github.merge_group.checks_requested"


class MergeGroupDestroyed(MergeGroupBase):
    """Payload for github.merge_group.destroyed events."""

    _dispatch_topic: ClassVar[str] = "github.merge_group.destroyed"
    reason: str = Field(
        description="Reason the merge group was destroyed: merged, invalidated, or dequeued"
    )


# =============================================================================
# Meta Event Payloads (webhook deletion)
# =============================================================================


class MetaBase(GitHubEventPayload):
    """Base class for meta.* events.

    Triggered when a webhook is deleted.
    """

    action: str = Field(description="Event action: deleted")
    hook_id: int = Field(description="ID of the modified webhook")
    hook: dict[str, Any] = Field(description="Modified webhook data")


class MetaDeleted(MetaBase):
    """Payload for github.meta.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.meta.deleted"


# =============================================================================
# Milestone Event Payloads
# =============================================================================


class MilestoneEventBase(GitHubEventPayload):
    """Base class for milestone.* events.

    Triggered when a milestone is created, closed, deleted, edited, or opened.
    Note: This is for milestone events, not the GitHubMilestone model used in
    issue/PR payloads.
    """

    action: str = Field(
        description="Event action: created, closed, deleted, edited, or opened"
    )
    milestone: GitHubMilestone = Field(description="Milestone that triggered the event")


class MilestoneClosed(MilestoneEventBase):
    """Payload for github.milestone.closed events."""

    _dispatch_topic: ClassVar[str] = "github.milestone.closed"


class MilestoneCreated(MilestoneEventBase):
    """Payload for github.milestone.created events."""

    _dispatch_topic: ClassVar[str] = "github.milestone.created"


class MilestoneDeleted(MilestoneEventBase):
    """Payload for github.milestone.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.milestone.deleted"


class MilestoneEdited(MilestoneEventBase):
    """Payload for github.milestone.edited events."""

    _dispatch_topic: ClassVar[str] = "github.milestone.edited"
    changes: GitHubChanges = Field(description="Changes made to the milestone")


class MilestoneOpened(MilestoneEventBase):
    """Payload for github.milestone.opened events."""

    _dispatch_topic: ClassVar[str] = "github.milestone.opened"


# =============================================================================
# Org Block Event Payloads
# =============================================================================


class OrgBlockBase(GitHubEventPayload):
    """Base class for org_block.* events.

    Triggered when an organization blocks or unblocks a user.
    """

    action: str = Field(description="Event action: blocked or unblocked")
    blocked_user: GitHubUser = Field(description="User who was blocked or unblocked")


class OrgBlockBlocked(OrgBlockBase):
    """Payload for github.org_block.blocked events."""

    _dispatch_topic: ClassVar[str] = "github.org_block.blocked"


class OrgBlockUnblocked(OrgBlockBase):
    """Payload for github.org_block.unblocked events."""

    _dispatch_topic: ClassVar[str] = "github.org_block.unblocked"


# =============================================================================
# Repository Event Payloads
# =============================================================================


class RepositoryEventBase(GitHubEventPayload):
    """Base class for repository.* events.

    Triggered when a repository is created, deleted, archived, unarchived,
    publicized, privatized, edited, renamed, or transferred.
    """

    action: str = Field(
        description="Event action: archived, created, deleted, edited, "
        "privatized, publicized, renamed, transferred, or unarchived"
    )


class RepositoryArchived(RepositoryEventBase):
    """Payload for github.repository.archived events."""

    _dispatch_topic: ClassVar[str] = "github.repository.archived"


class RepositoryCreated(RepositoryEventBase):
    """Payload for github.repository.created events."""

    _dispatch_topic: ClassVar[str] = "github.repository.created"


class RepositoryDeleted(RepositoryEventBase):
    """Payload for github.repository.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.repository.deleted"


class RepositoryEdited(RepositoryEventBase):
    """Payload for github.repository.edited events."""

    _dispatch_topic: ClassVar[str] = "github.repository.edited"
    changes: GitHubChanges = Field(description="Changes made to the repository")


class RepositoryPrivatized(RepositoryEventBase):
    """Payload for github.repository.privatized events."""

    _dispatch_topic: ClassVar[str] = "github.repository.privatized"


class RepositoryPublicized(RepositoryEventBase):
    """Payload for github.repository.publicized events."""

    _dispatch_topic: ClassVar[str] = "github.repository.publicized"


class RepositoryRenamed(RepositoryEventBase):
    """Payload for github.repository.renamed events."""

    _dispatch_topic: ClassVar[str] = "github.repository.renamed"
    changes: GitHubChanges = Field(description="Changes made to the repository name")


class RepositoryTransferred(RepositoryEventBase):
    """Payload for github.repository.transferred events."""

    _dispatch_topic: ClassVar[str] = "github.repository.transferred"
    changes: GitHubChanges = Field(description="Changes from the transfer")


class RepositoryUnarchived(RepositoryEventBase):
    """Payload for github.repository.unarchived events."""

    _dispatch_topic: ClassVar[str] = "github.repository.unarchived"


# =============================================================================
# Repository Dispatch Event Payload
# =============================================================================


class RepositoryDispatch(GitHubEventPayload):
    """Payload for github.repository_dispatch events.

    Triggered when a client sends a POST request to the repository dispatch
    endpoint. The action is user-defined.
    """

    _dispatch_topic: ClassVar[str] = "github.repository_dispatch"

    action: str = Field(description="User-defined event action")
    branch: str = Field(description="Branch the dispatch was triggered on")
    client_payload: dict[str, Any] = Field(
        description="User-defined payload sent with the dispatch"
    )
    installation: GitHubInstallation = Field(
        description="GitHub App installation that triggered the dispatch"
    )


# =============================================================================
# Discussion Event Payloads
# =============================================================================


class GitHubDiscussion(GitHubModel):
    """GitHub Discussions post."""

    id: int = Field(description="Unique numeric ID")
    node_id: str = Field(description="GraphQL node ID")
    number: int = Field(description="Discussion number in the repository")
    title: str = Field(description="Discussion title")
    body: str | None = Field(default=None, description="Discussion body text")
    state: str = Field(description="State: open or closed")
    category: dict[str, Any] = Field(description="Discussion category")
    author_association: str = Field(
        description="Author's association with the repository"
    )
    html_url: str = Field(description="Web URL for the discussion")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str = Field(description="ISO8601 last updated timestamp")
    answered_at: str | None = Field(
        default=None, description="ISO8601 timestamp when marked as answered"
    )
    answer_html_url: str | None = Field(
        default=None, description="URL of the answer comment"
    )


class DiscussionBase(GitHubEventPayload):
    """Base class for discussion.* events."""

    action: str = Field(
        description="Event action: answered, category_changed, created, deleted, "
        "edited, labeled, locked, pinned, transferred, unanswered, "
        "unlabeled, unlocked, or unpinned"
    )
    discussion: GitHubDiscussion = Field(
        description="Discussion that triggered the event"
    )


class DiscussionAnswered(DiscussionBase):
    """Payload for github.discussion.answered events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.answered"
    answer: dict[str, Any] = Field(description="Comment that was marked as the answer")


class DiscussionCategoryChanged(DiscussionBase):
    """Payload for github.discussion.category_changed events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.category_changed"
    changes: GitHubChanges = Field(description="Changes to the discussion category")


class DiscussionCreated(DiscussionBase):
    """Payload for github.discussion.created events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.created"


class DiscussionDeleted(DiscussionBase):
    """Payload for github.discussion.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.deleted"


class DiscussionEdited(DiscussionBase):
    """Payload for github.discussion.edited events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.edited"
    changes: GitHubChanges | None = Field(
        default=None, description="Changes made to the discussion"
    )


class DiscussionLabeled(DiscussionBase):
    """Payload for github.discussion.labeled events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.labeled"
    label: GitHubLabel = Field(description="Label added to the discussion")


class DiscussionLocked(DiscussionBase):
    """Payload for github.discussion.locked events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.locked"


class DiscussionPinned(DiscussionBase):
    """Payload for github.discussion.pinned events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.pinned"


class DiscussionTransferred(DiscussionBase):
    """Payload for github.discussion.transferred events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.transferred"
    changes: GitHubChanges = Field(description="Changes from the transfer")


class DiscussionUnanswered(DiscussionBase):
    """Payload for github.discussion.unanswered events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.unanswered"
    old_answer: dict[str, Any] = Field(
        description="Comment that was previously the answer"
    )


class DiscussionUnlabeled(DiscussionBase):
    """Payload for github.discussion.unlabeled events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.unlabeled"
    label: GitHubLabel = Field(description="Label removed from the discussion")


class DiscussionUnlocked(DiscussionBase):
    """Payload for github.discussion.unlocked events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.unlocked"


class DiscussionUnpinned(DiscussionBase):
    """Payload for github.discussion.unpinned events."""

    _dispatch_topic: ClassVar[str] = "github.discussion.unpinned"


# =============================================================================
# Discussion Comment Event Payloads
# =============================================================================


class DiscussionCommentBase(GitHubEventPayload):
    """Base class for discussion_comment.* events."""

    action: str = Field(description="Event action: created, deleted, or edited")
    comment: dict[str, Any] = Field(description="Comment on the discussion")
    discussion: GitHubDiscussion = Field(
        description="Discussion the comment belongs to"
    )
    installation: dict[str, Any] = Field(
        description="GitHub App installation (required for discussion_comment events)"
    )


class DiscussionCommentCreated(DiscussionCommentBase):
    """Payload for github.discussion_comment.created events."""

    _dispatch_topic: ClassVar[str] = "github.discussion_comment.created"


class DiscussionCommentDeleted(DiscussionCommentBase):
    """Payload for github.discussion_comment.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.discussion_comment.deleted"


class DiscussionCommentEdited(DiscussionCommentBase):
    """Payload for github.discussion_comment.edited events."""

    _dispatch_topic: ClassVar[str] = "github.discussion_comment.edited"
    changes: GitHubChanges = Field(description="Changes made to the comment")


# =============================================================================
# Team Event Payloads
# =============================================================================


class GitHubTeam(GitHubModel):
    """GitHub team."""

    id: int = Field(description="Unique numeric ID")
    node_id: str = Field(description="GraphQL node ID")
    name: str = Field(description="Team name")
    slug: str = Field(description="URL-friendly team name")
    description: str | None = Field(default=None, description="Team description")
    privacy: str = Field(description="Team privacy: closed or secret")
    permission: str = Field(description="Default permission: pull, push, or admin")
    url: str = Field(description="API URL")
    html_url: str = Field(description="Web URL")
    members_url: str = Field(description="API URL for members")
    repositories_url: str = Field(description="API URL for repositories")


class TeamBase(GitHubEventPayload):
    """Base class for team.* events.

    Triggered when a team is created, deleted, edited, or when
    repositories are added or removed from the team.
    """

    action: str = Field(
        description="Event action: added_to_repository, created, deleted, "
        "edited, or removed_from_repository"
    )
    team: GitHubTeam = Field(description="Team that triggered the event")


class TeamAddedToRepository(TeamBase):
    """Payload for github.team.added_to_repository events."""

    _dispatch_topic: ClassVar[str] = "github.team.added_to_repository"


class TeamCreated(TeamBase):
    """Payload for github.team.created events."""

    _dispatch_topic: ClassVar[str] = "github.team.created"


class TeamDeleted(TeamBase):
    """Payload for github.team.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.team.deleted"


class TeamEdited(TeamBase):
    """Payload for github.team.edited events."""

    _dispatch_topic: ClassVar[str] = "github.team.edited"
    changes: GitHubChanges = Field(description="Changes made to the team")


class TeamRemovedFromRepository(TeamBase):
    """Payload for github.team.removed_from_repository events."""

    _dispatch_topic: ClassVar[str] = "github.team.removed_from_repository"


# =============================================================================
# Organization Event Payloads
# =============================================================================


class OrganizationBase(GitHubEventPayload):
    """Base class for organization.* events.

    Triggered when membership in an organization changes.
    """

    action: str = Field(
        description="Event action: deleted, member_added, member_invited, "
        "member_removed, or renamed"
    )
    membership: dict[str, Any] | None = Field(
        default=None,
        description="Membership details (present for member_added/removed events)",
    )
    invitation: dict[str, Any] | None = Field(
        default=None,
        description="Invitation details (present for member_invited events)",
    )


class OrganizationDeleted(OrganizationBase):
    """Payload for github.organization.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.organization.deleted"


class OrganizationMemberAdded(OrganizationBase):
    """Payload for github.organization.member_added events."""

    _dispatch_topic: ClassVar[str] = "github.organization.member_added"


class OrganizationMemberInvited(OrganizationBase):
    """Payload for github.organization.member_invited events."""

    _dispatch_topic: ClassVar[str] = "github.organization.member_invited"
    user: GitHubUser = Field(description="User who was invited")


class OrganizationMemberRemoved(OrganizationBase):
    """Payload for github.organization.member_removed events."""

    _dispatch_topic: ClassVar[str] = "github.organization.member_removed"


class OrganizationRenamed(OrganizationBase):
    """Payload for github.organization.renamed events."""

    _dispatch_topic: ClassVar[str] = "github.organization.renamed"
    changes: GitHubChanges = Field(description="Changes to the organization (old name)")


# =============================================================================
# Code Scanning Alert Event Payloads
# =============================================================================


class GitHubCodeScanningAlert(GitHubModel):
    """A code scanning alert."""

    number: int = Field(description="Alert number within the repository")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    url: str = Field(description="API URL for this alert")
    html_url: str = Field(description="Web URL for this alert")
    state: str = Field(description="Alert state: open, dismissed, or fixed")
    dismissed_by: GitHubUser | None = Field(
        default=None, description="User who dismissed the alert"
    )
    dismissed_at: str | None = Field(
        default=None, description="ISO8601 dismiss timestamp"
    )
    dismissed_reason: str | None = Field(
        default=None, description="Reason for dismissal"
    )
    rule: dict[str, Any] = Field(description="Rule that triggered the alert")
    tool: dict[str, Any] = Field(description="Tool that generated the alert")
    most_recent_instance: dict[str, Any] = Field(
        description="Most recent instance of this alert"
    )


class CodeScanningAlertBase(GitHubEventPayload):
    """Base class for code_scanning_alert.* events."""

    action: str = Field(
        description="Event action: appeared_in_branch, closed_by_user, created, "
        "fixed, reopened, or reopened_by_user"
    )
    alert: GitHubCodeScanningAlert = Field(description="Code scanning alert data")
    ref: str = Field(description="Git ref the alert applies to")
    commit_oid: str = Field(description="Commit OID the alert applies to")


class CodeScanningAlertAppearedInBranch(CodeScanningAlertBase):
    """Payload for github.code_scanning_alert.appeared_in_branch events."""

    _dispatch_topic: ClassVar[str] = "github.code_scanning_alert.appeared_in_branch"


class CodeScanningAlertClosedByUser(CodeScanningAlertBase):
    """Payload for github.code_scanning_alert.closed_by_user events."""

    _dispatch_topic: ClassVar[str] = "github.code_scanning_alert.closed_by_user"


class CodeScanningAlertCreated(CodeScanningAlertBase):
    """Payload for github.code_scanning_alert.created events."""

    _dispatch_topic: ClassVar[str] = "github.code_scanning_alert.created"


class CodeScanningAlertFixed(CodeScanningAlertBase):
    """Payload for github.code_scanning_alert.fixed events."""

    _dispatch_topic: ClassVar[str] = "github.code_scanning_alert.fixed"


class CodeScanningAlertReopened(CodeScanningAlertBase):
    """Payload for github.code_scanning_alert.reopened events."""

    _dispatch_topic: ClassVar[str] = "github.code_scanning_alert.reopened"


class CodeScanningAlertReopenedByUser(CodeScanningAlertBase):
    """Payload for github.code_scanning_alert.reopened_by_user events."""

    _dispatch_topic: ClassVar[str] = "github.code_scanning_alert.reopened_by_user"


# =============================================================================
# Secret Scanning Alert Event Payloads
# =============================================================================


class GitHubSecretScanningAlert(GitHubModel):
    """A secret scanning alert."""

    number: int = Field(description="Alert number within the repository")
    created_at: str = Field(description="ISO8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO8601 update timestamp")
    url: str = Field(description="API URL for this alert")
    html_url: str = Field(description="Web URL for this alert")
    locations_url: str = Field(description="API URL to list alert locations")
    state: str = Field(description="Alert state: open or resolved")
    resolution: str | None = Field(
        default=None,
        description="Reason for resolution: false_positive, wont_fix, "
        "revoked, used_in_tests, or pattern_deleted",
    )
    resolved_at: str | None = Field(
        default=None, description="ISO8601 resolution timestamp"
    )
    resolved_by: GitHubUser | None = Field(
        default=None, description="User who resolved the alert"
    )
    secret_type: str = Field(description="Type of secret detected")
    secret_type_display_name: str | None = Field(
        default=None, description="Human-readable name of the secret type"
    )
    secret: str | None = Field(default=None, description="The detected secret value")
    push_protection_bypassed: bool | None = Field(
        default=None, description="Whether push protection was bypassed for this secret"
    )
    push_protection_bypassed_by: GitHubUser | None = Field(
        default=None, description="User who bypassed push protection"
    )
    push_protection_bypassed_at: str | None = Field(
        default=None, description="ISO8601 timestamp when push protection was bypassed"
    )


class SecretScanningAlertBase(GitHubEventPayload):
    """Base class for secret_scanning_alert.* events."""

    action: str = Field(
        description="Event action: created, reopened, resolved, or revoked"
    )
    alert: GitHubSecretScanningAlert = Field(description="Secret scanning alert data")


class SecretScanningAlertCreated(SecretScanningAlertBase):
    """Payload for github.secret_scanning_alert.created events."""

    _dispatch_topic: ClassVar[str] = "github.secret_scanning_alert.created"


class SecretScanningAlertReopened(SecretScanningAlertBase):
    """Payload for github.secret_scanning_alert.reopened events."""

    _dispatch_topic: ClassVar[str] = "github.secret_scanning_alert.reopened"


class SecretScanningAlertResolved(SecretScanningAlertBase):
    """Payload for github.secret_scanning_alert.resolved events."""

    _dispatch_topic: ClassVar[str] = "github.secret_scanning_alert.resolved"


class SecretScanningAlertRevoked(SecretScanningAlertBase):
    """Payload for github.secret_scanning_alert.revoked events."""

    _dispatch_topic: ClassVar[str] = "github.secret_scanning_alert.revoked"


# =============================================================================
# Secret Scanning Alert Location Event Payloads
# =============================================================================


class SecretScanningAlertLocationBase(GitHubEventPayload):
    """Base class for secret_scanning_alert_location.* events."""

    action: str = Field(description="Event action: created")
    alert: GitHubSecretScanningAlert = Field(
        description="The alert the location belongs to"
    )
    location: dict[str, Any] = Field(description="Location data for the secret")


class SecretScanningAlertLocationCreated(SecretScanningAlertLocationBase):
    """Payload for github.secret_scanning_alert_location.created events."""

    _dispatch_topic: ClassVar[str] = "github.secret_scanning_alert_location.created"


# =============================================================================
# Repository Vulnerability Alert Event Payloads
# =============================================================================


class RepositoryVulnerabilityAlertBase(GitHubEventPayload):
    """Base class for repository_vulnerability_alert.* events.

    Triggered when a vulnerability alert is created, dismissed, reopened,
    or resolved. Note: action values use non-standard names (create/dismiss
    instead of created/dismissed).
    """

    action: str = Field(description="Event action: create, dismiss, reopen, or resolve")
    alert: dict[str, Any] = Field(description="Vulnerability alert data")


class RepositoryVulnerabilityAlertCreate(RepositoryVulnerabilityAlertBase):
    """Payload for github.repository_vulnerability_alert.create events."""

    _dispatch_topic: ClassVar[str] = "github.repository_vulnerability_alert.create"


class RepositoryVulnerabilityAlertDismiss(RepositoryVulnerabilityAlertBase):
    """Payload for github.repository_vulnerability_alert.dismiss events."""

    _dispatch_topic: ClassVar[str] = "github.repository_vulnerability_alert.dismiss"


class RepositoryVulnerabilityAlertReopen(RepositoryVulnerabilityAlertBase):
    """Payload for github.repository_vulnerability_alert.reopen events."""

    _dispatch_topic: ClassVar[str] = "github.repository_vulnerability_alert.reopen"


class RepositoryVulnerabilityAlertResolve(RepositoryVulnerabilityAlertBase):
    """Payload for github.repository_vulnerability_alert.resolve events."""

    _dispatch_topic: ClassVar[str] = "github.repository_vulnerability_alert.resolve"


# =============================================================================
# Security Advisory Event Payloads
# =============================================================================


class SecurityAdvisoryBase(GitHubEventPayload):
    """Base class for security_advisory.* events.

    Triggered when a GitHub Security Advisory is published, updated,
    performed, or withdrawn.
    """

    action: str = Field(
        description="Event action: performed, published, updated, or withdrawn"
    )
    security_advisory: dict[str, Any] = Field(
        description="Security advisory data including CVE info, severity, and affected packages"
    )


class SecurityAdvisoryPerformed(SecurityAdvisoryBase):
    """Payload for github.security_advisory.performed events."""

    _dispatch_topic: ClassVar[str] = "github.security_advisory.performed"


class SecurityAdvisoryPublished(SecurityAdvisoryBase):
    """Payload for github.security_advisory.published events."""

    _dispatch_topic: ClassVar[str] = "github.security_advisory.published"


class SecurityAdvisoryUpdated(SecurityAdvisoryBase):
    """Payload for github.security_advisory.updated events."""

    _dispatch_topic: ClassVar[str] = "github.security_advisory.updated"


class SecurityAdvisoryWithdrawn(SecurityAdvisoryBase):
    """Payload for github.security_advisory.withdrawn events."""

    _dispatch_topic: ClassVar[str] = "github.security_advisory.withdrawn"


# =============================================================================
# Marketplace Purchase Event Payloads
# =============================================================================


class MarketplacePurchaseBase(GitHubEventPayload):
    """Base class for marketplace_purchase.* events.

    Triggered when a GitHub Marketplace purchase is made or changed.
    """

    action: str = Field(
        description="Event action: cancelled, changed, pending_change, "
        "pending_change_cancelled, or purchased"
    )
    effective_date: str = Field(description="ISO8601 date when the change takes effect")
    marketplace_purchase: dict[str, Any] = Field(
        description="Marketplace purchase details including plan and billing cycle"
    )
    previous_marketplace_purchase: dict[str, Any] | None = Field(
        default=None,
        description="Previous marketplace purchase (present for change events)",
    )


class MarketplacePurchaseCancelled(MarketplacePurchaseBase):
    """Payload for github.marketplace_purchase.cancelled events."""

    _dispatch_topic: ClassVar[str] = "github.marketplace_purchase.cancelled"


class MarketplacePurchaseChanged(MarketplacePurchaseBase):
    """Payload for github.marketplace_purchase.changed events."""

    _dispatch_topic: ClassVar[str] = "github.marketplace_purchase.changed"


class MarketplacePurchasePendingChange(MarketplacePurchaseBase):
    """Payload for github.marketplace_purchase.pending_change events."""

    _dispatch_topic: ClassVar[str] = "github.marketplace_purchase.pending_change"


class MarketplacePurchasePendingChangeCancelled(MarketplacePurchaseBase):
    """Payload for github.marketplace_purchase.pending_change_cancelled events."""

    _dispatch_topic: ClassVar[str] = (
        "github.marketplace_purchase.pending_change_cancelled"
    )


class MarketplacePurchasePurchased(MarketplacePurchaseBase):
    """Payload for github.marketplace_purchase.purchased events."""

    _dispatch_topic: ClassVar[str] = "github.marketplace_purchase.purchased"


# =============================================================================
# Package Event Payloads
# =============================================================================


class PackageBase(GitHubEventPayload):
    """Base class for package.* events.

    Triggered when a GitHub Packages package is published or updated.
    """

    action: str = Field(description="Event action: published or updated")
    package: dict[str, Any] = Field(description="Package data")


class PackagePublished(PackageBase):
    """Payload for github.package.published events."""

    _dispatch_topic: ClassVar[str] = "github.package.published"


class PackageUpdated(PackageBase):
    """Payload for github.package.updated events."""

    _dispatch_topic: ClassVar[str] = "github.package.updated"


# =============================================================================
# Registry Package Event Payloads
# =============================================================================


class RegistryPackageBase(GitHubEventPayload):
    """Base class for registry_package.* events.

    Triggered when a package is published or updated in the GitHub Container
    Registry (ghcr.io).
    """

    action: str = Field(description="Event action: published or updated")
    registry_package: dict[str, Any] = Field(description="Registry package data")


class RegistryPackagePublished(RegistryPackageBase):
    """Payload for github.registry_package.published events."""

    _dispatch_topic: ClassVar[str] = "github.registry_package.published"


class RegistryPackageUpdated(RegistryPackageBase):
    """Payload for github.registry_package.updated events."""

    _dispatch_topic: ClassVar[str] = "github.registry_package.updated"


# =============================================================================
# Project Event Payloads (classic Projects)
# =============================================================================


class ProjectBase(GitHubEventPayload):
    """Base class for project.* events (GitHub Classic Projects).

    Triggered when a project board is created, closed, deleted, edited,
    or reopened.
    """

    action: str = Field(
        description="Event action: closed, created, deleted, edited, or reopened"
    )
    project: dict[str, Any] = Field(description="Project board data")


class ProjectClosed(ProjectBase):
    """Payload for github.project.closed events."""

    _dispatch_topic: ClassVar[str] = "github.project.closed"


class ProjectCreated(ProjectBase):
    """Payload for github.project.created events."""

    _dispatch_topic: ClassVar[str] = "github.project.created"


class ProjectDeleted(ProjectBase):
    """Payload for github.project.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.project.deleted"


class ProjectEdited(ProjectBase):
    """Payload for github.project.edited events."""

    _dispatch_topic: ClassVar[str] = "github.project.edited"
    changes: GitHubChanges | None = Field(
        default=None, description="Changes made to the project"
    )


class ProjectReopened(ProjectBase):
    """Payload for github.project.reopened events."""

    _dispatch_topic: ClassVar[str] = "github.project.reopened"


# =============================================================================
# Project Card Event Payloads (classic Projects)
# =============================================================================


class ProjectCardBase(GitHubEventPayload):
    """Base class for project_card.* events (GitHub Classic Projects).

    Triggered when a project card is converted, created, deleted, edited,
    or moved.
    """

    action: str = Field(
        description="Event action: converted, created, deleted, edited, or moved"
    )
    project_card: dict[str, Any] = Field(description="Project card data")


class ProjectCardConverted(ProjectCardBase):
    """Payload for github.project_card.converted events."""

    _dispatch_topic: ClassVar[str] = "github.project_card.converted"
    changes: dict[str, Any] = Field(
        description="Changes made to the project card (previous note value)"
    )


class ProjectCardCreated(ProjectCardBase):
    """Payload for github.project_card.created events."""

    _dispatch_topic: ClassVar[str] = "github.project_card.created"


class ProjectCardDeleted(ProjectCardBase):
    """Payload for github.project_card.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.project_card.deleted"


class ProjectCardEdited(ProjectCardBase):
    """Payload for github.project_card.edited events."""

    _dispatch_topic: ClassVar[str] = "github.project_card.edited"
    changes: dict[str, Any] = Field(description="Changes made to the project card")


class ProjectCardMoved(ProjectCardBase):
    """Payload for github.project_card.moved events."""

    _dispatch_topic: ClassVar[str] = "github.project_card.moved"


# =============================================================================
# Project Column Event Payloads (classic Projects)
# =============================================================================


class ProjectColumnBase(GitHubEventPayload):
    """Base class for project_column.* events (GitHub Classic Projects).

    Triggered when a project column is created, deleted, edited, or moved.
    """

    action: str = Field(description="Event action: created, deleted, edited, or moved")
    project_column: dict[str, Any] = Field(description="Project column data")


class ProjectColumnCreated(ProjectColumnBase):
    """Payload for github.project_column.created events."""

    _dispatch_topic: ClassVar[str] = "github.project_column.created"


class ProjectColumnDeleted(ProjectColumnBase):
    """Payload for github.project_column.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.project_column.deleted"


class ProjectColumnEdited(ProjectColumnBase):
    """Payload for github.project_column.edited events."""

    _dispatch_topic: ClassVar[str] = "github.project_column.edited"
    changes: dict[str, Any] = Field(description="Changes made to the project column")


class ProjectColumnMoved(ProjectColumnBase):
    """Payload for github.project_column.moved events."""

    _dispatch_topic: ClassVar[str] = "github.project_column.moved"


# =============================================================================
# Projects V2 Item Event Payloads (new Projects)
# =============================================================================


class ProjectsV2ItemBase(GitHubEventPayload):
    """Base class for projects_v2_item.* events (GitHub Projects, new version).

    Triggered when a project item is archived, converted, created, deleted,
    edited, reordered, or restored.
    """

    action: str = Field(
        description="Event action: archived, converted, created, deleted, "
        "edited, reordered, or restored"
    )
    projects_v2_item: dict[str, Any] = Field(description="Projects V2 item data")


class ProjectsV2ItemArchived(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.archived events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.archived"
    changes: dict[str, Any] = Field(description="Changes made to the project item")


class ProjectsV2ItemConverted(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.converted events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.converted"
    changes: dict[str, Any] = Field(description="Changes made to the project item")


class ProjectsV2ItemCreated(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.created events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.created"


class ProjectsV2ItemDeleted(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.deleted"


class ProjectsV2ItemEdited(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.edited events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.edited"
    changes: dict[str, Any] = Field(description="Changes made to the project item")


class ProjectsV2ItemReordered(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.reordered events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.reordered"
    changes: dict[str, Any] = Field(description="Changes made to the project item")


class ProjectsV2ItemRestored(ProjectsV2ItemBase):
    """Payload for github.projects_v2_item.restored events."""

    _dispatch_topic: ClassVar[str] = "github.projects_v2_item.restored"
    changes: dict[str, Any] = Field(description="Changes made to the project item")


# =============================================================================
# Sponsorship Event Payloads
# =============================================================================


class SponsorshipBase(GitHubEventPayload):
    """Base class for sponsorship.* events.

    Triggered when a sponsorship is created, cancelled, edited,
    or when pending changes occur.
    """

    action: str = Field(
        description="Event action: cancelled, created, edited, "
        "pending_cancellation, pending_tier_change, or tier_changed"
    )
    sponsorship: dict[str, Any] = Field(description="Sponsorship details")


class SponsorshipCancelled(SponsorshipBase):
    """Payload for github.sponsorship.cancelled events."""

    _dispatch_topic: ClassVar[str] = "github.sponsorship.cancelled"


class SponsorshipCreated(SponsorshipBase):
    """Payload for github.sponsorship.created events."""

    _dispatch_topic: ClassVar[str] = "github.sponsorship.created"


class SponsorshipEdited(SponsorshipBase):
    """Payload for github.sponsorship.edited events."""

    _dispatch_topic: ClassVar[str] = "github.sponsorship.edited"
    changes: GitHubChanges = Field(description="Changes made to the sponsorship")


class SponsorshipPendingCancellation(SponsorshipBase):
    """Payload for github.sponsorship.pending_cancellation events."""

    _dispatch_topic: ClassVar[str] = "github.sponsorship.pending_cancellation"


class SponsorshipPendingTierChange(SponsorshipBase):
    """Payload for github.sponsorship.pending_tier_change events."""

    _dispatch_topic: ClassVar[str] = "github.sponsorship.pending_tier_change"
    changes: GitHubChanges = Field(description="Pending tier change details")


class SponsorshipTierChanged(SponsorshipBase):
    """Payload for github.sponsorship.tier_changed events."""

    _dispatch_topic: ClassVar[str] = "github.sponsorship.tier_changed"
    changes: GitHubChanges = Field(description="Tier change details")


# =============================================================================
# Custom Property Event Payloads
# =============================================================================


class CustomPropertyBase(GitHubEventPayload):
    """Base class for custom_property.* events.

    Triggered when a custom property definition is created or deleted
    at the organization level.
    """

    action: str = Field(description="Event action: created or deleted")
    definition: dict[str, Any] = Field(description="Custom property definition data")


class CustomPropertyCreated(CustomPropertyBase):
    """Payload for github.custom_property.created events."""

    _dispatch_topic: ClassVar[str] = "github.custom_property.created"


class CustomPropertyDeleted(CustomPropertyBase):
    """Payload for github.custom_property.deleted events."""

    _dispatch_topic: ClassVar[str] = "github.custom_property.deleted"


# =============================================================================
# Custom Property Values Event Payloads
# =============================================================================


class CustomPropertyValuesBase(GitHubEventPayload):
    """Base class for custom_property_values.* events.

    Triggered when custom property values are updated for a repository.
    """

    action: str = Field(description="Event action: updated")
    new_property_values: list[dict[str, Any]] = Field(
        description="New custom property values after the update"
    )
    old_property_values: list[dict[str, Any]] = Field(
        description="Previous custom property values before the update"
    )


class CustomPropertyValuesUpdated(CustomPropertyValuesBase):
    """Payload for github.custom_property_values.updated events."""

    _dispatch_topic: ClassVar[str] = "github.custom_property_values.updated"


# =============================================================================
# Deployment Protection Rule Event Payloads
# =============================================================================


class DeploymentProtectionRuleBase(GitHubEventPayload):
    """Base class for deployment_protection_rule.* events.

    Triggered when a deployment is waiting for a custom protection rule
    to be evaluated.
    """

    action: str = Field(description="Event action: requested")
    environment: str | None = Field(
        default=None, description="Target deployment environment"
    )
    event: str | None = Field(
        default=None, description="The event that triggered the protection rule"
    )
    deployment_callback_url: str | None = Field(
        default=None,
        description="URL to call back to when the protection rule is evaluated",
    )
    deployment: GitHubDeployment | None = Field(
        default=None, description="Deployment waiting for approval"
    )
    pull_requests: list[dict[str, Any]] = Field(
        default_factory=list, description="Pull requests associated with the deployment"
    )


class DeploymentProtectionRuleRequested(DeploymentProtectionRuleBase):
    """Payload for github.deployment_protection_rule.requested events."""

    _dispatch_topic: ClassVar[str] = "github.deployment_protection_rule.requested"


# =============================================================================
# Installation Target Event Payloads
# =============================================================================


class InstallationTargetBase(GitHubEventPayload):
    """Base class for installation_target.* events.

    Triggered when a GitHub App owner's account or organization is renamed.
    """

    action: str = Field(description="Event action: renamed")
    target_type: str = Field(
        description="Type of the installation target: User or Organization"
    )
    account: dict[str, Any] = Field(
        description="Account (user or org) that was renamed"
    )
    changes: dict[str, Any] = Field(description="Changes made (e.g., old login value)")


class InstallationTargetRenamed(InstallationTargetBase):
    """Payload for github.installation_target.renamed events."""

    _dispatch_topic: ClassVar[str] = "github.installation_target.renamed"
    installation: GitHubInstallation = Field(
        description="GitHub App installation associated with the renamed target"
    )


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
    | DeploymentBase
    | DeploymentStatusBase
    | DeploymentReviewBase
    | CommitStatus
    | WorkflowDispatch
    | DependabotAlertBase
    | LabelBase
    | WatchBase
    | Public
    | Gollum
    | Ping
    | RepositoryImport
    | PageBuild
    | GitHubAppAuthorizationBase
    | TeamAdd
    | BranchProtectionConfigurationBase
    | BranchProtectionRuleBase
    | CommitCommentBase
    | DeployKeyBase
    | MemberBase
    | MembershipBase
    | MergeGroupBase
    | MetaBase
    | MilestoneEventBase
    | OrgBlockBase
    | RepositoryEventBase
    | RepositoryDispatch
    | DiscussionBase
    | DiscussionCommentBase
    | TeamBase
    | OrganizationBase
    | CodeScanningAlertBase
    | SecretScanningAlertBase
    | SecretScanningAlertLocationBase
    | RepositoryVulnerabilityAlertBase
    | SecurityAdvisoryBase
    | MarketplacePurchaseBase
    | PackageBase
    | RegistryPackageBase
    | ProjectBase
    | ProjectCardBase
    | ProjectColumnBase
    | ProjectsV2ItemBase
    | SponsorshipBase
    | CustomPropertyBase
    | CustomPropertyValuesBase
    | DeploymentProtectionRuleBase
    | InstallationTargetBase
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
    # Deployment events
    "GitHubDeployment",
    "DeploymentBase",
    "DeploymentCreated",
    # Deployment Status events
    "GitHubDeploymentStatus",
    "DeploymentStatusBase",
    "DeploymentStatusCreated",
    # Deployment Review events
    "DeploymentReviewBase",
    "DeploymentReviewApproved",
    "DeploymentReviewRejected",
    "DeploymentReviewRequested",
    # Commit Status events (no action)
    "GitHubStatusCommit",
    "CommitStatus",
    # WorkflowDispatch event (no action)
    "WorkflowDispatch",
    # Dependabot Alert events
    "GitHubDependabotDependency",
    "GitHubDependabotAdvisory",
    "GitHubDependabotVulnerability",
    "GitHubDependabotAlert",
    "DependabotAlertBase",
    "DependabotAlertCreated",
    "DependabotAlertFixed",
    "DependabotAlertDismissed",
    "DependabotAlertReintroduced",
    "DependabotAlertAutoDismissed",
    "DependabotAlertAutoReopened",
    "DependabotAlertReopened",
    # Label events
    "LabelBase",
    "LabelCreated",
    "LabelEdited",
    "LabelDeleted",
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
    # Watch events
    "WatchBase",
    "WatchStarted",
    # Single-action events (no action field)
    "Public",
    "Gollum",
    "Ping",
    "RepositoryImport",
    "PageBuild",
    "TeamAdd",
    # GitHub App Authorization events
    "GitHubAppAuthorizationBase",
    "GitHubAppAuthorizationRevoked",
    # Branch Protection Configuration events
    "BranchProtectionConfigurationBase",
    "BranchProtectionConfigurationEnabled",
    "BranchProtectionConfigurationDisabled",
    # Branch Protection Rule events
    "BranchProtectionRuleBase",
    "BranchProtectionRuleCreated",
    "BranchProtectionRuleDeleted",
    "BranchProtectionRuleEdited",
    # Commit Comment events
    "CommitCommentBase",
    "CommitCommentCreated",
    # Deploy Key events
    "DeployKeyBase",
    "DeployKeyCreated",
    "DeployKeyDeleted",
    # Member events
    "MemberBase",
    "MemberAdded",
    "MemberEdited",
    "MemberRemoved",
    # Membership events
    "MembershipBase",
    "MembershipAdded",
    "MembershipRemoved",
    # Merge Group events
    "MergeGroupBase",
    "MergeGroupChecksRequested",
    "MergeGroupDestroyed",
    # Meta events
    "MetaBase",
    "MetaDeleted",
    # Milestone events
    "MilestoneEventBase",
    "MilestoneClosed",
    "MilestoneCreated",
    "MilestoneDeleted",
    "MilestoneEdited",
    "MilestoneOpened",
    # Org Block events
    "OrgBlockBase",
    "OrgBlockBlocked",
    "OrgBlockUnblocked",
    # Repository events (with actions)
    "RepositoryEventBase",
    "RepositoryArchived",
    "RepositoryCreated",
    "RepositoryDeleted",
    "RepositoryEdited",
    "RepositoryPrivatized",
    "RepositoryPublicized",
    "RepositoryRenamed",
    "RepositoryTransferred",
    "RepositoryUnarchived",
    "RepositoryDispatch",
    # Issue additional actions
    "IssueDeleted",
    # Pull Request additional actions
    "PullRequestMilestoned",
    "PullRequestDemilestoned",
    "PullRequestAutoMergeEnabled",
    "PullRequestAutoMergeDisabled",
    "PullRequestEnqueued",
    "PullRequestDequeued",
    # Discussion events
    "GitHubDiscussion",
    "DiscussionBase",
    "DiscussionAnswered",
    "DiscussionCategoryChanged",
    "DiscussionCreated",
    "DiscussionDeleted",
    "DiscussionEdited",
    "DiscussionLabeled",
    "DiscussionLocked",
    "DiscussionPinned",
    "DiscussionTransferred",
    "DiscussionUnanswered",
    "DiscussionUnlabeled",
    "DiscussionUnlocked",
    "DiscussionUnpinned",
    # Discussion Comment events
    "DiscussionCommentBase",
    "DiscussionCommentCreated",
    "DiscussionCommentDeleted",
    "DiscussionCommentEdited",
    # Team events
    "GitHubTeam",
    "TeamBase",
    "TeamAddedToRepository",
    "TeamCreated",
    "TeamDeleted",
    "TeamEdited",
    "TeamRemovedFromRepository",
    # Organization events
    "OrganizationBase",
    "OrganizationDeleted",
    "OrganizationMemberAdded",
    "OrganizationMemberInvited",
    "OrganizationMemberRemoved",
    "OrganizationRenamed",
    # Code Scanning Alert events
    "GitHubCodeScanningAlert",
    "CodeScanningAlertBase",
    "CodeScanningAlertAppearedInBranch",
    "CodeScanningAlertClosedByUser",
    "CodeScanningAlertCreated",
    "CodeScanningAlertFixed",
    "CodeScanningAlertReopened",
    "CodeScanningAlertReopenedByUser",
    # Secret Scanning Alert events
    "GitHubSecretScanningAlert",
    "SecretScanningAlertBase",
    "SecretScanningAlertCreated",
    "SecretScanningAlertReopened",
    "SecretScanningAlertResolved",
    "SecretScanningAlertRevoked",
    # Secret Scanning Alert Location events
    "SecretScanningAlertLocationBase",
    "SecretScanningAlertLocationCreated",
    # Repository Vulnerability Alert events
    "RepositoryVulnerabilityAlertBase",
    "RepositoryVulnerabilityAlertCreate",
    "RepositoryVulnerabilityAlertDismiss",
    "RepositoryVulnerabilityAlertReopen",
    "RepositoryVulnerabilityAlertResolve",
    # Security Advisory events
    "SecurityAdvisoryBase",
    "SecurityAdvisoryPerformed",
    "SecurityAdvisoryPublished",
    "SecurityAdvisoryUpdated",
    "SecurityAdvisoryWithdrawn",
    # Marketplace Purchase events
    "MarketplacePurchaseBase",
    "MarketplacePurchaseCancelled",
    "MarketplacePurchaseChanged",
    "MarketplacePurchasePendingChange",
    "MarketplacePurchasePendingChangeCancelled",
    "MarketplacePurchasePurchased",
    # Package events
    "PackageBase",
    "PackagePublished",
    "PackageUpdated",
    # Registry Package events
    "RegistryPackageBase",
    "RegistryPackagePublished",
    "RegistryPackageUpdated",
    # Project events
    "ProjectBase",
    "ProjectClosed",
    "ProjectCreated",
    "ProjectDeleted",
    "ProjectEdited",
    "ProjectReopened",
    # Project Card events
    "ProjectCardBase",
    "ProjectCardConverted",
    "ProjectCardCreated",
    "ProjectCardDeleted",
    "ProjectCardEdited",
    "ProjectCardMoved",
    # Project Column events
    "ProjectColumnBase",
    "ProjectColumnCreated",
    "ProjectColumnDeleted",
    "ProjectColumnEdited",
    "ProjectColumnMoved",
    # Projects V2 Item events
    "ProjectsV2ItemBase",
    "ProjectsV2ItemArchived",
    "ProjectsV2ItemConverted",
    "ProjectsV2ItemCreated",
    "ProjectsV2ItemDeleted",
    "ProjectsV2ItemEdited",
    "ProjectsV2ItemReordered",
    "ProjectsV2ItemRestored",
    # Sponsorship events
    "SponsorshipBase",
    "SponsorshipCancelled",
    "SponsorshipCreated",
    "SponsorshipEdited",
    "SponsorshipPendingCancellation",
    "SponsorshipPendingTierChange",
    "SponsorshipTierChanged",
    # Custom Property events
    "CustomPropertyBase",
    "CustomPropertyCreated",
    "CustomPropertyDeleted",
    # Custom Property Values events
    "CustomPropertyValuesBase",
    "CustomPropertyValuesUpdated",
    # Deployment Protection Rule events
    "DeploymentProtectionRuleBase",
    "DeploymentProtectionRuleRequested",
    # Installation Target events
    "InstallationTargetBase",
    "InstallationTargetRenamed",
]
