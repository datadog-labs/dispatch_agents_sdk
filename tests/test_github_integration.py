"""Tests for GitHub integration: payloads, types, and @on(github_event=...) decorator."""

import pytest
from pydantic import ValidationError

from dispatch_agents import (
    HANDLER_METADATA,
    REGISTERED_HANDLERS,
    TOPIC_HANDLERS,
    dispatch_message,
    on,
)
from dispatch_agents.integrations.github import (
    CheckRunBase,
    GitHubBranch,
    GitHubChanges,
    GitHubChangeValue,
    GitHubCheckRun,
    GitHubComment,
    GitHubCommit,
    GitHubCommitUser,
    GitHubInstallation,
    GitHubIssue,
    GitHubIssuePullRequest,
    GitHubLink,
    GitHubPullRequest,
    GitHubRelease,
    GitHubRepository,
    GitHubRequestedAction,
    GitHubReview,
    GitHubReviewComment,
    GitHubReviewThread,
    GitHubUser,
    GitHubWorkflow,
    GitHubWorkflowJob,
    GitHubWorkflowRun,
    GitHubWorkflowStep,
    IssueBase,
    IssueCommentBase,
    PullRequestBase,
    PullRequestReviewBase,
    Push,
    ReleaseBase,
    WorkflowRunBase,
)
from dispatch_agents.models import SuccessPayload, TopicMessage


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear handler registries before each test."""
    REGISTERED_HANDLERS.clear()
    HANDLER_METADATA.clear()
    TOPIC_HANDLERS.clear()
    yield
    REGISTERED_HANDLERS.clear()
    HANDLER_METADATA.clear()
    TOPIC_HANDLERS.clear()


# =============================================================================
# Sample GitHub Webhook Data
# =============================================================================


def make_github_user(login: str = "octocat", id: int = 1) -> GitHubUser:
    """Create a sample GitHub user."""
    return GitHubUser(
        id=id,
        login=login,
        type="User",
        avatar_url=f"https://avatars.githubusercontent.com/u/{id}",
    )


def make_github_repo(
    name: str = "test-repo", owner_login: str = "octocat"
) -> GitHubRepository:
    """Create a sample GitHub repository."""
    return GitHubRepository(
        id=123,
        name=name,
        full_name=f"{owner_login}/{name}",
        owner=make_github_user(owner_login),
        private=False,
        default_branch="main",
        html_url=f"https://github.com/{owner_login}/{name}",
    )


def make_github_installation(id: int = 12345) -> GitHubInstallation:
    """Create a sample GitHub installation."""
    return GitHubInstallation(
        id=id,
        account=make_github_user(),
    )


def make_github_organization(login: str = "my-org") -> GitHubUser:
    """Create a sample GitHub organization (uses GitHubUser type)."""
    return GitHubUser(
        id=456,
        login=login,
        type="Organization",
    )


def make_github_pr(number: int = 42, title: str = "Fix bug") -> GitHubPullRequest:
    """Create a sample GitHub pull request."""
    return GitHubPullRequest(
        id=999,
        number=number,
        title=title,
        state="open",
        body="This PR fixes a bug.",
        html_url="https://github.com/octocat/test-repo/pull/42",
        head=GitHubBranch(
            ref="feature-branch",
            sha="abc123def456",
            repo=make_github_repo(),
        ),
        base=GitHubBranch(
            ref="main",
            sha="789xyz",
            repo=make_github_repo(),
        ),
        user=make_github_user(),
        draft=False,
        merged=False,
        mergeable=True,
    )


def make_github_issue(number: int = 1, title: str = "Bug report") -> GitHubIssue:
    """Create a sample GitHub issue."""
    return GitHubIssue(
        id=888,
        number=number,
        title=title,
        state="open",
        body="Found a bug.",
        html_url="https://github.com/octocat/test-repo/issues/1",
        user=make_github_user(),
        labels=[],
    )


def make_github_comment(body: str = "Great work!") -> GitHubComment:
    """Create a sample GitHub comment."""
    return GitHubComment(
        id=777,
        body=body,
        html_url="https://github.com/octocat/test-repo/issues/1#issuecomment-777",
        user=make_github_user(),
        created_at="2024-01-15T10:30:00Z",
        updated_at="2024-01-15T10:30:00Z",
    )


# =============================================================================
# Test: GitHub Types
# =============================================================================


def test_github_user_validation():
    """Test GitHubUser type validation."""
    user = GitHubUser(id=1, login="octocat", type="User")
    assert user.id == 1
    assert user.login == "octocat"
    assert user.type == "User"


def test_github_user_with_extra_fields():
    """Test GitHubUser ignores extra fields (permissive validation)."""
    # GitHub payloads often include extra fields we don't model
    # Use model_validate to test dict parsing with extra fields
    user = GitHubUser.model_validate(
        {
            "id": 1,
            "login": "octocat",
            "type": "User",
            "node_id": "MDQ6VXNlcjE=",  # Known optional field
            "gravatar_id": "",  # Extra field not in model - should be ignored
        }
    )
    assert user.id == 1
    assert user.login == "octocat"


def test_github_repository_validation():
    """Test GitHubRepository type validation."""
    repo = make_github_repo()
    assert repo.name == "test-repo"
    assert repo.full_name == "octocat/test-repo"
    assert repo.owner.login == "octocat"


def test_github_pull_request_validation():
    """Test GitHubPullRequest type validation."""
    pr = make_github_pr()
    assert pr.number == 42
    assert pr.title == "Fix bug"
    assert pr.state == "open"
    assert pr.head.ref == "feature-branch"
    assert pr.base.ref == "main"


def test_github_issue_validation():
    """Test GitHubIssue type validation."""
    issue = make_github_issue()
    assert issue.number == 1
    assert issue.title == "Bug report"
    assert issue.state == "open"


def test_github_comment_validation():
    """Test GitHubComment type validation."""
    comment = make_github_comment()
    assert comment.id == 777
    assert comment.body == "Great work!"


def test_github_installation_validation():
    """Test GitHubInstallation type validation."""
    installation = make_github_installation()
    assert installation.id == 12345
    assert installation.account is not None
    assert installation.account.login == "octocat"


# =============================================================================
# Test: GitHub Event Payloads
# =============================================================================


def test_pull_request_base_payload():
    """Test PullRequestBase validation."""
    payload = PullRequestBase(
        action="opened",
        number=42,
        pull_request=make_github_pr(),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "opened"
    assert payload.number == 42
    assert payload.pull_request.title == "Fix bug"
    assert payload.repository.full_name == "octocat/test-repo"
    assert payload.sender.login == "octocat"


def test_issue_comment_base_payload():
    """Test IssueCommentBase validation."""
    payload = IssueCommentBase(
        action="created",
        issue=make_github_issue(),
        comment=make_github_comment(),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "created"
    assert payload.issue.number == 1
    assert payload.comment.body == "Great work!"


def test_push_payload():
    """Test Push payload validation."""
    payload = Push(
        ref="refs/heads/main",
        before="abc123",
        after="def456",
        compare="https://github.com/octocat/test-repo/compare/abc123...def456",
        commits=[
            GitHubCommit(
                id="def456",
                tree_id="tree123",
                message="Add new feature",
                timestamp="2024-01-15T10:30:00Z",
                author=GitHubCommitUser(name="Test User", email="test@example.com"),
                committer=GitHubCommitUser(name="Test User", email="test@example.com"),
                url="https://api.github.com/repos/octocat/test-repo/commits/def456",
            )
        ],
        pusher=GitHubCommitUser(name="octocat", email="octocat@github.com"),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.ref == "refs/heads/main"
    assert len(payload.commits) == 1
    assert payload.commits[0].message == "Add new feature"


def test_issue_base_payload():
    """Test IssueBase payload validation."""
    payload = IssueBase(
        action="opened",
        issue=make_github_issue(),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "opened"
    assert payload.issue.title == "Bug report"


def test_check_run_base_payload():
    """Test CheckRunBase payload validation."""
    payload = CheckRunBase(
        action="completed",
        check_run=GitHubCheckRun(
            id=123,
            name="test-suite",
            status="completed",
            conclusion="success",
            head_sha="abc123",
        ),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "completed"
    assert payload.check_run.name == "test-suite"
    assert payload.check_run.conclusion == "success"


def test_workflow_run_base_payload():
    """Test WorkflowRunBase payload validation."""
    payload = WorkflowRunBase(
        action="completed",
        workflow_run=GitHubWorkflowRun(
            id=456,
            name="CI",
            status="completed",
            conclusion="success",
            head_sha="abc123",
            head_branch="main",
            run_number=1,
            event="push",
            workflow_id=789,
        ),
        workflow=GitHubWorkflow(
            id=789,
            name="CI",
            path=".github/workflows/ci.yml",
            state="active",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            url="https://api.github.com/repos/octocat/test-repo/actions/workflows/789",
            html_url="https://github.com/octocat/test-repo/actions/workflows/ci.yml",
        ),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "completed"
    assert payload.workflow_run.name == "CI"


def test_release_base_payload():
    """Test ReleaseBase payload validation."""
    payload = ReleaseBase(
        action="published",
        release=GitHubRelease(
            id=111,
            tag_name="v1.0.0",
            name="Version 1.0.0",
            body="Release notes",
            draft=False,
            prerelease=False,
            created_at="2024-01-15T10:00:00Z",
            author=make_github_user(),
            html_url="https://github.com/octocat/test-repo/releases/tag/v1.0.0",
        ),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "published"
    assert payload.release.tag_name == "v1.0.0"


def test_pull_request_review_base_payload():
    """Test PullRequestReviewBase payload validation."""
    payload = PullRequestReviewBase(
        action="submitted",
        review=GitHubReview(
            id=222,
            body="LGTM!",
            state="approved",
            user=make_github_user(),
            html_url="https://github.com/octocat/test-repo/pull/1#pullrequestreview-222",
            commit_id="abc123",
        ),
        pull_request=make_github_pr(),
        repository=make_github_repo(),
        sender=make_github_user(),
        organization=make_github_organization(),
    )
    assert payload.action == "submitted"
    assert payload.review.state == "approved"


# =============================================================================
# Test: New Typed Models
# =============================================================================


def test_github_requested_action():
    """Test GitHubRequestedAction model."""
    action = GitHubRequestedAction(identifier="fix_it")
    assert action.identifier == "fix_it"


def test_github_workflow():
    """Test GitHubWorkflow model with required and optional fields."""
    workflow = GitHubWorkflow(
        id=1,
        name="CI",
        path=".github/workflows/ci.yml",
        state="active",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        url="https://api.github.com/repos/octocat/test-repo/actions/workflows/1",
        html_url="https://github.com/octocat/test-repo/actions/workflows/ci.yml",
    )
    assert workflow.name == "CI"
    assert workflow.state == "active"
    assert workflow.badge_url is None  # optional


def test_github_workflow_step():
    """Test GitHubWorkflowStep model."""
    step = GitHubWorkflowStep(
        name="Run tests",
        status="completed",
        conclusion="success",
        number=1,
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:01:00Z",
    )
    assert step.name == "Run tests"
    assert step.conclusion == "success"


def test_github_workflow_job():
    """Test GitHubWorkflowJob model with steps."""
    step = GitHubWorkflowStep(
        name="Checkout", status="completed", conclusion="success", number=1
    )
    job = GitHubWorkflowJob(
        id=100,
        run_id=200,
        head_sha="abc123",
        status="completed",
        conclusion="success",
        name="build",
        steps=[step],
    )
    assert job.name == "build"
    assert len(job.steps) == 1
    assert job.steps[0].name == "Checkout"
    assert job.runner_id is None  # optional


def test_github_review_thread():
    """Test GitHubReviewThread model."""
    thread = GitHubReviewThread(node_id="RT_123", comments=[])
    assert thread.node_id == "RT_123"
    assert thread.comments == []


def test_github_link():
    """Test GitHubLink model."""
    link = GitHubLink(href="https://github.com/octocat/test-repo/pull/1")
    assert link.href == "https://github.com/octocat/test-repo/pull/1"


def test_github_issue_pull_request():
    """Test GitHubIssuePullRequest model."""
    pr_info = GitHubIssuePullRequest(
        url="https://api.github.com/repos/octocat/test-repo/pulls/1",
        html_url="https://github.com/octocat/test-repo/pull/1",
    )
    assert pr_info.html_url == "https://github.com/octocat/test-repo/pull/1"
    assert pr_info.merged_at is None  # optional


def test_github_change_value():
    """Test GitHubChangeValue model with 'from' alias."""
    # Simulate the JSON deserialization path (GitHub sends {"from": "old title"})
    change = GitHubChangeValue.model_validate({"from": "old title"})
    assert change.from_ == "old title"


def test_github_changes_with_typed_values():
    """Test GitHubChanges with typed GitHubChangeValue fields."""
    changes = GitHubChanges.model_validate(
        {
            "title": {"from": "Old Title"},
            "body": {"from": "Old Body"},
        }
    )
    assert changes.title is not None
    assert changes.title.from_ == "Old Title"
    assert changes.body is not None
    assert changes.body.from_ == "Old Body"


def test_github_review_thread_with_comments():
    """Test GitHubReviewThread with actual review comments."""
    comment = GitHubReviewComment(
        id=1,
        html_url="https://github.com/octocat/test-repo/pull/1#discussion_r1",
        body="Looks good",
        user=make_github_user(),
        commit_id="abc123",
        diff_hunk="@@ -1,3 +1,4 @@",
        path="src/main.py",
        created_at="2024-01-15T10:00:00Z",
    )
    thread = GitHubReviewThread(node_id="RT_456", comments=[comment])
    assert len(thread.comments) == 1
    assert thread.comments[0].body == "Looks good"


# =============================================================================
# Test: Payload Validation Errors
# =============================================================================


def test_pr_payload_missing_required_field():
    """Test that missing required fields raise validation errors."""
    with pytest.raises(ValidationError):
        PullRequestBase(  # type: ignore[call-arg]
            action="opened",
            # Missing: number, pull_request, repository, sender
        )


def test_pr_payload_invalid_field_type():
    """Test that invalid field types raise validation errors."""
    with pytest.raises(ValidationError):
        PullRequestBase(
            action="opened",
            number="not_an_int",  # type: ignore[arg-type]  # Should be int
            pull_request=make_github_pr(),
            repository=make_github_repo(),
            sender=make_github_user(),
            organization=make_github_organization(),
        )


# =============================================================================
# Test: Class-Based API - dispatch_topic() method
# =============================================================================


def test_pull_request_opened_dispatch_topic():
    """Test dispatch_topic() class method returns correct topic."""
    from dispatch_agents.integrations.github import PullRequestOpened

    assert PullRequestOpened.dispatch_topic() == "github.pull_request.opened"


def test_pull_request_closed_dispatch_topic():
    """Test dispatch_topic() for PullRequestClosed."""
    from dispatch_agents.integrations.github import PullRequestClosed

    assert PullRequestClosed.dispatch_topic() == "github.pull_request.closed"


def test_issue_opened_dispatch_topic():
    """Test dispatch_topic() for IssueOpened."""
    from dispatch_agents.integrations.github import IssueOpened

    assert IssueOpened.dispatch_topic() == "github.issues.opened"


def test_push_dispatch_topic():
    """Test dispatch_topic() for Push (no action)."""
    from dispatch_agents.integrations.github import Push

    assert Push.dispatch_topic() == "github.push"


def test_issue_comment_created_dispatch_topic():
    """Test dispatch_topic() for IssueCommentCreated."""
    from dispatch_agents.integrations.github import IssueCommentCreated

    assert IssueCommentCreated.dispatch_topic() == "github.issue_comment.created"


# =============================================================================
# Test: Class-Based @on(github_event=...) Decorator
# =============================================================================


def test_on_github_class_single_event():
    """Test @on(github_event=...) with a single event class."""
    from dispatch_agents.integrations.github import PullRequestOpened

    @on(github_event=PullRequestOpened)
    async def handle_pr_class(payload: PullRequestOpened) -> None:
        pass

    # Check handler was registered
    assert "handle_pr_class" in REGISTERED_HANDLERS

    # Check topic was mapped
    assert "github.pull_request.opened" in TOPIC_HANDLERS
    assert "handle_pr_class" in TOPIC_HANDLERS["github.pull_request.opened"]

    # Check metadata
    metadata = HANDLER_METADATA["handle_pr_class"]
    assert metadata.topics == ["github.pull_request.opened"]


def test_on_github_class_multiple_events():
    """Test @on(github_event=...) with multiple event classes."""
    from dispatch_agents.integrations.github import (
        PullRequestBase,
        PullRequestOpened,
        PullRequestSynchronize,
    )

    @on(github_event=[PullRequestOpened, PullRequestSynchronize])
    async def handle_pr_multi_class(payload: PullRequestBase) -> None:
        pass

    # Check handler was registered
    assert "handle_pr_multi_class" in REGISTERED_HANDLERS

    # Check both topics were mapped
    assert "github.pull_request.opened" in TOPIC_HANDLERS
    assert "github.pull_request.synchronize" in TOPIC_HANDLERS
    assert "handle_pr_multi_class" in TOPIC_HANDLERS["github.pull_request.opened"]
    assert "handle_pr_multi_class" in TOPIC_HANDLERS["github.pull_request.synchronize"]

    # Check metadata contains all topics
    metadata = HANDLER_METADATA["handle_pr_multi_class"]
    assert "github.pull_request.opened" in metadata.topics
    assert "github.pull_request.synchronize" in metadata.topics


def test_on_github_class_issue_comment():
    """Test @on(github_event=...) with IssueCommentCreated class."""
    from dispatch_agents.integrations.github import IssueCommentCreated

    @on(github_event=IssueCommentCreated)
    async def handle_comment_class(payload: IssueCommentCreated) -> None:
        pass

    assert "github.issue_comment.created" in TOPIC_HANDLERS


def test_on_github_class_push():
    """Test @on(github_event=...) with Push class."""
    from dispatch_agents.integrations.github import Push

    @on(github_event=Push)
    async def handle_push_class(payload: Push) -> None:
        pass

    assert "github.push" in TOPIC_HANDLERS


def test_on_github_class_validation_success_base_class():
    """Test @on validation with base class accepts specific event subclasses."""
    from dispatch_agents.integrations.github import (
        PullRequestBase,
        PullRequestClosed,
        PullRequestOpened,
    )

    # This should work: using PullRequestBase as payload for multiple PR events
    @on(github_event=[PullRequestOpened, PullRequestClosed])
    async def handle_pr_base(payload: PullRequestBase) -> None:
        pass

    assert "handle_pr_base" in REGISTERED_HANDLERS


def test_on_github_class_validation_failure():
    """Test @on validation fails when payload type is incompatible."""
    from dispatch_agents.integrations.github import IssueBase, PullRequestOpened

    # This should fail: IssueBase is not a base class of PullRequestOpened
    with pytest.raises(TypeError, match="is not compatible with"):

        @on(github_event=PullRequestOpened)
        async def bad_handler_class(payload: IssueBase) -> None:
            pass


def test_on_github_class_invalid_type():
    """Test @on raises TypeError for invalid github_event type."""
    with pytest.raises(TypeError, match="Invalid github_event type"):

        @on(github_event="not_a_class")  # type: ignore[arg-type]
        async def invalid_type_handler(payload: PullRequestBase) -> None:
            pass


# =============================================================================
# Test: Class-Based Handler Dispatch
# =============================================================================


@pytest.mark.asyncio
async def test_on_github_class_dispatch_pr_opened():
    """Test dispatching a PR opened event to a class-based handler."""
    from dispatch_agents.integrations.github import PullRequestOpened

    received_payload = None

    @on(github_event=PullRequestOpened)
    async def handle_pr_dispatch(payload: PullRequestOpened) -> dict:
        nonlocal received_payload
        received_payload = payload
        return {"handled": True, "pr_number": payload.number}

    # Create topic message simulating a GitHub webhook
    message = TopicMessage.create(
        topic="github.pull_request.opened",
        payload={
            "action": "opened",
            "number": 123,
            "pull_request": make_github_pr(),
            "repository": make_github_repo(),
            "sender": make_github_user(),
            "installation": make_github_installation(),
            "organization": make_github_organization(),
        },
        sender_id="github-webhook-test",
    )

    # Dispatch
    result = await dispatch_message(message)

    # Verify
    assert isinstance(result, SuccessPayload)
    assert result.result == {"handled": True, "pr_number": 123}
    assert received_payload is not None
    assert received_payload.number == 123


@pytest.mark.asyncio
async def test_on_github_class_dispatch_push():
    """Test dispatching a push event to a class-based Push handler."""
    from dispatch_agents.integrations.github import Push

    @on(github_event=Push)
    async def handle_push_dispatch(payload: Push) -> dict:
        return {"ref": payload.ref, "commits": len(payload.commits)}

    message = TopicMessage.create(
        topic="github.push",
        payload={
            "ref": "refs/heads/feature",
            "before": "abc",
            "after": "def",
            "compare": "https://github.com/octocat/test-repo/compare/abc...def",
            "commits": [
                {
                    "id": "def",
                    "tree_id": "tree123",
                    "message": "commit 1",
                    "timestamp": "2024-01-15T10:00:00Z",
                    "author": {"name": "Test", "email": "test@test.com"},
                    "committer": {"name": "Test", "email": "test@test.com"},
                    "url": "https://api.github.com/repos/octocat/test-repo/commits/def",
                },
            ],
            "pusher": {"name": "octocat", "email": "octocat@github.com"},
            "repository": make_github_repo(),
            "sender": make_github_user(),
            "installation": make_github_installation(),
            "organization": make_github_organization(),
        },
        sender_id="github-webhook-test",
    )

    result = await dispatch_message(message)

    assert isinstance(result, SuccessPayload)
    assert result.result == {"ref": "refs/heads/feature", "commits": 1}


def test_pull_request_review_comment_validation_error_malformed_user():
    """Test that malformed user objects in PR review comment payloads raise ValidationError.

    This reproduces an issue where mock/generated payloads have user objects
    with only a 'description' field instead of required 'id' and 'login' fields.
    """
    from pydantic import ValidationError

    from dispatch_agents.integrations.github import PullRequestReviewCommentCreated

    # This payload has malformed user objects - they only have 'description'
    # instead of required fields like 'id' and 'login'
    malformed_payload = {
        "sender": {
            "id": -93062968,
            "login": "test_user",
            "type": "User",
            "site_admin": False,
        },
        "repository": {
            "id": 25974643,
            "name": "test-repo",
            "full_name": "owner/test-repo",
            "owner": {
                "description": "Repository owner"  # INVALID: missing id, login
            },
            "private": False,
            "default_branch": "main",
        },
        "organization": {
            "id": 84718925,
            "login": "test_org",
            "type": "Organization",
            "site_admin": False,
        },
        "action": "created",
        "pull_request": {
            "id": 27783226,
            "number": 123,
            "state": "open",
            "locked": False,
            "title": "Test PR",
            "user": {
                "id": 75264941,
                "login": "pr_author",
                "type": "User",
                "site_admin": False,
            },
            "labels": [],
            "assignees": [],
            "requested_reviewers": [],
            "head": {"ref": "feature", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "draft": False,
        },
        "comment": {
            "id": 12345,
            "diff_hunk": "@@ -1,5 +1,5 @@",
            "path": "src/main.py",
            "commit_id": "abc123",
            "user": {
                "description": "Comment author"  # INVALID: missing id, login
            },
            "body": "Test comment",
            "created_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/123#discussion_r12345",
            "author_association": "NONE",
        },
    }

    with pytest.raises(ValidationError) as exc_info:
        PullRequestReviewCommentCreated.model_validate(malformed_payload)

    # Verify the error mentions the missing required fields
    error_str = str(exc_info.value)
    assert "owner" in error_str or "user" in error_str


def test_pydantic_required_fields_match_octokit_spec():
    """Verify that required fields in our Pydantic models are also required in the spec.

    This test ensures we don't mark fields as required when GitHub's webhook spec
    considers them optional (e.g., 'organization' is only present for org-owned repos).
    """
    import json
    from pathlib import Path

    from pydantic import BaseModel

    from dispatch_agents.integrations import github as github_module

    # Load the octokit webhooks spec
    spec_path = Path(__file__).parent / "schemas" / "octokit-webhooks.json"
    with open(spec_path) as f:
        spec = json.load(f)

    definitions = spec.get("definitions", {})

    # Map our class names to spec definition names
    # Our naming: PullRequestOpened -> spec: pull_request$opened
    def class_name_to_spec_key(class_name: str) -> str:
        """Convert PullRequestOpened to pull_request$opened."""
        # Special cases
        special_mappings = {
            "Push": "push",
            "PullRequestReviewCommentCreated": "pull_request_review_comment$created",
            "PullRequestReviewCommentEdited": "pull_request_review_comment$edited",
            "PullRequestReviewCommentDeleted": "pull_request_review_comment$deleted",
            "PullRequestReviewThreadResolved": "pull_request_review_thread$resolved",
            "PullRequestReviewThreadUnresolved": "pull_request_review_thread$unresolved",
            "PullRequestReviewSubmitted": "pull_request_review$submitted",
            "PullRequestReviewEdited": "pull_request_review$edited",
            "PullRequestReviewDismissed": "pull_request_review$dismissed",
            "IssueCommentCreated": "issue_comment$created",
            "IssueCommentEdited": "issue_comment$edited",
            "IssueCommentDeleted": "issue_comment$deleted",
            "CheckRunCreated": "check_run$created",
            "CheckRunCompleted": "check_run$completed",
            "CheckRunRerequested": "check_run$rerequested",
            "CheckRunRequestedAction": "check_run$requested_action",
            "CheckSuiteCompleted": "check_suite$completed",
            "CheckSuiteRequested": "check_suite$requested",
            "CheckSuiteRerequested": "check_suite$rerequested",
            "WorkflowRunRequested": "workflow_run$requested",
            "WorkflowRunCompleted": "workflow_run$completed",
            "WorkflowRunInProgress": "workflow_run$in_progress",
            "ReleasePublished": "release$published",
            "ReleaseCreated": "release$created",
            "ReleaseEdited": "release$edited",
            "ReleaseDeleted": "release$deleted",
            "ReleaseReleased": "release$released",
            "ReleasePrereleased": "release$prereleased",
            "ReleaseUnpublished": "release$unpublished",
        }
        if class_name in special_mappings:
            return special_mappings[class_name]

        # General pattern: PullRequestOpened -> pull_request$opened
        # Split on capital letters
        import re

        parts = re.findall("[A-Z][a-z]*", class_name)
        if not parts:
            return class_name.lower()

        # Find where the action starts (last word that's a known action)
        actions = {
            "opened",
            "closed",
            "reopened",
            "edited",
            "labeled",
            "unlabeled",
            "assigned",
            "unassigned",
            "synchronize",
            "requested",
            "removed",
            "milestoned",
            "demilestoned",
            "locked",
            "unlocked",
            "transferred",
            "pinned",
            "unpinned",
            "created",
            "deleted",
            "completed",
            "submitted",
            "dismissed",
            "resolved",
            "unresolved",
            "published",
            "released",
            "prereleased",
            "unpublished",
        }

        # Convert parts to lowercase
        lower_parts = [p.lower() for p in parts]

        # Find the action (usually the last part)
        action_idx = None
        for i in range(len(lower_parts) - 1, -1, -1):
            if lower_parts[i] in actions:
                action_idx = i
                break

        if action_idx is not None:
            event_type = "_".join(lower_parts[:action_idx])
            action = "_".join(lower_parts[action_idx:])
            return f"{event_type}${action}"

        return "_".join(lower_parts)

    def get_required_fields(model_class: type[BaseModel]) -> set[str]:
        """Get the required fields for a Pydantic model (excluding ClassVars)."""
        required = set()
        for name, field_info in model_class.model_fields.items():
            # Field is required if it has no default and isn't Optional
            if field_info.is_required():
                required.add(name)
        return required

    def get_spec_required_fields(spec_key: str) -> set[str] | None:
        """Get required fields from the spec, or None if definition not found."""
        if spec_key not in definitions:
            return None
        return set(definitions[spec_key].get("required", []))

    # Get all event payload classes from our module
    from dispatch_agents.integrations.github import GitHubEventPayload

    event_classes = []
    for name in dir(github_module):
        obj = getattr(github_module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, GitHubEventPayload)
            and obj is not GitHubEventPayload
            and hasattr(obj, "_dispatch_topic")
            and obj._dispatch_topic  # Has a non-empty topic
        ):
            event_classes.append((name, obj))

    # Track failures
    failures = []

    for class_name, model_class in event_classes:
        spec_key = class_name_to_spec_key(class_name)
        spec_required = get_spec_required_fields(spec_key)

        if spec_required is None:
            # Skip classes without a matching spec definition
            continue

        model_required = get_required_fields(model_class)

        # Find fields required in our model but not in the spec
        extra_required = model_required - spec_required

        if extra_required:
            failures.append(
                f"{class_name} (spec: {spec_key}): "
                f"fields {extra_required} are required in model but not in spec"
            )

    if failures:
        pytest.fail(
            "Pydantic models have required fields not required by the spec:\n"
            + "\n".join(f"  - {f}" for f in failures)
        )
