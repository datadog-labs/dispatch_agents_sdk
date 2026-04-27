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
    GitHubBranch,
    GitHubChangeValue,
    GitHubComment,
    GitHubInstallation,
    GitHubIssue,
    GitHubPullRequest,
    GitHubRepository,
    GitHubTeam,
    GitHubUser,
    PullRequestBase,
    PullRequestReviewRequested,
    PullRequestReviewRequestRemoved,
    Push,
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
        created_at="2024-01-01T00:00:00Z",
        pushed_at="2024-01-02T00:00:00Z",
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


def make_github_team(name: str = "backend") -> GitHubTeam:
    """Create a sample GitHub team."""
    return GitHubTeam(
        id=789,
        node_id="MDQ6VGVhbTc4OQ==",
        name=name,
        slug=name,
        privacy="closed",
        permission="push",
        url=f"https://api.github.com/teams/{name}",
        html_url=f"https://github.com/orgs/octocat/teams/{name}",
        members_url=f"https://api.github.com/teams/{name}/members{{/member}}",
        repositories_url=f"https://api.github.com/teams/{name}/repos",
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
# Test: Model Configuration
# =============================================================================


def test_github_user_extra_fields_are_ignored():
    """Models use extra='ignore': unknown fields from GitHub payloads do not raise."""
    user = GitHubUser.model_validate(
        {
            "id": 1,
            "login": "octocat",
            "type": "User",
            "gravatar_id": "",  # not in model
            "unknown_future_field": True,
        }
    )
    assert user.id == 1


def test_github_change_value_from_alias():
    """GitHubChangeValue maps the JSON key 'from' to the Python attribute 'from_'."""
    change = GitHubChangeValue.model_validate({"from": "old title"})
    assert change.from_ == "old title"


# =============================================================================
# Test: Timestamp Schema Drift
# =============================================================================


def test_github_repository_accepts_integer_timestamps():
    """GitHubRepository accepts Unix epoch integers for created_at and pushed_at."""
    repo = GitHubRepository.model_validate(
        {
            "id": 123,
            "name": "test-repo",
            "full_name": "octocat/test-repo",
            "owner": {"id": 1, "login": "octocat", "type": "User"},
            "created_at": 1609459200,
            "pushed_at": 1609545600,
        }
    )
    assert repo.created_at == 1609459200
    assert repo.pushed_at == 1609545600


def test_github_repository_rejects_boolean_timestamps():
    """GitHubRepository rejects booleans for integer timestamp fields."""
    with pytest.raises(ValidationError) as exc_info:
        GitHubRepository.model_validate(
            {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "created_at": True,
                "pushed_at": False,
            }
        )
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("created_at",) in locs
    assert ("pushed_at",) in locs


def test_github_repository_rejects_float_timestamps():
    """StrictInt does not coerce floats — 1609459200.5 is rejected, not truncated."""
    with pytest.raises(ValidationError) as exc_info:
        GitHubRepository.model_validate(
            {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "created_at": 1609459200.5,
            }
        )
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("created_at",) in locs


def test_github_repository_rejects_missing_created_at():
    """GitHubRepository requires created_at to be present and non-null."""
    with pytest.raises(ValidationError) as exc_info:
        GitHubRepository.model_validate(
            {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "pushed_at": 1609545600,
            }
        )
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("created_at",) in locs


def test_github_repository_rejects_null_created_at():
    """GitHubRepository rejects null created_at per the octokit schema."""
    with pytest.raises(ValidationError) as exc_info:
        GitHubRepository.model_validate(
            {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "created_at": None,
                "pushed_at": 1609545600,
            }
        )
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("created_at",) in locs


def test_github_repository_accepts_null_pushed_at():
    """GitHubRepository accepts null pushed_at when the key is present."""
    repo = GitHubRepository.model_validate(
        {
            "id": 123,
            "name": "test-repo",
            "full_name": "octocat/test-repo",
            "owner": {"id": 1, "login": "octocat", "type": "User"},
            "created_at": 1609459200,
            "pushed_at": None,
        }
    )
    assert repo.pushed_at is None


def test_github_repository_rejects_missing_pushed_at():
    """GitHubRepository requires pushed_at to be present even when it is null."""
    with pytest.raises(ValidationError) as exc_info:
        GitHubRepository.model_validate(
            {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "created_at": 1609459200,
            }
        )
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("pushed_at",) in locs


def test_github_commit_user_accepts_null_email():
    """GitHubCommitUser accepts null email while keeping the field required."""
    from dispatch_agents.integrations.github import GitHubCommitUser

    user = GitHubCommitUser.model_validate({"name": "octocat", "email": None})
    assert user.email is None


def test_github_commit_user_rejects_missing_email():
    """GitHubCommitUser still requires the email key even when null is allowed."""
    from dispatch_agents.integrations.github import GitHubCommitUser

    with pytest.raises(ValidationError) as exc_info:
        GitHubCommitUser.model_validate({"name": "octocat"})
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("email",) in locs


def test_push_payload_accepts_integer_repository_timestamps():
    """Push payloads accept integer repository timestamps without validation errors."""
    payload = Push.model_validate(
        {
            "ref": "refs/heads/main",
            "before": "abc123",
            "after": "def456",
            "created": False,
            "deleted": False,
            "forced": False,
            "compare": "https://github.com/octocat/test-repo/compare/abc123...def456",
            "repository": {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "created_at": 1609459200,
                "pushed_at": 1609545600,
            },
            "pusher": {"name": "octocat", "email": "octocat@example.com"},
            "sender": {"id": 1, "login": "octocat", "type": "User"},
            "commits": [],
        }
    )
    assert payload.repository is not None
    assert payload.repository.created_at == 1609459200
    assert payload.repository.pushed_at == 1609545600


def test_push_payload_accepts_null_pusher_email():
    """Push payloads accept null pusher.email for private-email and bot pushes."""
    payload = Push.model_validate(
        {
            "ref": "refs/heads/main",
            "before": "abc123",
            "after": "def456",
            "created": False,
            "deleted": False,
            "forced": False,
            "compare": "https://github.com/octocat/test-repo/compare/abc123...def456",
            "repository": {
                "id": 123,
                "name": "test-repo",
                "full_name": "octocat/test-repo",
                "owner": {"id": 1, "login": "octocat", "type": "User"},
                "created_at": 1609459200,
                "pushed_at": 1609545600,
            },
            "pusher": {"name": "octocat", "email": None},
            "sender": {"id": 1, "login": "octocat", "type": "User"},
            "commits": [],
        }
    )
    assert payload.pusher.email is None


def test_push_payload_rejects_missing_pusher_email():
    """Push payloads still require pusher.email to be present."""
    with pytest.raises(ValidationError) as exc_info:
        Push.model_validate(
            {
                "ref": "refs/heads/main",
                "before": "abc123",
                "after": "def456",
                "created": False,
                "deleted": False,
                "forced": False,
                "compare": "https://github.com/octocat/test-repo/compare/abc123...def456",
                "repository": {
                    "id": 123,
                    "name": "test-repo",
                    "full_name": "octocat/test-repo",
                    "owner": {"id": 1, "login": "octocat", "type": "User"},
                    "created_at": 1609459200,
                    "pushed_at": 1609545600,
                },
                "pusher": {"name": "octocat"},
                "sender": {"id": 1, "login": "octocat", "type": "User"},
                "commits": [],
            }
        )
    locs = {error["loc"][:2] for error in exc_info.value.errors(include_input=False)}
    assert ("pusher", "email") in locs


def test_push_payload_rejects_boolean_repository_timestamps():
    """Push payloads reject boolean repository timestamps as malformed drift."""
    with pytest.raises(ValidationError) as exc_info:
        Push.model_validate(
            {
                "ref": "refs/heads/main",
                "before": "abc123",
                "after": "def456",
                "created": False,
                "deleted": False,
                "forced": False,
                "compare": "https://github.com/octocat/test-repo/compare/abc123...def456",
                "repository": {
                    "id": 123,
                    "name": "test-repo",
                    "full_name": "octocat/test-repo",
                    "owner": {"id": 1, "login": "octocat", "type": "User"},
                    "created_at": True,
                    "pushed_at": False,
                },
                "pusher": {"name": "octocat", "email": "octocat@example.com"},
                "sender": {"id": 1, "login": "octocat", "type": "User"},
                "commits": [],
            }
        )
    locs = {error["loc"][:2] for error in exc_info.value.errors(include_input=False)}
    assert ("repository", "created_at") in locs
    assert ("repository", "pushed_at") in locs


# =============================================================================
# Test: Review Target XOR Constraint
# =============================================================================


def test_pull_request_review_requested_accepts_user_target():
    """review_requested with only a user target passes the XOR constraint."""
    payload = PullRequestReviewRequested(
        action="review_requested",
        number=42,
        pull_request=make_github_pr(),
        repository=make_github_repo(),
        sender=make_github_user(),
        requested_reviewer=make_github_user("reviewer", 2),
    )
    assert payload.requested_reviewer is not None
    assert payload.requested_team is None


def test_pull_request_review_requested_accepts_team_target():
    """review_requested with only a team target passes the XOR constraint."""
    payload = PullRequestReviewRequested(
        action="review_requested",
        number=42,
        pull_request=make_github_pr(),
        repository=make_github_repo(),
        sender=make_github_user(),
        requested_team=make_github_team(),
    )
    assert payload.requested_reviewer is None
    assert payload.requested_team is not None


@pytest.mark.parametrize(
    ("model_class", "action"),
    [
        (PullRequestReviewRequested, "review_requested"),
        (PullRequestReviewRequestRemoved, "review_request_removed"),
    ],
)
def test_pull_request_review_target_requires_exactly_one_target(
    model_class: type[PullRequestReviewRequested | PullRequestReviewRequestRemoved],
    action: str,
):
    """Review target payloads reject missing and ambiguous targets."""
    base_kwargs = {
        "action": action,
        "number": 42,
        "pull_request": make_github_pr(),
        "repository": make_github_repo(),
        "sender": make_github_user(),
    }

    with pytest.raises(ValidationError) as exc_info:
        model_class.model_validate(base_kwargs)
    assert "Exactly one of" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        model_class.model_validate(
            {
                **base_kwargs,
                "requested_reviewer": make_github_user("reviewer", 2),
                "requested_team": make_github_team(),
            }
        )
    assert "Exactly one of" in str(exc_info.value)


def test_pull_request_review_request_removed_accepts_team_target():
    """review_request_removed with only a team target passes the XOR constraint."""
    payload = PullRequestReviewRequestRemoved(
        action="review_request_removed",
        number=42,
        pull_request=make_github_pr(),
        repository=make_github_repo(),
        sender=make_github_user(),
        requested_team=make_github_team("platform"),
    )
    assert payload.requested_reviewer is None
    assert payload.requested_team is not None


def test_pull_request_review_request_removed_accepts_user_target():
    """review_request_removed with only a user target passes the XOR constraint."""
    payload = PullRequestReviewRequestRemoved(
        action="review_request_removed",
        number=42,
        pull_request=make_github_pr(),
        repository=make_github_repo(),
        sender=make_github_user(),
        requested_reviewer=make_github_user("reviewer", 2),
    )
    assert payload.requested_reviewer is not None
    assert payload.requested_team is None


# =============================================================================
# Test: Payload Validation Errors
# =============================================================================


def test_pr_payload_missing_required_field():
    """Missing required fields raise a ValidationError pointing to those fields."""
    with pytest.raises(ValidationError) as exc_info:
        PullRequestBase(  # type: ignore[call-arg]
            action="opened",
            # Missing: number, pull_request, repository, sender
        )
    assert exc_info.value.error_count() > 0


def test_pr_payload_invalid_field_type():
    """A wrong type for 'number' (str instead of int) raises a ValidationError on that field."""
    with pytest.raises(ValidationError) as exc_info:
        PullRequestBase(
            action="opened",
            number="not_an_int",  # type: ignore[arg-type]
            pull_request=make_github_pr(),
            repository=make_github_repo(),
            sender=make_github_user(),
            organization=make_github_organization(),
        )
    locs = {error["loc"][:1] for error in exc_info.value.errors(include_input=False)}
    assert ("number",) in locs


# =============================================================================
# Test: dispatch_topic()
# =============================================================================


@pytest.mark.parametrize(
    ("event_class_name", "expected_topic"),
    [
        ("PullRequestOpened", "github.pull_request.opened"),
        ("PullRequestClosed", "github.pull_request.closed"),
        ("IssueOpened", "github.issues.opened"),
        ("Push", "github.push"),
        ("IssueCommentCreated", "github.issue_comment.created"),
    ],
)
def test_dispatch_topic(event_class_name: str, expected_topic: str):
    """Each event class returns the correct dispatch topic string."""
    import dispatch_agents.integrations.github as gh

    event_class = getattr(gh, event_class_name)
    assert event_class.dispatch_topic() == expected_topic


# =============================================================================
# Test: Class-Based @on(github_event=...) Decorator
# =============================================================================


def test_on_github_class_single_event():
    """@on(github_event=...) with a single event class registers handler and metadata."""
    from dispatch_agents.integrations.github import PullRequestOpened

    @on(github_event=PullRequestOpened)
    async def handle_pr_class(payload: PullRequestOpened) -> None:
        pass

    assert "handle_pr_class" in REGISTERED_HANDLERS
    assert "github.pull_request.opened" in TOPIC_HANDLERS
    assert "handle_pr_class" in TOPIC_HANDLERS["github.pull_request.opened"]
    metadata = HANDLER_METADATA["handle_pr_class"]
    assert metadata.topics == ["github.pull_request.opened"]


def test_on_github_class_multiple_events():
    """@on(github_event=...) with multiple event classes registers all topics."""
    from dispatch_agents.integrations.github import (
        PullRequestBase,
        PullRequestOpened,
        PullRequestSynchronize,
    )

    @on(github_event=[PullRequestOpened, PullRequestSynchronize])
    async def handle_pr_multi_class(payload: PullRequestBase) -> None:
        pass

    assert "handle_pr_multi_class" in REGISTERED_HANDLERS
    assert "github.pull_request.opened" in TOPIC_HANDLERS
    assert "github.pull_request.synchronize" in TOPIC_HANDLERS
    assert "handle_pr_multi_class" in TOPIC_HANDLERS["github.pull_request.opened"]
    assert "handle_pr_multi_class" in TOPIC_HANDLERS["github.pull_request.synchronize"]
    metadata = HANDLER_METADATA["handle_pr_multi_class"]
    assert "github.pull_request.opened" in metadata.topics
    assert "github.pull_request.synchronize" in metadata.topics


def test_on_github_class_validation_success_base_class():
    """@on accepts a base class as payload type for multiple PR event subclasses."""
    from dispatch_agents.integrations.github import (
        PullRequestBase,
        PullRequestClosed,
        PullRequestOpened,
    )

    @on(github_event=[PullRequestOpened, PullRequestClosed])
    async def handle_pr_base(payload: PullRequestBase) -> None:
        pass

    assert "handle_pr_base" in REGISTERED_HANDLERS


def test_on_github_class_validation_failure():
    """@on raises TypeError when the payload type is incompatible with the event."""
    from dispatch_agents.integrations.github import IssueBase, PullRequestOpened

    with pytest.raises(TypeError, match="is not compatible with"):

        @on(github_event=PullRequestOpened)
        async def bad_handler_class(payload: IssueBase) -> None:
            pass


def test_on_github_class_invalid_type():
    """@on raises TypeError when github_event is not a class."""
    with pytest.raises(TypeError, match="Invalid github_event type"):

        @on(github_event="not_a_class")  # type: ignore[arg-type]
        async def invalid_type_handler(payload: PullRequestBase) -> None:
            pass


# =============================================================================
# Test: Class-Based Handler Dispatch
# =============================================================================


@pytest.mark.asyncio
async def test_on_github_class_dispatch_pr_opened():
    """Dispatching a PR opened event reaches the registered class-based handler."""
    from dispatch_agents.integrations.github import PullRequestOpened

    received_payload = None

    @on(github_event=PullRequestOpened)
    async def handle_pr_dispatch(payload: PullRequestOpened) -> dict:
        nonlocal received_payload
        received_payload = payload
        return {"handled": True, "pr_number": payload.number}

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

    result = await dispatch_message(message)

    assert isinstance(result, SuccessPayload)
    assert result.result == {"handled": True, "pr_number": 123}
    assert received_payload is not None
    assert received_payload.number == 123


@pytest.mark.asyncio
async def test_on_github_class_dispatch_push():
    """Dispatching a push event reaches the registered Push handler."""
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


# =============================================================================
# Test: Regression
# =============================================================================


def test_pull_request_review_comment_validation_error_malformed_user():
    """Malformed user objects in PR review comment payloads raise ValidationError.

    Reproduces an issue where generated payloads have user objects with only
    a 'description' field instead of required 'id' and 'login' fields.
    """
    from dispatch_agents.integrations.github import PullRequestReviewCommentCreated

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
            "created_at": "2024-01-01T00:00:00Z",
            "pushed_at": "2024-01-02T00:00:00Z",
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

    error_str = str(exc_info.value)
    assert "owner" in error_str or "user" in error_str


# =============================================================================
# Test: Spec Alignment
# =============================================================================


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

    # Map our class names to spec definition names using _dispatch_topic.
    # Topic format: "github.pull_request.opened" -> spec key: "pull_request$opened"
    def class_to_spec_key(model_class: type[BaseModel]) -> str:
        """Derive the spec key from the class's _dispatch_topic attribute."""
        topic = model_class._dispatch_topic  # type: ignore[attr-defined]
        suffix = topic.removeprefix("github.")
        # Replace last "." with "$" to match spec key format (event_type$action)
        if "." in suffix:
            event_type, _, action = suffix.rpartition(".")
            return f"{event_type}${action}"
        return suffix

    def get_required_fields(model_class: type[BaseModel]) -> set[str]:
        """Get the required fields for a Pydantic model (excluding ClassVars).

        Uses the field alias when present, since spec keys use wire names (aliases).
        """
        required = set()
        for name, field_info in model_class.model_fields.items():
            # Field is required if it has no default and isn't Optional
            if field_info.is_required():
                # Use alias if defined (spec keys use wire/JSON names)
                required.add(field_info.alias if field_info.alias else name)
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

    # Fields that our per-action classes intentionally require even though the
    # generic spec marks them optional. The spec defines these at the event level
    # (across all actions), but our action-specific classes can safely require
    # fields that are always present for that particular action.
    known_extra_required: dict[str, set[str]] = {
        "IssueLabeled": {"label"},
        "IssueUnlabeled": {"label"},
        "InstallationTargetRenamed": {"repository", "sender"},
        "PullRequestReviewRequested": {
            "pull_request",
            "sender",
            "repository",
            "action",
            "number",
        },
        "PullRequestReviewRequestRemoved": {
            "pull_request",
            "sender",
            "repository",
            "action",
            "number",
        },
    }

    # Track failures
    failures = []

    for class_name, model_class in event_classes:
        spec_key = class_to_spec_key(model_class)
        spec_required = get_spec_required_fields(spec_key)

        if spec_required is None:
            # Skip classes without a matching spec definition
            continue

        model_required = get_required_fields(model_class)

        # Find fields required in our model but not in the spec
        extra_required = model_required - spec_required
        # Exclude known intentional differences
        extra_required -= known_extra_required.get(class_name, set())

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
