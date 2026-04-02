from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from dispatch_agents.events import _get_auth_headers, _get_router_url

# Module-level cache for the current agent process.
# The SDK assumes a single agent run does not switch orgs or API keys in-process;
# multi-org access is enforced by the backend, not modeled in this client cache.
_cached_token: tuple[GitHubAppToken, datetime] | None = None

_TOKEN_BUFFER_MINUTES = 5


@dataclass
class GitHubAppToken:
    token: str
    expires_at: datetime


async def get_github_app_token() -> GitHubAppToken:
    """Return a GitHub App installation token for use with any HTTP client.

    Fetches a GitHub App installation token from the Dispatch backend. The token
    is cached transparently; subsequent calls return the cached token until it is
    near expiry (< 5 min), at which point a fresh token is fetched automatically.

    Returns:
        GitHubAppToken: Dataclass with ``token`` (str) and
        ``expires_at`` (timezone-aware datetime) fields. Pass ``token`` as a
        Bearer credential to any GitHub API client of your choice.

    Raises:
        RuntimeError: If DISPATCH_API_KEY environment variable is not set.
        RuntimeError: If no GitHub installation is configured for this org (404).
        RuntimeError: If authentication fails — check DISPATCH_API_KEY (401).
        RuntimeError: If access is forbidden (403).
        RuntimeError: If the backend returns an unexpected HTTP status code.

    Note:
        Refresh is lazy and call-time: a new token is fetched on the NEXT call
        after the cached token nears expiry (<5 min remaining). Call
        get_github_app_token() at the start of each handler invocation
        rather than once at module load.

        Network errors (httpx.ConnectError, httpx.TimeoutException) propagate
        as-is and are not caught.

    Examples:
        Using PyGithub::

            from github import Auth, Github
            from dispatch_agents.integrations.github import get_github_app_token

            tok = await get_github_app_token()
            gh = Github(auth=Auth.Token(tok.token))
            repo = gh.get_repo("my-org/my-repo")

        Using httpx directly::

            import httpx
            from dispatch_agents.integrations.github import get_github_app_token

            tok = await get_github_app_token()
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.github.com/repos/my-org/my-repo",
                    headers={"Authorization": f"Bearer {tok.token}"},
                )
    """
    global _cached_token

    if not os.getenv("DISPATCH_API_KEY"):
        raise RuntimeError(
            "DISPATCH_API_KEY environment variable is not set. "
            "GitHub installation token requires an authenticated Dispatch agent."
        )

    if _cached_token is not None:
        cached, expires_at = _cached_token
        if expires_at >= datetime.now(UTC) + timedelta(minutes=_TOKEN_BUFFER_MINUTES):
            return cached

    url = _get_router_url() + "/api/unstable/integrations/github/installation-token"

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            url, headers=_get_auth_headers(), timeout=10.0
        )

    if response.status_code == 401:
        raise RuntimeError(
            "GitHub installation token request failed: unauthorized. "
            "Check that DISPATCH_API_KEY is valid."
        )
    if response.status_code == 403:
        raise RuntimeError("GitHub installation token request failed: forbidden.")
    if response.status_code == 404:
        raise RuntimeError(
            "No GitHub installation found for this organization. "
            "Ensure the GitHub App is installed and configured in Dispatch."
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch GitHub installation token: "
            f"backend returned HTTP {response.status_code}"
        )

    data = response.json()
    token: str = data["token"]
    expires_at = datetime.fromisoformat(data["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)

    result = GitHubAppToken(token=token, expires_at=expires_at)
    _cached_token = (result, expires_at)
    return result
