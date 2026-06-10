"""GitHub integration package for Dispatch agents.

``get_github_app_token`` and ``GitHubAppToken`` are re-exported here for
convenience; typed webhook payload models live in
:mod:`dispatch_agents.integrations.github.events`.
"""

from __future__ import annotations

from dispatch_agents.integrations.github.client import GitHubAppToken
from dispatch_agents.integrations.github.client import (
    get_github_app_token as _get_github_app_token,
)

__all__ = ["GitHubAppToken", "get_github_app_token"]


async def get_github_app_token() -> GitHubAppToken:
    """Return a short-lived GitHub App installation token.

    Fetches a GitHub App installation token from the Dispatch backend for the
    current organization. The token is cached transparently; subsequent calls
    return the cached token until it is near expiry, then fetch a fresh token.

    Call this at the start of each handler invocation rather than once at module
    load time, so refresh happens before the token expires.

    Returns:
        ``GitHubAppToken`` with ``token`` and timezone-aware ``expires_at`` fields.

    Raises:
        RuntimeError: If ``DISPATCH_API_KEY`` is not set.
        RuntimeError: If no GitHub installation is configured for the org.
        RuntimeError: If authentication or authorization fails.
        RuntimeError: If the backend returns an unexpected status.
        httpx.HTTPError: If the network request fails.

    Example::

        from github import Auth, Github
        from dispatch_agents.integrations.github import get_github_app_token

        tok = await get_github_app_token()
        gh = Github(auth=Auth.Token(tok.token))
        repo = gh.get_repo("my-org/my-repo")

    Example::

        import httpx
        from dispatch_agents.integrations.github import get_github_app_token

        tok = await get_github_app_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/repos/my-org/my-repo",
                headers={"Authorization": f"Bearer {tok.token}"},
            )
    """
    return await _get_github_app_token()
