"""GitHub integration client API."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import httpx

from dispatch_agents._internal.transport import (
    get_auth_headers as _get_auth_headers,
)
from dispatch_agents._internal.transport import (
    get_router_url as _get_router_url,
)
from dispatch_agents.models import GitHubAppToken

__all__ = ["get_github_app_token"]

# Module-level cache for the current agent process.
# The SDK assumes a single agent run does not switch orgs or API keys in-process;
# multi-org access is enforced by the backend, not modeled in this client cache.
_cached_token: tuple[GitHubAppToken, datetime] | None = None

_TOKEN_BUFFER_MINUTES = 5


async def get_github_app_token() -> GitHubAppToken:
    """Return a short-lived GitHub App installation token.

    See :func:`dispatch_agents.integrations.github.get_github_app_token` for
    usage, caching, errors, and examples.
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
