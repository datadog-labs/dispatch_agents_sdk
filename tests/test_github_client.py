"""Tests for get_github_app_token() in dispatch_agents.integrations.github.client."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest


def _set_sdk_env(monkeypatch) -> None:
    monkeypatch.setenv("DISPATCH_API_KEY", "dak_test_key")
    monkeypatch.setenv("BACKEND_URL", "http://test-backend:8000")


def _install_mock_transport(monkeypatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)


def _response(
    *,
    token: str = "ghs_test_token",
    expires_at: str | None = None,
    status_code: int = 200,
) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "token": token,
            "expires_at": expires_at
            or (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            "installation_id": 12345,
        },
    )


@pytest.fixture(autouse=True)
def reset_github_token_cache():
    import dispatch_agents.integrations.github.client as m

    m._cached_token = None
    yield
    m._cached_token = None


async def test_returns_token_and_calls_backend(monkeypatch):
    _set_sdk_env(monkeypatch)
    requests = []

    def responder(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(token="ghs_fresh_token")

    _install_mock_transport(monkeypatch, responder)

    from dispatch_agents.integrations.github.client import (
        GitHubAppToken,
        get_github_app_token,
    )

    result = await get_github_app_token()

    assert isinstance(result, GitHubAppToken)
    assert result.token == "ghs_fresh_token"
    assert result.expires_at.tzinfo is not None
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert (
        str(request.url)
        == "http://test-backend:8000/api/unstable/integrations/github/installation-token"
    )
    assert request.headers["authorization"] == "Bearer dak_test_key"
    assert request.headers["x-dispatch-client"] == "sdk"
    assert request.headers["x-dispatch-client-version"]


async def test_reuses_cached_token_within_single_run(monkeypatch):
    _set_sdk_env(monkeypatch)
    call_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _response(token="ghs_cached_token")

    _install_mock_transport(monkeypatch, responder)

    from dispatch_agents.integrations.github.client import get_github_app_token

    first = await get_github_app_token()
    second = await get_github_app_token()

    assert first.token == "ghs_cached_token"
    assert second.token == "ghs_cached_token"
    assert call_count == 1


async def test_reuses_cached_token_at_refresh_boundary(monkeypatch):
    import dispatch_agents.integrations.github.client as m

    _set_sdk_env(monkeypatch)
    fixed_now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
    boundary_expiry = fixed_now + timedelta(minutes=m._TOKEN_BUFFER_MINUTES)
    m._cached_token = (
        m.GitHubAppToken(token="ghs_boundary", expires_at=boundary_expiry),
        boundary_expiry,
    )
    call_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _response(token="ghs_refreshed")

    _install_mock_transport(monkeypatch, responder)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    with patch.object(m, "datetime", FrozenDateTime):
        result = await m.get_github_app_token()

    assert result.token == "ghs_boundary"
    assert call_count == 0


async def test_refreshes_near_expiry_token(monkeypatch):
    import dispatch_agents.integrations.github.client as m

    _set_sdk_env(monkeypatch)
    fixed_now = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
    near_expiry = fixed_now + timedelta(minutes=m._TOKEN_BUFFER_MINUTES, seconds=-1)
    m._cached_token = (
        m.GitHubAppToken(token="ghs_old", expires_at=near_expiry),
        near_expiry,
    )
    call_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _response(token="ghs_refreshed")

    _install_mock_transport(monkeypatch, responder)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    with patch.object(m, "datetime", FrozenDateTime):
        result = await m.get_github_app_token()

    assert result.token == "ghs_refreshed"
    assert call_count == 1
    assert m._cached_token is not None
    assert m._cached_token[0].token == "ghs_refreshed"


async def test_missing_api_key_does_not_return_cached_token(monkeypatch):
    import dispatch_agents.integrations.github.client as m

    cached_expiry = datetime.now(UTC) + timedelta(hours=1)
    m._cached_token = (
        m.GitHubAppToken(token="ghs_cached", expires_at=cached_expiry),
        cached_expiry,
    )

    monkeypatch.delenv("DISPATCH_API_KEY", raising=False)
    monkeypatch.setenv("BACKEND_URL", "http://test-backend:8000")
    call_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _response()

    _install_mock_transport(monkeypatch, responder)

    with pytest.raises(RuntimeError, match="DISPATCH_API_KEY"):
        await m.get_github_app_token()

    assert call_count == 0


async def test_normalizes_naive_expiry_timestamp(monkeypatch):
    _set_sdk_env(monkeypatch)

    def responder(request: httpx.Request) -> httpx.Response:
        return _response(expires_at="2026-03-30T12:45:00")

    _install_mock_transport(monkeypatch, responder)

    from dispatch_agents.integrations.github.client import get_github_app_token

    result = await get_github_app_token()

    assert result.expires_at == datetime(2026, 3, 30, 12, 45, tzinfo=UTC)


@pytest.mark.parametrize(
    ("status_code", "expected_message"),
    [
        (401, "DISPATCH_API_KEY"),
        (403, "request failed: forbidden"),
        (404, "No GitHub installation found"),
        (500, "backend returned HTTP 500"),
    ],
)
async def test_error_responses_raise_runtime_error(
    monkeypatch, status_code, expected_message
):
    _set_sdk_env(monkeypatch)
    _install_mock_transport(
        monkeypatch,
        lambda request: _response(status_code=status_code),
    )

    from dispatch_agents.integrations.github.client import get_github_app_token

    with pytest.raises(RuntimeError, match=expected_message):
        await get_github_app_token()
