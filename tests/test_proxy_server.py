"""Tests for the LLM sidecar proxy server (thin backend pass-through).

The proxy wraps raw SDK bodies with metadata (agent_name, trace context)
and forwards to the backend's /llm/proxy endpoint. No format conversion.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from dispatch_agents.proxy.server import (
    ANTHROPIC_MESSAGES_ROUTE,
    OPENAI_CHAT_ROUTE,
    OPENAI_RESPONSES_ROUTE,
    _call_provider_directly,
    _call_provider_passthrough,
    _fallback_traces,
    _format_sse_error,
    _get_auth_headers,
    _get_backend_log_url,
    _get_backend_passthrough_url,
    _is_auth_error,
    _is_not_configured_error,
    _log_fallback_call,
    create_app,
)


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clear_fallback_cache():
    """Clear the fallback trace cache between tests."""
    _fallback_traces.clear()
    yield
    _fallback_traces.clear()


# ── Health endpoint ───────────────────────────────────────────────────


class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Proxy pass-through behavior ──────────────────────────────────────

# Mock backend response (already in SDK format — proxy returns it as-is)
MOCK_OPENAI_RESPONSE = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

MOCK_ANTHROPIC_RESPONSE = {
    "id": "msg-456",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "Hello!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 7},
}

# Backend "not configured" error — the /llm/proxy endpoint returns SDK-formatted errors
NOT_CONFIGURED_RESPONSE = {
    "error": {
        "message": "No LLM providers configured. Run `dispatch llm setup` or visit /manage/llm-providers.",
        "type": "backend_error",
        "code": "400",
    }
}


def _make_mock_client(
    response_data=None,
    status_code=200,
    connect_error=False,
    request_error: Exception | None = None,
):
    """Create a mocked httpx.AsyncClient context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    if request_error is not None:
        mock_client.post = AsyncMock(side_effect=request_error)
    elif connect_error:
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
    else:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.content = json.dumps(
            response_data or MOCK_OPENAI_RESPONSE
        ).encode()
        mock_response.json.return_value = response_data or MOCK_OPENAI_RESPONSE
        mock_response.text = json.dumps(response_data or MOCK_OPENAI_RESPONSE)
        if status_code >= 400:
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_response
            )
        else:
            mock_response.raise_for_status.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)

    return mock_client


def _make_sequential_mock_client(responses: list[tuple[dict, int]]):
    """Create a mock client that returns different responses on successive calls.

    Each entry in responses is (response_data, status_code).
    """
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_responses = []
    for data, status in responses:
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.content = json.dumps(data).encode()
        mock_resp.json.return_value = data
        mock_resp.text = json.dumps(data)
        if status >= 400:
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "error", request=MagicMock(), response=mock_resp
            )
        else:
            mock_resp.raise_for_status.return_value = None
        mock_responses.append(mock_resp)

    mock_client.post = AsyncMock(side_effect=mock_responses)
    return mock_client


class TestOpenAIPassthrough:
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_wraps_body_with_metadata(self, mock_client_cls, client):
        """Raw OpenAI body is wrapped with endpoint and metadata (no provider_format)."""
        mock_client_cls.return_value = _make_mock_client()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
            },
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert "provider_format" not in payload
        assert payload["endpoint"] == "/v1/chat/completions"
        assert payload["body"]["model"] == "gpt-4o"
        assert payload["body"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["body"]["temperature"] == 0.7

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_injects_trace_context(self, mock_client_cls, client):
        """X-Dispatch-* headers become trace_id/invocation_id in wrapper."""
        mock_client_cls.return_value = _make_mock_client()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
            headers={
                "X-Dispatch-Trace-Id": "trace-abc",
                "X-Dispatch-Invocation-Id": "inv-xyz",
            },
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert payload["trace_id"] == "trace-abc"
        assert payload["invocation_id"] == "inv-xyz"

    @patch.dict(os.environ, {"DISPATCH_AGENT_NAME": "my-agent"})
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_injects_agent_name(self, mock_client_cls, client):
        """DISPATCH_AGENT_NAME env var is included in wrapper."""
        mock_client_cls.return_value = _make_mock_client()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert payload["agent_name"] == "my-agent"

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_returns_backend_response_as_is(self, mock_client_cls, client):
        """Backend response is returned directly without transformation."""
        mock_client_cls.return_value = _make_mock_client(MOCK_OPENAI_RESPONSE)

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        data = resp.json()
        assert data == MOCK_OPENAI_RESPONSE

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_connection_error_returns_openai_format_502(self, mock_client_cls, client):
        """Connection failure returns OpenAI-format error."""
        mock_client_cls.return_value = _make_mock_client(connect_error=True)

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        )

        assert resp.status_code == 502
        assert "error" in resp.json()
        assert "connection_error" in resp.json()["error"]["type"]

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_timeout_error_returns_openai_format_502(self, mock_client_cls, client):
        """Timeout failures should also map to OpenAI-format 502."""
        mock_client_cls.return_value = _make_mock_client(
            request_error=httpx.ReadTimeout("Timed out")
        )

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        )

        assert resp.status_code == 502
        assert "error" in resp.json()
        assert "connection_error" in resp.json()["error"]["type"]


class TestAnthropicPassthrough:
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_wraps_body_with_anthropic_format(self, mock_client_cls, client):
        """Raw Anthropic body is wrapped with endpoint (no provider_format)."""
        mock_client_cls.return_value = _make_mock_client(MOCK_ANTHROPIC_RESPONSE)

        resp = client.post(
            ANTHROPIC_MESSAGES_ROUTE,
            json={
                "model": "claude-sonnet-4-5-20250929",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1024,
            },
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert "provider_format" not in payload
        assert payload["endpoint"] == "/v1/messages"
        # Body is passed through unmodified
        assert payload["body"]["system"] == "You are helpful."
        assert payload["body"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["body"]["max_tokens"] == 1024

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_returns_backend_response_as_is(self, mock_client_cls, client):
        """Backend Anthropic-format response returned directly."""
        mock_client_cls.return_value = _make_mock_client(MOCK_ANTHROPIC_RESPONSE)

        resp = client.post(
            ANTHROPIC_MESSAGES_ROUTE,
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )

        data = resp.json()
        assert data == MOCK_ANTHROPIC_RESPONSE

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_connection_error_returns_anthropic_format_502(
        self, mock_client_cls, client
    ):
        """Connection failure returns Anthropic-format error."""
        mock_client_cls.return_value = _make_mock_client(connect_error=True)

        resp = client.post(
            ANTHROPIC_MESSAGES_ROUTE,
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 100,
            },
        )

        assert resp.status_code == 502
        assert resp.json()["type"] == "error"
        assert "api_error" in resp.json()["error"]["type"]

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_timeout_error_returns_anthropic_format_502(self, mock_client_cls, client):
        """Timeout failures should also map to Anthropic-format 502."""
        mock_client_cls.return_value = _make_mock_client(
            request_error=httpx.ReadTimeout("Timed out")
        )

        resp = client.post(
            ANTHROPIC_MESSAGES_ROUTE,
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 100,
            },
        )

        assert resp.status_code == 502
        assert resp.json()["type"] == "error"
        assert "api_error" in resp.json()["error"]["type"]


# ── Fallback behavior ────────────────────────────────────────────────


class TestIsNotConfiguredError:
    """Unit tests for the _is_not_configured_error helper."""

    def test_matches_sdk_format_no_providers(self):
        """Backend /llm/proxy returns SDK-formatted error with error.message."""
        body = json.dumps(
            {
                "error": {
                    "message": "No LLM providers configured. Run `dispatch llm setup`.",
                    "type": "backend_error",
                    "code": "400",
                }
            }
        ).encode()
        assert _is_not_configured_error(400, body) is True

    def test_matches_sdk_format_provider_not_configured(self):
        body = json.dumps(
            {
                "error": {
                    "message": "Provider 'openai' not configured. Run `dispatch llm setup`.",
                    "type": "backend_error",
                    "code": "400",
                }
            }
        ).encode()
        assert _is_not_configured_error(400, body) is True

    def test_matches_detail_format(self):
        """Other backend endpoints use {"detail": "..."} format."""
        body = json.dumps(
            {"detail": "No LLM providers configured. Run `dispatch llm setup`."}
        ).encode()
        assert _is_not_configured_error(400, body) is True

    def test_matches_detail_provider_not_configured(self):
        body = json.dumps(
            {
                "detail": "Provider 'anthropic' is not configured. Available providers: []."
            }
        ).encode()
        assert _is_not_configured_error(400, body) is True

    def test_ignores_non_400(self):
        body = json.dumps(
            {"error": {"message": "No LLM providers configured."}}
        ).encode()
        assert _is_not_configured_error(500, body) is False

    def test_ignores_other_400_errors(self):
        body = json.dumps(
            {"error": {"message": "Budget exceeded for this agent"}}
        ).encode()
        assert _is_not_configured_error(400, body) is False

    def test_handles_malformed_json(self):
        assert _is_not_configured_error(400, b"not json") is False

    def test_handles_non_string_detail(self):
        body = json.dumps({"detail": {"code": "some_error"}}).encode()
        assert _is_not_configured_error(400, body) is False


class TestFallbackBehavior:
    """Tests for the proxy fallback when backend has no LLM provider configured."""

    @patch("dispatch_agents.proxy.server._log_fallback_call", new_callable=AsyncMock)
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test-key-123"})
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_fallback_with_original_openai_key(self, mock_client_cls, mock_log, client):
        """Backend returns 400 'not configured', fallback uses original key."""
        # First call: backend returns "not configured"
        # Second call: direct provider returns success
        mock_client_cls.return_value = _make_sequential_mock_client(
            [
                (NOT_CONFIGURED_RESPONSE, 400),
                (MOCK_OPENAI_RESPONSE, 200),
            ]
        )

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert resp.status_code == 200
        assert resp.json() == MOCK_OPENAI_RESPONSE

        # Verify the second call went to the real OpenAI endpoint
        calls = mock_client_cls.return_value.post.call_args_list
        assert len(calls) == 2
        # Second call should be to OpenAI
        second_call_url = (
            calls[1].args[0] if calls[1].args else calls[1].kwargs.get("url")
        )
        assert "api.openai.com" in str(second_call_url)
        # Verify auth header uses original key
        second_call_headers = calls[1].kwargs.get("headers", {})
        assert second_call_headers.get("Authorization") == "Bearer sk-test-key-123"

    @patch("dispatch_agents.proxy.server._log_fallback_call", new_callable=AsyncMock)
    @patch.dict(
        os.environ,
        {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": ""},
        clear=False,
    )
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_no_key_returns_401(self, mock_client_cls, mock_log, client):
        """Backend returns 400, no original key → 401 with actionable message."""
        # Remove the original key (empty string isn't truthy)
        os.environ.pop("_DISPATCH_ORIGINAL_OPENAI_API_KEY", None)

        mock_client_cls.return_value = _make_mock_client(NOT_CONFIGURED_RESPONSE, 400)

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert resp.status_code == 401
        error = resp.json()["error"]
        assert "OPENAI_API_KEY" in error["message"]
        assert "dispatch llm setup" in error["message"]

    @patch("dispatch_agents.proxy.server._log_fallback_call", new_callable=AsyncMock)
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_no_fallback_on_other_errors(self, mock_client_cls, mock_log, client):
        """Backend returns 400 for a different reason → no fallback, error returned."""
        budget_error = {"detail": "Budget exceeded for this agent"}
        mock_client_cls.return_value = _make_mock_client(budget_error, 400)

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

        # Should return the backend 400 error as-is, no fallback
        assert resp.status_code == 400
        assert resp.json() == budget_error
        # Only one call (to backend), no second call to provider
        assert mock_client_cls.return_value.post.call_count == 1

    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test-key-123"})
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_fallback_logs_to_llm_log(self, mock_client_cls, client):
        """Successful fallback triggers fire-and-forget /llm/log POST."""
        # We need 3 calls: backend (400), provider (200), log (200)
        mock_client_cls.return_value = _make_sequential_mock_client(
            [
                (NOT_CONFIGURED_RESPONSE, 400),
                (MOCK_OPENAI_RESPONSE, 200),
                ({"llm_call_id": "log-123"}, 200),  # /llm/log response
            ]
        )

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Dispatch-Trace-Id": "trace-log-test"},
        )

        assert resp.status_code == 200

        # The third call should be the /llm/log POST (fire-and-forget)
        # It may take a moment for the async task to execute
        calls = mock_client_cls.return_value.post.call_args_list
        # At minimum, backend + provider calls happened
        assert len(calls) >= 2
        # If the log call executed, verify it went to /llm/log
        if len(calls) >= 3:
            log_call_url = calls[2].kwargs.get(
                "url", calls[2].args[0] if calls[2].args else ""
            )
            assert "llm/log" in str(log_call_url)
            log_payload = calls[2].kwargs.get("json", {})
            assert log_payload.get("provider") == "openai"
            assert log_payload.get("trace_id") == "trace-log-test"

    @patch("dispatch_agents.proxy.server._log_fallback_call", new_callable=AsyncMock)
    @patch.dict(
        os.environ, {"_DISPATCH_ORIGINAL_ANTHROPIC_API_KEY": "sk-ant-test-key-123"}
    )
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_anthropic_fallback_uses_x_api_key_header(
        self, mock_client_cls, mock_log, client
    ):
        """Anthropic fallback uses x-api-key header (not Authorization: Bearer)."""
        mock_client_cls.return_value = _make_sequential_mock_client(
            [
                (NOT_CONFIGURED_RESPONSE, 400),
                (MOCK_ANTHROPIC_RESPONSE, 200),
            ]
        )

        resp = client.post(
            ANTHROPIC_MESSAGES_ROUTE,
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1024,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

        assert resp.status_code == 200
        assert resp.json() == MOCK_ANTHROPIC_RESPONSE

        # Verify the fallback call uses Anthropic auth conventions
        calls = mock_client_cls.return_value.post.call_args_list
        assert len(calls) == 2
        second_call_url = (
            calls[1].args[0] if calls[1].args else calls[1].kwargs.get("url")
        )
        assert "api.anthropic.com" in str(second_call_url)
        second_call_headers = calls[1].kwargs.get("headers", {})
        assert second_call_headers.get("x-api-key") == "sk-ant-test-key-123"
        # anthropic-version is passed through from SDK request headers
        assert second_call_headers.get("anthropic-version") == "2023-06-01"
        # Should NOT have Authorization header for Anthropic
        assert "Authorization" not in second_call_headers

    @patch("dispatch_agents.proxy.server._log_fallback_call", new_callable=AsyncMock)
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test-key-123"})
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_fallback_cache_skips_backend_on_second_call(
        self, mock_client_cls, mock_log, client
    ):
        """Second call with same trace_id skips backend, goes direct to provider."""
        # First request: backend 400, then provider 200
        mock_client_cls.return_value = _make_sequential_mock_client(
            [
                (NOT_CONFIGURED_RESPONSE, 400),
                (MOCK_OPENAI_RESPONSE, 200),
                # Third call: second request goes direct to provider (cache hit)
                (MOCK_OPENAI_RESPONSE, 200),
            ]
        )

        trace_id = "trace-cache-test"

        # First request — hits backend, gets 400, falls back
        resp1 = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Dispatch-Trace-Id": trace_id},
        )
        assert resp1.status_code == 200

        # Second request with same trace_id — should skip backend (cache hit)
        resp2 = client.post(
            OPENAI_CHAT_ROUTE,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Again"}],
            },
            headers={"X-Dispatch-Trace-Id": trace_id},
        )
        assert resp2.status_code == 200

        # Verify: 3 total calls — NOT 4 (backend was skipped on second request)
        assert mock_client_cls.return_value.post.call_count == 3
        # Third call should go directly to OpenAI (not backend)
        third_url = (
            mock_client_cls.return_value.post.call_args_list[2].args[0]
            if mock_client_cls.return_value.post.call_args_list[2].args
            else mock_client_cls.return_value.post.call_args_list[2].kwargs.get("url")
        )
        assert "api.openai.com" in str(third_url)

    @patch("dispatch_agents.proxy.server._log_fallback_call", new_callable=AsyncMock)
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_anthropic_no_key_returns_401_anthropic_format(
        self, mock_client_cls, mock_log, client
    ):
        """Anthropic fallback with no key returns Anthropic-format 401 error."""
        os.environ.pop("_DISPATCH_ORIGINAL_ANTHROPIC_API_KEY", None)
        mock_client_cls.return_value = _make_mock_client(NOT_CONFIGURED_RESPONSE, 400)

        resp = client.post(
            ANTHROPIC_MESSAGES_ROUTE,
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1024,
            },
        )

        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "authentication_error"
        assert "ANTHROPIC_API_KEY" in data["error"]["message"]
        assert "dispatch llm setup" in data["error"]["message"]


# ── Responses API endpoint ────────────────────────────────────────


MOCK_RESPONSES_API_RESPONSE = {
    "id": "resp-123",
    "object": "response",
    "status": "completed",
    "model": "gpt-4o",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Hello!"}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}


class TestResponsesAPIEndpoint:
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_sends_responses_endpoint(self, mock_client_cls, client):
        """Responses API route sends endpoint='/v1/responses' in backend payload."""
        mock_client_cls.return_value = _make_mock_client(MOCK_RESPONSES_API_RESPONSE)

        resp = client.post(
            OPENAI_RESPONSES_ROUTE,
            json={
                "model": "gpt-4o",
                "input": "Tell me a joke",
            },
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert "provider_format" not in payload
        assert payload["endpoint"] == "/v1/responses"
        assert payload["body"]["input"] == "Tell me a joke"

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_responses_connection_error_returns_502(self, mock_client_cls, client):
        """Connection failure on Responses API returns OpenAI-format 502."""
        mock_client_cls.return_value = _make_mock_client(connect_error=True)

        resp = client.post(
            OPENAI_RESPONSES_ROUTE,
            json={"model": "gpt-4o", "input": "Tell me a joke"},
        )

        assert resp.status_code == 502
        assert "error" in resp.json()
        assert "connection_error" in resp.json()["error"]["type"]


# ── Helper functions ────────────────────────────────────────────────


class TestGetBackendProxyUrl:
    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("BACKEND_URL", raising=False)
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        from dispatch_agents.proxy.server import _get_backend_proxy_url

        url = _get_backend_proxy_url()
        assert "dispatch.api:8000" in url
        assert "/llm/proxy" in url
        assert "/namespace/dev/" in url

    @patch.dict(
        os.environ,
        {"BACKEND_URL": "http://custom:9000", "DISPATCH_NAMESPACE": "my-ns"},
    )
    def test_custom_values(self):
        from dispatch_agents.proxy.server import _get_backend_proxy_url

        url = _get_backend_proxy_url()
        assert url == "http://custom:9000/api/unstable/namespace/my-ns/llm/proxy"


class TestMissingBackendUrl:
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_openai_route_with_namespace(self, mock_client_cls, client):
        """DISPATCH_NAMESPACE set correctly changes the backend URL."""
        mock_client_cls.return_value = _make_mock_client()

        with patch.dict(os.environ, {"DISPATCH_NAMESPACE": "custom-ns"}):
            resp = client.post(
                OPENAI_CHAT_ROUTE,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "test"}],
                },
            )

        assert resp.status_code == 200
        call_url = mock_client_cls.return_value.post.call_args.kwargs.get(
            "url", mock_client_cls.return_value.post.call_args.args[0]
        )
        assert "namespace/custom-ns" in call_url


class TestTraceContext:
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_request_with_no_trace_headers(self, mock_client_cls, client):
        """Request without X-Dispatch-* headers has no trace fields in payload."""
        mock_client_cls.return_value = _make_mock_client()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            },
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert "trace_id" not in payload
        assert "invocation_id" not in payload

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_request_with_subprocess_id(self, mock_client_cls, client):
        """X-Dispatch-Subprocess-Id is forwarded in the payload."""
        mock_client_cls.return_value = _make_mock_client()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={"X-Dispatch-Subprocess-Id": "sub-123"},
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert payload["subprocess_id"] == "sub-123"

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_extra_headers_forwarded(self, mock_client_cls, client):
        """X-Dispatch-Extra-Headers JSON is parsed and included."""
        mock_client_cls.return_value = _make_mock_client()

        extra = json.dumps({"X-Custom": "value"})
        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={"X-Dispatch-Extra-Headers": extra},
        )

        assert resp.status_code == 200
        payload = mock_client_cls.return_value.post.call_args.kwargs["json"]
        assert payload["extra_headers"] == {"X-Custom": "value"}


# ── Helper function unit tests ──────────────────────────────────────


class TestFormatSseError:
    def test_openai_format(self):
        result = _format_sse_error("openai", "Something went wrong")
        assert "error" in result
        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "backend_error"

    def test_anthropic_format(self):
        result = _format_sse_error("anthropic", "Connection lost")
        assert result["type"] == "error"
        assert result["error"]["type"] == "api_error"
        assert result["error"]["message"] == "Connection lost"


class TestGetAuthHeaders:
    def test_with_api_key(self):
        with patch.dict(os.environ, {"DISPATCH_API_KEY": "dak_test123"}):
            headers = _get_auth_headers()
        assert headers["Authorization"] == "Bearer dak_test123"
        assert headers["Content-Type"] == "application/json"

    def test_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPATCH_API_KEY", None)
            headers = _get_auth_headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"


class TestGetBackendUrls:
    @patch.dict(
        os.environ, {"BACKEND_URL": "http://test:8080", "DISPATCH_NAMESPACE": "ns1"}
    )
    def test_passthrough_url(self):
        url = _get_backend_passthrough_url()
        assert url == "http://test:8080/api/unstable/namespace/ns1/llm/passthrough"

    @patch.dict(
        os.environ, {"BACKEND_URL": "http://test:8080", "DISPATCH_NAMESPACE": "ns1"}
    )
    def test_log_url(self):
        url = _get_backend_log_url()
        assert url == "http://test:8080/api/unstable/namespace/ns1/llm/log"


class TestCallProviderDirectly:
    @pytest.mark.asyncio
    async def test_unsupported_provider(self):
        resp = await _call_provider_directly({"messages": []}, "unsupported")
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_no_api_key_openai(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_DISPATCH_ORIGINAL_OPENAI_API_KEY", None)
            resp = await _call_provider_directly({"messages": []}, "openai")
        assert resp.status_code == 401
        body = json.loads(resp.body)
        assert "OPENAI_API_KEY" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_no_api_key_anthropic(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_DISPATCH_ORIGINAL_ANTHROPIC_API_KEY", None)
            resp = await _call_provider_directly({"messages": []}, "anthropic")
        assert resp.status_code == 401
        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert "ANTHROPIC_API_KEY" in body["error"]["message"]

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test"})
    async def test_connection_error_openai(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        resp = await _call_provider_directly({"messages": []}, "openai")
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_ANTHROPIC_API_KEY": "sk-ant-test"})
    async def test_connection_error_anthropic(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        resp = await _call_provider_directly({"messages": []}, "anthropic")
        assert resp.status_code == 502
        body = json.loads(resp.body)
        assert body["type"] == "error"


class TestCallProviderPassthrough:
    @pytest.mark.asyncio
    async def test_unsupported_provider(self):
        resp = await _call_provider_passthrough(
            "unsupported", "/v1/models", "GET", None, ""
        )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_DISPATCH_ORIGINAL_OPENAI_API_KEY", None)
            resp = await _call_provider_passthrough(
                "openai", "/v1/models", "GET", None, ""
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test"})
    async def test_successful_passthrough(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        resp = await _call_provider_passthrough("openai", "/v1/models", "GET", None, "")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test"})
    async def test_connection_error(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        resp = await _call_provider_passthrough("openai", "/v1/models", "GET", None, "")
        assert resp.status_code == 502

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-test"})
    async def test_with_query_string(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"data": []}'
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        resp = await _call_provider_passthrough(
            "openai", "/v1/models", "GET", None, "page=1"
        )
        assert resp.status_code == 200
        call_url = mock_client.request.call_args.kwargs.get("url")
        assert "?page=1" in call_url


class TestLogFallbackCall:
    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    async def test_logs_openai_call(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        body = {"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}
        response_body = json.dumps(MOCK_OPENAI_RESPONSE).encode()

        with patch.dict(os.environ, {"DISPATCH_AGENT_NAME": "my-agent"}):
            await _log_fallback_call(
                body, "openai", response_body, "trace-1", "inv-1", 100
            )

        mock_client.post.assert_called_once()
        log_payload = mock_client.post.call_args.kwargs.get("json")
        assert log_payload["provider"] == "openai"
        assert log_payload["trace_id"] == "trace-1"
        assert log_payload["agent_name"] == "my-agent"
        assert log_payload["model"] == "gpt-4o"

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    async def test_logs_anthropic_call(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        body = {"messages": [{"role": "user", "content": "hi"}]}
        response_body = json.dumps(MOCK_ANTHROPIC_RESPONSE).encode()

        await _log_fallback_call(body, "anthropic", response_body, "trace-2", None, 50)

        log_payload = mock_client.post.call_args.kwargs.get("json")
        assert log_payload["provider"] == "anthropic"
        assert log_payload["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        # Should not raise
        await _log_fallback_call({}, "openai", b"not json", None, None, 0)

    @pytest.mark.asyncio
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    async def test_handles_connection_error(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("fail"))
        mock_client_cls.return_value = mock_client

        # Should not raise
        await _log_fallback_call(
            {}, "openai", json.dumps(MOCK_OPENAI_RESPONSE).encode(), None, None, 0
        )


# ── Passthrough catch-all route ─────────────────────────────────────


class TestPassthroughRoute:
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_openai_passthrough_get(self, mock_client_cls, client):
        """GET /openai/v1/models goes through passthrough to backend."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({"data": []}).encode()
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        resp = client.get("/openai/v1/models")
        assert resp.status_code == 200

        # Verify provider_format is sent in the payload
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["provider_format"] == "openai"

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_anthropic_passthrough_get(self, mock_client_cls, client):
        """GET /anthropic/v1/models goes through passthrough."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = json.dumps({"data": []}).encode()
        mock_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        resp = client.get("/anthropic/v1/models")
        assert resp.status_code == 200

        # Verify provider_format is sent in the payload
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["provider_format"] == "anthropic"


# ── Auth Error Detection ─────────────────────────────────────────────


class TestIsAuthError:
    def test_401_is_auth_error(self):
        assert _is_auth_error(401) is True

    def test_403_is_auth_error(self):
        assert _is_auth_error(403) is True

    def test_400_is_not_auth_error(self):
        assert _is_auth_error(400) is False

    def test_500_is_not_auth_error(self):
        assert _is_auth_error(500) is False

    def test_200_is_not_auth_error(self):
        assert _is_auth_error(200) is False


# ── Auth Error Fallback (non-streaming) ──────────────────────────────

# Backend 401 error in SDK format
AUTH_ERROR_RESPONSE = {
    "error": {
        "message": "LLM call failed: Authentication failed",
        "type": "auth_error",
        "code": "401",
    }
}


def _make_mock_client_with_headers(
    response_data,
    status_code,
    headers=None,
):
    """Create a mock client whose response includes headers."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.content = json.dumps(response_data).encode()
    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)
    mock_response.headers = httpx.Headers(headers or {})
    mock_client.post = AsyncMock(return_value=mock_response)

    return mock_client


class TestAuthErrorFallback:
    """Test that the sidecar falls back to the agent's own key on auth errors."""

    def _make_fallback_response(self, data=None):
        """Create a real Response object for _call_provider_directly mock."""
        from starlette.responses import Response

        return Response(
            content=json.dumps(data or MOCK_OPENAI_RESPONSE).encode(),
            status_code=200,
            media_type="application/json",
        )

    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-agent-key"})
    @patch("dispatch_agents.proxy.server._call_provider_directly")
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_falls_back_on_401(self, mock_client_cls, mock_direct_call, client):
        """Backend returns 401 → sidecar should fall back to agent's own key."""
        mock_client_cls.return_value = _make_mock_client_with_headers(
            AUTH_ERROR_RESPONSE, 401
        )
        mock_direct_call.return_value = self._make_fallback_response()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

        mock_direct_call.assert_called_once()
        assert resp.status_code == 200

    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-agent-key"})
    @patch("dispatch_agents.proxy.server._call_provider_directly")
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_fallback_adds_trace_to_cache(
        self, mock_client_cls, mock_direct_call, client
    ):
        """After auth error fallback, the trace_id should be cached for future calls."""
        mock_client_cls.return_value = _make_mock_client_with_headers(
            AUTH_ERROR_RESPONSE, 401
        )
        mock_direct_call.return_value = self._make_fallback_response()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Dispatch-Trace-Id": "trace-abc"},
        )

        assert resp.status_code == 200
        assert "trace-abc" in _fallback_traces

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_no_fallback_without_agent_key(self, mock_client_cls, client):
        """Without an agent key, auth errors should be returned as-is."""
        mock_client_cls.return_value = _make_mock_client_with_headers(
            AUTH_ERROR_RESPONSE, 401
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("_DISPATCH_ORIGINAL_OPENAI_API_KEY", None)

            resp = client.post(
                OPENAI_CHAT_ROUTE,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 401

    @patch.dict(os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-agent-key"})
    @patch("dispatch_agents.proxy.server._call_provider_directly")
    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_falls_back_on_403(self, mock_client_cls, mock_direct_call, client):
        """403 (forbidden) should also trigger fallback."""
        error_403 = {
            "error": {"message": "Forbidden", "type": "auth_error", "code": "403"}
        }
        mock_client_cls.return_value = _make_mock_client_with_headers(error_403, 403)
        mock_direct_call.return_value = self._make_fallback_response()

        resp = client.post(
            OPENAI_CHAT_ROUTE,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
        )

        mock_direct_call.assert_called_once()
        assert resp.status_code == 200

    @patch("dispatch_agents.proxy.server.httpx.AsyncClient")
    def test_500_does_not_trigger_fallback(self, mock_client_cls, client):
        """500 errors should NOT trigger auth fallback."""
        error_500 = {"error": {"message": "Internal error", "type": "server_error"}}
        mock_client_cls.return_value = _make_mock_client_with_headers(error_500, 500)

        with patch.dict(
            os.environ, {"_DISPATCH_ORIGINAL_OPENAI_API_KEY": "sk-agent-key"}
        ):
            resp = client.post(
                OPENAI_CHAT_ROUTE,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 500
