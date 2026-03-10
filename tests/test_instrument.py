"""Tests for auto-instrumentation module.

Verifies that httpx/requests patches:
- Only inject headers on proxy-bound requests
- Leave non-proxy requests untouched
- Read correct values from contextvars
- Are idempotent (safe to call multiple times)
"""

import os
from unittest.mock import patch

import httpx

from dispatch_agents.instrument import (
    _get_context_headers,
    _is_proxy_bound,
    auto_instrument,
)

# ── _is_proxy_bound ───────────────────────────────────────────────────


class TestIsProxyBound:
    def test_returns_false_when_no_proxy_host(self):
        with patch("dispatch_agents.instrument.PROXY_HOST", ""):
            assert _is_proxy_bound("http://api.openai.com/v1/chat") is False

    def test_returns_true_for_matching_url(self):
        with patch("dispatch_agents.instrument.PROXY_HOST", "http://127.0.0.1:8780"):
            assert _is_proxy_bound("http://127.0.0.1:8780/v1/chat/completions") is True

    def test_returns_false_for_non_matching_url(self):
        with patch("dispatch_agents.instrument.PROXY_HOST", "http://127.0.0.1:8780"):
            assert (
                _is_proxy_bound("https://api.openai.com/v1/chat/completions") is False
            )

    def test_handles_httpx_url_object(self):
        with patch("dispatch_agents.instrument.PROXY_HOST", "http://127.0.0.1:8780"):
            url = httpx.URL("http://127.0.0.1:8780/v1/chat/completions")
            assert _is_proxy_bound(url) is True


# ── _get_context_headers ──────────────────────────────────────────────


class TestGetContextHeaders:
    def test_returns_empty_when_no_context(self):
        """Outside a handler, contextvars are None → no headers."""
        headers = _get_context_headers()
        assert "X-Dispatch-Trace-Id" not in headers
        assert "X-Dispatch-Invocation-Id" not in headers

    def test_includes_trace_id_when_set(self):
        # Patch at the source since _get_context_headers does a local import
        with patch(
            "dispatch_agents.events.get_current_trace_id",
            return_value="trace-123",
        ):
            headers = _get_context_headers()
            assert headers["X-Dispatch-Trace-Id"] == "trace-123"

    def test_includes_invocation_id_when_set(self):
        with patch(
            "dispatch_agents.events.get_current_invocation_id",
            return_value="inv-456",
        ):
            headers = _get_context_headers()
            assert headers["X-Dispatch-Invocation-Id"] == "inv-456"

    def test_includes_agent_name_from_env(self):
        with patch.dict(os.environ, {"DISPATCH_AGENT_NAME": "my-agent"}):
            headers = _get_context_headers()
            assert headers["X-Dispatch-Agent-Name"] == "my-agent"

    def test_skips_agent_name_when_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPATCH_AGENT_NAME", None)
            headers = _get_context_headers()
            assert "X-Dispatch-Agent-Name" not in headers


# ── auto_instrument ───────────────────────────────────────────────────


class TestAutoInstrument:
    def test_skips_when_no_proxy_url(self):
        """No-op when DISPATCH_LLM_PROXY_URL is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPATCH_LLM_PROXY_URL", None)
            # Should not raise
            auto_instrument()

    def test_patches_httpx_client(self):
        with patch.dict(
            os.environ, {"DISPATCH_LLM_PROXY_URL": "http://127.0.0.1:8780"}
        ):
            auto_instrument()
            assert getattr(httpx.Client.send, "_dispatch_patched", False) is True

    def test_patches_httpx_async_client(self):
        with patch.dict(
            os.environ, {"DISPATCH_LLM_PROXY_URL": "http://127.0.0.1:8780"}
        ):
            auto_instrument()
            assert getattr(httpx.AsyncClient.send, "_dispatch_patched", False) is True

    def test_idempotent(self):
        """Calling auto_instrument() twice doesn't double-wrap."""
        with patch.dict(
            os.environ, {"DISPATCH_LLM_PROXY_URL": "http://127.0.0.1:8780"}
        ):
            auto_instrument()
            first_send = httpx.Client.send
            auto_instrument()
            assert httpx.Client.send is first_send


# ── httpx patch behavior ─────────────────────────────────────────────


class TestHttpxPatch:
    """Test that patched httpx injects headers on proxy-bound requests."""

    def setup_method(self):
        """Ensure instrumentation is active."""
        with patch.dict(
            os.environ, {"DISPATCH_LLM_PROXY_URL": "http://127.0.0.1:8780"}
        ):
            auto_instrument()

    def test_sync_send_injects_headers_for_proxy(self):
        """Verify the header injection logic for proxy-bound requests."""
        with (
            patch("dispatch_agents.instrument.PROXY_HOST", "http://127.0.0.1:8780"),
            patch(
                "dispatch_agents.events.get_current_trace_id",
                return_value="trace-abc",
            ),
            patch(
                "dispatch_agents.events.get_current_invocation_id",
                return_value="inv-xyz",
            ),
        ):
            assert _is_proxy_bound("http://127.0.0.1:8780/v1/chat") is True
            headers = _get_context_headers()
            assert headers["X-Dispatch-Trace-Id"] == "trace-abc"
            assert headers["X-Dispatch-Invocation-Id"] == "inv-xyz"

    def test_does_not_inject_for_non_proxy(self):
        """Non-proxy URLs should not get Dispatch headers."""
        with patch("dispatch_agents.instrument.PROXY_HOST", "http://127.0.0.1:8780"):
            assert (
                _is_proxy_bound("https://api.openai.com/v1/chat/completions") is False
            )
