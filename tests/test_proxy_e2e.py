"""End-to-end test for the LLM sidecar proxy.

Starts a real proxy server on a free port and verifies infrastructure:
process startup, health check, backend forwarding behavior, and
auto-instrumentation wiring.
"""

import multiprocessing
import os
import socket
import time
from unittest.mock import patch

import httpx
import pytest

from dispatch_agents.instrument import auto_instrument
from dispatch_agents.proxy.server import (
    ANTHROPIC_MESSAGES_ROUTE,
    OPENAI_CHAT_ROUTE,
    run_server,
)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 5.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            resp = httpx.get(url, timeout=0.5)
            if resp.status_code in (200, 404):
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(0.1)
    return False


class TestProxyE2E:
    """E2E tests with a real proxy subprocess."""

    @pytest.fixture(autouse=True)
    def setup_server(self):
        self.proxy_port = _get_free_port()
        self.proxy_url = f"http://127.0.0.1:{self.proxy_port}"

        self.proxy_process = multiprocessing.Process(
            target=run_server,
            args=(self.proxy_port,),
            daemon=True,
        )
        self.proxy_process.start()

        assert _wait_for_server(f"{self.proxy_url}/health"), (
            "Proxy did not start in time"
        )

        yield

        self.proxy_process.terminate()
        self.proxy_process.join(timeout=2)

    def test_health_check(self):
        resp = httpx.get(f"{self.proxy_url}/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_chat_completions_returns_502_without_backend(self):
        """Without a backend running, proxy returns 502 (connection error)."""
        resp = httpx.post(
            f"{self.proxy_url}{OPENAI_CHAT_ROUTE}",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert resp.status_code == 502
        assert "backend_unavailable" in resp.json()["error"]["code"]

    def test_anthropic_messages_returns_502_without_backend(self):
        """Without a backend running, /v1/messages returns 502."""
        resp = httpx.post(
            f"{self.proxy_url}{ANTHROPIC_MESSAGES_ROUTE}",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )
        assert resp.status_code == 502

    def test_trace_headers_dont_crash(self):
        """Proxy accepts X-Dispatch-* headers without 500 errors."""
        resp = httpx.post(
            f"{self.proxy_url}{OPENAI_CHAT_ROUTE}",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            },
            headers={
                "X-Dispatch-Trace-Id": "e2e-trace-123",
                "X-Dispatch-Invocation-Id": "e2e-inv-456",
                "X-Dispatch-Agent-Name": "e2e-test-agent",
            },
        )
        # 502 because no backend, but NOT 500 from header parsing
        assert resp.status_code == 502


class TestAutoInstrumentWiring:
    def test_instrument_activates_with_proxy_url(self):
        proxy_url = "http://127.0.0.1:9999"
        with patch.dict(os.environ, {"DISPATCH_LLM_PROXY_URL": proxy_url}):
            auto_instrument()

            from dispatch_agents.instrument import _is_proxy_bound

            assert _is_proxy_bound(f"{proxy_url}/v1/chat/completions") is True
            assert (
                _is_proxy_bound("https://api.openai.com/v1/chat/completions") is False
            )

    def test_instrument_is_noop_without_proxy_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPATCH_LLM_PROXY_URL", None)
            auto_instrument()

            from dispatch_agents.instrument import _is_proxy_bound

            assert _is_proxy_bound("http://127.0.0.1:8780/v1/chat") is False
