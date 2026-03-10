"""Tests for dispatch_agents.grpc_server module.

Covers the testable functions without requiring a running gRPC server.
"""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dispatch_agents.grpc_server import (
    AgentServiceServicer,
    _is_local_dev_mode,
    _subscribe_registered_triggers,
    _SubscribeLogFilter,
    _subscription_loop,
)
from dispatch_agents.models import ErrorPayload

# ── _SubscribeLogFilter ──────────────────────────────────────────────


class TestSubscribeLogFilter:
    def setup_method(self):
        self.filt = _SubscribeLogFilter()

    def _make_record(self, msg: str) -> logging.LogRecord:
        record = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        return record

    def test_suppresses_subscribe_200(self):
        record = self._make_record("POST /events/subscribe HTTP/1.1 200 OK")
        assert self.filt.filter(record) is False

    def test_allows_subscribe_500(self):
        record = self._make_record(
            "POST /events/subscribe HTTP/1.1 500 Internal Server Error"
        )
        assert self.filt.filter(record) is True

    def test_allows_other_endpoints(self):
        record = self._make_record("POST /api/unstable/events/emit HTTP/1.1 200 OK")
        assert self.filt.filter(record) is True

    def test_allows_subscribe_without_status(self):
        record = self._make_record("Connecting to /events/subscribe")
        assert self.filt.filter(record) is True


# ── _is_local_dev_mode ───────────────────────────────────────────────


class TestIsLocalDevMode:
    def test_not_set(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOCAL_DEV", raising=False)
        assert _is_local_dev_mode() is False

    def test_true_values(self, monkeypatch):
        for val in ("1", "true", "yes"):
            monkeypatch.setenv("DISPATCH_LOCAL_DEV", val)
            assert _is_local_dev_mode() is True, f"Expected True for {val!r}"

    def test_false_values(self, monkeypatch):
        for val in ("0", "no", "false", ""):
            monkeypatch.setenv("DISPATCH_LOCAL_DEV", val)
            assert _is_local_dev_mode() is False, f"Expected False for {val!r}"


# ── AgentServiceServicer ────────────────────────────────────────────


class TestAgentServiceServicer:
    @pytest.mark.asyncio
    async def test_health_check(self):
        from agentservice.v1 import request_response_pb2

        servicer = AgentServiceServicer(agent_name="test-agent")
        request = request_response_pb2.HealthCheckRequest()
        context = AsyncMock()

        response = await servicer.HealthCheck(request, context)
        assert (
            response.status
            == request_response_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
        )

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.dispatch_message")
    async def test_invoke_function_message(self, mock_dispatch):
        from agentservice.v1 import message_pb2, request_response_pb2
        from dispatch_agents.models import SuccessPayload

        mock_dispatch.return_value = SuccessPayload(result={"answer": 42})

        servicer = AgentServiceServicer(agent_name="test-agent")
        payload = message_pb2.Payload(
            metadata={},
            data=json.dumps({"query": "hello"}).encode(),
        )
        request = request_response_pb2.InvokeRequest(
            function_name="my_func",
            payload=payload,
            uid="uid-1",
            trace_id="trace-1",
            ts="2025-01-01T00:00:00Z",
            message_type="function",
        )
        context = AsyncMock()

        response = await servicer.Invoke(request, context)
        assert response.is_error is False

        # Verify dispatch_message was called with a FunctionMessage
        call_args = mock_dispatch.call_args[0][0]
        assert call_args.function_name == "my_func"
        assert call_args.payload == {"query": "hello"}

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.dispatch_message")
    async def test_invoke_topic_message(self, mock_dispatch):
        from agentservice.v1 import message_pb2, request_response_pb2
        from dispatch_agents.models import SuccessPayload

        mock_dispatch.return_value = SuccessPayload(result={"ok": True})

        servicer = AgentServiceServicer(agent_name="test-agent")
        payload = message_pb2.Payload(
            metadata={},
            data=json.dumps({"data": "event"}).encode(),
        )
        request = request_response_pb2.InvokeRequest(
            topic="user.created",
            payload=payload,
            uid="uid-2",
            trace_id="trace-2",
            ts="2025-01-01T00:00:00Z",
            message_type="topic",
        )
        context = AsyncMock()

        response = await servicer.Invoke(request, context)
        assert response.is_error is False

        call_args = mock_dispatch.call_args[0][0]
        assert call_args.topic == "user.created"
        assert call_args.payload == {"data": "event"}

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.dispatch_message")
    async def test_invoke_error_result(self, mock_dispatch):
        from agentservice.v1 import message_pb2, request_response_pb2

        mock_dispatch.return_value = ErrorPayload(
            error="something failed", error_type="ValueError"
        )

        servicer = AgentServiceServicer(agent_name="test-agent")
        payload = message_pb2.Payload(
            metadata={},
            data=json.dumps({}).encode(),
        )
        request = request_response_pb2.InvokeRequest(
            function_name="failing_func",
            payload=payload,
            uid="uid-3",
            trace_id="trace-3",
            ts="2025-01-01T00:00:00Z",
            message_type="function",
        )
        context = AsyncMock()

        response = await servicer.Invoke(request, context)
        assert response.is_error is True

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.dispatch_message")
    async def test_invoke_handler_exception(self, mock_dispatch):
        from agentservice.v1 import message_pb2, request_response_pb2

        mock_dispatch.side_effect = ValueError("unexpected error")

        servicer = AgentServiceServicer(agent_name="test-agent")
        payload = message_pb2.Payload(
            metadata={},
            data=json.dumps({}).encode(),
        )
        request = request_response_pb2.InvokeRequest(
            function_name="broken_func",
            payload=payload,
            uid="uid-4",
            trace_id="trace-4",
            ts="2025-01-01T00:00:00Z",
            message_type="function",
        )
        context = AsyncMock()
        context.abort = AsyncMock(side_effect=Exception("aborted"))

        with pytest.raises(Exception, match="aborted"):
            await servicer.Invoke(request, context)

        context.abort.assert_called_once()


# ── _subscribe_registered_triggers ───────────────────────────────────


class TestSubscribeRegisteredTriggers:
    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.HANDLER_METADATA", {})
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {})
    async def test_no_handlers_returns_true(self):
        result = await _subscribe_registered_triggers("test-agent")
        assert result is True

    @pytest.mark.asyncio
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_no_backend_url_raises(self, monkeypatch):
        monkeypatch.delenv("BACKEND_URL", raising=False)
        with pytest.raises(RuntimeError, match="BACKEND_URL"):
            await _subscribe_registered_triggers("test-agent")

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_subscribe_success(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await _subscribe_registered_triggers("test-agent")
        assert result is True
        mock_client.post.assert_called_once()

        # Verify URL used non-namespaced endpoint
        call_args = mock_client.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert "/api/unstable/events/subscribe" in url

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=["user.created"],
                handler_doc="doc",
                input_schema={},
                output_schema={},
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {"user.created": ["my_fn"]})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_subscribe_with_namespace(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "my-ns")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await _subscribe_registered_triggers("test-agent")
        assert result is True

        call_args = mock_client.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert "/api/unstable/namespace/my-ns/events/subscribe" in url

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_connect_error_local_dev_returns_false(
        self, mock_client_cls, monkeypatch
    ):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.setenv("DISPATCH_LOCAL_DEV", "1")
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        result = await _subscribe_registered_triggers("test-agent")
        assert result is False

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_connect_error_production_raises(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.delenv("DISPATCH_LOCAL_DEV", raising=False)
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Failed to connect"):
            await _subscribe_registered_triggers("test-agent")

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_http_status_error_raises(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="error status 401"):
            await _subscribe_registered_triggers("test-agent")

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_timeout_error_local_dev_returns_false(
        self, mock_client_cls, monkeypatch
    ):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.setenv("DISPATCH_LOCAL_DEV", "1")
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_cls.return_value = mock_client

        result = await _subscribe_registered_triggers("test-agent")
        assert result is False

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_timeout_error_production_raises(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.delenv("DISPATCH_LOCAL_DEV", raising=False)
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Timeout"):
            await _subscribe_registered_triggers("test-agent")

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server.httpx.AsyncClient")
    @patch(
        "dispatch_agents.grpc_server.HANDLER_METADATA",
        {
            "my_fn": MagicMock(
                topics=[], handler_doc="doc", input_schema={}, output_schema={}
            )
        },
    )
    @patch("dispatch_agents.grpc_server.TOPIC_HANDLERS", {})
    @patch("dispatch_agents.grpc_server.REGISTERED_HANDLERS", {"my_fn": AsyncMock()})
    async def test_unexpected_error_raises(self, mock_client_cls, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://localhost:8080")
        monkeypatch.delenv("DISPATCH_NAMESPACE", raising=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=OSError("unexpected"))
        mock_client_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="Unexpected error"):
            await _subscribe_registered_triggers("test-agent")


# ── _subscription_loop ───────────────────────────────────────────────


class TestSubscriptionLoop:
    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server._subscribe_registered_triggers")
    async def test_initial_success_then_cancel(self, mock_subscribe):
        mock_subscribe.return_value = True

        with pytest.raises(asyncio.CancelledError):
            task = asyncio.create_task(
                _subscription_loop("test-agent", interval_seconds=0)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            await task

        mock_subscribe.assert_called()

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server._is_local_dev_mode", return_value=True)
    @patch("dispatch_agents.grpc_server._subscribe_registered_triggers")
    async def test_consecutive_failures_trigger_shutdown(
        self, mock_subscribe, mock_dev
    ):
        mock_subscribe.return_value = False

        shutdown_event = asyncio.Event()
        await _subscription_loop(
            "test-agent", interval_seconds=0, shutdown_event=shutdown_event
        )
        assert shutdown_event.is_set()

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server._is_local_dev_mode", return_value=True)
    @patch("dispatch_agents.grpc_server._subscribe_registered_triggers")
    async def test_initial_exception_counts_as_failure(self, mock_subscribe, mock_dev):
        # First call raises, second call returns False (in the loop)
        mock_subscribe.side_effect = [RuntimeError("fail"), False]

        shutdown_event = asyncio.Event()
        await _subscription_loop(
            "test-agent", interval_seconds=0, shutdown_event=shutdown_event
        )
        assert shutdown_event.is_set()

    @pytest.mark.asyncio
    @patch("dispatch_agents.grpc_server._is_local_dev_mode", return_value=True)
    @patch("dispatch_agents.grpc_server._subscribe_registered_triggers")
    async def test_loop_exception_triggers_shutdown(self, mock_subscribe, mock_dev):
        # Initial success, then two exceptions in loop
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True
            raise RuntimeError("loop failure")

        mock_subscribe.side_effect = side_effect

        shutdown_event = asyncio.Event()
        await _subscription_loop(
            "test-agent", interval_seconds=0, shutdown_event=shutdown_event
        )
        assert shutdown_event.is_set()
