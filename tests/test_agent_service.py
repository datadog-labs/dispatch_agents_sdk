"""Tests for dispatch_agents.agent_service module.

Covers the AgentServiceClient gRPC client without needing a running server.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dispatch_agents.agent_service import AgentServiceClient


class TestAgentServiceClientNotInitialized:
    @pytest.mark.asyncio
    async def test_invoke_raises_without_context(self):
        client = AgentServiceClient("localhost:50051", insecure=True)
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client.invoke(MagicMock())

    @pytest.mark.asyncio
    async def test_health_check_raises_without_context(self):
        client = AgentServiceClient("localhost:50051", insecure=True)
        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client.health_check(MagicMock())


class TestAgentServiceClientInsecure:
    @pytest.mark.asyncio
    @patch("dispatch_agents.agent_service.service_pb2_grpc.AgentServiceStub")
    @patch("dispatch_agents.agent_service.grpc.aio.insecure_channel")
    async def test_creates_insecure_channel(self, mock_insecure, mock_stub_cls):
        mock_channel = AsyncMock()
        mock_insecure.return_value = mock_channel

        client = AgentServiceClient("localhost:50051", insecure=True)
        async with client:
            pass

        mock_insecure.assert_called_once_with("localhost:50051")
        mock_channel.close.assert_called_once()


class TestAgentServiceClientSecure:
    @pytest.mark.asyncio
    @patch("dispatch_agents.agent_service.service_pb2_grpc.AgentServiceStub")
    @patch("dispatch_agents.agent_service.grpc.aio.secure_channel")
    @patch("dispatch_agents.agent_service.grpc.ssl_channel_credentials")
    async def test_default_ssl_credentials(
        self, mock_ssl_creds, mock_secure, mock_stub_cls
    ):
        mock_channel = AsyncMock()
        mock_secure.return_value = mock_channel
        mock_creds = MagicMock()
        mock_ssl_creds.return_value = mock_creds

        client = AgentServiceClient("localhost:50051", insecure=False)
        async with client:
            pass

        mock_ssl_creds.assert_called_once()
        mock_secure.assert_called_once_with("localhost:50051", mock_creds)
        mock_channel.close.assert_called_once()

    @pytest.mark.asyncio
    @patch("dispatch_agents.agent_service.service_pb2_grpc.AgentServiceStub")
    @patch("dispatch_agents.agent_service.grpc.aio.secure_channel")
    async def test_custom_credentials(self, mock_secure, mock_stub_cls):
        mock_channel = AsyncMock()
        mock_secure.return_value = mock_channel
        custom_creds = MagicMock()

        client = AgentServiceClient(
            "localhost:50051", insecure=False, credentials=custom_creds
        )
        async with client:
            pass

        mock_secure.assert_called_once_with("localhost:50051", custom_creds)


class TestAgentServiceClientDelegation:
    @pytest.mark.asyncio
    @patch("dispatch_agents.agent_service.service_pb2_grpc.AgentServiceStub")
    @patch("dispatch_agents.agent_service.grpc.aio.insecure_channel")
    async def test_invoke_delegates_to_stub(self, mock_insecure, mock_stub_cls):
        mock_channel = AsyncMock()
        mock_insecure.return_value = mock_channel

        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_stub.Invoke = AsyncMock(return_value=mock_response)
        mock_stub_cls.return_value = mock_stub

        mock_request = MagicMock()

        client = AgentServiceClient("localhost:50051", insecure=True)
        async with client:
            result = await client.invoke(mock_request, timeout=5.0)

        mock_stub.Invoke.assert_called_once_with(mock_request, timeout=5.0)
        assert result is mock_response

    @pytest.mark.asyncio
    @patch("dispatch_agents.agent_service.service_pb2_grpc.AgentServiceStub")
    @patch("dispatch_agents.agent_service.grpc.aio.insecure_channel")
    async def test_health_check_delegates_to_stub(self, mock_insecure, mock_stub_cls):
        mock_channel = AsyncMock()
        mock_insecure.return_value = mock_channel

        mock_stub = MagicMock()
        mock_response = MagicMock()
        mock_stub.HealthCheck = AsyncMock(return_value=mock_response)
        mock_stub_cls.return_value = mock_stub

        mock_request = MagicMock()

        client = AgentServiceClient("localhost:50051", insecure=True)
        async with client:
            result = await client.health_check(mock_request, timeout=3.0)

        mock_stub.HealthCheck.assert_called_once_with(mock_request, timeout=3.0)
        assert result is mock_response

    @pytest.mark.asyncio
    @patch("dispatch_agents.agent_service.service_pb2_grpc.AgentServiceStub")
    @patch("dispatch_agents.agent_service.grpc.aio.insecure_channel")
    async def test_invoke_without_timeout(self, mock_insecure, mock_stub_cls):
        mock_channel = AsyncMock()
        mock_insecure.return_value = mock_channel

        mock_stub = MagicMock()
        mock_stub.Invoke = AsyncMock(return_value=MagicMock())
        mock_stub_cls.return_value = mock_stub

        client = AgentServiceClient("localhost:50051", insecure=True)
        async with client:
            await client.invoke(MagicMock())

        # timeout defaults to None
        mock_stub.Invoke.assert_called_once()
        _, kwargs = mock_stub.Invoke.call_args
        assert kwargs["timeout"] is None
