"""gRPC client for the AgentService.

This module provides a high-level async client for invoking agents via gRPC.
"""

import grpc

from agentservice.v1 import (
    request_response_pb2,
    service_pb2_grpc,
)

__all__ = [
    "AgentServiceClient",
]


class AgentServiceClient:
    """Async gRPC client for the AgentService.

    This client wraps the generated gRPC stub to provide a clean, typed interface
    for invoking agents.

    Example:
        >>> from agentservice.v1 import request_response_pb2
        >>> from google.protobuf import json_format, struct_pb2
        >>> async with AgentServiceClient("localhost:50051") as client:
        ...     payload = struct_pb2.Value()
        ...     json_format.ParseDict({"query": "Hello"}, payload)
        ...     request = request_response_pb2.InvokeRequest(
        ...         agent_name="my-agent",
        ...         org_id="my-org",
        ...         topic="user.query",
        ...         payload=payload,
        ...         uid="msg-123",
        ...         trace_id="trace-456",
        ...         sender_id="user-1",
        ...         ts="2025-01-01T00:00:00Z",
        ...     )
        ...     response = await client.invoke(request)
        ...     print(response.result)
    """

    def __init__(
        self,
        target: str,
        *,
        insecure: bool = False,
        credentials: grpc.ChannelCredentials | None = None,
    ):
        """Initialize the AgentService client.

        Args:
            target: The gRPC server address (e.g., "localhost:50051")
            insecure: If True, use an insecure channel (for development only)
            credentials: Optional gRPC channel credentials for TLS/mTLS
        """
        self.target = target
        self.insecure = insecure
        self.credentials = credentials
        self._channel: grpc.aio.Channel | None = None
        self._stub: service_pb2_grpc.AgentServiceStub | None = None

    async def __aenter__(self) -> "AgentServiceClient":
        """Enter async context and establish gRPC channel."""
        if self.insecure:
            self._channel = grpc.aio.insecure_channel(self.target)
        else:
            if self.credentials is None:
                # Default to SSL with system root certificates
                self.credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.aio.secure_channel(self.target, self.credentials)

        self._stub = service_pb2_grpc.AgentServiceStub(self._channel)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and close gRPC channel."""
        if self._channel:
            await self._channel.close()
        return False

    async def invoke(
        self,
        request: request_response_pb2.InvokeRequest,
        *,
        timeout: float | None = None,
    ) -> request_response_pb2.InvokeResponse:
        """Invoke an agent and await the result.

        Args:
            request: The InvokeRequest protobuf message
            timeout: Optional timeout in seconds for the RPC call

        Returns:
            InvokeResponse protobuf message containing the agent's result

        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        if self._stub is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with AgentServiceClient(...)' context manager."
            )

        # Make the gRPC call
        response = await self._stub.Invoke(request, timeout=timeout)
        return response

    async def health_check(
        self,
        request: request_response_pb2.HealthCheckRequest,
        *,
        timeout: float | None = None,
    ) -> request_response_pb2.HealthCheckResponse:
        """Check agent health status.

        Args:
            request: The HealthCheckRequest protobuf message
            timeout: Optional timeout in seconds for the RPC call

        Returns:
            HealthCheckResponse protobuf message containing the agent's health status

        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        if self._stub is None:
            raise RuntimeError(
                "Client not initialized. Use 'async with AgentServiceClient(...)' context manager."
            )

        # Make the gRPC call
        response = await self._stub.HealthCheck(request, timeout=timeout)
        return response
