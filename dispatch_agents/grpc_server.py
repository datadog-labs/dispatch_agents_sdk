"""gRPC server implementation for dispatch agents.

This module provides a gRPC server that implements the AgentService interface,
allowing agents to be invoked via gRPC instead of HTTP.
"""

import asyncio
import json
import logging
import os
import signal
import time
from pathlib import Path

import grpc
import httpx
from grpc import aio

from agentservice.v1 import (
    message_pb2,
    request_response_pb2,
    service_pb2_grpc,
)
from dispatch_agents.events import (
    HANDLER_METADATA,
    REGISTERED_HANDLERS,
    TOPIC_HANDLERS,
    dispatch_message,
    run_init_hook,
)
from dispatch_agents.logging_config import get_logger
from dispatch_agents.models import ErrorPayload, FunctionMessage, Message, TopicMessage

logger = get_logger(__name__)


class _SubscribeLogFilter(logging.Filter):
    """Filter to suppress successful subscription heartbeat logs from httpx.

    Only suppresses httpx logs for /events/subscribe that return 200 OK.
    Failed subscriptions (4xx, 5xx) are still logged for debugging.
    All other HTTP requests (emit_event, invoke, memory API) remain visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        # Only suppress successful subscription heartbeats
        if "/events/subscribe" in msg and "200" in msg:
            return False  # Suppress successful subscription logs
        return True  # Allow failures and all other logs


# Apply filter to httpx logger - only suppresses successful subscription heartbeats
logging.getLogger("httpx").addFilter(_SubscribeLogFilter())

# File-based health signal for ECS container health checks.
# Contains a Unix timestamp updated on each successful subscription.
# The ECS health check verifies the timestamp is recent (< 90s old),
# so a stale file from a crashed process won't fool the check.
# Uses /tmp/ which is a writable tmpfs mount on containers with
# readonlyRootFilesystem. The /app/ directory is read-only.
_HEALTH_FILE = Path("/tmp/.dispatch_healthy")


def _mark_healthy() -> None:
    """Best-effort: write current timestamp to health marker file."""
    try:
        _HEALTH_FILE.write_text(str(int(time.time())))
    except OSError as exc:
        logger.warning("Could not write health file %s: %s", _HEALTH_FILE, exc)


def _mark_unhealthy() -> None:
    """Best-effort: remove health marker file."""
    try:
        _HEALTH_FILE.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove health file %s: %s", _HEALTH_FILE, exc)


class AgentServiceServicer(service_pb2_grpc.AgentServiceServicer):
    """Implementation of the AgentService gRPC interface."""

    def __init__(self, agent_name: str):
        """Initialize the servicer.

        Args:
            agent_name: The name of the agent being served
        """
        self.agent_name = agent_name

    async def Invoke(
        self,
        request: request_response_pb2.InvokeRequest,
        context: grpc.aio.ServicerContext,
    ) -> request_response_pb2.InvokeResponse:
        """Invoke an agent function and return the result.

        Args:
            request: The InvokeRequest containing function name and payload
            context: The gRPC context

        Returns:
            InvokeResponse containing the result payload
        """
        logger.info(
            f"Received Invoke request: message_type={request.message_type}, "
            f"topic={request.topic}, function_name={request.function_name}, "
            f"trace_id={request.trace_id}"
        )

        try:
            # Decode the payload from protobuf
            payload_data = json.loads(request.payload.data.decode("utf-8"))

            # Create appropriate message type based on message_type field
            # - "topic": Creates TopicMessage, routes via TOPIC_HANDLERS[topic]
            # - "function": Creates FunctionMessage, routes via REGISTERED_HANDLERS[function_name]
            message: Message
            if request.message_type == "topic":
                message = TopicMessage(
                    topic=request.topic,
                    payload=payload_data,
                    uid=request.uid,
                    trace_id=request.trace_id,
                    sender_id="grpc-client",
                    ts=request.ts,
                    parent_id=None,
                )
            else:
                # Default to function message (for backwards compatibility and direct calls)
                message = FunctionMessage(
                    function_name=request.function_name,
                    payload=payload_data,
                    uid=request.uid,
                    trace_id=request.trace_id,
                    sender_id="grpc-client",
                    ts=request.ts,
                    parent_id=None,
                )

            # Dispatch the message to the appropriate handler
            result = await dispatch_message(message)

            # Serialize the result (SuccessPayload or ErrorPayload) to JSON
            is_error = isinstance(result, ErrorPayload)
            result_payload = message_pb2.Payload(
                metadata={},
                data=json.dumps(result.model_dump()).encode("utf-8"),
            )

            return request_response_pb2.InvokeResponse(
                result=result_payload,
                is_error=is_error,
            )

        except Exception as e:
            logger.error(f"Error processing Invoke request: {e}", exc_info=True)
            # Return error as gRPC status
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Error processing request: {str(e)}",
            )
            # This line won't be reached, but satisfies type checker
            raise

    async def HealthCheck(
        self,
        request: request_response_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> request_response_pb2.HealthCheckResponse:
        """Check the health of the agent.

        Args:
            request: The HealthCheckRequest
            context: The gRPC context

        Returns:
            HealthCheckResponse with serving status
        """
        logger.debug("Received HealthCheck request")
        return request_response_pb2.HealthCheckResponse(
            status=request_response_pb2.HealthCheckResponse.SERVING_STATUS_SERVING
        )


def _is_local_dev_mode() -> bool:
    """Check if running in local development mode.

    Returns True only if DISPATCH_LOCAL_DEV is explicitly set to a truthy value.
    This enables dev-friendly behaviors like auto-shutdown on backend connection failure.

    Note: We use explicit opt-in rather than heuristics (like checking for AWS env vars)
    because localstack Docker containers should behave like production agents, not dev mode.
    """
    local_dev = os.getenv("DISPATCH_LOCAL_DEV", "").lower()
    return local_dev in ("1", "true", "yes")


async def _subscribe_registered_triggers(
    agent_name: str, *, is_initial: bool = False
) -> bool:
    """Subscribe the agent to all registered topics with the backend.

    This function performs a single subscription attempt. For continuous re-subscription,
    use _subscription_loop().

    Args:
        agent_name: The name of the agent to subscribe
        is_initial: Whether this is the initial subscription (affects log level)

    Returns:
        True if subscription succeeded, False if it failed (for local dev tracking)

    Raises:
        RuntimeError: If subscription fails and backend is expected to be available
    """
    topics = list(TOPIC_HANDLERS.keys())

    # Count total handlers (all registered handlers)
    # Topic-based (@on) handlers have topics, callable (@fn) handlers have empty topics
    fn_handlers = [
        name
        for name, meta in HANDLER_METADATA.items()
        if not meta.topics  # @fn handlers have empty topics list
    ]
    total_handlers = len(REGISTERED_HANDLERS)

    if total_handlers == 0:
        logger.info("No registered handlers found; skipping subscription.")
        return True  # No handlers = nothing to subscribe, counts as success

    # Get backend URL - REQUIRED
    backend_url = os.getenv("BACKEND_URL")
    if not backend_url:
        error_msg = (
            "BACKEND_URL environment variable is required but not set. "
            "This should be configured during agent deployment."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Get namespace - optional for backwards compatibility with simple local router
    namespace = os.getenv("DISPATCH_NAMESPACE")
    if namespace:
        # Use namespace-scoped endpoints (backend infrastructure)
        api_base_url = f"{backend_url}/api/unstable/namespace/{namespace}"
    else:
        # Use simple non-namespaced endpoints (local router)
        api_base_url = f"{backend_url}/api/unstable"
        logger.info(
            "DISPATCH_NAMESPACE not set, using non-namespaced endpoints (local router mode)"
        )

    url = f"{api_base_url}/events/subscribe"

    # Get auth headers - API key is required for deployed agents
    api_key = os.getenv("DISPATCH_API_KEY")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    agent_version = os.getenv("DISPATCH_AGENT_VERSION")
    if agent_version:
        headers["X-Dispatch-Agent-Version"] = agent_version
    if api_key:
        # Mask the API key for logging (show first 12 chars only)
        api_key_preview = api_key[:12] + "..." if len(api_key) > 12 else "***"
        headers["Authorization"] = f"Bearer {api_key}"
        logger.debug(f"Using API key for authentication: {api_key_preview}")
    else:
        # In local dev mode, API key is optional - use debug level to reduce noise
        # In production, backend will enforce authentication
        logger.debug(
            "No DISPATCH_API_KEY found in environment (optional for local dev)"
        )

    # Build functions list from unified handler registry
    from dispatch_agents.models import AgentFunction, FunctionTrigger

    functions = []

    # All handlers are in HANDLER_METADATA - build functions list from there
    for handler_name, metadata in HANDLER_METADATA.items():
        handler_topics = metadata.topics

        # Build triggers based on handler type
        triggers = []

        # Add topic triggers for @on handlers
        for topic in handler_topics:
            triggers.append(FunctionTrigger(type="topic", topic=topic))

        # All handlers are callable by name (even @on handlers)
        triggers.append(FunctionTrigger(type="callable", function_name=handler_name))

        function = AgentFunction(
            name=handler_name,
            description=metadata.handler_doc,
            input_schema=metadata.input_schema,
            output_schema=metadata.output_schema,
            triggers=triggers,
        )
        functions.append(function.model_dump())

    payload = {"topics": topics, "agent_name": agent_name, "functions": functions}

    # Log configuration for debugging (use debug for subsequent calls)
    logger.debug(
        f"Subscribing agent '{agent_name}': {len(topics)} topic(s), {len(fn_handlers)} callable function(s)"
    )
    logger.debug(f"Backend URL: {backend_url}")
    logger.debug(f"Subscription endpoint: {url}")
    logger.debug(f"Sending {len(functions)} function(s) with schemas")

    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Sending POST request to {url}")
            response = await client.post(
                url, json=payload, headers=headers, timeout=10.0
            )

            # Log response details for debugging
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")

            response.raise_for_status()

            # Log at INFO for initial subscription, DEBUG for re-subscriptions to reduce noise
            log_msg = f"✅ Successfully subscribed {len(topics)} topic(s) for agent {agent_name}"
            if is_initial:
                logger.info(log_msg)
            else:
                logger.debug(log_msg)

        # Signal healthy to ECS container health check
        _mark_healthy()
        return True  # Success

    except httpx.ConnectError as e:
        _mark_unhealthy()
        error_msg = (
            f"Failed to connect to backend at {url}. "
            f"Connection error: {e}. "
            f"Ensure BACKEND_URL is set correctly and the backend is accessible."
        )
        logger.error(error_msg)
        # Exit with error if we're in a deployed environment (ECS)
        if not _is_local_dev_mode():
            raise RuntimeError(error_msg) from e
        logger.warning("Connection failed (local development mode)")
        return False  # Signal failure for retry tracking

    except httpx.HTTPStatusError as e:
        _mark_unhealthy()
        error_msg = (
            f"Backend returned error status {e.response.status_code} for {url}. "
            f"Response: {e.response.text}"
        )
        logger.error(error_msg)
        # Always exit on HTTP errors (401, 403, 404, 500, etc)
        raise RuntimeError(error_msg) from e

    except httpx.TimeoutException as e:
        _mark_unhealthy()
        error_msg = f"Timeout connecting to backend at {url} after 10s: {e}"
        logger.error(error_msg)
        # Exit with error if we're in a deployed environment
        if not _is_local_dev_mode():
            raise RuntimeError(error_msg) from e
        logger.warning("Connection timed out (local development mode)")
        return False  # Signal failure for retry tracking

    except Exception as e:
        _mark_unhealthy()
        error_msg = (
            f"Unexpected error during subscription to {url}: {type(e).__name__}: {e}"
        )
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


async def _subscription_loop(
    agent_name: str,
    interval_seconds: int = 30,
    *,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Continuously re-subscribe the agent to registered topics.

    This background task ensures that the agent maintains its subscription even if
    the backend server is restarted or becomes temporarily unavailable.

    In local development mode, the agent will exit after 2 consecutive connection
    failures to avoid running indefinitely when the router is not available.

    Args:
        agent_name: The name of the agent to subscribe
        interval_seconds: Time to wait between subscription attempts (default: 30)
        shutdown_event: Optional event to signal shutdown on fatal errors
    """
    logger.info(
        f"Starting subscription loop for agent '{agent_name}' "
        f"(interval: {interval_seconds}s)"
    )

    consecutive_failures = 0
    max_consecutive_failures = 2  # Exit after 2 failures in local dev mode

    # Perform initial subscription immediately
    try:
        success = await _subscribe_registered_triggers(agent_name, is_initial=True)
        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if (
                _is_local_dev_mode()
                and consecutive_failures >= max_consecutive_failures
            ):
                logger.error(
                    f"Failed to connect to backend {consecutive_failures} times in a row. "
                    f"Shutting down. Is the local router running? "
                    f"Start it with: dispatch router start"
                )
                if shutdown_event:
                    shutdown_event.set()
                return
    except Exception as e:
        logger.error(f"Initial subscription failed: {e}")
        consecutive_failures += 1

    # Continuously re-subscribe at the specified interval
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            success = await _subscribe_registered_triggers(agent_name, is_initial=False)
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if (
                    _is_local_dev_mode()
                    and consecutive_failures >= max_consecutive_failures
                ):
                    logger.error(
                        f"Failed to connect to backend {consecutive_failures} times in a row. "
                        f"Shutting down. Is the local router running? "
                        f"Start it with: dispatch router start"
                    )
                    if shutdown_event:
                        shutdown_event.set()
                    return
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Subscription loop cancelled, shutting down")
            raise
        except Exception as e:
            logger.error(f"Error in subscription loop: {e}", exc_info=True)
            consecutive_failures += 1
            if (
                _is_local_dev_mode()
                and consecutive_failures >= max_consecutive_failures
            ):
                logger.error(
                    f"Failed to connect to backend {consecutive_failures} times in a row. "
                    f"Shutting down. Is the local router running? "
                    f"Start it with: dispatch router start"
                )
                if shutdown_event:
                    shutdown_event.set()
                return


async def serve(
    agent_name: str,
    port: int = 50051,
    *,
    insecure: bool = True,
    cert_dir: str | None = None,
    subscription_interval: int = 30,
) -> None:
    """Start the gRPC server for the agent.

    Args:
        agent_name: The name of the agent
        port: The port to listen on (default: 50051)
        insecure: Whether to use an insecure server (default: True for development)
        cert_dir: Directory containing server.crt, server.key, ca.crt for mTLS
        subscription_interval: Seconds between subscription attempts (default: 30)
    """
    server = aio.server()
    servicer = AgentServiceServicer(agent_name=agent_name)
    service_pb2_grpc.add_AgentServiceServicer_to_server(servicer, server)

    listen_addr = f"0.0.0.0:{port}"
    if insecure:
        server.add_insecure_port(listen_addr)
        logger.info(f"Starting insecure gRPC server on {listen_addr}")
    else:
        if not cert_dir:
            raise ValueError("cert_dir is required when insecure=False")
        tls_dir = Path(cert_dir)
        server_key = (tls_dir / "server.key").read_bytes()
        server_cert = (tls_dir / "server.crt").read_bytes()
        ca_cert = (tls_dir / "ca.crt").read_bytes()
        server_creds = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=ca_cert,
            require_client_auth=True,
        )
        server.add_secure_port(listen_addr, server_creds)
        logger.info("Starting mTLS gRPC server on %s", listen_addr)

    await server.start()
    logger.info(f"gRPC server started for agent '{agent_name}' on {listen_addr}")

    # Run @init function before handling any requests
    await run_init_hook()

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    # Start background subscription loop (pass shutdown_event for fatal error handling)
    subscription_task = asyncio.create_task(
        _subscription_loop(
            agent_name, subscription_interval, shutdown_event=shutdown_event
        )
    )

    def signal_handler():
        logger.info("Received shutdown signal, initiating graceful shutdown")
        shutdown_event.set()

    # Register handlers for SIGTERM (container stop) and SIGINT (Ctrl+C)
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        # Wait for either server termination or shutdown signal
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(server.wait_for_termination()),
                asyncio.create_task(shutdown_event.wait()),
            ],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        # If shutdown signal was received, stop the server
        if shutdown_event.is_set():
            logger.info("Stopping gRPC server...")
            await server.stop(grace=5)

    except KeyboardInterrupt:
        # Fallback for systems where signal handlers don't work
        logger.info("Received keyboard interrupt, shutting down")
        await server.stop(grace=5)
    finally:
        # Cancel the subscription loop when server terminates
        subscription_task.cancel()
        try:
            await subscription_task
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
