"""Cross-agent function invocation API."""

from __future__ import annotations

import asyncio
import uuid
from typing import TypeVar, overload

import httpx
from pydantic import BaseModel

from dispatch_agents._internal.dispatch import (
    get_current_invocation_id as _get_current_invocation_id,
)
from dispatch_agents._internal.dispatch import (
    get_current_trace_id as _get_current_trace_id,
)
from dispatch_agents._internal.models import (
    InvokeFunctionRequest as _InvokeFunctionRequest,
)
from dispatch_agents._internal.transport import get_api_base_url as _get_api_base_url
from dispatch_agents._internal.transport import get_auth_headers as _get_auth_headers
from dispatch_agents.models import BasePayload as _BasePayload
from dispatch_agents.models import InvocationResult as _InvocationResult
from dispatch_agents.models import JsonObject as _JsonObject

__all__ = ["invoke"]

# Bind to the public payload base so a concrete ``response_model`` is preserved
# in the return type (e.g. ``invoke(..., response_model=Weather) -> Weather``).
_ResponseT = TypeVar("_ResponseT", bound=_BasePayload)


@overload
async def invoke(
    agent_name: str,
    function_name: str,
    payload: _BasePayload | _JsonObject,
    *,
    response_model: type[_ResponseT],
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> _ResponseT: ...


@overload
async def invoke(
    agent_name: str,
    function_name: str,
    payload: _BasePayload | _JsonObject,
    *,
    response_model: None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> _InvocationResult: ...


async def invoke(
    agent_name: str,
    function_name: str,
    payload: _BasePayload | _JsonObject,
    *,
    response_model: type[_ResponseT] | None = None,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> _ResponseT | _InvocationResult:
    """Call a function on another agent and await the response.

    The target agent must register the function with
    :func:`dispatch_agents.handlers.fn`. The SDK starts the invocation, polls
    until completion or failure, and returns either a raw
    :class:`dispatch_agents.models.InvocationResult` or a validated
    ``response_model``.

    Args:
        agent_name: Target agent name.
        function_name: Registered function name on the target agent.
        payload: Input payload as a dict or :class:`BasePayload` instance.
        response_model: Optional Pydantic model used to validate and parse the
            response. When supplied, the return type is that model.
        timeout: Maximum time to wait for completion, in seconds.
        poll_interval: Time between status checks, in seconds.

    Returns:
        ``InvocationResult`` when ``response_model`` is omitted, otherwise an
        instance of ``response_model``.

    Raises:
        httpx.HTTPStatusError: If the backend returns an HTTP error.
        RuntimeError: If the remote invocation fails.
        TimeoutError: If the invocation does not complete before ``timeout``.
        pydantic.ValidationError: If ``response_model`` validation fails.

    Example::

        from dispatch_agents import BasePayload, invoke

        class WeatherResponse(BasePayload):
            temperature: int

        result = await invoke(
            "weather-agent",
            "get_weather",
            {"city": "NYC"},
            response_model=WeatherResponse,
        )
        print(result.temperature)
    """
    payload_dict = payload.model_dump() if isinstance(payload, BaseModel) else payload
    trace_id = _get_current_trace_id() or str(uuid.uuid4())
    parent_id = _get_current_invocation_id()

    invoke_request = _InvokeFunctionRequest(
        agent_name=agent_name,
        function_name=function_name,
        payload=payload_dict,
        trace_id=trace_id,
        parent_id=parent_id,
        timeout_seconds=int(timeout),
    )
    invoke_body = invoke_request.model_dump(exclude_none=True)

    api_base_url = _get_api_base_url()
    auth_headers = _get_auth_headers()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_base_url}/invoke",
            json=invoke_body,
            headers=auth_headers,
            timeout=10.0,
        )
        response.raise_for_status()
        start_result = response.json()
        invocation_id = start_result["invocation_id"]

        loop = asyncio.get_running_loop()
        start_time = loop.time()
        while True:
            elapsed = loop.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Invocation {invocation_id} did not complete within {timeout}s"
                )

            status_response = await client.get(
                f"{api_base_url}/invoke/{invocation_id}",
                headers=auth_headers,
                timeout=10.0,
            )
            status_response.raise_for_status()
            status = status_response.json()

            if status["status"] == "completed":
                result = status.get("result") or {}
                if response_model is not None:
                    return response_model.model_validate(result)
                return _InvocationResult(result=result)

            if status["status"] == "error":
                raise RuntimeError(
                    f"Invoke failed: {status.get('error', 'Unknown error')}"
                )

            await asyncio.sleep(poll_interval)
