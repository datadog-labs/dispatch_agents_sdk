"""Event publication API."""

from __future__ import annotations

import uuid

import httpx

from dispatch_agents._internal.dispatch import (
    get_current_invocation_id as _get_current_invocation_id,
)
from dispatch_agents._internal.dispatch import (
    get_current_trace_id as _get_current_trace_id,
)
from dispatch_agents._internal.models import PublishEventBody as _PublishEventBody
from dispatch_agents._internal.transport import get_api_base_url as _get_api_base_url
from dispatch_agents._internal.transport import get_auth_headers as _get_auth_headers
from dispatch_agents.config import config as _config
from dispatch_agents.models import BasePayload as _BasePayload
from dispatch_agents.models import JsonObject as _JsonObject

__all__ = ["emit_event"]


async def emit_event(
    topic: str,
    payload: _BasePayload | _JsonObject,
    sender_id: str | None = None,
) -> str:
    """Publish an event payload to a topic.

    A ``dict`` payload is published as-is. A Pydantic payload model is serialized
    by the dispatch layer and delivered as ``{"data": <model fields>}``.
    Child events automatically inherit the current trace context when emitted
    from inside a handler.

    Args:
        topic: Event topic to publish to.
        payload: Event payload as a dict or :class:`BasePayload` instance.
        sender_id: Optional sender identifier. Defaults to the current agent.

    Returns:
        The unique event ID of the published message.

    Raises:
        httpx.HTTPStatusError: If the backend rejects the publish request.

    Example::

        from dispatch_agents import BasePayload, emit_event, fn

        class UserCreated(BasePayload):
            user_id: str

        @fn()
        async def create_user(payload: UserCreated) -> None:
            await emit_event("user.created", payload)
    """
    if sender_id is None:
        sender_id = _config.agent_name or "unknown-agent"

    # A dict is published as-is; a BasePayload is wrapped as ``{"data": <model>}``
    # and serialized by ``PublishEventBody.model_dump()`` below.
    body_payload: _JsonObject | dict[str, _BasePayload]
    if isinstance(payload, dict):
        body_payload = payload
    else:
        body_payload = {"data": payload}

    event_body = _PublishEventBody(
        topic=topic,
        payload=body_payload,
        sender_id=sender_id,
        trace_id=_get_current_trace_id(),
        parent_id=_get_current_invocation_id(),
    )

    api_base_url = _get_api_base_url()
    auth_headers = _get_auth_headers()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_base_url}/events/publish",
            json=event_body.model_dump(),
            headers=auth_headers,
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("event_uid", str(uuid.uuid4()))
