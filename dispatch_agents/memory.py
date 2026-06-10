"""Memory client API for Dispatch agents.

Use :data:`memory` for the module-level singleton, or instantiate the client
classes directly in tests. Long-term memory is persistent key/value storage.
Short-term memory is session-scoped JSON state.
"""

from __future__ import annotations

import httpx

from dispatch_agents._internal.transport import get_api_base_url as _get_api_base_url
from dispatch_agents._internal.transport import get_auth_headers as _get_auth_headers
from dispatch_agents.config import config as _config
from dispatch_agents.models import JsonObject as _JsonObject
from dispatch_agents.models import KVGetResponse as _KVGetResponse
from dispatch_agents.models import KVListResponse as _KVListResponse
from dispatch_agents.models import MemoryWriteResponse as _MemoryWriteResponse
from dispatch_agents.models import SessionGetResponse as _SessionGetResponse

__all__ = [
    "LongTermMemoryClient",
    "MemoryClient",
    "ShortTermMemoryClient",
    "memory",
]


def _kv_store_payload(
    *,
    agent_name: str,
    key: str,
    value: str | None = None,
) -> _JsonObject:
    payload: _JsonObject = {"agent_name": agent_name, "key": key}
    if value is not None:
        payload["value"] = value
    return payload


def _session_store_payload(
    *,
    agent_name: str,
    session_id: str,
    session_data: _JsonObject | None = None,
) -> _JsonObject:
    payload: _JsonObject = {"agent_name": agent_name, "session_id": session_id}
    if session_data is not None:
        payload["session_data"] = session_data
    return payload


def _get_agent_name(agent_name: str | None = None) -> str:
    """Return explicit agent name or the configured ``agent_name`` fallback."""
    if agent_name is not None:
        return agent_name
    config_agent_name = _config.agent_name
    if not config_agent_name:
        raise ValueError(
            "agent_name not provided and DISPATCH_AGENT_NAME environment variable not set. "
            "Either pass agent_name explicitly or ensure DISPATCH_AGENT_NAME is set."
        )
    return config_agent_name


class LongTermMemoryClient:
    """Long-term memory operations for persistent key/value records.

    The agent name is auto-detected from ``DISPATCH_AGENT_NAME`` when
    ``agent_name`` is not provided explicitly.

    Example::

        from dispatch_agents import memory

        await memory.long_term.add("preference", "dark-mode")
        item = await memory.long_term.get("preference")
        await memory.long_term.delete("preference")
    """

    def __init__(self) -> None:
        self._api_base_url: str | None = None

    def _ensure_api_base_url(self) -> str:
        if self._api_base_url is None:
            self._api_base_url = _get_api_base_url()
        return self._api_base_url

    async def add(
        self, mem_key: str, mem_val: str, *, agent_name: str | None = None
    ) -> _MemoryWriteResponse:
        """Store a value in long-term memory.

        Args:
            mem_key: Memory key.
            mem_val: Memory value.
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = _kv_store_payload(
            agent_name=resolved_agent_name, key=mem_key, value=mem_val
        )
        url = f"{api_base_url}/memory/long-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return _MemoryWriteResponse.model_validate(response.json())

    async def get(
        self, mem_key: str, *, agent_name: str | None = None
    ) -> _KVGetResponse:
        """Read a value from long-term memory.

        Args:
            mem_key: Memory key.
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = _kv_store_payload(agent_name=resolved_agent_name, key=mem_key)
        url = f"{api_base_url}/memory/long-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "GET", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return _KVGetResponse.model_validate(response.json())

    async def delete(
        self, mem_key: str, *, agent_name: str | None = None
    ) -> _MemoryWriteResponse:
        """Delete a value from long-term memory.

        Args:
            mem_key: Memory key.
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = _kv_store_payload(agent_name=resolved_agent_name, key=mem_key)
        url = f"{api_base_url}/memory/long-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "DELETE", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return _MemoryWriteResponse.model_validate(response.json())

    async def list(self, *, agent_name: str | None = None) -> _KVListResponse:
        """List all long-term memory records for an agent.

        Args:
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        url = f"{api_base_url}/memory/long-term/agent/{resolved_agent_name}"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=auth_headers, timeout=10.0)
            response.raise_for_status()
            return _KVListResponse.model_validate(response.json())


class ShortTermMemoryClient:
    """Short-term memory operations for session-scoped JSON state.

    The agent name is auto-detected from ``DISPATCH_AGENT_NAME`` when
    ``agent_name`` is not provided explicitly.

    Example::

        from dispatch_agents import memory

        await memory.short_term.add("thread-123", {"turns": 3})
        session = await memory.short_term.get("thread-123")
        await memory.short_term.delete("thread-123")
    """

    def __init__(self) -> None:
        self._api_base_url: str | None = None

    def _ensure_api_base_url(self) -> str:
        if self._api_base_url is None:
            self._api_base_url = _get_api_base_url()
        return self._api_base_url

    async def add(
        self,
        session_id: str,
        session_data: _JsonObject,
        *,
        agent_name: str | None = None,
    ) -> _MemoryWriteResponse:
        """Store session data in short-term memory.

        Args:
            session_id: Session identifier.
            session_data: JSON object to store.
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = _session_store_payload(
            agent_name=resolved_agent_name,
            session_id=session_id,
            session_data=session_data,
        )
        url = f"{api_base_url}/memory/short-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return _MemoryWriteResponse.model_validate(response.json())

    async def get(
        self, session_id: str, *, agent_name: str | None = None
    ) -> _SessionGetResponse:
        """Read session data from short-term memory.

        Args:
            session_id: Session identifier.
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = _session_store_payload(
            agent_name=resolved_agent_name, session_id=session_id
        )
        url = f"{api_base_url}/memory/short-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "GET", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return _SessionGetResponse.model_validate(response.json())

    async def delete(
        self, session_id: str, *, agent_name: str | None = None
    ) -> _MemoryWriteResponse:
        """Delete session data from short-term memory.

        Args:
            session_id: Session identifier.
            agent_name: Optional agent name. Defaults to ``DISPATCH_AGENT_NAME``.

        Raises:
            ValueError: If ``agent_name`` is omitted and ``DISPATCH_AGENT_NAME`` is unset.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = _session_store_payload(
            agent_name=resolved_agent_name, session_id=session_id
        )
        url = f"{api_base_url}/memory/short-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "DELETE", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return _MemoryWriteResponse.model_validate(response.json())


class MemoryClient:
    """Top-level memory client grouped by retention scope."""

    def __init__(self) -> None:
        self.long_term = LongTermMemoryClient()
        self.short_term = ShortTermMemoryClient()


memory = MemoryClient()
"""Singleton client for Dispatch memory APIs.

Use ``memory.long_term`` for persistent key/value memory and
``memory.short_term`` for session-scoped memory.
"""
