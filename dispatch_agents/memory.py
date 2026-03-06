import os
from typing import Any

import httpx

from .events import _get_api_base_url, _get_auth_headers
from .models import (
    KVGetResponse,
    KVStoreRequest,
    MemoryWriteResponse,
    SessionGetResponse,
    SessionStoreRequest,
)


def _get_agent_name(agent_name: str | None = None) -> str:
    """Get agent name from argument or DISPATCH_AGENT_NAME environment variable.

    Args:
        agent_name: Optional explicit agent name. If provided, used directly.

    Returns:
        The resolved agent name.

    Raises:
        ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
    """
    if agent_name is not None:
        return agent_name
    env_agent_name = os.environ.get("DISPATCH_AGENT_NAME")
    if not env_agent_name:
        raise ValueError(
            "agent_name not provided and DISPATCH_AGENT_NAME environment variable not set. "
            "Either pass agent_name explicitly or ensure DISPATCH_AGENT_NAME is set."
        )
    return env_agent_name


class LongTermMemoryClient:
    """Long-term memory (KV store) operations: add/get/delete.

    Agent name is auto-detected from the DISPATCH_AGENT_NAME environment variable
    if not explicitly provided.
    """

    def __init__(self) -> None:
        self._api_base_url: str | None = None

    def _ensure_api_base_url(self) -> str:
        """Lazily initialize API base URL when first needed."""
        if self._api_base_url is None:
            self._api_base_url = _get_api_base_url()
        return self._api_base_url

    async def add(
        self, mem_key: str, mem_val: str, *, agent_name: str | None = None
    ) -> MemoryWriteResponse:
        """Add a value to long-term memory.

        Args:
            mem_key: The key of the memory.
            mem_val: The value of the memory.
            agent_name: Optional agent name. If not provided, uses DISPATCH_AGENT_NAME env var.

        Returns:
            The response from the API.

        Raises:
            ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = KVStoreRequest(
            agent_name=resolved_agent_name, key=mem_key, value=mem_val
        ).model_dump()
        url = f"{api_base_url}/memory/long-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return MemoryWriteResponse.model_validate(response.json())

    async def get(
        self, mem_key: str, *, agent_name: str | None = None
    ) -> KVGetResponse:
        """Get a value from long-term memory.

        Args:
            mem_key: The key of the memory.
            agent_name: Optional agent name. If not provided, uses DISPATCH_AGENT_NAME env var.

        Returns:
            The value of the memory.

        Raises:
            ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = KVStoreRequest(
            agent_name=resolved_agent_name, key=mem_key
        ).model_dump()
        url = f"{api_base_url}/memory/long-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "GET", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return KVGetResponse.model_validate(response.json())

    async def delete(
        self, mem_key: str, *, agent_name: str | None = None
    ) -> MemoryWriteResponse:
        """Delete a value from long-term memory.

        Args:
            mem_key: The key of the memory.
            agent_name: Optional agent name. If not provided, uses DISPATCH_AGENT_NAME env var.

        Returns:
            The response from the API.

        Raises:
            ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = KVStoreRequest(
            agent_name=resolved_agent_name, key=mem_key
        ).model_dump()
        url = f"{api_base_url}/memory/long-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "DELETE", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return MemoryWriteResponse.model_validate(response.json())


class ShortTermMemoryClient:
    """Short-term conversational memory (session store): add/get/delete.

    Agent name is auto-detected from the DISPATCH_AGENT_NAME environment variable
    if not explicitly provided.
    """

    def __init__(self) -> None:
        self._api_base_url: str | None = None

    def _ensure_api_base_url(self) -> str:
        """Lazily initialize API base URL when first needed."""
        if self._api_base_url is None:
            self._api_base_url = _get_api_base_url()
        return self._api_base_url

    async def add(
        self,
        session_id: str,
        session_data: dict[str, Any],
        *,
        agent_name: str | None = None,
    ) -> MemoryWriteResponse:
        """Add session data to short-term memory.

        Args:
            session_id: The session identifier.
            session_data: The session data to store.
            agent_name: Optional agent name. If not provided, uses DISPATCH_AGENT_NAME env var.

        Returns:
            The response from the API.

        Raises:
            ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = SessionStoreRequest(
            agent_name=resolved_agent_name,
            session_id=session_id,
            session_data=session_data,
        ).model_dump()
        url = f"{api_base_url}/memory/short-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return MemoryWriteResponse.model_validate(response.json())

    async def get(
        self, session_id: str, *, agent_name: str | None = None
    ) -> SessionGetResponse:
        """Get session data from short-term memory.

        Args:
            session_id: The session identifier.
            agent_name: Optional agent name. If not provided, uses DISPATCH_AGENT_NAME env var.

        Returns:
            The session data.

        Raises:
            ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = SessionStoreRequest(
            agent_name=resolved_agent_name, session_id=session_id
        ).model_dump()
        url = f"{api_base_url}/memory/short-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "GET", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return SessionGetResponse.model_validate(response.json())

    async def delete(
        self, session_id: str, *, agent_name: str | None = None
    ) -> MemoryWriteResponse:
        """Delete session data from short-term memory.

        Args:
            session_id: The session identifier.
            agent_name: Optional agent name. If not provided, uses DISPATCH_AGENT_NAME env var.

        Returns:
            The response from the API.

        Raises:
            ValueError: If agent_name not provided and DISPATCH_AGENT_NAME env var not set.
        """
        resolved_agent_name = _get_agent_name(agent_name)
        api_base_url = self._ensure_api_base_url()
        payload = SessionStoreRequest(
            agent_name=resolved_agent_name, session_id=session_id
        ).model_dump()
        url = f"{api_base_url}/memory/short-term"
        auth_headers = _get_auth_headers()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                "DELETE", url, json=payload, headers=auth_headers, timeout=10.0
            )
            response.raise_for_status()
            return MemoryWriteResponse.model_validate(response.json())


class MemoryClient:
    """Top-level Memory client grouping long-term and short-term memory."""

    def __init__(self) -> None:
        self.long_term = LongTermMemoryClient()
        self.short_term = ShortTermMemoryClient()


# Convenient module-level singleton
memory = MemoryClient()
