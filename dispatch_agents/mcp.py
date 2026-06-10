"""MCP client utilities for Dispatch agents.

These helpers construct MCP server configuration from ``dispatch.yaml`` (plus any
user-provided ``.mcp.json``) and open typed clients that automatically attach
Dispatch trace context to tool calls. Server configs are built in memory at
runtime from environment variables -- no config file is written to disk.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mcp import ClientSession as _ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

import dispatch_agents.models as _models
from dispatch_agents._internal.dispatch import (
    get_current_invocation_id,
    get_current_trace_id,
    get_invocation_id_for_trace,
)

if TYPE_CHECKING:
    from dispatch_agents.models import (
        JsonObject,
        MCPGetPromptResult,
        McpHttpServerConfig,
        MCPListPromptsResult,
        MCPListResourcesResult,
        MCPListToolsResult,
        MCPReadResourceResult,
        MCPToolCallResult,
    )

__all__ = ["MCPClient", "get_mcp_client", "get_mcp_servers_config"]

MCP_CONFIG_PATH = os.environ.get("MCP_CONFIG_PATH", ".mcp.json")


class MCPClient(Protocol):
    """Typed MCP client interface returned by :func:`get_mcp_client`.

    Instances are yielded by the ``get_mcp_client`` async context manager and
    should not be constructed directly.
    """

    async def call_tool(
        self,
        name: str,
        arguments: JsonObject | None = None,
        *,
        read_timeout_seconds: float | None = None,
        meta: JsonObject | None = None,
    ) -> MCPToolCallResult:
        """Call a tool with automatic Dispatch trace context.

        Args:
            name: Tool name.
            arguments: Optional JSON object passed as tool arguments.
            read_timeout_seconds: Optional per-call read timeout override.
            meta: Optional MCP ``_meta`` fields. Dispatch trace fields are merged
                in automatically when available.
        """

    async def list_tools(self) -> MCPListToolsResult:
        """List available tools from the MCP server."""

    async def list_resources(self) -> MCPListResourcesResult:
        """List available resources from the MCP server."""

    async def read_resource(self, uri: str) -> MCPReadResourceResult:
        """Read a resource from the MCP server.

        Args:
            uri: Resource URI returned by :meth:`list_resources`.
        """

    async def list_prompts(self) -> MCPListPromptsResult:
        """List available prompts from the MCP server."""

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> MCPGetPromptResult:
        """Get a prompt from the MCP server.

        Args:
            name: Prompt name.
            arguments: Optional string arguments for prompt rendering.
        """


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self, *, by_alias: bool = False, mode: str = "python") -> object: ...


def _build_trace_meta() -> _models.JsonObject | None:
    """Build trace context metadata for MCP _meta field."""
    meta: _models.JsonObject = {}

    trace_id = get_current_trace_id()
    invocation_id = get_current_invocation_id()

    if not invocation_id and trace_id:
        invocation_id = get_invocation_id_for_trace(trace_id)

    if trace_id:
        meta["dispatch_trace_id"] = trace_id
    if invocation_id:
        meta["dispatch_invocation_id"] = invocation_id

    return meta if meta else None


def _to_json_value(value: object) -> _models.JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode(errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_to_json_value(item) for item in value]
    return str(value)


def _mcp_model_dump(value: object) -> _models.JsonObject:
    """Convert third-party MCP SDK objects into JSON-compatible objects."""
    if isinstance(value, _ModelDumpable):
        value = value.model_dump(by_alias=True, mode="json")

    json_value = _to_json_value(value)
    if isinstance(json_value, dict):
        return json_value
    return {"data": json_value}


def _wrap_mcp_items(items: object) -> list[_models.JsonObject]:
    json_value = _to_json_value(items)
    if not isinstance(json_value, list):
        return []

    wrapped: list[_models.JsonObject] = []
    for item in json_value:
        if isinstance(item, dict):
            wrapped.append(item)
        else:
            wrapped.append({"data": item})
    return wrapped


def _extract_object_list(
    data: _models.JsonObject, key: str
) -> list[_models.JsonObject]:
    return _wrap_mcp_items(data.get(key))


def _extract_string_map(value: _models.JsonValue | None) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(item, str)}


class _TracingMCPClient:
    """MCP client implementation that injects Dispatch trace metadata."""

    def __init__(self, session: _ClientSession) -> None:
        self._session = session

    async def call_tool(
        self,
        name: str,
        arguments: JsonObject | None = None,
        *,
        read_timeout_seconds: float | None = None,
        meta: JsonObject | None = None,
    ) -> MCPToolCallResult:
        trace_meta = _build_trace_meta()
        merged_meta: _models.JsonObject | None = meta
        if trace_meta:
            merged: _models.JsonObject = {}
            for key, value in trace_meta.items():
                merged[key] = value
            if meta:
                merged.update(meta)
            merged_meta = merged

        result = await self._session.call_tool(
            name=name,
            arguments=arguments,
            read_timeout_seconds=(
                timedelta(seconds=read_timeout_seconds)
                if read_timeout_seconds is not None
                else None
            ),
            meta=merged_meta,
        )
        return _models.MCPToolCallResult.model_validate(_mcp_model_dump(result))

    async def list_tools(self) -> MCPListToolsResult:
        result = await self._session.list_tools()
        data = _mcp_model_dump(result)
        return _models.MCPListToolsResult.model_validate(
            {**data, "tools": _extract_object_list(data, "tools")}
        )

    async def list_resources(self) -> MCPListResourcesResult:
        result = await self._session.list_resources()
        data = _mcp_model_dump(result)
        resources = [
            _models.MCPResource(data=resource)
            for resource in _extract_object_list(data, "resources")
        ]
        return _models.MCPListResourcesResult(resources=resources)

    async def read_resource(self, uri: str) -> MCPReadResourceResult:
        result = await self._session.read_resource(AnyUrl(uri))
        data = _mcp_model_dump(result)
        return _models.MCPReadResourceResult.model_validate(
            {**data, "contents": _extract_object_list(data, "contents")}
        )

    async def list_prompts(self) -> MCPListPromptsResult:
        result = await self._session.list_prompts()
        data = _mcp_model_dump(result)
        prompts = [{"data": prompt} for prompt in _extract_object_list(data, "prompts")]
        return _models.MCPListPromptsResult.model_validate({"prompts": prompts})

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> MCPGetPromptResult:
        result = await self._session.get_prompt(name, arguments)
        return _models.MCPGetPromptResult(data=_mcp_model_dump(result))


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` references in strings.

    Strings are expanded using the process environment. If a variable is not set
    and no default is provided, the original placeholder is left intact (never
    silently replaced with an empty string). Dicts and lists are expanded
    recursively; all other types are returned as-is.
    """
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            inner = match.group(1)
            if ":-" in inner:
                var, default = inner.split(":-", 1)
                return os.environ.get(var, default)
            return os.environ.get(inner, match.group(0))

        return re.sub(r"\$\{([^}]+)\}", replace, value)

    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]

    return value


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is required when dispatch.yaml declares mcp_servers."
        )
    return value


def _get_dispatch_mcp_servers() -> dict[str, Any]:
    """Build MCP server configs for servers declared in ``dispatch.yaml``.

    Constructs URLs and auth headers from runtime environment variables
    (``DISPATCH_MCP_GATEWAY_URL``, ``DISPATCH_NAMESPACE``, ``DISPATCH_API_KEY``).
    No file I/O.
    """
    from dispatch_agents.config import _load_runtime_config

    cfg = _load_runtime_config()
    if not cfg.mcp_servers:
        return {}

    gateway = _require_env("DISPATCH_MCP_GATEWAY_URL")
    namespace = os.environ.get("DISPATCH_NAMESPACE") or cfg.namespace
    if not namespace:
        raise RuntimeError(
            "DISPATCH_NAMESPACE is required when dispatch.yaml declares mcp_servers."
        )
    api_key = _require_env("DISPATCH_API_KEY")

    return {
        s.server: {
            "url": f"{gateway}/api/v1/mcp/namespaces/{namespace}/proxy/{s.server}",
            "headers": {"Authorization": f"Bearer {api_key}"},
        }
        for s in cfg.mcp_servers
    }


def _get_user_mcp_servers() -> dict[str, Any]:
    """Load user-provided ``.mcp.json`` servers if the file exists.

    Environment variable references (``${VAR}``, ``${VAR:-default}``) are expanded
    in memory at load time. The file is never modified.
    """
    if not os.path.exists(MCP_CONFIG_PATH):
        return {}
    with open(MCP_CONFIG_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    expanded = _expand_env_vars(raw)
    servers = expanded.get("mcpServers", {}) if isinstance(expanded, dict) else {}
    return servers if isinstance(servers, dict) else {}


def _get_merged_mcp_servers() -> dict[str, Any]:
    """Merge dispatch-managed servers with user-provided ``.mcp.json`` servers.

    Dispatch-managed servers come from the ``mcp_servers`` list in
    ``dispatch.yaml``; their URLs and credentials are constructed from runtime
    environment variables. User-provided servers (from ``.mcp.json``) may extend
    the config additively but must not shadow dispatch-managed server names.
    """
    dispatch_servers = _get_dispatch_mcp_servers()
    user_servers = _get_user_mcp_servers()

    duplicate_names = sorted(dispatch_servers.keys() & user_servers.keys())
    if duplicate_names:
        names = ", ".join(duplicate_names)
        raise ValueError(
            f"User .mcp.json duplicates dispatch-managed MCP server name(s): {names}. "
            "Rename the user-provided server or remove the duplicate entry."
        )

    return {**dispatch_servers, **user_servers}


def _get_server_config(server_name: str) -> dict[str, Any]:
    """Get configuration for a specific MCP server."""
    servers = _get_merged_mcp_servers()
    if server_name not in servers:
        available = list(servers.keys())
        raise ValueError(
            f"MCP server '{server_name}' not found in config. "
            f"Available servers: {available}"
        )
    return servers[server_name]


def get_mcp_servers_config() -> dict[str, McpHttpServerConfig]:
    """Return configured MCP servers as HTTP transport configs.

    Servers declared in ``dispatch.yaml`` under ``mcp_servers`` are constructed
    from runtime environment variables (``DISPATCH_MCP_GATEWAY_URL``,
    ``DISPATCH_NAMESPACE``, ``DISPATCH_API_KEY``). Any user-provided ``.mcp.json``
    is merged additively; duplicate names with dispatch-managed servers are
    rejected.

    Returns:
        Mapping of server name to HTTP URL and headers.

    Raises:
        ValueError: If a user server name collides with a dispatch-managed one.
        RuntimeError: If required environment variables are missing.
    """
    mcp_servers: dict[str, McpHttpServerConfig] = {}
    for server_name, server_config in _get_merged_mcp_servers().items():
        raw_url = server_config.get("url")
        url = raw_url if isinstance(raw_url, str) else ""
        headers = _extract_string_map(server_config.get("headers"))
        mcp_servers[server_name] = _models.McpHttpServerConfig(url=url, headers=headers)

    return mcp_servers


@asynccontextmanager
async def get_mcp_client(
    server_name: str,
    *,
    timeout: float = 30.0,
    read_timeout: float = 300.0,
) -> AsyncIterator[MCPClient]:
    """Open an MCP client for a configured server.

    The yielded client supports tools, resources, and prompts. Tool calls
    automatically include the current Dispatch trace and invocation IDs when
    called from inside a handler.

    Args:
        server_name: Name of the configured MCP server.
        timeout: HTTP connection timeout in seconds.
        read_timeout: MCP session read timeout in seconds.

    Raises:
        ValueError: If the named server is missing or malformed.
        RuntimeError: If required environment variables are missing.

    Example::

        from dispatch_agents import fn, get_mcp_client

        @fn()
        async def search(payload: SearchRequest) -> SearchResponse:
            async with get_mcp_client("datadog") as client:
                result = await client.call_tool(
                    "search_docs",
                    {"query": payload.query},
                )
            return SearchResponse(result=result.content)
    """
    server_config = _get_server_config(server_name)

    url = server_config.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError(f"MCP server '{server_name}' missing 'url' in config")

    headers = _extract_string_map(server_config.get("headers"))

    async with streamablehttp_client(url=url, headers=headers, timeout=timeout) as (
        read_stream,
        write_stream,
        _,
    ):
        async with _ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timedelta(seconds=read_timeout),
        ) as session:
            await session.initialize()
            yield _TracingMCPClient(session)
