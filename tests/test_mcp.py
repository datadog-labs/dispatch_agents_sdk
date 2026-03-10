"""Tests for dispatch_agents.mcp module.

Covers _build_trace_meta, TracingClientSession, and config loading functions.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dispatch_agents.mcp import (
    TracingClientSession,
    _build_trace_meta,
    _get_server_config,
    _load_mcp_config,
    get_mcp_servers_config,
)

# ── _build_trace_meta ────────────────────────────────────────────────


class TestBuildTraceMeta:
    @patch("dispatch_agents.mcp.get_current_invocation_id", return_value="inv-123")
    @patch("dispatch_agents.mcp.get_current_trace_id", return_value="trace-abc")
    def test_with_both_context_vars(self, mock_trace, mock_inv):
        result = _build_trace_meta()
        assert result == {
            "dispatch_trace_id": "trace-abc",
            "dispatch_invocation_id": "inv-123",
        }

    @patch("dispatch_agents.mcp.get_current_invocation_id", return_value=None)
    @patch("dispatch_agents.mcp.get_current_trace_id", return_value=None)
    def test_no_context(self, mock_trace, mock_inv):
        result = _build_trace_meta()
        assert result is None

    @patch(
        "dispatch_agents.mcp.get_invocation_id_for_trace", return_value="inv-fallback"
    )
    @patch("dispatch_agents.mcp.get_current_invocation_id", return_value=None)
    @patch("dispatch_agents.mcp.get_current_trace_id", return_value="trace-xyz")
    def test_fallback_invocation_id(self, mock_trace, mock_inv, mock_fallback):
        result = _build_trace_meta()
        assert result == {
            "dispatch_trace_id": "trace-xyz",
            "dispatch_invocation_id": "inv-fallback",
        }

    @patch("dispatch_agents.mcp.get_invocation_id_for_trace", return_value=None)
    @patch("dispatch_agents.mcp.get_current_invocation_id", return_value=None)
    @patch("dispatch_agents.mcp.get_current_trace_id", return_value="trace-only")
    def test_trace_id_only(self, mock_trace, mock_inv, mock_fallback):
        result = _build_trace_meta()
        assert result == {"dispatch_trace_id": "trace-only"}


# ── TracingClientSession ─────────────────────────────────────────────


class TestTracingClientSession:
    @pytest.mark.asyncio
    @patch(
        "dispatch_agents.mcp._build_trace_meta",
        return_value={"dispatch_trace_id": "t1"},
    )
    async def test_call_tool_injects_trace(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = TracingClientSession(inner)
        await session.call_tool("my_tool", {"arg": "val"})

        inner.call_tool.assert_called_once()
        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"] == {"dispatch_trace_id": "t1"}

    @pytest.mark.asyncio
    @patch(
        "dispatch_agents.mcp._build_trace_meta",
        return_value={"dispatch_trace_id": "t2"},
    )
    async def test_call_tool_merges_user_meta(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = TracingClientSession(inner)
        await session.call_tool("my_tool", meta={"user_key": "user_val"})

        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"]["dispatch_trace_id"] == "t2"
        assert call_kwargs["meta"]["user_key"] == "user_val"

    @pytest.mark.asyncio
    @patch("dispatch_agents.mcp._build_trace_meta", return_value=None)
    async def test_call_tool_no_trace(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = TracingClientSession(inner)
        await session.call_tool("my_tool", meta={"custom": "data"})

        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"] == {"custom": "data"}

    @pytest.mark.asyncio
    @patch("dispatch_agents.mcp._build_trace_meta", return_value=None)
    async def test_call_tool_no_trace_no_meta(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = TracingClientSession(inner)
        await session.call_tool("my_tool")

        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"] is None

    @pytest.mark.asyncio
    async def test_list_tools_delegates(self):
        inner = AsyncMock()
        inner.list_tools = AsyncMock(return_value=MagicMock())

        session = TracingClientSession(inner)
        await session.list_tools()

        inner.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_resources_delegates(self):
        inner = AsyncMock()
        inner.list_resources = AsyncMock(return_value=MagicMock())

        session = TracingClientSession(inner)
        await session.list_resources()

        inner.list_resources.assert_called_once()


# ── Config loading functions ─────────────────────────────────────────


class TestLoadMcpConfig:
    def test_missing_file_raises(self):
        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", "/nonexistent/.mcp.json"):
            with pytest.raises(
                FileNotFoundError, match="MCP configuration file not found"
            ):
                _load_mcp_config()

    def test_loads_valid_config(self, tmp_path):
        config = {"mcpServers": {"my-server": {"url": "http://localhost:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            result = _load_mcp_config()
        assert result == config


class TestGetServerConfig:
    def test_not_found_raises(self, tmp_path):
        config = {"mcpServers": {"other-server": {"url": "http://localhost:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            with pytest.raises(ValueError, match="not found in config"):
                _get_server_config("missing-server")

    def test_returns_server_config(self, tmp_path):
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:3000",
                    "headers": {"Authorization": "Bearer tok"},
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            result = _get_server_config("my-server")
        assert result["url"] == "http://localhost:3000"


class TestGetMcpServersConfig:
    def test_returns_http_configs(self, tmp_path):
        config = {
            "mcpServers": {
                "server-a": {"url": "http://a.com", "headers": {"X-Key": "abc"}},
                "server-b": {"url": "http://b.com"},
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            result = get_mcp_servers_config()

        assert len(result) == 2
        assert result["server-a"]["type"] == "http"
        assert result["server-a"]["url"] == "http://a.com"
        assert result["server-a"]["headers"] == {"X-Key": "abc"}
        assert result["server-b"]["type"] == "http"
        assert result["server-b"]["url"] == "http://b.com"
        assert result["server-b"]["headers"] == {}
