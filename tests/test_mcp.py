"""Tests for dispatch_agents.mcp module.

Covers _build_trace_meta, the tracing MCP client, and the in-memory MCP server
config builders.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dispatch_agents.mcp import (
    _build_trace_meta,
    _expand_env_vars,
    _get_dispatch_mcp_servers,
    _get_merged_mcp_servers,
    _get_server_config,
    _get_user_mcp_servers,
    _TracingMCPClient,
    get_mcp_servers_config,
)
from dispatch_agents.models import DispatchConfig, MCPServerConfig


@pytest.fixture(autouse=True)
def isolate_runtime_config(monkeypatch):
    from dispatch_agents.config import _load_runtime_config

    monkeypatch.delenv("DISPATCH_CONFIG_PATH", raising=False)
    _load_runtime_config.cache_clear()
    yield
    _load_runtime_config.cache_clear()


@pytest.fixture
def sdk_caplog(caplog):
    """``caplog`` that captures the ``dispatch_agents`` logger.

    The SDK logger sets ``propagate=False``, so caplog's root handler never sees
    its records. Attach caplog's handler directly and detach it afterwards.
    """
    sdk_logger = logging.getLogger("dispatch_agents")
    sdk_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        sdk_logger.removeHandler(caplog.handler)


# ── _build_trace_meta ────────────────────────────────────────────────


class TestBuildTraceMeta:
    @patch(
        "dispatch_agents.mcp.get_current_invocation_id",
        return_value="inv-123",
    )
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
        "dispatch_agents.mcp.get_invocation_id_for_trace",
        return_value="inv-fallback",
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


# ── MCPClient ────────────────────────────────────────────


class TestMCPClient:
    @pytest.mark.asyncio
    @patch(
        "dispatch_agents.mcp._build_trace_meta",
        return_value={"dispatch_trace_id": "t1"},
    )
    async def test_call_tool_injects_trace(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = _TracingMCPClient(inner)
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

        session = _TracingMCPClient(inner)
        await session.call_tool("my_tool", meta={"user_key": "user_val"})

        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"]["dispatch_trace_id"] == "t2"
        assert call_kwargs["meta"]["user_key"] == "user_val"

    @pytest.mark.asyncio
    @patch("dispatch_agents.mcp._build_trace_meta", return_value=None)
    async def test_call_tool_no_trace(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = _TracingMCPClient(inner)
        await session.call_tool("my_tool", meta={"custom": "data"})

        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"] == {"custom": "data"}

    @pytest.mark.asyncio
    @patch("dispatch_agents.mcp._build_trace_meta", return_value=None)
    async def test_call_tool_no_trace_no_meta(self, mock_meta):
        inner = AsyncMock()
        inner.call_tool = AsyncMock(return_value=MagicMock())

        session = _TracingMCPClient(inner)
        await session.call_tool("my_tool")

        call_kwargs = inner.call_tool.call_args.kwargs
        assert call_kwargs["meta"] is None

    @pytest.mark.asyncio
    async def test_list_tools_delegates(self):
        inner = AsyncMock()
        inner.list_tools = AsyncMock(return_value=MagicMock())

        session = _TracingMCPClient(inner)
        await session.list_tools()

        inner.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_resources_delegates(self):
        inner = AsyncMock()
        inner.list_resources = AsyncMock(return_value=MagicMock())

        session = _TracingMCPClient(inner)
        await session.list_resources()

        inner.list_resources.assert_called_once()


# ── _expand_env_vars ─────────────────────────────────────────────────


class TestExpandEnvVars:
    def test_simple_var_set(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        assert _expand_env_vars("Bearer ${MY_TOKEN}") == "Bearer secret123"

    def test_simple_var_unset_leaves_placeholder(self, monkeypatch):
        monkeypatch.delenv("UNSET_VAR", raising=False)
        assert _expand_env_vars("${UNSET_VAR}") == "${UNSET_VAR}"

    def test_default_syntax_var_set(self, monkeypatch):
        monkeypatch.setenv("MY_HOST", "prod.example.com")
        assert _expand_env_vars("${MY_HOST:-localhost}") == "prod.example.com"

    def test_default_syntax_var_unset(self, monkeypatch):
        monkeypatch.delenv("MY_HOST", raising=False)
        assert _expand_env_vars("${MY_HOST:-localhost}") == "localhost"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "tok-abc")
        config = {"headers": {"Authorization": "Bearer ${API_KEY}"}}
        result = _expand_env_vars(config)
        assert result == {"headers": {"Authorization": "Bearer tok-abc"}}

    def test_list(self, monkeypatch):
        monkeypatch.setenv("ITEM_VAL", "hello")
        result = _expand_env_vars(["${ITEM_VAL}", "literal"])
        assert result == ["hello", "literal"]

    def test_non_string_passthrough(self):
        assert _expand_env_vars(42) == 42
        assert _expand_env_vars(3.14) == 3.14
        assert _expand_env_vars(None) is None
        assert _expand_env_vars(True) is True

    def test_dict_keys_not_expanded(self, monkeypatch):
        monkeypatch.setenv("KEY_VAR", "expanded")
        result = _expand_env_vars({"${KEY_VAR}": "value"})
        # Keys should NOT be expanded — only values
        assert "${KEY_VAR}" in result
        assert "value" == result["${KEY_VAR}"]


# ── Config builder functions ─────────────────────────────────────────


class TestGetUserMcpServers:
    def test_missing_file_returns_empty(self):
        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", "/nonexistent/.mcp.json"):
            result = _get_user_mcp_servers()
        assert result == {}

    def test_loads_valid_config(self, tmp_path):
        config = {"mcpServers": {"my-server": {"url": "http://localhost:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            result = _get_user_mcp_servers()
        assert result == {"my-server": {"url": "http://localhost:3000"}}

    def test_expands_env_vars_in_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_TOKEN", "secret-value")
        config = {
            "mcpServers": {
                "my-server": {
                    "url": "http://localhost:3000",
                    "headers": {"Authorization": "Bearer ${TEST_TOKEN}"},
                }
            }
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            result = _get_user_mcp_servers()

        assert result["my-server"]["headers"]["Authorization"] == "Bearer secret-value"
        # The original file must not have been modified
        raw = json.loads(config_file.read_text())
        assert (
            raw["mcpServers"]["my-server"]["headers"]["Authorization"]
            == "Bearer ${TEST_TOKEN}"
        )


class TestGetDispatchMcpServers:
    def test_no_mcp_servers_returns_empty(self):
        config = DispatchConfig()
        with patch("dispatch_agents.config._load_runtime_config", return_value=config):
            result = _get_dispatch_mcp_servers()
        assert result == {}

    def test_builds_urls_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_MCP_GATEWAY_URL", "https://gateway.example.com")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "my-ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "test-key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="my-server")],
        )

        with patch("dispatch_agents.config._load_runtime_config", return_value=config):
            result = _get_dispatch_mcp_servers()

        assert "my-server" in result
        assert (
            result["my-server"]["url"]
            == "https://gateway.example.com/api/v1/mcp/namespaces/my-ns/proxy/my-server"
        )
        assert result["my-server"]["headers"]["Authorization"] == "Bearer test-key"

    def test_missing_required_env_raises(self, monkeypatch):
        monkeypatch.delenv("DISPATCH_MCP_GATEWAY_URL", raising=False)
        monkeypatch.delenv("DISPATCH_LOCAL_DEV", raising=False)
        monkeypatch.setenv("DISPATCH_NAMESPACE", "ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="my-server")],
        )

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            pytest.raises(RuntimeError, match="DISPATCH_MCP_GATEWAY_URL is required"),
        ):
            _get_dispatch_mcp_servers()

    def test_local_dev_skips_dispatch_servers(self, monkeypatch, sdk_caplog):
        # In local dev the MCP gateway is unavailable, so dispatch-managed
        # servers are skipped (with a warning) instead of crashing.
        monkeypatch.setenv("DISPATCH_LOCAL_DEV", "true")
        monkeypatch.delenv("DISPATCH_MCP_GATEWAY_URL", raising=False)

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="my-server")],
        )

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            sdk_caplog.at_level("WARNING", logger="dispatch_agents"),
        ):
            result = _get_dispatch_mcp_servers()

        assert result == {}
        assert "local dev mode" in sdk_caplog.text
        assert "my-server" in sdk_caplog.text


class TestGetMergedMcpServers:
    def test_merges_dispatch_and_user_servers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISPATCH_MCP_GATEWAY_URL", "https://gw.example.com")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="dispatch-server")],
        )

        user_config = {"mcpServers": {"user-server": {"url": "http://user:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(user_config))

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)),
        ):
            result = _get_merged_mcp_servers()

        assert "dispatch-server" in result
        assert "user-server" in result

    def test_missing_user_file_returns_dispatch_servers(self, monkeypatch):
        monkeypatch.setenv("DISPATCH_MCP_GATEWAY_URL", "https://gw.example.com")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="dispatch-server")],
        )

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            patch("dispatch_agents.mcp.MCP_CONFIG_PATH", "/nonexistent/.mcp.json"),
        ):
            result = _get_merged_mcp_servers()

        assert list(result) == ["dispatch-server"]

    def test_local_dev_uses_only_user_servers(self, tmp_path, monkeypatch):
        # In local dev, dispatch-managed servers are skipped, so the merged set
        # comes entirely from the user's .mcp.json -- no gateway URL required.
        monkeypatch.setenv("DISPATCH_LOCAL_DEV", "true")
        monkeypatch.delenv("DISPATCH_MCP_GATEWAY_URL", raising=False)

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="dispatch-server")],
        )

        user_config = {"mcpServers": {"user-server": {"url": "http://user:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(user_config))

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)),
        ):
            result = _get_merged_mcp_servers()

        assert list(result) == ["user-server"]

    def test_dispatch_server_overrides_colliding_user_server(
        self, tmp_path, monkeypatch
    ):
        # On a name collision the dispatch-managed server takes precedence over
        # the .mcp.json entry (no exception): the managed gateway URL wins.
        monkeypatch.setenv("DISPATCH_MCP_GATEWAY_URL", "https://gw.example.com")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="shared-server")],
        )

        user_config = {
            "mcpServers": {"shared-server": {"url": "http://user-override:9999"}}
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(user_config))

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)),
        ):
            result = _get_merged_mcp_servers()

        assert list(result) == ["shared-server"]
        assert (
            result["shared-server"]["url"]
            == "https://gw.example.com/api/v1/mcp/namespaces/ns/proxy/shared-server"
        )

    def test_override_logs_debug_message(self, tmp_path, monkeypatch, sdk_caplog):
        monkeypatch.setenv("DISPATCH_MCP_GATEWAY_URL", "https://gw.example.com")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="shared-server")],
        )

        user_config = {
            "mcpServers": {"shared-server": {"url": "http://user-override:9999"}}
        }
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(user_config))

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)),
            sdk_caplog.at_level("DEBUG", logger="dispatch_agents"),
        ):
            _get_merged_mcp_servers()

        assert "override .mcp.json" in sdk_caplog.text
        assert "shared-server" in sdk_caplog.text

    def test_no_override_log_without_collision(self, tmp_path, monkeypatch, sdk_caplog):
        # When dispatch-managed and .mcp.json servers have distinct names, the
        # override log must stay silent -- it should only fire on a real collision.
        monkeypatch.setenv("DISPATCH_MCP_GATEWAY_URL", "https://gw.example.com")
        monkeypatch.setenv("DISPATCH_NAMESPACE", "ns")
        monkeypatch.setenv("DISPATCH_API_KEY", "key")

        config = DispatchConfig(
            mcp_servers=[MCPServerConfig(server="dispatch-server")],
        )

        user_config = {"mcpServers": {"user-server": {"url": "http://user:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(user_config))

        with (
            patch("dispatch_agents.config._load_runtime_config", return_value=config),
            patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)),
            sdk_caplog.at_level("DEBUG", logger="dispatch_agents"),
        ):
            _get_merged_mcp_servers()

        assert "override .mcp.json" not in sdk_caplog.text


class TestGetServerConfig:
    def test_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DISPATCH_LOCAL_DEV", raising=False)
        config = {"mcpServers": {"other-server": {"url": "http://localhost:3000"}}}
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps(config))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            with pytest.raises(ValueError, match="not found in config") as exc_info:
                _get_server_config("missing-server")
        # Outside local dev, no dev-mode hint is appended.
        assert "dispatch agent dev" not in str(exc_info.value)

    def test_not_found_in_local_dev_hints_at_mcp_json(self, tmp_path, monkeypatch):
        # In local dev a dispatch-managed server is skipped, so it is absent here.
        # The error should point the author at the .mcp.json escape hatch instead
        # of leaving them with a bare "not found".
        monkeypatch.setenv("DISPATCH_LOCAL_DEV", "true")
        config_file = tmp_path / ".mcp.json"
        config_file.write_text(json.dumps({"mcpServers": {}}))

        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", str(config_file)):
            with pytest.raises(ValueError, match="dispatch agent dev"):
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
        assert result["server-a"].type == "http"
        assert result["server-a"].url == "http://a.com"
        assert result["server-a"].headers == {"X-Key": "abc"}
        assert result["server-b"].type == "http"
        assert result["server-b"].url == "http://b.com"
        assert result["server-b"].headers == {}
