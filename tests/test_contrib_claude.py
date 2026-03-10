"""Tests for dispatch_agents.contrib.claude module.

These tests verify that the Claude contrib package correctly loads MCP
configuration, creates proxy SDK servers, and injects trace context headers.
"""

import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_claude_sdk():
    """Mock the Claude Agent SDK modules."""
    # Create mock module
    claude_agent_sdk_mock = MagicMock()

    # Create mock McpSdkServerConfig TypedDict-like type
    claude_agent_sdk_mock.McpSdkServerConfig = dict

    # Create mock SdkMcpTool class that stores the tool definition
    # and supports generic subscripting
    class MockSdkMcpTool:
        def __init__(self, name, description, input_schema, handler):
            self.name = name
            self.description = description
            self.input_schema = input_schema
            self.handler = handler

        def __class_getitem__(cls, item):
            return cls

    claude_agent_sdk_mock.SdkMcpTool = MockSdkMcpTool

    # Create mock create_sdk_mcp_server that returns proper configs
    def create_sdk_mcp_server(name: str, version: str, tools: list) -> dict:
        return {"type": "sdk", "name": name, "instance": MagicMock(), "tools": tools}

    claude_agent_sdk_mock.create_sdk_mcp_server = create_sdk_mcp_server

    with patch.dict(
        "sys.modules",
        {
            "claude_agent_sdk": claude_agent_sdk_mock,
        },
    ):
        yield claude_agent_sdk_mock


@pytest.fixture
def mcp_config_file(mock_claude_sdk):
    """Create a temporary .mcp.json config file and patch the path."""
    config = {
        "mcpServers": {
            "test-server": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer test-token"},
            },
            "another-server": {
                "url": "https://other.com/mcp",
                "headers": {"X-Custom": "value"},
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        temp_path = f.name

    # Patch the module-level constant
    with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", temp_path):
        yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def reset_singleton(mock_claude_sdk):
    """Reset the singleton state before and after each test."""
    # Import fresh module with mocked dependencies
    import importlib

    import dispatch_agents.contrib.claude as claude_module

    # Reload to get fresh module with mocked dependencies
    importlib.reload(claude_module)
    claude_module._mcp_servers = None
    yield claude_module
    claude_module._mcp_servers = None


@pytest.fixture
def mock_upstream_tools():
    """Mock the upstream MCP server tools/list response."""
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"arg1": {"type": "string"}},
                "required": ["arg1"],
            },
        },
        {
            "name": "another_tool",
            "description": "Another test tool",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
            },
        },
    ]


def _create_mock_aiohttp_session(response_data: dict) -> MagicMock:
    """Create a mock aiohttp session that returns the given response data."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=response_data)

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock()

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_context)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()

    return mock_session


class TestGetMcpServers:
    """Tests for get_mcp_servers function."""

    @pytest.mark.asyncio
    async def test_returns_dict_of_sdk_server_configs(
        self,
        mcp_config_file: str,
        mock_upstream_tools: list[dict[str, Any]],
        reset_singleton: MagicMock,
    ) -> None:
        """Test that get_mcp_servers returns a dict of SDK server configs."""
        from dispatch_agents.contrib.claude import get_mcp_servers

        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": mock_upstream_tools},
        }
        mock_session = _create_mock_aiohttp_session(response_data)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            servers = await get_mcp_servers()

        assert isinstance(servers, dict)
        assert len(servers) == 2
        assert "test-server" in servers
        assert "another-server" in servers

    @pytest.mark.asyncio
    async def test_server_config_has_sdk_type(
        self,
        mcp_config_file: str,
        mock_upstream_tools: list[dict[str, Any]],
        reset_singleton: MagicMock,
    ) -> None:
        """Test that each server config has type='sdk'."""
        from dispatch_agents.contrib.claude import get_mcp_servers

        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": mock_upstream_tools},
        }
        mock_session = _create_mock_aiohttp_session(response_data)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            servers = await get_mcp_servers()

        for name, config in servers.items():
            assert config["type"] == "sdk"
            assert config["name"] == name
            assert "instance" in config

    @pytest.mark.asyncio
    async def test_returns_cached_servers_on_subsequent_calls(
        self,
        mcp_config_file: str,
        mock_upstream_tools: list[dict[str, Any]],
        reset_singleton: MagicMock,
    ) -> None:
        """Test that subsequent calls return cached servers."""
        from dispatch_agents.contrib.claude import get_mcp_servers

        response_data = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": mock_upstream_tools},
        }
        mock_session = _create_mock_aiohttp_session(response_data)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            servers1 = await get_mcp_servers()
            servers2 = await get_mcp_servers()

        # Should be the exact same object (cached)
        assert servers1 is servers2

    @pytest.mark.asyncio
    async def test_raises_file_not_found_when_no_config(
        self, mock_claude_sdk: MagicMock, reset_singleton: MagicMock
    ) -> None:
        """Test that FileNotFoundError is raised when config doesn't exist."""
        from dispatch_agents.contrib.claude import get_mcp_servers

        with patch(
            "dispatch_agents.mcp.MCP_CONFIG_PATH", "/nonexistent/path/.mcp.json"
        ):
            with pytest.raises(FileNotFoundError):
                await get_mcp_servers()

    @pytest.mark.asyncio
    async def test_handles_upstream_server_unavailable(
        self, mcp_config_file: str, reset_singleton: MagicMock
    ) -> None:
        """Test graceful handling when upstream server is unavailable."""
        from dispatch_agents.contrib.claude import get_mcp_servers

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=Exception("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Should not raise, but return servers with no tools
            servers = await get_mcp_servers()

        assert isinstance(servers, dict)
        assert len(servers) == 2


class TestProxyToolTraceContext:
    """Tests for trace context injection in proxy tools."""

    @pytest.mark.asyncio
    async def test_proxy_tool_uses_get_mcp_client(
        self,
        mcp_config_file: str,
        reset_singleton: MagicMock,
    ) -> None:
        """Test that proxy tools use get_mcp_client for trace context injection."""
        from mcp.types import CallToolResult, TextContent, Tool

        from dispatch_agents.contrib.claude import _create_proxy_tool

        # Track call_tool invocations
        captured_tool_name: str | None = None
        captured_args: dict[str, Any] | None = None

        # Create a mock MCP client
        class MockMcpClient:
            async def call_tool(
                self, name: str, arguments: dict[str, Any] | None = None
            ) -> CallToolResult:
                nonlocal captured_tool_name, captured_args
                captured_tool_name = name
                captured_args = arguments
                return CallToolResult(
                    content=[TextContent(type="text", text="mock result")],
                    isError=False,
                )

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        # Create a mock Tool
        mock_tool = Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={"type": "object", "properties": {"arg1": {"type": "string"}}},
        )

        # Patch get_mcp_client to return our mock
        with patch(
            "dispatch_agents.contrib.claude.get_mcp_client",
            return_value=MockMcpClient(),
        ):
            # Create proxy tool and invoke it
            proxy_tool = _create_proxy_tool("test-server", mock_tool)
            result = await proxy_tool.handler({"arg1": "value"})

        # Verify the tool was called correctly
        assert captured_tool_name == "test_tool"
        assert captured_args == {"arg1": "value"}
        assert result["content"][0]["text"] == "mock result"
        assert result["is_error"] is False


class TestMcpSdkServerConfigExport:
    """Tests for McpSdkServerConfig re-export."""

    def test_mcp_sdk_server_config_is_exported(
        self, mock_claude_sdk: MagicMock
    ) -> None:
        """Test that McpSdkServerConfig is exported from the module."""
        import importlib

        import dispatch_agents.contrib.claude as claude_module

        importlib.reload(claude_module)
        from dispatch_agents.contrib.claude import McpSdkServerConfig

        # Verify it's exported (will be the mock)
        assert McpSdkServerConfig is not None
