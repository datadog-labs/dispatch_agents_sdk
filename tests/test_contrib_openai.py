"""Tests for dispatch_agents.contrib.openai module.

These tests verify that the OpenAI contrib package correctly loads MCP
configuration and returns types compatible with the OpenAI Agents SDK.
"""

import json
import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_agents_sdk():
    """Mock the OpenAI Agents SDK modules."""
    # Create mock modules
    agents_mock = MagicMock()
    agents_mcp_mock = MagicMock()
    agents_mcp_util_mock = MagicMock()

    # Create a mock MCPServerStreamableHttp class that returns unique instances
    mock_server_class = MagicMock()

    def create_mock_instance(*args, **kwargs):
        mock_instance = MagicMock()
        mock_instance.connect = AsyncMock()
        return mock_instance

    mock_server_class.side_effect = create_mock_instance
    agents_mcp_mock.MCPServerStreamableHttp = mock_server_class

    # Create a mock MCPToolMetaContext
    agents_mcp_util_mock.MCPToolMetaContext = MagicMock()

    with patch.dict(
        "sys.modules",
        {
            "agents": agents_mock,
            "agents.mcp": agents_mcp_mock,
            "agents.mcp.util": agents_mcp_util_mock,
        },
    ):
        yield mock_server_class


@pytest.fixture
def mcp_config_file():
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
def reset_singleton():
    """Reset the singleton state before and after each test."""
    # Import and reset the singleton
    import dispatch_agents.contrib.openai as openai_module

    openai_module._mcp_servers = None
    yield
    openai_module._mcp_servers = None


class TestGetMcpServers:
    """Tests for get_mcp_servers function."""

    @pytest.mark.asyncio
    async def test_returns_list_of_servers(
        self, mcp_config_file: str, mock_agents_sdk: MagicMock, reset_singleton: None
    ) -> None:
        """Test that get_mcp_servers returns a list of server instances."""
        from dispatch_agents.contrib.openai import get_mcp_servers

        servers = await get_mcp_servers()

        assert isinstance(servers, list)
        assert len(servers) == 2

    @pytest.mark.asyncio
    async def test_creates_server_with_correct_params(
        self, mcp_config_file: str, mock_agents_sdk: MagicMock, reset_singleton: None
    ) -> None:
        """Test that MCPServerStreamableHttp is created with correct params."""
        from dispatch_agents.contrib.openai import get_mcp_servers

        await get_mcp_servers()

        # Check that MCPServerStreamableHttp was called for each server
        assert mock_agents_sdk.call_count == 2

        # Check that servers were created with correct params
        calls = mock_agents_sdk.call_args_list
        first_call = calls[0]
        assert first_call.kwargs["name"] in ["test-server", "another-server"]
        assert "url" in first_call.kwargs["params"]
        assert first_call.kwargs["cache_tools_list"] is True
        # Should have tool_meta_resolver set
        assert "tool_meta_resolver" in first_call.kwargs
        assert first_call.kwargs["tool_meta_resolver"] is not None

    @pytest.mark.asyncio
    async def test_connects_all_servers(
        self, mcp_config_file: str, mock_agents_sdk: MagicMock, reset_singleton: None
    ) -> None:
        """Test that connect() is called on all servers."""
        from dispatch_agents.contrib.openai import get_mcp_servers

        servers = await get_mcp_servers()

        # Each server should have had connect() called
        for server in servers:
            server.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_singleton_returns_same_servers(
        self, mcp_config_file: str, mock_agents_sdk: MagicMock, reset_singleton: None
    ) -> None:
        """Test that subsequent calls return the same server instances."""
        from dispatch_agents.contrib.openai import get_mcp_servers

        servers1 = await get_mcp_servers()
        servers2 = await get_mcp_servers()

        # Should be the exact same list
        assert servers1 is servers2
        # MCPServerStreamableHttp should only be called on first invocation
        assert mock_agents_sdk.call_count == 2  # Still 2, not 4

    @pytest.mark.asyncio
    async def test_raises_file_not_found_when_no_config(
        self, mock_agents_sdk: MagicMock, reset_singleton: None
    ) -> None:
        """Test that FileNotFoundError is raised when config doesn't exist."""
        from dispatch_agents.contrib.openai import get_mcp_servers

        with patch(
            "dispatch_agents.mcp.MCP_CONFIG_PATH", "/nonexistent/path/.mcp.json"
        ):
            with pytest.raises(FileNotFoundError):
                await get_mcp_servers()


class TestTraceMetaResolver:
    """Tests for _trace_meta_resolver function."""

    def test_returns_none_when_no_trace_context(
        self, mock_agents_sdk: MagicMock
    ) -> None:
        """Test that None is returned when there's no trace context."""
        from dispatch_agents.contrib.openai import _trace_meta_resolver

        # Mock MCPToolMetaContext
        mock_context = MagicMock()

        with patch(
            "dispatch_agents.contrib.openai.get_current_trace_id", return_value=None
        ):
            with patch(
                "dispatch_agents.contrib.openai.get_current_invocation_id",
                return_value=None,
            ):
                result = _trace_meta_resolver(mock_context)

        assert result is None

    def test_returns_trace_id_when_available(self, mock_agents_sdk: MagicMock) -> None:
        """Test that trace_id is included when available."""
        from dispatch_agents.contrib.openai import _trace_meta_resolver

        mock_context = MagicMock()

        with patch(
            "dispatch_agents.contrib.openai.get_current_trace_id",
            return_value="test-trace-id",
        ):
            with patch(
                "dispatch_agents.contrib.openai.get_current_invocation_id",
                return_value=None,
            ):
                result = _trace_meta_resolver(mock_context)

        assert result == {"dispatch_trace_id": "test-trace-id"}

    def test_returns_invocation_id_when_available(
        self, mock_agents_sdk: MagicMock
    ) -> None:
        """Test that invocation_id is included when available."""
        from dispatch_agents.contrib.openai import _trace_meta_resolver

        mock_context = MagicMock()

        with patch(
            "dispatch_agents.contrib.openai.get_current_trace_id", return_value=None
        ):
            with patch(
                "dispatch_agents.contrib.openai.get_current_invocation_id",
                return_value="test-invocation-id",
            ):
                result = _trace_meta_resolver(mock_context)

        assert result == {"dispatch_invocation_id": "test-invocation-id"}

    def test_returns_both_when_available(self, mock_agents_sdk: MagicMock) -> None:
        """Test that both trace_id and invocation_id are included when available."""
        from dispatch_agents.contrib.openai import _trace_meta_resolver

        mock_context = MagicMock()

        with patch(
            "dispatch_agents.contrib.openai.get_current_trace_id",
            return_value="test-trace-id",
        ):
            with patch(
                "dispatch_agents.contrib.openai.get_current_invocation_id",
                return_value="test-invocation-id",
            ):
                result = _trace_meta_resolver(mock_context)

        assert result == {
            "dispatch_trace_id": "test-trace-id",
            "dispatch_invocation_id": "test-invocation-id",
        }


# Type checking examples that will be verified by mypy
if TYPE_CHECKING:
    from agents.mcp import MCPServerStreamableHttp

    from dispatch_agents.contrib.openai import get_mcp_servers

    async def example_openai_agent_usage() -> None:
        """Example showing usage with OpenAI Agents SDK.

        This function demonstrates the intended usage pattern and serves
        as a compile-time type check.

        Example:
            >>> from dispatch_agents.contrib.openai import get_mcp_servers
            >>> from agents import Agent, Runner
            >>>
            >>> # Get MCP servers (singleton, auto-connected)
            >>> mcp_servers = await get_mcp_servers()
            >>>
            >>> agent = Agent(
            ...     name="MyAgent",
            ...     instructions="Use MCP tools.",
            ...     mcp_servers=mcp_servers,
            ... )
            >>> result = await Runner.run(agent, "Hello")
        """
        servers: list[MCPServerStreamableHttp] = await get_mcp_servers()
        # Type should be list[MCPServerStreamableHttp]
        _: list[MCPServerStreamableHttp] = servers
