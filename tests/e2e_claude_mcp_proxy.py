#!/usr/bin/env python3
"""
End-to-end test for the Claude SDK MCP server proxy.

This test:
1. Creates a simple HTTP MCP server with test tools
2. Creates a .mcp.json config pointing to it
3. Tests the Claude SDK proxy by calling get_mcp_servers() and invoking tools
4. Verifies trace context headers are properly injected

Run with: uv run python tests/e2e_claude_mcp_proxy.py
"""

import asyncio
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# Track captured headers for verification
captured_requests: list[dict[str, Any]] = []


class MockMCPServer(BaseHTTPRequestHandler):
    """A simple HTTP MCP server for testing."""

    # Define test tools
    TOOLS = [
        {
            "name": "echo",
            "description": "Echoes back the input message",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"],
            },
        },
        {
            "name": "add",
            "description": "Adds two numbers together",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
        {
            "name": "get_time",
            "description": "Returns the current server time",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def do_POST(self) -> None:
        """Handle MCP JSON-RPC requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request = json.loads(body)

        # Capture the request and headers for verification
        captured_requests.append(
            {
                "headers": dict(self.headers),
                "body": request,
            }
        )

        # Handle JSON-RPC methods
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", 1)

        if method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": self.TOOLS},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self._call_tool(tool_name, arguments)
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        elif method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test-server", "version": "1.0.0"},
                },
            }
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool and return the result."""
        if tool_name == "echo":
            message = arguments.get("message", "")
            return {
                "content": [{"type": "text", "text": f"Echo: {message}"}],
                "isError": False,
            }
        elif tool_name == "add":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            result = a + b
            return {
                "content": [{"type": "text", "text": f"Result: {result}"}],
                "isError": False,
            }
        elif tool_name == "get_time":
            import datetime

            now = datetime.datetime.now().isoformat()
            return {
                "content": [{"type": "text", "text": f"Current time: {now}"}],
                "isError": False,
            }
        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }


def run_server(port: int, ready_event: threading.Event) -> HTTPServer:
    """Run the mock MCP server."""
    server = HTTPServer(("localhost", port), MockMCPServer)
    ready_event.set()
    server.serve_forever()
    return server


async def test_claude_mcp_proxy():
    """Test the Claude SDK MCP proxy end-to-end."""
    global captured_requests
    captured_requests = []

    # Find an available port
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    print(f"\n{'=' * 60}")
    print("Claude SDK MCP Proxy End-to-End Test")
    print(f"{'=' * 60}\n")

    # Start the mock MCP server in a background thread
    ready_event = threading.Event()
    server_thread = threading.Thread(
        target=run_server,
        args=(port, ready_event),
        daemon=True,
    )
    server_thread.start()
    ready_event.wait(timeout=5)

    print(f"[1/5] Started mock MCP server on port {port}")

    # Create a temporary .mcp.json config
    mcp_config = {
        "mcpServers": {
            "test-server": {
                "url": f"http://localhost:{port}",
                "headers": {
                    "Authorization": "Bearer test-token",
                    "X-Custom-Header": "custom-value",
                },
            }
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as config_file:
        json.dump(mcp_config, config_file)
        config_path = config_file.name

    print(f"[2/5] Created .mcp.json config at {config_path}")

    try:
        # Mock the claude_agent_sdk module since it may not be installed
        import sys
        from unittest.mock import MagicMock

        # Create mock claude_agent_sdk
        claude_sdk_mock = MagicMock()

        class MockSdkMcpTool:
            def __init__(self, name, description, input_schema, handler):
                self.name = name
                self.description = description
                self.input_schema = input_schema
                self.handler = handler

            def __class_getitem__(cls, item):
                return cls

        def create_sdk_mcp_server(name, version, tools):
            return {
                "type": "sdk",
                "name": name,
                "instance": MagicMock(),
                "tools": tools,
            }

        claude_sdk_mock.McpSdkServerConfig = dict
        claude_sdk_mock.SdkMcpTool = MockSdkMcpTool
        claude_sdk_mock.create_sdk_mcp_server = create_sdk_mcp_server

        sys.modules["claude_agent_sdk"] = claude_sdk_mock

        # Now import and reload our module with the mock
        import importlib
        from unittest.mock import patch

        # Patch the config path and reload
        with patch("dispatch_agents.mcp.MCP_CONFIG_PATH", config_path):
            import dispatch_agents.contrib.claude as claude_module
            import dispatch_agents.mcp as mcp_module

            importlib.reload(claude_module)
            claude_module._mcp_servers = None  # Reset singleton

            # Set up trace context
            test_trace_id = "test-trace-id-12345"
            test_invocation_id = "test-invocation-id-67890"

            # Patch trace context functions in the mcp module where they're used
            with (
                patch.object(
                    mcp_module,
                    "get_current_trace_id",
                    return_value=test_trace_id,
                ),
                patch.object(
                    mcp_module,
                    "get_current_invocation_id",
                    return_value=test_invocation_id,
                ),
            ):
                print("[3/5] Calling get_mcp_servers() to create proxy servers...")

                # Get MCP servers (this fetches tool list from upstream)
                servers = await claude_module.get_mcp_servers()

                print(f"      Created {len(servers)} proxy server(s)")
                for name, config in servers.items():
                    print(f"      - {name}: type={config['type']}")
                    if "tools" in config:
                        print(f"        Tools: {[t.name for t in config['tools']]}")

                # Verify tools were fetched
                assert "test-server" in servers, "test-server not found in servers"
                server_config = servers["test-server"]
                assert server_config["type"] == "sdk", "Server type should be 'sdk'"

                tools = server_config.get("tools", [])
                tool_names = [t.name for t in tools]
                assert "echo" in tool_names, "echo tool not found"
                assert "add" in tool_names, "add tool not found"
                assert "get_time" in tool_names, "get_time tool not found"

                print("[4/5] Invoking tools through the proxy...")

                # Test each tool through the proxy
                # Clear captured requests before tool calls
                tools_list_requests = len(captured_requests)
                captured_requests = captured_requests[tools_list_requests:]

                # Test echo tool
                echo_tool = next(t for t in tools if t.name == "echo")
                result = await echo_tool.handler({"message": "Hello, MCP!"})
                print(f"      echo('Hello, MCP!') -> {result}")
                assert "Echo: Hello, MCP!" in str(result), "Echo result mismatch"

                # Test add tool
                add_tool = next(t for t in tools if t.name == "add")
                result = await add_tool.handler({"a": 5, "b": 3})
                print(f"      add(5, 3) -> {result}")
                assert "Result: 8" in str(result), "Add result mismatch"

                # Test get_time tool
                time_tool = next(t for t in tools if t.name == "get_time")
                result = await time_tool.handler({})
                print(f"      get_time() -> {result}")
                assert "Current time:" in str(result), "Time result mismatch"

                print("[5/5] Verifying trace context was injected via _meta...")

                # Verify trace context was injected in _meta field of tool calls
                tool_call_requests = [
                    r
                    for r in captured_requests
                    if r["body"].get("method") == "tools/call"
                ]

                assert len(tool_call_requests) >= 3, (
                    f"Expected at least 3 tool calls, got {len(tool_call_requests)}"
                )

                for i, req in enumerate(tool_call_requests):
                    headers = req["headers"]
                    params = req["body"].get("params", {})
                    meta = params.get("_meta", {})
                    tool_name = params.get("name")

                    print(f"\n      Tool call {i + 1} ({tool_name}):")
                    print(
                        f"        _meta.dispatch_trace_id: {meta.get('dispatch_trace_id', 'MISSING')}"
                    )
                    print(
                        f"        _meta.dispatch_invocation_id: {meta.get('dispatch_invocation_id', 'MISSING')}"
                    )
                    print(
                        f"        Authorization header: {headers.get('Authorization', 'MISSING')}"
                    )

                    assert meta.get("dispatch_trace_id") == test_trace_id, (
                        f"Trace ID mismatch for {tool_name}"
                    )
                    assert meta.get("dispatch_invocation_id") == test_invocation_id, (
                        f"Invocation ID mismatch for {tool_name}"
                    )
                    assert headers.get("Authorization") == "Bearer test-token", (
                        f"Auth header mismatch for {tool_name}"
                    )
                    # Verify trace context is NOT in HTTP headers anymore
                    assert "X-Dispatch-Trace-ID" not in headers, (
                        f"Trace ID should not be in HTTP headers for {tool_name}"
                    )

        print(f"\n{'=' * 60}")
        print("ALL TESTS PASSED!")
        print(f"{'=' * 60}\n")
        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"TEST FAILED: {e}")
        print(f"{'=' * 60}\n")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        os.unlink(config_path)


if __name__ == "__main__":
    success = asyncio.run(test_claude_mcp_proxy())
    sys.exit(0 if success else 1)
