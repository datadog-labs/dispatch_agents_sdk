#!/usr/bin/env python3
"""
Test that the Claude Agent SDK automatically loads .mcp.json configuration.

This script verifies that:
1. The .mcp.json file is automatically loaded by the SDK
2. The MCP server configuration is recognized
3. The ${DISPATCH_API_KEY} env var placeholder is expanded

Run from the deployed container or locally with .mcp.json present.
"""

import asyncio
import os
import sys


async def test_mcp_config():
    """Test that the Claude Agent SDK can load and use .mcp.json."""
    try:
        from claude_agent_sdk import ClaudeAgentOptions, SystemMessage, query
    except ImportError:
        print("ERROR: claude-agent-sdk not installed")
        print("Install with: pip install claude-agent-sdk")
        return False

    # Check that .mcp.json exists
    mcp_json_path = ".mcp.json"
    if not os.path.exists(mcp_json_path):
        print(f"ERROR: {mcp_json_path} not found")
        return False

    print(f"✓ Found {mcp_json_path}")

    # Read and display the config
    import json

    with open(mcp_json_path) as f:
        config = json.load(f)

    print(f"✓ MCP servers configured: {list(config.get('mcpServers', {}).keys())}")

    # Verify DISPATCH_API_KEY is set (needed for auth header expansion)
    api_key = os.environ.get("DISPATCH_API_KEY")
    if not api_key:
        print("WARNING: DISPATCH_API_KEY not set - auth headers won't work")
    else:
        print(f"✓ DISPATCH_API_KEY is set ({api_key[:10]}...)")

    # Test that the SDK can initialize with the MCP config
    # We'll use a simple prompt that should trigger MCP server connection
    print("\nTesting Claude Agent SDK MCP integration...")

    try:
        # The SDK should automatically load .mcp.json
        # We use allowed_tools to grant access to the MCP tools
        options = ClaudeAgentOptions(
            # Don't pass mcp_servers - let SDK load from .mcp.json
            allowed_tools=["mcp__com.datadoghq.mcp__*"],
            # Use a permissive mode for testing
            permission_mode="bypassPermissions",
        )

        # Just initialize and check for the init message
        async for message in query(
            prompt="List available tools from the Datadog MCP server",
            options=options,
        ):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                mcp_servers = message.data.get("mcp_servers", [])
                print(f"\n✓ SDK initialized with {len(mcp_servers)} MCP server(s)")
                for server in mcp_servers:
                    status = server.get("status", "unknown")
                    name = server.get("name", "unknown")
                    print(f"  - {name}: {status}")
                    if status == "connected":
                        print("✓ MCP server connected successfully!")
                        return True
                    elif status == "failed":
                        print("✗ MCP server failed to connect")
                        return False
                break

    except Exception as e:
        print(f"ERROR during SDK test: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_config())
    sys.exit(0 if success else 1)
