# Dispatch Agents Plugin for Claude Code

Dispatch Agents is a platform for building and deploying AI agents. This Claude Code plugin brings agent management tools directly into your editor, so you can create, run, and deploy agents without leaving your workflow.

It bundles:

- **getting-started** skill — step-by-step guide to building your first Dispatch Agent
- **dispatch-operator** MCP server — tools for creating, deploying, and managing agents

## Installation

Add the marketplace and install the plugin:

```bash
claude plugin marketplace add datadog-labs/dispatch_agents_sdk && claude plugin install dispatch-agents@dispatch-agents
```

## What's included

### Getting Started Skill

Invoke with `/dispatch-agents:getting-started` to get a guided walkthrough of:

1. Installing and authenticating the Dispatch CLI
2. Scaffolding a new agent
3. Writing handler functions (`@fn` and `@on` patterns)
4. Adding dependencies, secrets, and storage
5. Testing locally
6. Deploying to production

### Dispatch Operator MCP Server

Provides MCP tools for agent lifecycle management:

- `create_agent` — scaffold a new agent directory
- `deploy_agent` — deploy an agent to the cloud
- `start_local_agent_dev` — run an agent locally for testing
- `invoke_function` / `invoke_local_function` — call agent functions
- `publish_event` / `send_local_test_event` — emit events to topics
- `get_agent_logs` / `read_local_agent_logs` — view agent logs
- And more — use `get_agent_functions` to discover available functions on any agent

## Prerequisites

- Access to the [Dispatch Agents Claude Plugins](https://github.com/datadog-labs/dispatch_agents_sdk) GitHub repo
- [uv](https://docs.astral.sh/uv/) installed for Python tooling
- The Dispatch CLI will be installed as part of the getting-started skill flow

## Support

- **GitHub Issues:** [Open an issue](https://github.com/datadog-labs/dispatch_agents_sdk/issues) for bugs, feature requests, or questions
- **Slack:** `#dispatch-agents` (internal)
