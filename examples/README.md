# Dispatch Agent Examples

Example agents built with the [Dispatch Agent SDK](https://github.com/datadog-labs/dispatch_agents_sdk). Each example is a self-contained agent demonstrating specific patterns.

## Prerequisites

- [Dispatch CLI](https://github.com/datadog-labs/dispatch_agents_cli) installed
- [Dispatch Agent SDK](https://github.com/datadog-labs/dispatch_agents_sdk) installed

## Examples

| Directory | Description | Key Patterns |
|---|---|---|
| `hello_world` | Starter agent with typed payloads, error handling, and GitHub event integration | `@on`, `@fn`, GitHub webhooks |
| `weather-service` | Callable weather lookup with simulated data | `@fn` (inter-agent calls) |
| `weather-assistant` | Orchestrator that calls `weather-service` via `invoke()` | Inter-agent communication |
| `datadog_mcp` | Three approaches to Datadog MCP integration (Dispatch SDK, Claude SDK, OpenAI SDK) | MCP servers, multi-SDK |
| `dd_mcp_agent` | LangGraph Datadog agent with memory and human-in-the-loop | LangGraph, MCP, long-term memory |
| `dd_incident_multi_agent` | Multi-step incident investigation workflow | LangGraph, structured workflow |
| `github_pr_responder` | Handles PR comments, reviews, and inline review comments | GitHub event handlers |
| `storage-tester` | Validates persistent volume behavior and filesystem boundaries | Volumes, `@fn` test suite |
| `llm_gateway_tester` | Tests OpenAI, Anthropic, Dispatch LLM surface, and MCP tool calls | Multi-LLM, tool calls |
| `trace-stress-test` | Exercises nested and parallel subagent tracing | Tracing, concurrency, Claude SDK |

## Running an Example

```bash
cd hello_world

# Run the agent locally
dispatch agent run

# Send a test event
dispatch agent send-event --topic hello_world --payload '{"message": "hello"}'
```

## Support

- **Issues:** [GitHub Issues](https://github.com/datadog-labs/dispatch_agents_examples/issues)
- **Slack:** `#dispatch-agents` (internal)
