# Dispatch Agent Examples

Example agents built with the [Dispatch Agent SDK](https://github.com/datadog-labs/dispatch_agents_sdk). Each example is a self-contained agent demonstrating specific patterns.

## Prerequisites

- [Dispatch CLI](https://github.com/datadog-labs/dispatch_agents_cli) installed
- [Dispatch Agent SDK](https://github.com/datadog-labs/dispatch_agents_sdk) installed

## Examples

### Getting Started

| Directory | Description | Key Patterns |
|---|---|---|
| `hello_world` | Starter agent with typed payloads, error handling, and GitHub event integration | `@on`, `@fn`, GitHub webhooks |
| `conversational-agent` | Multi-turn chat assistant with session memory | **LLM Gateway**, **short-term memory**, `@fn` |
| `daily-digest` | Scheduled news digest that tracks reported stories | **Schedules (cron)**, **LLM Gateway**, **long-term memory** |
| `deep-research` | Parallel web research with supervisor/researcher pattern | OpenAI SDK, Tavily search, parallel execution |
| `multi-framework` | Rotates between Claude SDK, OpenAI Agents, and LangGraph | **Long-term memory**, multi-framework |

### Integration Examples

| Directory | Description | Key Patterns |
|---|---|---|
| `company-researcher` | Web research report on any company | Claude Agent SDK, WebSearch |
| `weather-service` | Callable weather lookup with simulated data | `@fn` (inter-agent calls) |
| `weather-assistant` | Orchestrator that calls `weather-service` via `invoke()` | Inter-agent communication |
| `github_pr_responder` | Handles PR comments, reviews, and inline review comments | GitHub event handlers |
| `datadog_mcp` | Three approaches to Datadog MCP integration (Dispatch SDK, Claude SDK, OpenAI SDK) | MCP servers, multi-SDK |
| `dd_incident_multi_agent` | Multi-step incident investigation workflow | LangGraph, structured workflow |

### Testing & Utilities

| Directory | Description | Key Patterns |
|---|---|---|
| `storage-tester` | Validates persistent volume behavior and filesystem boundaries | Volumes, `@fn` test suite |
| `llm_gateway_tester` | Tests OpenAI, Anthropic, Dispatch LLM surface, and MCP tool calls | Multi-LLM, tool calls |
| `trace-stress-test` | Exercises nested and parallel subagent tracing | Tracing, concurrency, Claude SDK |

## Running an Example

```bash
cd hello_world

# Run the agent locally (with auto-reload on file changes)
dispatch agent dev --reload

# Invoke a function
dispatch agent invoke reverse '{"text": "hello"}'
```

## Support

- **Issues:** [GitHub Issues](https://github.com/datadog-labs/dispatch_agents_examples/issues)
