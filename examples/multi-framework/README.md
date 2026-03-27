# Multi-Framework Agent

A framework-agnostic agent that rotates between Claude Agent SDK, OpenAI Agents
SDK, and LangGraph on each invocation — demonstrating that Dispatch works with
any AI framework.

## How It Works

1. Reads the last-used framework index from long-term memory
2. Rotates to the next framework in the list
3. Sends the user's question to that framework
4. Saves the updated index to memory
5. Returns the answer along with which framework generated it

Rotation order: **Claude Agent SDK** → **OpenAI Agents SDK** → **LangGraph + OpenAI** → (repeat)

## Key Patterns Demonstrated

| Feature | How it's used |
|---|---|
| `@fn()` | Exposes `ask` as a callable function |
| **Long-term Memory** | `memory.long_term` to persist rotation state across invocations |
| Claude Agent SDK | `query()` with `ClaudeAgentOptions` |
| OpenAI Agents SDK | `Agent` + `Runner.run()` |
| LangGraph | `StateGraph` with `ChatOpenAI` |

## Prerequisites

- **ANTHROPIC_API_KEY** — for Claude Agent SDK
- **OPENAI_API_KEY** — for OpenAI Agents SDK and LangGraph

## Running

```bash
cd examples/multi-framework

# Run locally
dispatch agent dev

# Ask a question (uses next framework in rotation)
dispatch agent invoke ask '{"question": "What makes a good AI agent?"}'

# Ask again (rotates to next framework)
dispatch agent invoke ask '{"question": "What makes a good AI agent?"}'

# Compare the answers!
```

## Customisation

- Add or remove frameworks by editing the `FRAMEWORKS` list and `CALLERS` dict
- Change model defaults in each `_call_*` function
- Use a cron schedule to automatically cycle through frameworks on a timer
