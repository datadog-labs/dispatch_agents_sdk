# Conversational Agent

A multi-turn chat assistant that remembers conversation history using Dispatch's
short-term memory and calls LLMs through the built-in LLM Gateway.

## How It Works

1. Receives a message with a `session_id`
2. Loads previous conversation history from short-term memory
3. Sends the full conversation to the LLM via `llm.inference()`
4. Saves the updated history back to memory
5. Returns the assistant's reply

Send multiple messages with the same `session_id` to have a multi-turn conversation.

## Key Patterns Demonstrated

| Feature | How it's used |
|---|---|
| `@fn()` | Exposes `chat` as a callable function |
| LLM Gateway | `llm.inference()` — call any configured LLM provider without API keys in code |
| Short-term Memory | `memory.short_term` — persist conversation history per session |
| Typed payloads | Pydantic `BasePayload` for input/output validation |

## Prerequisites

- LLM Gateway configured (`dispatch llm setup`)

## Running

```bash
cd examples/conversational-agent

# Run locally
dispatch agent dev

# First message
dispatch agent invoke chat '{"session_id": "demo-1", "message": "What is the capital of France?"}'

# Follow-up (same session — agent remembers context)
dispatch agent invoke chat '{"session_id": "demo-1", "message": "What about Germany?"}'
```

## Customisation

Edit the `SYSTEM_PROMPT` in `agent.py` to change the assistant's personality,
domain expertise, or response style. For example, make it a customer support
agent, a coding tutor, or a domain-specific advisor.
