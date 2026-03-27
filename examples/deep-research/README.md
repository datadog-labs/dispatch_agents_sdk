# Deep Research Agent

A comprehensive research agent that uses a supervisor/researcher pattern to
conduct deep, parallel web research on any topic and produce a detailed report.

## How It Works

1. **Research Brief** — the LLM refines your query into a structured research plan
2. **Supervisor** — delegates sub-topics to parallel researcher agents
3. **Researchers** — each searches the web via Tavily, reflects, and iterates
4. **Compression** — raw findings are cleaned and de-duplicated
5. **Final Report** — all findings are synthesised into a comprehensive markdown report with citations

## Key Patterns Demonstrated

| Feature | How it's used |
|---|---|
| `@fn()` | Exposes `deep_research` as a callable function |
| OpenAI SDK | Tool-calling loop for supervisor and researcher agents |
| Parallel execution | `asyncio.gather` for concurrent research and search |
| Tavily web search | Real-time web search with content summarisation |
| Structured output | JSON mode for research briefs and clarification |

## Prerequisites

- **OPENAI_API_KEY** — OpenAI API key
- **TAVILY_API_KEY** — Tavily search API key (free tier available at [tavily.com](https://tavily.com))

## Running

```bash
cd examples/deep-research

# Run locally
dispatch agent dev

# Invoke the function
dispatch agent invoke deep_research '{"query": "What are the latest breakthroughs in fusion energy?"}'
```

## Configuration

Edit `configuration.py` to adjust:
- Models (default: `gpt-4o-mini` for research, `gpt-4o` for final report)
- Max concurrent researchers (default: 5)
- Max iterations and tool calls
- Content length limits and timeouts
