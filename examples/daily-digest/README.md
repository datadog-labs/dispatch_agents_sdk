# Daily Digest Agent

A scheduled agent that fetches top Hacker News stories, filters out previously
reported ones, and uses an LLM to generate a concise newsletter-style digest.

## How It Works

1. Fetches top stories from the [Hacker News API](https://github.com/HackerNews/API) (no key required)
2. Checks long-term memory to skip stories from previous runs
3. Sends new stories to the LLM Gateway for summarisation
4. Saves reported story IDs to memory for next time
5. Returns a formatted digest

## Key Patterns Demonstrated

| Feature | How it's used |
|---|---|
| `@fn()` | Exposes `generate_digest` as a callable function |
| **Schedules (cron)** | Designed to run on a recurring schedule |
| **LLM Gateway** | `llm.inference()` to summarise stories |
| **Long-term Memory** | `memory.long_term` to track reported stories across runs |
| External API | Fetches from Hacker News (no API key needed) |

## Prerequisites

- LLM Gateway configured (`dispatch llm setup`)

## Running

```bash
cd examples/daily-digest

# Run locally
dispatch agent dev

# Invoke manually
dispatch agent invoke generate_digest '{}'

# Set up a cron schedule (weekdays at 9am ET)
dispatch schedule create daily-digest generate_digest \
    --cron "0 9 * * MON-FRI" \
    --timezone "America/New_York" \
    --payload '{}'
```

## Customisation

- Change `MAX_STORIES` to control how many stories per digest
- Swap the Hacker News API for any data source (RSS feeds, GitHub activity, etc.)
- Edit the system prompt to change the digest style (formal, casual, bullet-point)
