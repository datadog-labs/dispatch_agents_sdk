# Dispatch Agents SDK

Build and deploy event-driven AI agents with simple Python decorators.
[dispatchagents.ai](https://dispatchagents.ai/) — currently in public preview.

## Installation

```bash
uv add git+ssh://git@github.com/datadog-labs/dispatch_agents_sdk.git@vX.Y.Z
```

Replace `vX.Y.Z` with the latest tag:

```bash
git ls-remote --tags git@github.com:datadog-labs/dispatch_agents_sdk.git 'v*' \
  | sort -t'/' -k3 -V | tail -1 | awk -F'/' '{print $3}'
```

## Core Concepts

### `@on` — Subscribe to events

Handle events published to a topic. The payload is automatically validated
against your Pydantic model before your handler is called.

```python
from dispatch_agents import BasePayload, on

class OrderPayload(BasePayload):
    order_id: str
    customer_email: str

class OrderResult(BasePayload):
    status: str

@on(topic="orders.process")
async def process_order(payload: OrderPayload) -> OrderResult:
    await send_confirmation(payload.customer_email, payload.order_id)
    return OrderResult(status="confirmed")
```

Subscribe to GitHub events by passing a GitHub event class:

```python
from dispatch_agents import on
from dispatch_agents.integrations.github import PullRequestOpened

@on(github_event=PullRequestOpened)
async def handle_pr(payload: PullRequestOpened) -> None:
    print(f"PR opened: {payload.pull_request.title}")
```

### `@fn` — Expose a callable function

Register a function that other agents can call directly by name using `invoke()`.

```python
from dispatch_agents import BasePayload, fn

class WeatherRequest(BasePayload):
    city: str

class WeatherResponse(BasePayload):
    temperature: float
    conditions: str

@fn()
async def get_weather(request: WeatherRequest) -> WeatherResponse:
    data = await fetch_weather_api(request.city)
    return WeatherResponse(temperature=data["temp"], conditions=data["sky"])
```

### `@init` — One-time async setup

Runs once before any events are processed. Use for connecting to databases,
loading models, or initializing shared state.

```python
from dispatch_agents import BasePayload, init, on

_client = None

@init
async def setup():
    global _client
    _client = await connect_to_database()

class QueryPayload(BasePayload):
    user_id: str

@on(topic="users.query")
async def query_users(payload: QueryPayload) -> QueryResult:
    rows = await _client.fetch(
        "select * from users where id = $1",
        payload.user_id,
    )
    return QueryResult(rows=rows)
```

### `emit_event` — Fire-and-forget publishing

Publish an event without waiting for handlers to process it.

```python
from dispatch_agents import emit_event, on

@on(topic="orders.process")
async def process_order(payload: OrderPayload) -> OrderResult:
    await fulfill(payload)
    # Notify other agents — returns immediately
    await emit_event("notifications.send", {"email": payload.customer_email})
    return OrderResult(status="fulfilled")
```

### `invoke` — Call another agent directly

Call a `@fn`-decorated function on another agent and wait for the result.

```python
from dispatch_agents import invoke, on

@on(topic="reports.request")
async def handle_report(payload: ReportPayload) -> ReportResult:
    weather = await invoke(
        agent_name="weather-service",
        function_name="get_weather",
        payload={"city": payload.city},
        response_model=WeatherResponse,
        timeout=30.0,
    )
    return ReportResult(weather=weather.conditions)
```

## Error Handling

- **`OSError` and subclasses** → automatic retry with exponential backoff
- **`ValueError` and subclasses** → terminal failure, not retried

```python
from dispatch_agents import on

@on(topic="data.fetch")
async def fetch(payload: FetchPayload) -> FetchResult:
    if not payload.url.startswith("https://"):
        raise ValueError("Only HTTPS URLs allowed")  # not retried

    # TimeoutError (an OSError subclass) will be retried automatically
    response = await httpx_client.get(payload.url, timeout=10.0)
    return FetchResult(data=response.json())
```

## Payloads

All handler inputs and outputs must inherit from `BasePayload`:

```python
from dispatch_agents import BasePayload
from typing import Optional

class TaskPayload(BasePayload):
    task_id: str
    priority: int = 1
    tags: list[str] = []
    metadata: Optional[dict[str, str]] = None
```

## Examples

- [`hello_world`](../examples/hello_world/) — typed payloads, error handling, persistent storage
- [`weather-service`](../examples/weather-service/) + [`weather-assistant`](../examples/weather-assistant/) — inter-agent `invoke()`
- [`conversational-agent`](../examples/conversational-agent/) — multi-turn LLM chat with memory
- [`daily-digest`](../examples/daily-digest/) — scheduled agents
- [`deep-research`](../examples/deep-research/) — parallel supervisor/researcher pattern
- [`multi-framework`](../examples/multi-framework/) — Claude SDK, OpenAI Agents, LangGraph

## Full Documentation

[dispatchagents.ai/docs/sdk](https://dispatchagents.ai/docs/sdk)

## Support

GitHub Issues: [github.com/datadog-labs/dispatch_agents_sdk/issues](https://github.com/datadog-labs/dispatch_agents_sdk/issues)
