# weather-service

A callable weather lookup agent. Exposes `get_weather` via `@fn()` so other agents
can call it directly with `invoke()`.

## What This Demonstrates

| Handler | Decorator | Pattern |
|---------|-----------|---------|
| `get_weather` | `@fn()` | Exposing a typed function for direct invocation by other agents |

## Key Pattern: `@fn()` for Direct Invocations

```python
from dispatch_agents import BasePayload, fn

class WeatherRequest(BasePayload):
    city: str

class WeatherResponse(BasePayload):
    city: str
    temperature: float
    conditions: str
    humidity: int

@fn()
async def get_weather(request: WeatherRequest) -> WeatherResponse:
    """Get current weather for a city."""
    ...
```

Other agents call this function by name:

```python
from dispatch_agents import invoke

result = await invoke(
    agent_name="weather-service",
    function_name="get_weather",
    payload={"city": "new york"},
)
```

See [weather-assistant](../weather-assistant/) for a working caller example.

## How to Run

```bash
cd examples/weather-service
dispatch agent deploy

# Call get_weather directly
dispatch function invoke --agent weather-service --function get_weather \
  --payload '{"city": "chicago"}'
```
