# weather-assistant

An orchestrator agent that answers weather questions by calling `weather-service`
via `invoke()`. The primary example of inter-agent communication.

## What This Demonstrates

| Handler | Decorator | Pattern |
|---------|-----------|---------|
| `handle_weather_question` | `@on(topic="weather-assistant.ask")` | Calling another agent with `invoke()` and using the result |

## Key Pattern: `invoke()` for Inter-Agent Calls

```python
from dispatch_agents import BasePayload, invoke, on

@on(topic="weather-assistant.ask")
async def handle_weather_question(payload: CityRequest) -> AssistantResponse:
    # Call the weather-service agent's get_weather function directly
    result = await invoke(
        agent_name="weather-service",
        function_name="get_weather",
        payload={"city": payload.city},
    )
    message = f"Weather in {result['city']}: {result['temperature']}°F, {result['conditions']}"
    return AssistantResponse(message=message)
```

`invoke()` blocks until the target function returns. For durable fire-and-forget
work, publish an event with `emit_event()` instead.

## How to Run

`weather-service` must be deployed first:

```bash
cd examples/weather-service && dispatch agent deploy
cd examples/weather-assistant && dispatch agent deploy

# Ask a weather question
dispatch event publish --topic "weather-assistant.ask" \
  --payload '{"city": "Seattle"}'
```

## Related

- [weather-service](../weather-service/) — the `@fn()` service this agent calls
- [Inter-Agent Communication guide](/docs/sdk-inter-agent)
