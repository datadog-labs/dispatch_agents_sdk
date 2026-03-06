# Dispatch SDK

Dispatch Agents is a platform for building and deploying AI agents, currently in public preview at [dispatchagents.ai](https://dispatchagents.ai/). This SDK lets you define agent handlers using simple Python decorators and run them anywhere — locally or in the cloud.

## Installation

### From GitHub (Recommended)

```bash
# Get the latest SDK version tag
git ls-remote --tags git@github.com:datadog-labs/dispatch_agents_sdk.git 'v*' | sort -t'/' -k3 -V | tail -1 | awk -F'/' '{print $3}'

# Install using the version tag (replace vX.Y.Z with the output above)
uv add git+ssh://git@github.com/datadog-labs/dispatch_agents_sdk.git@vX.Y.Z
```

## Usage

Create agents by decorating functions with `@dispatch_agents.fn`:

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
    """Get current weather for a city.

    This function is callable by other agents using:
        result = await invoke("weather-service", "get_weather", {"city": "new york"})
    """
    return WeatherResponse(city=request.city, temperature=72.0, conditions="sunny", humidity=40)
```

## Support

- **GitHub Issues:** [github.com/datadog-labs/dispatch_agents_sdk/issues](https://github.com/datadog-labs/dispatch_agents_sdk/issues)
