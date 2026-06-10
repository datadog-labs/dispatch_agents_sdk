"""Static type-checking fixture for the public SDK surface.

This module is intentionally NOT named ``test_*`` so pytest does not collect it,
but it lives under ``sdk/`` so ``uv run mypy .`` checks it. Its job is to fail the
type-check gate if the public facades ever re-narrow the generics from
``_internal`` and erase concrete payload/response types again (the regression
class fixed alongside this file). ``examples/`` is excluded from mypy, so without
this fixture nothing exercises concretely-typed handlers under a checked path.

Nothing here is executed; the value is in the type assertions.
"""

from __future__ import annotations

from dispatch_agents import BasePayload, fn, invoke, on
from dispatch_agents.llm import parse_json
from dispatch_agents.models import InvocationResult, JsonValue, LLMResponse


class WeatherRequest(BasePayload):
    city: str


class WeatherResponse(BasePayload):
    temp: int


@fn()
async def get_weather(payload: WeatherRequest) -> WeatherResponse:
    return WeatherResponse(temp=72)


@on(topic="weather")
async def on_weather(payload: WeatherRequest) -> None:
    return None


async def _check_decorated_signature_preserved() -> None:
    # @fn must preserve the concrete return type, not erase it to BasePayload.
    resp: WeatherResponse = await get_weather(WeatherRequest(city="NYC"))
    _ = resp.temp


async def _check_invoke_typed() -> None:
    # invoke(..., response_model=Concrete) must return the concrete model.
    typed: WeatherResponse = await invoke(
        "weather-agent", "get_weather", {"city": "NYC"}, response_model=WeatherResponse
    )
    _ = typed.temp


async def _check_invoke_untyped() -> None:
    # invoke without response_model returns an InvocationResult wrapper.
    untyped: InvocationResult = await invoke(
        "weather-agent", "get_weather", {"city": "NYC"}
    )
    _ = untyped.result


def _check_parse_json(response: LLMResponse) -> None:
    # parse_json(resp, Concrete) narrows to the model; bare parse_json -> JsonValue.
    parsed: WeatherResponse = parse_json(response, WeatherResponse)
    _ = parsed.temp
    raw: JsonValue = parse_json(response)
    _ = raw
