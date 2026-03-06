"""Weather assistant that calls weather-service using invoke()."""

from dispatch_agents import BasePayload, invoke, on


class CityRequest(BasePayload):
    """Request to get weather info."""

    city: str


class AssistantResponse(BasePayload):
    """Response with formatted weather info."""

    message: str


@on(topic="weather-assistant.ask")
async def handle_weather_question(payload: CityRequest) -> AssistantResponse:
    """Handle a weather question by calling the weather-service.

    This demonstrates using invoke() to call a function on another agent.
    """
    print(f"[Assistant] Received question about weather in: {payload.city}")

    # Call the weather-service agent's get_weather function
    # This uses the new @fn / invoke() feature!
    result = await invoke(
        agent_name="weather-service",
        function_name="get_weather",
        payload={"city": payload.city},
    )

    print(f"[Assistant] Got response from weather-service: {result}")
    if "result" in result:
        result = result["result"]
    # Format the response nicely
    city = result.get("city", payload.city)

    temp = result.get("temperature", "unknown")
    conditions = result.get("conditions", "unknown")
    humidity = result.get("humidity", "unknown")

    message = f"Weather in {city}: {temp}°F, {conditions}, {humidity}% humidity"

    return AssistantResponse(message=message)
