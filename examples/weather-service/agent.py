"""Weather service agent with callable function."""

import logging

from dispatch_agents import BasePayload, fn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeatherRequest(BasePayload):
    """Request payload for weather lookup."""

    city: str


class WeatherResponse(BasePayload):
    """Response with weather data."""

    city: str
    temperature: float
    conditions: str
    humidity: int


# Simulated weather data
WEATHER_DATA = {
    "new york": {"temperature": 45.0, "conditions": "cloudy", "humidity": 65},
    "los angeles": {"temperature": 72.0, "conditions": "sunny", "humidity": 40},
    "chicago": {"temperature": 38.0, "conditions": "windy", "humidity": 55},
    "miami": {"temperature": 82.0, "conditions": "humid", "humidity": 85},
    "seattle": {"temperature": 52.0, "conditions": "rainy", "humidity": 75},
    "berlin": {"temperature": 35.0, "conditions": "cloudy", "humidity": 70},
}


@fn()
async def get_weather(request: WeatherRequest) -> WeatherResponse:
    """Get current weather for a city.

    This function is callable by other agents using:
        result = await invoke("weather-service", "get_weather", {"city": "new york"})
    """
    logger.info(f"[weather-service] get_weather called with request: {request}")
    city_lower = request.city.lower()

    if city_lower in WEATHER_DATA:
        data = WEATHER_DATA[city_lower]
        response = WeatherResponse(
            city=request.city,
            temperature=data["temperature"],
            conditions=data["conditions"],
            humidity=data["humidity"],
        )
        logger.info(f"[weather-service] Returning response: {response}")
        logger.info(f"[weather-service] Response as dict: {response.model_dump()}")
        return response

    # Default response for unknown cities
    response = WeatherResponse(
        city=request.city,
        temperature=70.0,
        conditions="unknown",
        humidity=50,
    )
    logger.info(f"[weather-service] Returning default response: {response}")
    logger.info(f"[weather-service] Response as dict: {response.model_dump()}")
    return response
