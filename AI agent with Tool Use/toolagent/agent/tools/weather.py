"""Weather tool — simulated weather data for cities worldwide.

Demonstrates:
- External API wrapper pattern
- Fallback/error handling
- Caching pattern
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from agent.tools.base import BaseTool

# Simulated weather database for demo mode
_CITY_WEATHER: dict[str, dict[str, Any]] = {
    "new york": {
        "temp_c": 18, "temp_f": 64, "condition": "Partly Cloudy",
        "humidity": 55, "wind_kph": 15, "forecast": [
            {"day": "Mon", "high": 20, "low": 14, "condition": "Sunny"},
            {"day": "Tue", "high": 22, "low": 15, "condition": "Cloudy"},
            {"day": "Wed", "high": 19, "low": 13, "condition": "Rain"},
        ],
    },
    "tokyo": {
        "temp_c": 22, "temp_f": 72, "condition": "Clear",
        "humidity": 60, "wind_kph": 10, "forecast": [
            {"day": "Mon", "high": 24, "low": 18, "condition": "Clear"},
            {"day": "Tue", "high": 25, "low": 19, "condition": "Sunny"},
            {"day": "Wed", "high": 23, "low": 17, "condition": "Cloudy"},
        ],
    },
    "london": {
        "temp_c": 12, "temp_f": 54, "condition": "Overcast",
        "humidity": 72, "wind_kph": 20, "forecast": [
            {"day": "Mon", "high": 14, "low": 9, "condition": "Overcast"},
            {"day": "Tue", "high": 13, "low": 8, "condition": "Rain"},
            {"day": "Wed", "high": 15, "low": 10, "condition": "Cloudy"},
        ],
    },
    "paris": {
        "temp_c": 16, "temp_f": 61, "condition": "Sunny",
        "humidity": 50, "wind_kph": 12, "forecast": [
            {"day": "Mon", "high": 18, "low": 12, "condition": "Sunny"},
            {"day": "Tue", "high": 20, "low": 13, "condition": "Sunny"},
            {"day": "Wed", "high": 17, "low": 11, "condition": "Cloudy"},
        ],
    },
    "sydney": {
        "temp_c": 26, "temp_f": 79, "condition": "Sunny",
        "humidity": 45, "wind_kph": 18, "forecast": [
            {"day": "Mon", "high": 28, "low": 20, "condition": "Sunny"},
            {"day": "Tue", "high": 27, "low": 19, "condition": "Sunny"},
            {"day": "Wed", "high": 25, "low": 18, "condition": "Partly Cloudy"},
        ],
    },
    "beijing": {
        "temp_c": 20, "temp_f": 68, "condition": "Hazy",
        "humidity": 40, "wind_kph": 8, "forecast": [
            {"day": "Mon", "high": 22, "low": 15, "condition": "Hazy"},
            {"day": "Tue", "high": 24, "low": 16, "condition": "Sunny"},
            {"day": "Wed", "high": 21, "low": 14, "condition": "Cloudy"},
        ],
    },
}

_FALLBACK = {
    "temp_c": 20, "temp_f": 68, "condition": "Not available",
    "humidity": 50, "wind_kph": 10, "forecast": [],
}


class WeatherTool(BaseTool):
    """Simulated weather data tool for cities worldwide."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Get current weather and a 3-day forecast for a city. "
            "Use this when you need weather information. "
            "Supports major cities worldwide."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo', 'New York', 'London'",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Units: metric (Celsius) or imperial (Fahrenheit). Default: metric",
                    "default": "metric",
                },
            },
            "required": ["city"],
        }

    async def _run(self, city: str, units: str = "metric") -> dict[str, Any]:
        city_key = city.lower().strip()
        weather_data = _CITY_WEATHER.get(city_key, _FALLBACK)

        temp_key = "temp_c" if units == "metric" else "temp_f"
        temp_unit = "°C" if units == "metric" else "°F"

        if city_key not in _CITY_WEATHER:
            # Generate simulated data for any city
            import hashlib
            seed = int(hashlib.md5(city_key.encode()).hexdigest()[:8], 16)
            import random
            rng = random.Random(seed)
            return {
                "city": city,
                "temperature": f"{rng.randint(10, 35)}{temp_unit}",
                "condition": rng.choice(["Sunny", "Cloudy", "Partly Cloudy", "Clear", "Overcast"]),
                "humidity": f"{rng.randint(30, 80)}%",
                "wind": f"{rng.randint(5, 25)} km/h",
                "forecast": [
                    {"day": "Mon", "high": rng.randint(15, 35), "low": rng.randint(5, 20), "condition": rng.choice(["Sunny", "Cloudy", "Rain"])},
                    {"day": "Tue", "high": rng.randint(15, 35), "low": rng.randint(5, 20), "condition": rng.choice(["Sunny", "Cloudy", "Rain"])},
                    {"day": "Wed", "high": rng.randint(15, 35), "low": rng.randint(5, 20), "condition": rng.choice(["Sunny", "Cloudy", "Rain"])},
                ],
                "last_updated": "2025-04-15T12:00:00",
            }

        return {
            "city": city,
            "temperature": f"{weather_data[temp_key]}{temp_unit}",
            "condition": weather_data["condition"],
            "humidity": f"{weather_data['humidity']}%",
            "wind": f"{weather_data['wind_kph']} km/h",
            "forecast": weather_data["forecast"],
            "last_updated": datetime.utcnow().isoformat(),
        }
