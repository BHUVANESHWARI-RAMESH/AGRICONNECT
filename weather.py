"""Fetch a seven-day weather forecast from Open-Meteo."""

from typing import Any

import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherAPIError(Exception):
    """Raised when Open-Meteo cannot provide a usable response."""


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Request JSON from an API endpoint and report network/API errors clearly."""
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        raise WeatherAPIError(f"Could not contact Open-Meteo: {error}") from error

    if not isinstance(data, dict):
        raise WeatherAPIError("Open-Meteo returned an unexpected response.")
    if data.get("error"):
        reason = data.get("reason", "The request was rejected.")
        raise WeatherAPIError(f"Open-Meteo error: {reason}")
    return data


def get_location_coordinates(location: str) -> dict[str, Any]:
    """Find a location and return its name, country, latitude, and longitude."""
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Please provide a non-empty location name.")

    geocoding_data = _get_json(
        GEOCODING_URL,
        {"name": location.strip(), "count": 1, "language": "en", "format": "json"},
    )
    results = geocoding_data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(f"Location not found: {location}")

    place = results[0]
    if not isinstance(place, dict):
        raise WeatherAPIError("Open-Meteo returned invalid location data.")

    latitude = place.get("latitude")
    longitude = place.get("longitude")
    if latitude is None or longitude is None:
        raise WeatherAPIError("Open-Meteo did not return coordinates for this location.")

    return {
        "name": place.get("name", location.strip()),
        "country": place.get("country"),
        "latitude": latitude,
        "longitude": longitude,
    }


def get_7_day_forecast(location: str) -> dict[str, Any]:
    """Return a seven-day forecast for a city name such as ``Thanjavur``."""
    place = get_location_coordinates(location)

    # Ask for daily values in the location's local time zone.
    forecast_data = _get_json(
        FORECAST_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "relative_humidity_2m_mean",
                    "wind_speed_10m_max",
                ]
            ),
            "forecast_days": 7,
            "timezone": "auto",
        },
    )
    daily = forecast_data.get("daily", {})
    dates = daily.get("time", [])
    fields = {
        "temperature_2m_max": "temperature_max_c",
        "temperature_2m_min": "temperature_min_c",
        "precipitation_sum": "precipitation_mm",
        "relative_humidity_2m_mean": "humidity_mean_percent",
        "wind_speed_10m_max": "wind_speed_max_kmh",
    }
    if not isinstance(dates, list) or len(dates) != 7:
        raise WeatherAPIError("Open-Meteo did not return seven forecast dates.")
    if any(
        not isinstance(daily.get(field), list) or len(daily[field]) != 7
        for field in fields
    ):
        raise WeatherAPIError("Open-Meteo returned incomplete daily weather data.")

    forecast = []
    for day_index, date in enumerate(dates):
        day = {"date": date}
        for api_field, result_field in fields.items():
            day[result_field] = daily[api_field][day_index]
        forecast.append(day)

    return {
        "location": {
            "name": place["name"],
            "country": place["country"],
            "latitude": place["latitude"],
            "longitude": place["longitude"],
        },
        "forecast": forecast,
    }