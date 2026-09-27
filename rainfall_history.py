"""Fetch daily rainfall history from Open-Meteo."""

from datetime import date, timedelta
from typing import Any

from weather import WeatherAPIError, _get_json, get_location_coordinates


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_rainfall_history(location: str, days: int) -> dict[str, Any]:
    """Return daily rainfall totals for the completed days before today."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive whole number.")

    place = get_location_coordinates(location)
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)

    # Open-Meteo's archive API supplies historical daily liquid rain in millimeters.
    data = _get_json(
        ARCHIVE_URL,
        {
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "rain_sum",
            "timezone": "auto",
            "precipitation_unit": "mm",
        },
    )
    daily = data.get("daily")
    if not isinstance(daily, dict):
        raise WeatherAPIError("Open-Meteo did not return daily rainfall data.")

    dates = daily.get("time")
    rainfall_values = daily.get("rain_sum")
    if (
        not isinstance(dates, list)
        or not isinstance(rainfall_values, list)
        or len(dates) != days
        or len(rainfall_values) != days
    ):
        raise WeatherAPIError(
            f"Open-Meteo did not return rainfall data for all {days} requested days."
        )

    rainfall_by_day = []
    for forecast_date, rainfall_mm in zip(dates, rainfall_values):
        if isinstance(rainfall_mm, bool) or not isinstance(rainfall_mm, (int, float)):
            raise WeatherAPIError(
                f"Open-Meteo has no rainfall value for {forecast_date}."
            )
        rainfall_by_day.append(
            {"date": forecast_date, "rainfall_mm": rainfall_mm}
        )

    return {
        "location": place,
        "number_of_days": days,
        "rainfall_by_day": rainfall_by_day,
        "total_rainfall_mm": sum(item["rainfall_mm"] for item in rainfall_by_day),
    }