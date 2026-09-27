"""Live checks for Module 2's rainfall history function."""

import json
import sys

from rainfall_history import get_rainfall_history
from weather import WeatherAPIError


def expect_value_error(location: str, days: int, label: str) -> None:
    try:
        get_rainfall_history(location, days)
    except ValueError as error:
        print(f"PASS {label}: {error}")
    except Exception as error:
        raise AssertionError(f"{label} raised the wrong error: {error}") from error
    else:
        raise AssertionError(f"{label} did not raise ValueError")


def main() -> None:
    try:
        result = get_rainfall_history("Thanjavur", 7)
        assert result["location"]["name"]
        assert isinstance(result["location"]["latitude"], (int, float))
        assert isinstance(result["location"]["longitude"], (int, float))
        assert result["number_of_days"] == 7
        assert len(result["rainfall_by_day"]) == 7
        assert all(
            isinstance(day["rainfall_mm"], (int, float))
            for day in result["rainfall_by_day"]
        )
        assert result["total_rainfall_mm"] == sum(
            day["rainfall_mm"] for day in result["rainfall_by_day"]
        )
    except (AssertionError, ValueError, WeatherAPIError) as error:
        print(f"FAIL Thanjavur rainfall history: {error}", file=sys.stderr)
        sys.exit(1)

    print("PASS Thanjavur: location, 7 daily values, and total verified")
    print(json.dumps(result, indent=2))
    expect_value_error("InvalidRandomPlace123456", 7, "unknown location")
    expect_value_error("Thanjavur", 0, "invalid day count")


if __name__ == "__main__":
    main()