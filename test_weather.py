"""Small live API check for the weather module."""

import json
import sys

from weather import WeatherAPIError, get_7_day_forecast


def main() -> None:
    try:
        result = get_7_day_forecast("Thanjavur")
    except (WeatherAPIError, ValueError) as error:
        print(f"Weather lookup failed: {error}", file=sys.stderr)
        sys.exit(1)

    if len(result["forecast"]) != 7:
        print("Expected exactly seven forecast days.", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()