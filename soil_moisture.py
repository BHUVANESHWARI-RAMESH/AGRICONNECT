"""Provide a clearly labeled mock soil-moisture estimate."""

import hashlib
from typing import Any


def get_soil_moisture(location: str) -> dict[str, Any]:
    """Return a stable mock estimate for a non-empty location name.

    This value is for development only; it is not measured soil or sensor data.
    """
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Please provide a non-empty location name.")

    clean_location = location.strip()
    normalized_location = clean_location.casefold()

    # A digest keeps this mock repeatable across calls and Python processes.
    digest = hashlib.sha256(normalized_location.encode("utf-8")).digest()
    number = int.from_bytes(digest[:4], byteorder="big")
    estimated_percent = round(15 + (number % 451) / 10, 1)

    return {
        "location": clean_location,
        "soil_moisture_percent": estimated_percent,
        "status": "MOCKED/ESTIMATED",
        "source": "Deterministic mock value; not real sensor data.",
    }