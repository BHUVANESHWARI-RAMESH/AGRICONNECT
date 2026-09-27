"""Unit tests for the mocked soil-moisture module."""

import unittest

from soil_moisture import get_soil_moisture


class SoilMoistureTests(unittest.TestCase):
    def test_returns_explicitly_mocked_estimate(self) -> None:
        result = get_soil_moisture("Thanjavur")

        self.assertEqual(result["location"], "Thanjavur")
        self.assertEqual(result["status"], "MOCKED/ESTIMATED")
        self.assertIn("not real sensor data", result["source"])
        self.assertIsInstance(result["soil_moisture_percent"], float)
        self.assertGreaterEqual(result["soil_moisture_percent"], 15.0)
        self.assertLessEqual(result["soil_moisture_percent"], 60.0)

    def test_estimate_is_deterministic_and_case_insensitive(self) -> None:
        first = get_soil_moisture("Thanjavur")
        second = get_soil_moisture("  THANJAVUR  ")

        self.assertEqual(
            first["soil_moisture_percent"], second["soil_moisture_percent"]
        )

    def test_empty_or_non_string_location_is_rejected(self) -> None:
        for location in ("", "   ", None, 42):
            with self.subTest(location=location):
                with self.assertRaisesRegex(ValueError, "non-empty location"):
                    get_soil_moisture(location)


if __name__ == "__main__":
    unittest.main()