"""Tests the deterministic report writer with a paddy irrigation example."""

import unittest

from report_writer_agent import write_advisory_report


class ReportWriterTests(unittest.TestCase):
    def test_renders_supplied_paddy_report_fields(self) -> None:
        report = write_advisory_report(
            {
                "farmer_question": "I grow paddy near Thanjavur. Should I irrigate this week?",
                "crop": "paddy",
                "location": "Thanjavur",
                "weather_summary": "Forecast indicates a chance of rain later this week.",
                "weather_uncertainty": "The forecast is an estimate and may change.",
                "rainfall_history": {
                    "number_of_days": 7,
                    "total_rainfall_mm": 12.5,
                },
                "soil_moisture": {
                    "soil_moisture_percent": 32.4,
                    "status": "MOCKED/ESTIMATED",
                },
                "agricultural_guidance": {
                    "overall_recommendation": "Check field water before irrigating.",
                    "recommendations": [
                        {
                            "guidance": "Maintain the stage-specific water level.",
                            "reason": "The retrieved TNAU paddy source describes critical water stages.",
                            "source_documents": ["paddy/cultivation_and_irrigation.md"],
                        }
                    ],
                    "what_to_avoid": ["Do not rely on the forecast as a guarantee."],
                    "sources": [
                        {
                            "title": "Paddy water management",
                            "source_document": "paddy/cultivation_and_irrigation.md",
                            "source_url": "https://agritech.tnau.ac.in/example",
                            "source_organization": "TNAU",
                        }
                    ],
                    "pesticide_dosage_note": "Pesticide dosage is not available in the retrieved sources.",
                },
                "current_alerts": {
                    "alert_found": False,
                    "alert_description": "No current regional alert was confirmed in the supplied search results.",
                },
                "day_by_day_advisory": ["Day 1: Review the latest local forecast."],
            }
        )

        self.assertEqual(report["crop"], "paddy")
        self.assertEqual(report["location"], "Thanjavur")
        self.assertEqual(len(report["what_to_do"]), 1)
        self.assertIn("12.5 mm", report["rainfall_history"])
        self.assertIn("not real sensor data", report["soil_moisture"])
        self.assertEqual(len(report["day_by_day_advisory"]), 1)
        self.assertIn("No current regional alert", report["active_alerts"][0])
        self.assertIn("https://agritech.tnau.ac.in/example", report["report_text"])
        self.assertIn("forecast is an estimate", report["weather_uncertainty"])

    def test_unverified_pesticide_dosage_is_omitted(self) -> None:
        report = write_advisory_report(
            {
                "crop": "paddy",
                "location": "Thanjavur",
                "agricultural_guidance": {
                    "recommendations": [
                        {"guidance": "Apply pesticide 2 ml/l."}
                    ]
                },
                "sources": [
                    {
                        "source_document": "paddy/irrigation.md",
                        "excerpt": "Maintain water during critical crop stages.",
                    }
                ],
            }
        )

        self.assertEqual(report["what_to_do"], [])
        self.assertIn("unverified pesticide dosage", report["pesticide_dosage_note"])
        self.assertNotIn("2 ml/l", report["report_text"])

    def test_missing_inputs_are_reported_instead_of_filled(self) -> None:
        report = write_advisory_report({"crop": "cotton", "location": "Erode"})

        self.assertEqual(report["overall_recommendation"], "No overall recommendation was supplied; the report only summarizes provided information.")
        self.assertIn("No current alert information", report["active_alerts"][0])
        self.assertIn("No agricultural sources", report["report_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)