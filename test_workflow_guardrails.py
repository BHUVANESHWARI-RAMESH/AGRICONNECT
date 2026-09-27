"""Unit tests for Module 12 safety checks used by the LangGraph workflow."""

import unittest

from workflow_guardrails import (
    validate_alert_result,
    validate_crop_guidance,
    validate_supervisor_result,
)


class WorkflowGuardrailTests(unittest.TestCase):
    def test_missing_crop_and_location_are_returned_for_clarification(self) -> None:
        result = validate_supervisor_result({"question": "Should I irrigate?"})

        self.assertEqual(result["missing_information"], ["crop", "location"])
        self.assertTrue(result["needs_clarification"])
        self.assertIsNone(result["agricultural_advice"])

    def test_alert_requires_recent_regional_cited_warning(self) -> None:
        result = validate_alert_result(
            {
                "alert_found": True,
                "alerts": [
                    {
                        "title": "Undated alert",
                        "current_region_warning": True,
                        "date_status": "date_unknown",
                        "region_match": True,
                        "source_url": "https://example.test/alert",
                    }
                ],
            }
        )

        self.assertFalse(result["alert_found"])
        self.assertEqual(result["alerts"], [])

    def test_agricultural_recommendation_must_cite_retrieved_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            validate_crop_guidance(
                {
                    "sources": [{"source_document": "paddy/guide.md", "excerpt": "water"}],
                    "recommendations": [
                        {"guidance": "Do this", "source_documents": ["invented.md"]}
                    ],
                }
            )

    def test_pesticide_dosage_must_match_a_retrieved_excerpt(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported pesticide dosage"):
            validate_crop_guidance(
                {
                    "sources": [{"source_document": "paddy/guide.md", "excerpt": "Use careful water management."}],
                    "recommendations": [
                        {
                            "guidance": "Apply pesticide 2 ml/l.",
                            "source_documents": ["paddy/guide.md"],
                        }
                    ],
                }
            )

    def test_weather_uncertainty_is_always_marked(self) -> None:
        result = validate_crop_guidance(
            {
                "sources": [{"source_document": "paddy/guide.md", "excerpt": "water"}],
                "recommendations": [
                    {"guidance": "Check field water.", "source_documents": ["paddy/guide.md"]}
                ],
            }
        )

        self.assertTrue(result["weather_is_estimated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)