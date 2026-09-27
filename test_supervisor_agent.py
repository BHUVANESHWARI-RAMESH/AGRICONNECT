"""Tests for explicit request extraction and missing-field clarification."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from supervisor_agent import extract_farmer_request


class FakeSupervisorClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **request: object) -> SimpleNamespace:
        message = SimpleNamespace(content=json.dumps(self.response))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class SupervisorAgentTests(unittest.TestCase):
    def test_extracts_crop_location_activity_and_time(self) -> None:
        message = "I grow cotton in Erode. Can I spray tomorrow?"
        result = asyncio.run(
            extract_farmer_request(
                message,
                llm_client=FakeSupervisorClient(
                    {
                        "crop": "cotton",
                        "location": "Erode",
                        "question": "Can I spray tomorrow?",
                        "activity": "spraying",
                        "time_period": "tomorrow",
                    }
                ),
            )
        )

        self.assertEqual(result["crop"], "cotton")
        self.assertEqual(result["location"], "Erode")
        self.assertEqual(result["question"], "Can I spray tomorrow?")
        self.assertEqual(result["activity"], "spraying")
        self.assertEqual(result["time_period"], "tomorrow")
        self.assertFalse(result["needs_clarification"])
        self.assertIsNone(result["agricultural_advice"])

    def test_missing_crop_prompts_without_guessing(self) -> None:
        result = asyncio.run(
            extract_farmer_request(
                "I am farming near Erode. Should I spray tomorrow?",
                llm_client=FakeSupervisorClient(
                    {
                        "crop": "cotton",
                        "location": "Erode",
                        "question": "Should I spray tomorrow?",
                        "activity": "spraying",
                        "time_period": "tomorrow",
                    }
                ),
            )
        )

        self.assertIsNone(result["crop"])
        self.assertEqual(result["location"], "Erode")
        self.assertEqual(result["missing_information"], ["crop"])
        self.assertEqual(result["clarifying_question"], "What is the crop?")

    def test_missing_location_prompts_without_guessing(self) -> None:
        result = asyncio.run(
            extract_farmer_request(
                "I grow paddy. Should I irrigate this week?",
                llm_client=FakeSupervisorClient(
                    {
                        "crop": "paddy",
                        "location": "Thanjavur",
                        "question": "Should I irrigate this week?",
                        "activity": "irrigation",
                        "time_period": "this week",
                    }
                ),
            )
        )

        self.assertEqual(result["crop"], "paddy")
        self.assertIsNone(result["location"])
        self.assertEqual(result["missing_information"], ["location"])
        self.assertEqual(
            result["clarifying_question"], "What is the location?"
        )

    def test_missing_crop_and_location_prompts_for_both(self) -> None:
        result = asyncio.run(
            extract_farmer_request(
                "Can I irrigate this week?",
                llm_client=FakeSupervisorClient(
                    {
                        "crop": None,
                        "location": None,
                        "question": "Can I irrigate this week?",
                        "activity": "irrigation",
                        "time_period": "this week",
                    }
                ),
            )
        )

        self.assertEqual(result["missing_information"], ["crop", "location"])
        self.assertTrue(result["needs_clarification"])


if __name__ == "__main__":
    unittest.main(verbosity=2)