"""Tests combining the Forecast Agent, RAG citations, and crop guidance."""

import asyncio
import json
import unittest
from types import SimpleNamespace

from crop_advisor_agent import CropAdvisorError, run_crop_advisor_agent
from forecast_agent import run_forecast_agent
from test_forecast_agent import FakeOpenAIClient


QUESTION = "I grow paddy near Thanjavur. Should I irrigate this week?"


class FakeCropAdviceClient:
    def __init__(
        self,
        *,
        invent_source: bool = False,
        include_dosage: bool = False,
    ) -> None:
        self.invent_source = invent_source
        self.include_dosage = include_dosage
        self.context: dict[str, object] = {}
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **request: object) -> SimpleNamespace:
        self.context = json.loads(request["messages"][1]["content"])
        sources = self.context["retrieved_sources"]
        source_document = (
            "invented/source.md"
            if self.invent_source
            else sources[0]["source_document"]
        )
        output = {
            "weather_summary": "The forecast is an estimate; review local conditions.",
            "recommendations": [
                {
                    "guidance": (
                        "Use pesticide 2 ml/l."
                        if self.include_dosage
                        else "Check field water before deciding whether to irrigate."
                    ),
                    "reason": "The retrieved TNAU paddy source describes water needs at different crop stages.",
                    "source_documents": [source_document],
                }
            ],
            "uncertainties": ["The weather forecast may change."],
            "pesticide_dosage": None,
        }
        message = SimpleNamespace(content=json.dumps(output))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class CropAdvisorAgentTests(unittest.TestCase):
    def test_requested_paddy_question_combines_forecast_and_rag_sources(self) -> None:
        weather = asyncio.run(
            run_forecast_agent(
                QUESTION,
                "paddy",
                "Thanjavur",
                llm_client=FakeOpenAIClient(
                    [
                        "get_forecast",
                        "get_rainfall_history",
                        "get_soil_moisture",
                    ]
                ),
                model="test-model",
            )
        )
        fake_llm = FakeCropAdviceClient()
        result = asyncio.run(
            run_crop_advisor_agent(
                QUESTION,
                "paddy",
                "Thanjavur",
                weather_information=weather,
                llm_client=fake_llm,
                model="test-model",
            )
        )

        self.assertEqual(result["crop"], "paddy")
        self.assertEqual(result["location"], "Thanjavur")
        self.assertTrue(result["weather_is_estimated"])
        self.assertTrue(result["sources"])
        self.assertTrue(
            result["recommendations"][0]["source_documents"][0].startswith("paddy/")
        )
        self.assertIsNone(result["pesticide_dosage"])
        self.assertIn("No pesticide dosage was requested", result["pesticide_dosage_note"])

        allowed_sources = {source["source_document"] for source in result["sources"]}
        for recommendation in result["recommendations"]:
            self.assertTrue(
                set(recommendation["source_documents"]) <= allowed_sources
            )

    def test_unavailable_pesticide_dosage_is_not_invented(self) -> None:
        source = {
            "chunk": "For paddy, maintain shallow water after transplanting.",
            "source": {
                "source_document": "paddy/cultivation_and_irrigation.md",
                "title": "Paddy irrigation",
                "source_url": "https://example.test/tnau",
                "source_organization": "TNAU",
            },
        }
        result = asyncio.run(
            run_crop_advisor_agent(
                "What pesticide dosage should I apply?",
                "paddy",
                "Thanjavur",
                weather_information={"summary": "Forecast estimate."},
                retrieved_sources=[source],
                llm_client=FakeCropAdviceClient(),
                model="test-model",
            )
        )

        self.assertIsNone(result["pesticide_dosage"])
        self.assertEqual(
            result["pesticide_dosage_note"],
            "Pesticide dosage is not available in the retrieved sources.",
        )

    def test_unsupported_dosage_in_recommendation_text_is_rejected(self) -> None:
        source = {
            "chunk": "For paddy, maintain shallow water after transplanting.",
            "source": {
                "source_document": "paddy/cultivation_and_irrigation.md",
                "title": "Paddy irrigation",
                "source_url": "https://example.test/tnau",
                "source_organization": "TNAU",
            },
        }

        with self.assertRaisesRegex(CropAdvisorError, "dosage.*retrieved sources"):
            asyncio.run(
                run_crop_advisor_agent(
                    "Should I spray pesticide?",
                    "paddy",
                    "Thanjavur",
                    weather_information={"summary": "Forecast estimate."},
                    retrieved_sources=[source],
                    llm_client=FakeCropAdviceClient(include_dosage=True),
                    model="test-model",
                )
            )

    def test_advice_with_unretrieved_citation_is_rejected(self) -> None:
        source = {
            "chunk": "Maintain shallow water after transplanting.",
            "source": {
                "source_document": "paddy/cultivation_and_irrigation.md",
                "title": "Paddy irrigation",
                "source_url": "https://example.test/tnau",
                "source_organization": "TNAU",
            },
        }
        fake_llm = FakeCropAdviceClient(invent_source=True)

        with self.assertRaisesRegex(CropAdvisorError, "not retrieved"):
            asyncio.run(
                run_crop_advisor_agent(
                    QUESTION,
                    "paddy",
                    "Thanjavur",
                    weather_information={"summary": "Forecast estimate."},
                    retrieved_sources=[source],
                    llm_client=fake_llm,
                    model="test-model",
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)