"""LangGraph scenario tests and one integrated existing-agent execution."""

import asyncio
import json
import logging
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from workflow import build_workflow, run_workflow


SCENARIOS = [
    (
        "I grow paddy near Thanjavur. Should I irrigate this week?",
        "paddy",
        "Thanjavur",
        "irrigation",
        False,
    ),
    (
        "I grow cotton in Erode. Can I spray tomorrow?",
        "cotton",
        "Erode",
        "spraying",
        False,
    ),
    (
        "I grow banana near Thanjavur. A cyclone warning was issued. What should I do?",
        "banana",
        "Thanjavur",
        None,
        True,
    ),
    (
        "I grow groundnut in Madurai. When should I harvest?",
        "groundnut",
        "Madurai",
        "harvesting",
        False,
    ),
]


class ScenarioComponents:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def supervisor(self, farmer_message: str, _options: dict[str, Any]) -> dict[str, Any]:
        for message, crop, location, activity, _extreme in SCENARIOS:
            if farmer_message == message:
                return {
                    "crop": crop,
                    "location": location,
                    "question": farmer_message,
                    "activity": activity,
                    "time_period": "tomorrow" if "tomorrow" in message else "this week" if "this week" in message else None,
                }
        raise AssertionError(f"Unexpected farmer input: {farmer_message}")

    async def forecast(self, state: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("forecast")
        return {
            "summary": "Forecast estimate: conditions are variable this week.",
            "forecasts_are_estimates": True,
            "forecast_notice": "Weather forecasts are estimates and may change.",
            "tool_results": [
                {
                    "tool": "get_forecast",
                    "result": {
                        "location": {"name": state["location"]},
                        "forecast": [],
                    },
                }
            ],
        }

    async def alert(self, state: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("alert")
        return {
            "alert_found": True,
            "alert_status": "current_region_alert_candidate_found",
            "alert_description": "A recent alert candidate matched the search.",
            "alerts": [
                {
                    "title": "Cyclone warning candidate",
                    "source_url": "https://example.test/cyclone",
                    "published_date": datetime.now(UTC).isoformat(),
                    "date_status": "recent",
                    "region_match": True,
                    "current_region_warning": True,
                    "affected_region": state["location"].casefold(),
                }
            ],
        }

    async def crop_advisor(self, state: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("crop_advisor")
        doc = f"{state['crop']}/trusted_source.md"
        return {
            "overall_recommendation": "Use only the supplied source-backed guidance.",
            "weather_summary": state["weather_information"]["summary"],
            "recommendations": [
                {
                    "guidance": "Review crop conditions and the cited source.",
                    "reason": "The retrieved crop source covers this activity.",
                    "source_documents": [doc],
                }
            ],
            "what_to_avoid": [],
            "reasons": ["This is based on the retrieved source."],
            "uncertainties": ["Weather forecasts can change."],
            "pesticide_dosage": None,
            "pesticide_dosage_note": "No pesticide dosage was added.",
            "sources": [
                {
                    "title": "Trusted crop source",
                    "source_document": doc,
                    "source_url": "https://example.test/crop-guide",
                    "source_organization": "Test source",
                }
            ],
        }


class IntegratedFakeLLM:
    """Fake model interface that dispatches by the existing agent prompt."""

    def __init__(self, farmer_message: str) -> None:
        self.farmer_message = farmer_message
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **request: object) -> SimpleNamespace:
        messages = request["messages"]
        system_message = messages[0]["content"]

        if "Extract information from the farmer's message" in system_message:
            result = {
                "crop": "banana",
                "location": "Thanjavur",
                "question": self.farmer_message,
                "activity": None,
                "time_period": None,
            }
        elif "You are a weather-data summarizer" in system_message:
            tool_messages = [item for item in messages if item.get("role") == "tool"]
            if not tool_messages:
                calls = [
                    SimpleNamespace(
                        id=f"weather_call_{name}",
                        function=SimpleNamespace(
                            name=name,
                            arguments=json.dumps({"location": "Thanjavur", "days": 7}),
                        ),
                    )
                    for name in ("get_forecast", "get_rainfall_history", "get_soil_moisture")
                ]
                message = SimpleNamespace(content=None, tool_calls=calls)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])
            results = {
                item["name"]: json.loads(item["content"])
                for item in tool_messages
            }
            forecast = results["get_forecast"]["forecast"][0]
            result = (
                "Seven-day forecast estimate: max temperature "
                f"{forecast['temperature_max_c']} C, rainfall "
                f"{forecast['precipitation_mm']} mm. Recent rainfall total: "
                f"{results['get_rainfall_history']['total_rainfall_mm']} mm."
            )
            message = SimpleNamespace(content=result, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        elif "You are a cautious, farmer-friendly crop advisor" in system_message:
            context = json.loads(messages[1]["content"])
            source = context["retrieved_sources"][0]
            result = {
                "weather_summary": context["weather_information"]["summary"],
                "recommendations": [
                    {
                        "guidance": "Consider the retrieved banana cultivation guidance.",
                        "reason": "This note is directly sourced from the retrieved crop document.",
                        "source_documents": [source["source_document"]],
                    }
                ],
                "what_to_avoid": [],
                "reasons": [],
                "uncertainties": ["The forecast may change."],
                "pesticide_dosage": None,
                "pesticide_dosage_note": "No pesticide dosage was added.",
                "sources": context["retrieved_sources"],
            }
        else:
            raise AssertionError(f"Unrecognized agent prompt: {system_message[:40]}")

        message = SimpleNamespace(content=json.dumps(result), tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class IntegratedFakeTavily:
    def search(self, **_parameters: object) -> dict[str, object]:
        return {
            "results": [
                {
                    "title": "Cyclone warning issued for Thanjavur",
                    "url": "https://example.test/thanjavur-cyclone-warning",
                    "content": "A cyclone warning is active for Thanjavur district.",
                    "published_date": datetime.now(UTC).isoformat(),
                }
            ]
        }


class WorkflowScenarioTests(unittest.TestCase):
    def test_four_requested_scenarios_and_alert_routing(self) -> None:
        for message, crop, location, activity, expect_alert in SCENARIOS:
            with self.subTest(crop=crop):
                components = ScenarioComponents()
                graph = build_workflow(
                    {
                        "supervisor": components.supervisor,
                        "forecast": components.forecast,
                        "alert": components.alert,
                        "crop_advisor": components.crop_advisor,
                    }
                )
                with self.assertLogs("farmer_crop_weather.workflow", level="INFO") as logs:
                    state = asyncio.run(
                        graph.ainvoke(
                            {"farmer_message": message, "errors": [], "log": []}
                        )
                    )

                self.assertEqual(state["crop"], crop)
                self.assertEqual(state["location"], location)
                self.assertEqual(state["question"], message)
                self.assertEqual(state["activity"], activity)
                self.assertEqual(state["extreme_weather_signal"], expect_alert)
                self.assertEqual("alert" in components.calls, expect_alert)
                self.assertIn("report_writer: final report created", state["log"])
                self.assertIn("workflow node started: supervisor", " ".join(logs.output))
                self.assertIn("extreme_weather_check result=", " ".join(logs.output))
                self.assertTrue(state["final_answer"])
                if expect_alert:
                    self.assertTrue(state["alert_information"]["alert_found"])
                self.assertTrue(state["report"]["agricultural_sources"])

    def test_default_nodes_run_through_real_mcp_and_rag(self) -> None:
        message = (
            "I grow banana near Thanjavur and a cyclone warning was issued. "
            "What should I do?"
        )
        fake_llm = IntegratedFakeLLM(message)
        state = asyncio.run(
            run_workflow(
                message,
                components={
                    "llm_client": fake_llm,
                    "forecast_llm_client": fake_llm,
                    "model": "test-model",
                    "tavily_client": IntegratedFakeTavily(),
                },
            )
        )

        self.assertEqual(state["crop"], "banana")
        self.assertEqual(state["location"], "Thanjavur")
        self.assertTrue(state["extreme_weather_signal"])
        self.assertTrue(state["alert_information"]["alert_found"])
        self.assertTrue(state["report"]["agricultural_sources"])
        self.assertTrue(state["final_answer"])

    def test_missing_crop_routes_to_clarification_without_downstream_calls(self) -> None:
        calls: list[str] = []

        async def supervisor(_message: str, _options: dict[str, Any]) -> dict[str, Any]:
            return {
                "crop": None,
                "location": "Erode",
                "question": "Should I spray tomorrow?",
            }

        async def should_not_run(*_args: Any) -> dict[str, Any]:
            calls.append("unexpected")
            return {}

        graph = build_workflow(
            {
                "supervisor": supervisor,
                "forecast": should_not_run,
                "alert": should_not_run,
                "crop_advisor": should_not_run,
            }
        )
        state = asyncio.run(graph.ainvoke({"farmer_message": "Should I spray tomorrow?"}))

        self.assertTrue(state["needs_clarification"])
        self.assertEqual(state["missing_information"], ["crop"])
        self.assertIn("crop", state["final_answer"])
        self.assertEqual(calls, [])

    def test_supervisor_service_failure_is_not_reported_as_missing_crop(self) -> None:
        async def failed_supervisor(_message: str, _options: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("OPENAI_API_KEY is not set")

        graph = build_workflow({"supervisor": failed_supervisor})
        state = asyncio.run(
            graph.ainvoke(
                {
                    "farmer_message": "I grow paddy near Thanjavur. Should I irrigate?",
                    "errors": [],
                    "log": [],
                }
            )
        )

        self.assertIn("Check the configured OpenAI", state["final_answer"])
        self.assertFalse(state.get("needs_clarification", False))
        self.assertFalse(state.get("weather_information"))


if __name__ == "__main__":
    unittest.main(verbosity=2)