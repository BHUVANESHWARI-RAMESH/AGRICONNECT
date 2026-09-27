"""Exercise the Forecast Agent with a fake LLM and the real MCP server."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from forecast_agent import ForecastAgentError, run_forecast_agent


QUESTION = "I grow paddy near Thanjavur. Should I irrigate this week?"


class FakeOpenAIClient:
    """Stand-in for Chat Completions; MCP tools are still called for real."""

    def __init__(self, selected_tools: list[str]) -> None:
        self.selected_tools = selected_tools
        self.call_count = 0
        self.chat = SimpleNamespace(completions=self)
        self.tool_inputs: list[dict[str, object]] = []
        self.tool_payloads: dict[str, dict[str, object]] = {}

    async def create(self, **request: object) -> SimpleNamespace:
        if self.call_count == 0:
            advertised = {
                item["function"]["name"] for item in request["tools"]  # type: ignore[index]
            }
            if set(self.selected_tools) - advertised:
                raise AssertionError("The MCP tools were not advertised to the LLM")

            tool_calls = []
            for index, name in enumerate(self.selected_tools):
                arguments = {"location": "ignored-by-agent"}
                if name == "get_rainfall_history":
                    arguments["days"] = 365
                tool_calls.append(
                    SimpleNamespace(
                        id=f"call_{index}",
                        function=SimpleNamespace(
                            name=name,
                            arguments=json.dumps(arguments),
                        ),
                    )
                )
            self.call_count += 1
            message = SimpleNamespace(content=None, tool_calls=tool_calls)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.tool_payloads = {
            item["name"]: json.loads(item["content"])
            for item in request["messages"]  # type: ignore[index]
            if item.get("role") == "tool"
        }
        self.tool_inputs = [
            self.tool_payloads[name] for name in self.selected_tools
        ]
        forecast = self.tool_payloads.get("get_forecast", {})
        rainfall = self.tool_payloads.get("get_rainfall_history", {})
        soil = self.tool_payloads.get("get_soil_moisture", {})
        if forecast.get("error"):
            summary = "Forecast data is unavailable; no irrigation recommendation is provided."
        else:
            first_day = forecast["forecast"][0]
            summary = (
                f"Forecast estimate: {first_day['temperature_min_c']} to "
                f"{first_day['temperature_max_c']} C, {first_day['humidity_mean_percent']}% "
                f"humidity, wind up to {first_day['wind_speed_max_kmh']} km/h. "
                f"Recent 7-day rainfall total: {rainfall['total_rainfall_mm']} mm. "
                f"Soil moisture: {soil['soil_moisture_percent']}%, MOCKED/ESTIMATED. "
                "This is a weather summary only, not an irrigation recommendation."
            )

        self.call_count += 1
        message = SimpleNamespace(content=summary, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ForecastAgentTests(unittest.TestCase):
    def test_paddy_question_uses_mcp_tools_and_returns_weather_only_summary(self) -> None:
        fake_llm = FakeOpenAIClient(
            ["get_forecast", "get_rainfall_history", "get_soil_moisture"]
        )
        result = asyncio.run(
            run_forecast_agent(
                QUESTION,
                "paddy",
                "Thanjavur",
                llm_client=fake_llm,
                model="test-model",
            )
        )

        self.assertEqual(result["question"], QUESTION)
        self.assertEqual(result["crop"], "paddy")
        self.assertEqual(result["location"], "Thanjavur")
        self.assertCountEqual(
            result["tools_used"],
            ["get_forecast", "get_rainfall_history", "get_soil_moisture"],
        )
        self.assertEqual(fake_llm.tool_payloads["get_forecast"]["location"]["name"], "Thanjavur")
        self.assertEqual(fake_llm.tool_payloads["get_rainfall_history"]["number_of_days"], 7)
        self.assertEqual(fake_llm.tool_payloads["get_soil_moisture"]["status"], "MOCKED/ESTIMATED")
        self.assertTrue(result["forecasts_are_estimates"])
        self.assertTrue(result["soil_moisture_is_mocked"])
        self.assertIsNone(result["agricultural_recommendation"])
        self.assertIn("weather summary only", result["summary"])

    def test_missing_api_configuration_returns_clear_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ForecastAgentError, "OPENAI_API_KEY"):
                asyncio.run(run_forecast_agent(QUESTION, "paddy", "Thanjavur"))

    def test_tool_failure_is_reported_without_crashing(self) -> None:
        fake_llm = FakeOpenAIClient(["get_forecast"])
        result = asyncio.run(
            run_forecast_agent(
                QUESTION,
                "paddy",
                "InvalidRandomPlace123456",
                llm_client=fake_llm,
                model="test-model",
            )
        )

        self.assertTrue(result["tool_results"][0]["result"]["error"])
        self.assertIn("unavailable", result["summary"])
        self.assertIsNone(result["agricultural_recommendation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)