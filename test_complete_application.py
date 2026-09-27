"""End-to-end Module 14 validation scenarios using injected component doubles."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import unittest

from report_writer_agent import write_advisory_report
from workflow import run_workflow


SCENARIOS = [
    {
        "name": "Paddy + Thanjavur + irrigation",
        "message": "I grow paddy near Thanjavur. Should I irrigate this week?",
        "crop": "paddy",
        "location": "Thanjavur",
        "activity": "irrigation",
    },
    {
        "name": "Cotton + spraying + tomorrow",
        "message": "I grow cotton in Erode. Can I spray tomorrow?",
        "crop": "cotton",
        "location": "Erode",
        "activity": "spraying",
        "time_period": "tomorrow",
    },
    {
        "name": "Banana + cyclone warning",
        "message": "I grow banana near Thanjavur. Cyclone warning issued. What should I do?",
        "crop": "banana",
        "location": "Thanjavur",
        "extreme": True,
        "alert": True,
    },
    {
        "name": "Groundnut + harvesting",
        "message": "I grow groundnut in Madurai. When should I harvest?",
        "crop": "groundnut",
        "location": "Madurai",
        "activity": "harvesting",
    },
    {
        "name": "Missing crop",
        "message": "I farm near Erode. Should I irrigate?",
        "crop": None,
        "location": "Erode",
        "clarification": True,
    },
    {
        "name": "Missing location",
        "message": "I grow paddy. Should I irrigate?",
        "crop": "paddy",
        "location": None,
        "clarification": True,
    },
    {
        "name": "Invalid location",
        "message": "I grow paddy in InvalidRandomPlace123456. Should I irrigate?",
        "crop": "paddy",
        "location": "InvalidRandomPlace123456",
        "invalid_location": True,
    },
    {
        "name": "Weather API failure",
        "message": "I grow cotton in Erode. Is weather suitable for spraying?",
        "crop": "cotton",
        "location": "Erode",
        "weather_failure": True,
    },
    {
        "name": "Tavily failure",
        "message": "I grow banana near Thanjavur. There is a cyclone warning.",
        "crop": "banana",
        "location": "Thanjavur",
        "extreme": True,
        "alert_failure": True,
    },
    {
        "name": "RAG retrieval failure",
        "message": "I grow groundnut in Madurai. How should I manage the crop?",
        "crop": "groundnut",
        "location": "Madurai",
        "rag_failure": True,
    },
    {
        "name": "Missing pesticide dosage information",
        "message": "I grow paddy near Thanjavur. What pesticide dosage should I use?",
        "crop": "paddy",
        "location": "Thanjavur",
        "activity": "pest_management",
    },
    {
        "name": "Extreme weather without a verified alert",
        "message": "I grow cotton in Erode. A heatwave warning is mentioned; is there a current alert?",
        "crop": "cotton",
        "location": "Erode",
        "extreme": True,
        "alert": False,
    },
]


def _components_for(scenario: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    calls: dict[str, list[str]] = {name: [] for name in ("forecast", "alert", "advisor")}
    crop = scenario.get("crop")
    location = scenario.get("location")
    source_document = f"{crop or 'unknown'}/verified_source.md"

    async def supervisor(message: str, _options: dict[str, Any]) -> dict[str, Any]:
        return {
            "crop": crop,
            "location": location,
            "question": message,
            "activity": scenario.get("activity"),
            "time_period": scenario.get("time_period"),
        }

    async def forecast(state: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
        calls["forecast"].append("get_forecast")
        calls["forecast"].append("get_rainfall_history")
        if scenario.get("weather_failure"):
            raise RuntimeError("simulated Open-Meteo outage")
        if scenario.get("invalid_location"):
            return {
                "summary": "Location lookup failed; no weather values are available.",
                "forecasts_are_estimates": True,
                "forecast_notice": "No forecast was returned for this location.",
                "tool_results": [
                    {"tool": "get_forecast", "result": {"error": True, "message": "Location not found."}}
                ],
            }
        return {
            "summary": "Seven-day weather forecast estimate for the requested area.",
            "forecasts_are_estimates": True,
            "forecast_notice": "Weather forecasts are estimates and may change.",
            "tool_results": [
                {
                    "tool": "get_forecast",
                    "result": {
                        "location": {"name": location},
                        "forecast": [
                            {
                                "date": "2026-09-28",
                                "temperature_min_c": 24,
                                "temperature_max_c": 34,
                                "precipitation_mm": 2.0,
                                "humidity_mean_percent": 72,
                                "wind_speed_max_kmh": 12,
                            }
                        ],
                    },
                },
                {
                    "tool": "get_rainfall_history",
                    "result": {
                        "location": {"name": location},
                        "number_of_days": 7,
                        "rainfall_by_day": [{"date": "2026-09-27", "rainfall_mm": 1.5}],
                        "total_rainfall_mm": 1.5,
                    },
                },
            ],
        }

    async def alert(state: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
        calls["alert"].append("tavily_search")
        if scenario.get("alert_failure"):
            raise RuntimeError("simulated Tavily outage")
        found = bool(scenario.get("alert", False))
        alert_results = []
        if found:
            alert_results = [
                {
                    "title": "Cyclone warning for Thanjavur",
                    "source_url": "https://example.test/current-alert",
                    "published_date": datetime.now(UTC).isoformat(),
                    "date_status": "recent",
                    "region_match": True,
                    "current_region_warning": True,
                    "affected_region": location,
                }
            ]
        return {
            "alert_found": found,
            "alert_status": "current_region_alert_candidate_found" if found else "no_current_region_alert_found",
            "alert_description": "A current candidate was found." if found else "No current regional alert was confirmed in search results.",
            "alerts": alert_results,
        }

    async def advisor(state: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
        calls["advisor"].append("rag_retrieval")
        if scenario.get("rag_failure"):
            raise RuntimeError("simulated local vector database failure")
        recommendations = []
        sources = []
        if crop:
            source = {
                "title": f"Verified {crop} guide",
                "source_document": source_document,
                "source_url": "https://example.test/agriculture/guide",
                "source_organization": "Verified agricultural source",
                "excerpt": "The guide provides crop-specific field management information.",
            }
            sources.append(source)
            if not scenario.get("invalid_location") and not scenario.get("weather_failure"):
                recommendations.append(
                    {
                        "guidance": "Use the retrieved crop guidance and check actual field conditions.",
                        "reason": "This is the management topic covered by the cited source.",
                        "source_documents": [source_document],
                    }
                )
        note = None
        if "dosage" in state["question"].casefold():
            note = "Pesticide dosage is not available in the retrieved sources."
        return {
            "overall_recommendation": (
                "No recommendation is available from weather data alone."
                if not recommendations
                else "Use the source-backed field guidance and consider the provided weather information."
            ),
            "weather_summary": state.get("weather_information", {}).get("summary", "Weather unavailable."),
            "recommendations": recommendations,
            "what_to_avoid": [],
            "reasons": [],
            "uncertainties": ["Weather forecasts are estimates and may change."],
            "pesticide_dosage": None,
            "pesticide_dosage_note": note or "No pesticide dosage was added.",
            "sources": sources,
        }

    return {
        "supervisor": supervisor,
        "forecast": forecast,
        "alert": alert,
        "crop_advisor": advisor,
    }, calls


class CompleteApplicationValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results: list[tuple[str, str, str]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        lines = [
            "# Module 14 Complete Application Test Report",
            "",
            "Test scenarios use deterministic backend component doubles to cover success and failure branches.",
            "The separate workflow integration test uses real MCP weather tools and the local RAG vector database.",
            "",
            "| Scenario | Result | Details |",
            "|---|---|---|",
        ]
        lines.extend(
            f"| {name} | {status} | {details.replace('|', '/')} |"
            for name, status, details in cls.results
        )
        (Path(__file__).resolve().parent / "test_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _run_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        components, calls = _components_for(scenario)
        state = asyncio.run(
            run_workflow(scenario["message"], components=components)
        )
        state["test_calls"] = calls
        return state

    def _record_assertions(self, scenario: dict[str, Any], checks: list[tuple[str, bool]]) -> None:
        failures = [name for name, passed in checks if not passed]
        status = "FAIL" if failures else "PASS"
        details = "all requested invariants verified" if not failures else "failed: " + ", ".join(failures)
        self.results.append((scenario["name"], status, details))
        self.assertFalse(failures, details)

    def _validate_normal_scenario(self, index: int) -> None:
        scenario = SCENARIOS[index]
        state = self._run_scenario(scenario)
        expect_alert = bool(scenario.get("extreme"))
        report = state.get("report", {})
        guidance = state.get("agricultural_guidance", {})
        source_docs = {
            item.get("source_document")
            for item in guidance.get("sources", [])
            if isinstance(item, dict)
        }
        recommendation_docs = {
            citation
            for item in guidance.get("recommendations", [])
            if isinstance(item, dict)
            for citation in item.get("source_documents", [])
        }
        checks = [
            ("crop extraction", state.get("crop") == scenario["crop"]),
            ("location extraction", state.get("location") == scenario["location"]),
            ("question preserved", state.get("question") == scenario["message"]),
            (
                "weather tools used",
                {"get_forecast", "get_rainfall_history"}
                <= set(state["test_calls"]["forecast"]),
            ),
            ("alert routing", bool(state["test_calls"]["alert"]) == expect_alert),
            ("RAG/advisor called", bool(state["test_calls"]["advisor"])),
            ("sources preserved", bool(report.get("agricultural_sources"))),
            ("recommendations cite retrieved sources", recommendation_docs <= source_docs),
            ("forecast uncertainty shown", "estimate" in report.get("weather_uncertainty", "").casefold()),
            ("final answer understandable", len(state.get("final_answer", "").strip()) > 20),
        ]
        if "activity" in scenario:
            checks.append(("requested activity extracted", state.get("activity") == scenario["activity"]))
        if "time_period" in scenario:
            checks.append(("requested time extracted", state.get("time_period") == scenario["time_period"]))
        if expect_alert and scenario.get("alert"):
            checks.append(("alert evidence preserved", bool(report.get("active_alerts"))))
        self._record_assertions(scenario, checks)

    def test_paddy_irrigation(self) -> None:
        self._validate_normal_scenario(0)

    def test_cotton_spraying_tomorrow(self) -> None:
        self._validate_normal_scenario(1)

    def test_banana_cyclone_warning(self) -> None:
        self._validate_normal_scenario(2)

    def test_groundnut_harvesting(self) -> None:
        self._validate_normal_scenario(3)

    def test_missing_crop(self) -> None:
        state = self._run_scenario(SCENARIOS[4])
        self._record_assertions(
            SCENARIOS[4],
            [
                ("crop remains missing", state.get("crop") is None),
                ("clarification requested", "crop" in state.get("final_answer", "").casefold()),
                ("no downstream fabricated data", not state["test_calls"]["forecast"]),
            ],
        )

    def test_missing_location(self) -> None:
        state = self._run_scenario(SCENARIOS[5])
        self._record_assertions(
            SCENARIOS[5],
            [
                ("location remains missing", state.get("location") is None),
                ("clarification requested", "location" in state.get("final_answer", "").casefold()),
                ("no downstream fabricated data", not state["test_calls"]["forecast"]),
            ],
        )

    def test_invalid_location(self) -> None:
        state = self._run_scenario(SCENARIOS[6])
        final_text = state.get("final_answer", "")
        self._record_assertions(
            SCENARIOS[6],
            [
                ("invalid location preserved", state.get("location") == "InvalidRandomPlace123456"),
                ("forecast failure visible", "no weather values are available" in final_text.casefold()),
                ("no weather values fabricated", "temperature 24-34" not in final_text),
                ("sources preserved", bool(state.get("report", {}).get("agricultural_sources"))),
            ],
        )

    def test_weather_api_failure(self) -> None:
        state = self._run_scenario(SCENARIOS[7])
        self._record_assertions(
            SCENARIOS[7],
            [
                ("failure recorded", any("Forecast failed" in error for error in state.get("errors", []))),
                ("unavailable stated", "unavailable" in state.get("final_answer", "").casefold()),
                ("uncertainty stated", bool(state.get("report", {}).get("weather_uncertainty"))),
                ("no fabricated temperatures", "temperature 24-34" not in state.get("final_answer", "")),
            ],
        )

    def test_tavily_failure(self) -> None:
        state = self._run_scenario(SCENARIOS[8])
        self._record_assertions(
            SCENARIOS[8],
            [
                ("extreme routed to alert", bool(state.get("extreme_weather_signal"))),
                ("Tavily failure recorded", any("Alert search failed" in error for error in state.get("errors", []))),
                ("alert status unavailable", state.get("alert_information", {}).get("alert_found") is None),
                ("no unsupported current alert", "search failed" in " ".join(state.get("report", {}).get("active_alerts", [])).casefold()),
            ],
        )

    def test_rag_retrieval_failure(self) -> None:
        state = self._run_scenario(SCENARIOS[9])
        self._record_assertions(
            SCENARIOS[9],
            [
                ("RAG failure recorded", any("Crop Advisor failed safely" in error for error in state.get("errors", []))),
                ("no unsupported advice", not state.get("agricultural_guidance", {}).get("recommendations")),
                ("no invented source", not state.get("report", {}).get("agricultural_sources")),
                ("understandable fallback", "no agricultural recommendation" in state.get("final_answer", "").casefold()),
            ],
        )

    def test_missing_pesticide_dosage_information(self) -> None:
        state = self._run_scenario(SCENARIOS[10])
        self._record_assertions(
            SCENARIOS[10],
            [
                ("no dosage in output", "pesticide 2 ml/l" not in state.get("final_answer", "").casefold()),
                ("dosage absence stated", "not available" in state.get("final_answer", "").casefold()),
                ("source preserved", bool(state.get("report", {}).get("agricultural_sources"))),
            ],
        )

    def test_extreme_weather_without_confirmed_alert(self) -> None:
        state = self._run_scenario(SCENARIOS[11])
        self._record_assertions(
            SCENARIOS[11],
            [
                ("extreme signal detected", bool(state.get("extreme_weather_signal"))),
                ("alert agent routed", bool(state["test_calls"]["alert"])),
                ("no unconfirmed alert", state.get("alert_information", {}).get("alert_found") is False),
                ("absence stated clearly", "no current regional alert" in " ".join(state.get("report", {}).get("active_alerts", [])).casefold()),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)