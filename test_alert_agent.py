"""Tests for current-date and regional validation of Tavily results."""

import os
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from alert_agent import AlertAgentError, run_alert_agent


class FakeTavilyClient:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.queries: list[dict[str, object]] = []

    def search(self, **parameters: object) -> dict[str, object]:
        self.queries.append(parameters)
        return {"results": self.results}


class AlertAgentTests(unittest.TestCase):
    def test_only_recent_region_specific_warning_is_confirmed(self) -> None:
        now = datetime.now(UTC)
        client = FakeTavilyClient(
            [
                {
                    "title": "Flood warning issued for Thanjavur district",
                    "url": "https://example.test/current-flood-warning",
                    "content": "Authorities issued a current flood warning in Thanjavur.",
                    "published_date": now.isoformat(),
                },
                {
                    "title": "Old cyclone warning for Thanjavur",
                    "url": "https://example.test/old-cyclone",
                    "content": "Cyclone warning from a past event in Thanjavur.",
                    "published_date": (now - timedelta(days=30)).isoformat(),
                },
                {
                    "title": "Flood warning for another region",
                    "url": "https://example.test/other-region",
                    "content": "A flood warning is active in a distant region.",
                    "published_date": now.isoformat(),
                },
            ]
        )

        result = run_alert_agent(
            "Thanjavur",
            "paddy",
            "Any weather alerts?",
            tavily_client=client,
            now=now,
        )

        self.assertTrue(result["alert_found"])
        self.assertEqual(result["alert_status"], "current_region_alert_candidate_found")
        self.assertEqual(result["alerts"][0]["affected_region"], "thanjavur")
        self.assertEqual(result["alerts"][1]["date_status"], "old")
        self.assertFalse(result["market_search_performed"])
        self.assertEqual(client.queries[0]["topic"], "news")
        self.assertTrue(client.queries[0]["include_published_date"])

    def test_undated_or_old_results_are_not_claimed_as_current(self) -> None:
        now = datetime.now(UTC)
        client = FakeTavilyClient(
            [
                {
                    "title": "Heatwave warning in Thanjavur",
                    "url": "https://example.test/undated",
                    "content": "Heatwave warning mentions Thanjavur.",
                },
                {
                    "title": "Flood warning in Thanjavur",
                    "url": "https://example.test/old",
                    "content": "Flood warning mentions Thanjavur.",
                    "published_date": (now - timedelta(days=20)).isoformat(),
                },
            ]
        )

        result = run_alert_agent(
            "Thanjavur", "paddy", "Any alerts?", tavily_client=client, now=now
        )

        self.assertFalse(result["alert_found"])
        self.assertEqual(result["alerts"][0]["date_status"], "date_unknown")
        self.assertEqual(result["alerts"][1]["date_status"], "old")
        self.assertIn("No current", result["alert_description"])

    def test_location_matching_uses_whole_name_boundaries(self) -> None:
        client = FakeTavilyClient(
            [
                {
                    "title": "Flood warning for Thanjavuram",
                    "url": "https://example.test/unmatched-region",
                    "content": "A flood warning is active for Thanjavuram district.",
                    "published_date": datetime.now(UTC).isoformat(),
                }
            ]
        )
        result = run_alert_agent(
            "Thanjavur", "paddy", "Any flood alert?", tavily_client=client
        )

        self.assertFalse(result["alert_found"])
        self.assertFalse(result["alerts"][0]["region_match"])

    def test_market_search_only_runs_for_harvest_or_selling_questions(self) -> None:
        client = FakeTavilyClient([])
        result = run_alert_agent(
            "Thanjavur",
            "paddy",
            "Where can I sell this harvest?",
            tavily_client=client,
        )

        self.assertTrue(result["market_search_performed"])
        self.assertEqual(len(client.queries), 2)
        self.assertEqual(client.queries[1]["topic"], "general")

    def test_missing_api_key_is_reported(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(AlertAgentError, "TAVILY_API_KEY"):
                run_alert_agent("Thanjavur", "paddy", "Any alerts?")


if __name__ == "__main__":
    unittest.main(verbosity=2)