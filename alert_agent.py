"""Search Tavily for current regional alerts and, when requested, market news."""

import argparse
import json
import os
import re
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


ALERT_TERMS = re.compile(
    r"\b(cyclone|typhoon|hurricane|tornado|flood|heat ?wave|storm|heavy rain|"
    r"landslide|drought|wildfire|tsunami|earthquake|weather alert|weather warning|"
    r"disaster warning|red alert|orange alert|yellow alert|evacuation warning)\b",
    re.IGNORECASE,
)
WARNING_TERMS = re.compile(
    r"\b(alert|warning|advisory|watch|evacuat\w*|issued|active|ongoing)\b",
    re.IGNORECASE,
)
RETROSPECTIVE_TERMS = re.compile(
    r"\b(historical|anniversary|last year|archive|retrospective|past event|ended)\b",
    re.IGNORECASE,
)
MARKET_TERMS = re.compile(
    r"\b(harvest\w*|sell(?:ing|er)?|market|mandi|price|procure\w*)\b",
    re.IGNORECASE,
)


class AlertAgentError(Exception):
    """Raised when the alert search cannot be configured or completed."""


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.date()


def _date_status(published_date: Any, today: date) -> tuple[str, int | None]:
    published = _parse_date(published_date)
    if published is None:
        return "date_unknown", None
    age_days = (today - published).days
    if age_days < 0:
        return "future_dated", age_days
    if age_days <= 7:
        return "recent", age_days
    return "old", age_days


def _location_terms(location: str) -> list[str]:
    return [part.strip().casefold() for part in re.split(r"[,;]", location) if part.strip()]


def _matching_region(location: str, text: str) -> str | None:
    normalized_text = re.sub(r"\s+", " ", text.casefold())
    return next(
        (
            term
            for term in _location_terms(location)
            if re.search(
                rf"(?<!\w){re.escape(term).replace(r'\ ', r'\s+')}(?!\w)",
                normalized_text,
            )
        ),
        None,
    )


def _normalise_result(result: dict[str, Any], location: str, today: date) -> dict[str, Any]:
    title = str(result.get("title") or "")
    content = str(result.get("content") or "")
    published_date = result.get("published_date")
    date_status, age_days = _date_status(published_date, today)
    region = _matching_region(location, f"{title} {content}")
    mentions_hazard = bool(ALERT_TERMS.search(f"{title} {content}"))
    mentions_warning = bool(WARNING_TERMS.search(f"{title} {content}"))
    retrospective = bool(RETROSPECTIVE_TERMS.search(f"{title} {content}"))
    current_region_warning = (
        date_status == "recent"
        and region is not None
        and mentions_hazard
        and mentions_warning
        and not retrospective
    )

    return {
        "title": title or "Untitled result",
        "source_url": result.get("url"),
        "description": content or "No result description was returned.",
        "published_date": published_date,
        "date_status": date_status,
        "age_days": age_days,
        "affected_region": region or "Not specified in source",
        "region_match": region is not None,
        "current_region_warning": current_region_warning,
    }


def _client_or_error(client: Any | None) -> Any:
    if client is not None:
        return client
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise AlertAgentError("TAVILY_API_KEY is not set in the environment.")
    try:
        from tavily import TavilyClient
    except ImportError as error:
        raise AlertAgentError(
            "Tavily is not installed. Install dependencies from requirements.txt."
        ) from error
    return TavilyClient(api_key=api_key)


def _search(client: Any, *, query: str, topic: str) -> dict[str, Any]:
    response = client.search(
        query=query,
        topic=topic,
        search_depth="basic",
        time_range="week",
        include_published_date=True,
        filter_by_published_date=False,
        max_results=8,
        include_answer=False,
    )
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise AlertAgentError("Tavily returned an unexpected search response.")
    return response


def run_alert_agent(
    location: str,
    crop: str,
    question: str,
    *,
    tavily_client: Any | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Find recent, region-matching weather alerts and optional market results.

    Results without detectable dates are preserved but never treated as confirmed
    current alerts. A search hit is not itself proof that an alert is active.
    """
    for value, name in ((location, "location"), (crop, "crop"), (question, "question")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Please provide a non-empty {name}.")

    client = _client_or_error(tavily_client)
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=UTC)
    today = checked_at.date()
    clean_location = location.strip()
    clean_crop = crop.strip()

    alert_query = (
        f"current active cyclone flood heatwave weather alert warning advisory "
        f"{clean_location} {clean_crop}"
    )
    alert_search_error = None
    try:
        alert_response = _search(client, query=alert_query, topic="news")
        alert_results = [
            _normalise_result(item, clean_location, today)
            for item in alert_response["results"]
            if isinstance(item, dict)
            and ALERT_TERMS.search(f"{item.get('title', '')} {item.get('content', '')}")
        ]
        alert_found: bool | None = any(
            item["current_region_warning"] for item in alert_results
        )
    except Exception as error:
        alert_results = []
        alert_found = None
        alert_search_error = f"Tavily alert search failed: {error}"

    market_requested = bool(MARKET_TERMS.search(question))
    market_results = []
    market_search_error = None
    if market_requested:
        market_query = (
            f"current {clean_crop} market prices mandi selling harvest "
            f"{clean_location}"
        )
        try:
            market_response = _search(client, query=market_query, topic="general")
            market_results = [
                _normalise_result(item, clean_location, today)
                for item in market_response["results"]
                if isinstance(item, dict)
            ]
        except Exception as error:
            market_search_error = f"Tavily market search failed: {error}"

    if alert_found is None:
        alert_status = "unavailable"
        alert_description = (
            "The alert search could not be completed; no conclusion about current alerts can be made."
        )
    elif alert_found:
        alert_status = "current_region_alert_candidate_found"
        alert_description = (
            "A recently dated result mentions a weather hazard and warning for the requested region. "
            "Check the cited source for the active official notice."
        )
    else:
        alert_status = "no_current_region_alert_found"
        alert_description = (
            "No current, region-matching weather alert was confirmed in the recent search results. "
            "Search coverage is not a guarantee that no alert exists."
        )

    return {
        "location": clean_location,
        "crop": clean_crop,
        "question": question.strip(),
        "checked_at": checked_at.isoformat(),
        "alert_found": alert_found,
        "alert_status": alert_status,
        "alert_description": alert_description,
        "alerts": alert_results,
        "market_search_performed": market_requested,
        "market_information": market_results,
        "errors": [
            error
            for error in (alert_search_error, market_search_error)
            if error is not None
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search current local weather alerts.")
    parser.add_argument("--location", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--question", required=True)
    args = parser.parse_args()
    try:
        result = run_alert_agent(args.location, args.crop, args.question)
    except (AlertAgentError, ValueError) as error:
        parser.exit(1, f"Alert Agent failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()