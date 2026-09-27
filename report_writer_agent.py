"""Format supplied component outputs into a farmer-friendly advisory report."""

import re
from typing import Any


DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|litre|liter|ppm)\s*/\s*"
    r"(?:l|lit|ha|acre|plant|kg|hectare)\b",
    re.IGNORECASE,
)


def _tool_result(weather_information: dict[str, Any], name: str) -> dict[str, Any] | None:
    for item in weather_information.get("tool_results", []):
        if item.get("tool") == name and isinstance(item.get("result"), dict):
            return item["result"]
    return None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _source_records(
    supplied_sources: Any,
    agricultural_guidance: dict[str, Any],
) -> list[dict[str, str]]:
    candidates = _items(supplied_sources) + _items(agricultural_guidance.get("sources"))
    sources = []
    seen = set()
    for source in candidates:
        if not isinstance(source, dict):
            continue
        document = _text(source.get("source_document"))
        url = _text(source.get("source_url"))
        title = _text(source.get("title")) or document or url
        identity = (document, url, title)
        if not any(identity) or identity in seen:
            continue
        seen.add(identity)
        sources.append(
            {
                "title": title,
                "source_document": document,
                "source_url": url,
                "source_organization": _text(source.get("source_organization")),
            }
        )
    return sources


def _safe_text(value: Any, source_excerpts: list[str], suppressed: list[str]) -> str:
    text = _text(value)
    for dosage in DOSAGE_PATTERN.finditer(text):
        normalized = re.sub(r"\s+", "", dosage.group(0).casefold())
        source_text = re.sub(r"\s+", "", " ".join(source_excerpts).casefold())
        if normalized not in source_text:
            suppressed.append(dosage.group(0))
            return ""
    return text


def _recommendation_parts(guidance: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any]]:
    recommendations = _items(guidance.get("recommendations"))
    what_to_do = _items(guidance.get("what_to_do"))
    what_to_avoid = _items(guidance.get("what_to_avoid"))
    reasons = _items(guidance.get("reasons"))

    for recommendation in recommendations:
        if isinstance(recommendation, dict):
            what_to_do.extend(_items(recommendation.get("guidance")))
            what_to_avoid.extend(_items(recommendation.get("what_to_avoid")))
            reasons.extend(_items(recommendation.get("reason")))
    return what_to_do, what_to_avoid, reasons


def _format_alerts(alerts: Any) -> list[str]:
    if not isinstance(alerts, dict):
        return ["No current alert information was supplied."]
    if alerts.get("alert_found") is None:
        return [_text(alerts.get("alert_description")) or "Alert status is unavailable."]
    if alerts.get("alert_found") is False:
        return [_text(alerts.get("alert_description")) or "No alert was confirmed in the supplied search results."]

    formatted = []
    for item in alerts.get("alerts", []):
        if not isinstance(item, dict) or not item.get("current_region_warning"):
            continue
        title = _text(item.get("title")) or "Weather warning candidate"
        published = _text(item.get("published_date")) or "date not supplied"
        url = _text(item.get("source_url"))
        formatted.append(f"Candidate: {title} (published {published})" + (f" - {url}" if url else ""))
    return formatted or ["An alert was flagged, but no matching alert details were supplied."]


def _day_rows(weather_information: dict[str, Any], rainfall_history: Any) -> list[str]:
    forecast_result = _tool_result(weather_information, "get_forecast") or {}
    rainfall_result = rainfall_history if isinstance(rainfall_history, dict) else None
    if rainfall_result is None:
        rainfall_result = _tool_result(weather_information, "get_rainfall_history") or {}
    rainfall_by_date = {
        item.get("date"): item.get("rainfall_mm")
        for item in rainfall_result.get("rainfall_by_day", [])
        if isinstance(item, dict)
    }

    rows = []
    for day in forecast_result.get("forecast", []):
        if not isinstance(day, dict):
            continue
        details = [str(day.get("date", "Date not supplied"))]
        if day.get("temperature_min_c") is not None and day.get("temperature_max_c") is not None:
            details.append(
                f"temperature {day['temperature_min_c']}-{day['temperature_max_c']} C"
            )
        if day.get("humidity_mean_percent") is not None:
            details.append(f"humidity {day['humidity_mean_percent']}%")
        if day.get("wind_speed_max_kmh") is not None:
            details.append(f"wind up to {day['wind_speed_max_kmh']} km/h")
        rain = rainfall_by_date.get(day.get("date"), day.get("precipitation_mm"))
        if rain is not None:
            details.append(f"rainfall {rain} mm")
        rows.append("; ".join(details))
    return rows


def write_advisory_report(report_input: dict[str, Any]) -> dict[str, Any]:
    """Format supplied weather, guidance, alerts, and sources without adding facts."""
    if not isinstance(report_input, dict):
        raise ValueError("Report input must be a dictionary.")

    guidance = report_input.get("agricultural_guidance")
    guidance = guidance if isinstance(guidance, dict) else {}
    weather_information = report_input.get("weather_information")
    weather_information = weather_information if isinstance(weather_information, dict) else {}
    weather_summary = _text(
        report_input.get("weather_summary") or weather_information.get("summary")
    )
    rainfall = report_input.get("rainfall_history")
    soil_moisture = report_input.get("soil_moisture")
    if soil_moisture is None:
        soil_moisture = _tool_result(weather_information, "get_soil_moisture")
    alerts = report_input.get("current_alerts")
    sources = _source_records(report_input.get("sources"), guidance)
    source_excerpts = [
        _text(source.get("excerpt"))
        for source in _items(report_input.get("sources"))
        if isinstance(source, dict)
    ]

    suppressed_dosages: list[str] = []
    raw_what_to_do, raw_what_to_avoid, raw_reasons = _recommendation_parts(guidance)
    what_to_do = [
        clean
        for item in raw_what_to_do
        if (clean := _safe_text(item, source_excerpts, suppressed_dosages))
    ]
    what_to_avoid = [
        clean
        for item in raw_what_to_avoid
        if (clean := _safe_text(item, source_excerpts, suppressed_dosages))
    ]
    reasons = [
        clean
        for item in raw_reasons
        if (clean := _safe_text(item, source_excerpts, suppressed_dosages))
    ]

    overall = _safe_text(
        guidance.get("overall_recommendation"), source_excerpts, suppressed_dosages
    ) or "No overall recommendation was supplied; the report only summarizes provided information."
    if suppressed_dosages:
        dosage_note = "An unverified pesticide dosage was omitted because it was not present in supplied source excerpts."
        what_to_avoid.append(dosage_note)
    else:
        dosage_note = _text(guidance.get("pesticide_dosage_note")) or "No pesticide dosage was added by the Report Writer."

    explicit_day_advisory = report_input.get("day_by_day_advisory")
    day_by_day = [
        _safe_text(item, source_excerpts, suppressed_dosages)
        for item in _items(explicit_day_advisory)
    ]
    day_by_day = [item for item in day_by_day if item]
    day_by_day.extend(_day_rows(weather_information, rainfall))

    if isinstance(rainfall, dict):
        rainfall_summary = (
            f"{rainfall.get('number_of_days', 'Unknown')} days; "
            f"total {rainfall.get('total_rainfall_mm', 'not supplied')} mm."
        )
    else:
        rainfall_result = _tool_result(weather_information, "get_rainfall_history")
        rainfall_summary = (
            f"{rainfall_result.get('number_of_days')} days; total "
            f"{rainfall_result.get('total_rainfall_mm')} mm."
            if rainfall_result and not rainfall_result.get("error")
            else "Recent rainfall history was not supplied."
        )

    if isinstance(soil_moisture, dict):
        soil_summary = (
            f"{soil_moisture.get('soil_moisture_percent')}% "
            f"({soil_moisture.get('status', 'status not supplied')}); "
            "this is not real sensor data."
        )
    else:
        soil_summary = "Soil-moisture information was not supplied."

    uncertainty = _text(
        report_input.get("weather_uncertainty")
        or weather_information.get("forecast_notice")
    )
    if not uncertainty and weather_information:
        uncertainty = "Weather forecasts are estimates and may change."
    if not uncertainty:
        uncertainty = "Weather uncertainty information was not supplied."

    report = {
        "crop": _text(report_input.get("crop")) or "Not supplied",
        "location": _text(report_input.get("location")) or "Not supplied",
        "farmer_question": _text(report_input.get("farmer_question")) or "Not supplied",
        "overall_recommendation": overall,
        "day_by_day_advisory": day_by_day,
        "what_to_do": what_to_do,
        "what_to_avoid": what_to_avoid,
        "reasons": reasons,
        "weather_summary": weather_summary or "Weather summary was not supplied.",
        "rainfall_history": rainfall_summary,
        "soil_moisture": soil_summary,
        "active_alerts": _format_alerts(alerts),
        "agricultural_sources": sources,
        "weather_uncertainty": uncertainty,
        "pesticide_dosage_note": dosage_note,
    }
    report["report_text"] = _render_report(report)
    return report


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "Farmer Advisory Report",
        f"Crop: {report['crop']}",
        f"Location: {report['location']}",
        f"Question: {report['farmer_question']}",
        "",
        "Overall recommendation",
        report["overall_recommendation"],
        "",
        "Weather and rainfall",
        report["weather_summary"],
        f"Recent rainfall: {report['rainfall_history']}",
        f"Soil moisture: {report['soil_moisture']}",
        "",
        "Day-by-day information",
    ]
    lines.extend(f"- {item}" for item in report["day_by_day_advisory"])
    if not report["day_by_day_advisory"]:
        lines.append("- No day-by-day details were supplied.")

    lines.extend(["", "What to do"])
    lines.extend(f"- {item}" for item in report["what_to_do"])
    if not report["what_to_do"]:
        lines.append("- No action guidance was supplied.")

    lines.extend(["", "What to avoid"])
    lines.extend(f"- {item}" for item in report["what_to_avoid"])
    if not report["what_to_avoid"]:
        lines.append("- No avoidance guidance was supplied.")

    lines.extend(["", "Reasons"])
    lines.extend(f"- {item}" for item in report["reasons"])
    if not report["reasons"]:
        lines.append("- No reasons were supplied.")

    lines.extend(["", "Current alerts"])
    lines.extend(f"- {item}" for item in report["active_alerts"])

    lines.extend(["", "Agricultural sources"])
    for source in report["agricultural_sources"]:
        citation = source["title"]
        if source["source_document"]:
            citation += f" ({source['source_document']})"
        if source["source_url"]:
            citation += f" - {source['source_url']}"
        lines.append(f"- {citation}")
    if not report["agricultural_sources"]:
        lines.append("- No agricultural sources were supplied.")

    lines.extend(
        [
            "",
            "Weather uncertainty",
            report["weather_uncertainty"],
            "",
            report["pesticide_dosage_note"],
        ]
    )
    return "\n".join(lines)


def main() -> None:
    raise SystemExit(
        "Report Writer accepts component data as a Python dictionary. "
        "See README.md for a usage example."
    )


if __name__ == "__main__":
    main()