"""Small validation gates for the farmer-advisory workflow."""

import re
from typing import Any


DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|litre|liter|ppm)\s*/\s*"
    r"(?:l|lit|ha|acre|plant|kg|hectare)\b",
    re.IGNORECASE,
)


def validate_supervisor_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure required values are present and represent explicit extraction."""
    validated = dict(result)
    missing = [
        key
        for key in ("crop", "location")
        if not isinstance(validated.get(key), str) or not validated[key].strip()
    ]
    validated["crop"] = validated.get("crop", "")
    validated["location"] = validated.get("location", "")
    validated["missing_information"] = missing
    validated["needs_clarification"] = bool(missing)
    if missing:
        if len(missing) == 2:
            validated["clarifying_question"] = (
                "What crop are you growing, and where is your farm located?"
            )
        else:
            field = "crop" if missing[0] == "crop" else "location"
            validated["clarifying_question"] = f"What is the {field}?"
        validated["agricultural_advice"] = None
    return validated


def validate_alert_result(result: dict[str, Any]) -> dict[str, Any]:
    """Only retain an alert claim backed by recent, regional source evidence."""
    validated = dict(result)
    if validated.get("alert_found") is None:
        validated["alert_status"] = "unavailable"
        return validated
    if not validated.get("alert_found"):
        validated["alert_found"] = False
        return validated

    verified = []
    for alert in validated.get("alerts", []):
        if not isinstance(alert, dict):
            continue
        if (
            alert.get("current_region_warning") is True
            and alert.get("date_status") == "recent"
            and alert.get("region_match") is True
            and isinstance(alert.get("source_url"), str)
            and alert["source_url"].startswith("https://")
            and isinstance(alert.get("title"), str)
            and alert["title"].strip()
        ):
            verified.append(alert)

    validated["alerts"] = verified
    if not verified:
        validated["alert_found"] = False
        validated["alert_status"] = "no_verified_current_region_alert"
        validated["alert_description"] = (
            "No current regional alert could be verified from dated search results with a source URL."
        )
    return validated


def validate_crop_guidance(result: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported citations or dosage-like text in generated guidance."""
    validated = dict(result)
    sources = validated.get("sources", [])
    source_documents = {
        source.get("source_document")
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("source_document"), str)
    }
    source_text = re.sub(
        r"\s+",
        "",
        " ".join(
            source.get("excerpt", "")
            for source in sources
            if isinstance(source, dict)
        ).casefold(),
    )

    recommendations = validated.get("recommendations", [])
    if not isinstance(recommendations, list):
        raise ValueError("Agricultural recommendations have an invalid structure.")

    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise ValueError("An agricultural recommendation has an invalid structure.")
        citations = recommendation.get("source_documents")
        if not isinstance(citations, list) or not citations:
            raise ValueError("Every agricultural recommendation must cite a retrieved source.")
        if any(citation not in source_documents for citation in citations):
            raise ValueError("A recommendation cited an agricultural source that was not retrieved.")
        statement = " ".join(
            str(recommendation.get(key, "")) for key in ("guidance", "reason")
        )
        for dosage in DOSAGE_PATTERN.finditer(statement):
            normalized = re.sub(r"\s+", "", dosage.group(0).casefold())
            if normalized not in source_text:
                raise ValueError("An unsupported pesticide dosage was detected in crop guidance.")

    validated["pesticide_dosage"] = None
    validated["weather_is_estimated"] = True
    return validated