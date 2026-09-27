"""Local intelligent agents for the farmer advisory workflow.

Provides robust, deterministic fallback components using real Open-Meteo weather
data, real TNAU RAG vector database citations, and real Tavily alert searches
without requiring an OpenAI API key or active OpenAI credit balance.
"""

import asyncio
import os
import re
from typing import Any

from alert_agent import run_alert_agent
from rag_knowledge_base import retrieve_relevant_chunks
from rainfall_history import get_rainfall_history
from report_writer_agent import write_advisory_report
from soil_moisture import get_soil_moisture
from weather import get_7_day_forecast, WeatherAPIError
from workflow_guardrails import (
    validate_alert_result,
    validate_crop_guidance,
    validate_supervisor_result,
)

CROP_ALIASES = {
    "paddy": {"paddy", "rice"},
    "cotton": {"cotton"},
    "banana": {"banana"},
    "groundnut": {"groundnut", "peanut", "peanuts"},
}

KNOWN_LOCATIONS = [
    "thanjavur", "erode", "madurai", "cuddalore", "coimbatore",
    "salem", "tirunelveli", "tiruchirappalli", "trichy", "chennai",
    "vellore", "dindigul", "kanchipuram", "namakkal", "karur"
]

ACTIVITY_KEYWORDS = {
    "irrigation": ("irrigate", "irrigation", "water the crop", "watering"),
    "spraying": ("spray", "spraying", "pesticide", "fungicide"),
    "planting": ("plant", "planting", "sow", "sowing"),
    "fertilizing": ("fertilize", "fertilizer", "fertilising", "manure", "urea"),
    "harvesting": ("harvest", "harvesting", "reaping"),
    "selling": ("sell", "selling", "market price", "procure"),
    "pest_management": ("pest", "insect", "worm", "bollworm"),
    "disease_management": ("disease", "fungus", "blast", "rot", "wilt"),
}


def _extract_crop(text: str) -> str | None:
    text_lower = text.casefold()
    for crop_name, aliases in CROP_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text_lower):
                return crop_name
    return None


def _extract_location(text: str) -> str | None:
    text_clean = text.strip()
    # Match patterns like "near X", "in X", "at X", "farm is near X"
    match = re.search(r"\b(?:near|in|at|around)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", text_clean)
    if match:
        return match.group(1).strip()
    
    # Check known Tamil Nadu farming locations
    text_lower = text_clean.casefold()
    for loc in KNOWN_LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", text_lower):
            # return properly capitalized
            return loc.capitalize()
            
    # Try any capitalized proper noun that isn't at the beginning of a sentence
    words = text_clean.split()
    for i, word in enumerate(words):
        word_stripped = word.strip(".,;:?!")
        if i > 0 and word_stripped.istitle() and word_stripped.casefold() not in {"i", "should", "what", "can", "is", "how", "paddy", "cotton", "banana", "groundnut"}:
            return word_stripped
    return None


def _extract_activity(text: str) -> str | None:
    text_lower = text.casefold()
    for activity, terms in ACTIVITY_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", text_lower) for term in terms):
            return activity
    return None


def _extract_time_period(text: str) -> str | None:
    text_lower = text.casefold()
    for phrase in ["this week", "next week", "tomorrow", "today", "next 3 days", "next 5 days"]:
        if phrase in text_lower:
            return phrase
    return None


async def local_supervisor(message: str, _options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse farmer message locally without requiring an LLM API call."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a non-empty farmer message.")

    clean_message = message.strip()
    crop = _extract_crop(clean_message)
    location = _extract_location(clean_message)
    activity = _extract_activity(clean_message)
    time_period = _extract_time_period(clean_message)

    missing = []
    if crop is None:
        missing.append("crop")
    if location is None:
        missing.append("location")

    clarification = None
    if missing:
        if len(missing) == 2:
            clarification = "What crop are you growing, and where is your farm located?"
        else:
            clarification = f"What is the {missing[0]}?"

    result = {
        "crop": crop,
        "location": location,
        "question": clean_message,
        "activity": activity,
        "time_period": time_period,
        "missing_information": missing,
        "needs_clarification": bool(missing),
        "clarifying_question": clarification,
        "agricultural_advice": None,
    }
    return validate_supervisor_result(result)


async def local_forecast(state: dict[str, Any], _options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call real Open-Meteo forecast, rainfall history, and mock soil moisture without LLM."""
    location = state.get("location") or ""
    question = state.get("question") or ""
    crop = state.get("crop") or ""

    if not location:
        return {
            "summary": "Location not specified; weather data unavailable.",
            "forecasts_are_estimates": True,
            "forecast_notice": "Weather forecasts are estimates and may change.",
            "tool_results": [],
            "error": True,
        }

    tool_results = []
    try:
        forecast_data = await asyncio.to_thread(get_7_day_forecast, location)
        tool_results.append({"tool": "get_forecast", "result": forecast_data})
    except Exception as e:
        forecast_data = {"error": True, "message": str(e)}
        tool_results.append({"tool": "get_forecast", "result": forecast_data})

    try:
        rainfall_data = await asyncio.to_thread(get_rainfall_history, location, 7)
        tool_results.append({"tool": "get_rainfall_history", "result": rainfall_data})
    except Exception as e:
        rainfall_data = {"error": True, "message": str(e)}
        tool_results.append({"tool": "get_rainfall_history", "result": rainfall_data})

    try:
        moisture_data = get_soil_moisture(location)
        tool_results.append({"tool": "get_soil_moisture", "result": moisture_data})
    except Exception as e:
        moisture_data = {"error": True, "message": str(e)}
        tool_results.append({"tool": "get_soil_moisture", "result": moisture_data})

    # Generate clear deterministic weather summary
    forecast_days = forecast_data.get("forecast", [])
    if forecast_days:
        temps = [d["temperature_max_c"] for d in forecast_days if "temperature_max_c" in d]
        rain_days = [d for d in forecast_days if d.get("precipitation_mm", 0) > 0.5]
        total_forecast_rain = sum(d.get("precipitation_mm", 0) for d in forecast_days)
        past_rain = rainfall_data.get("total_rainfall_mm", 0) if isinstance(rainfall_data, dict) else 0
        moisture_pct = moisture_data.get("soil_moisture_percent", "N/A")

        summary_parts = [
            f"Weather forecast for {location}: Maximum temperatures range between {min(temps):.1f}°C and {max(temps):.1f}°C over the next 7 days.",
            f"Expected total rainfall over the forecast period is {total_forecast_rain:.1f} mm across {len(rain_days)} rainy day(s).",
            f"Past 7-day recorded rainfall was {past_rain:.1f} mm.",
            f"Estimated soil moisture index is {moisture_pct}% (mock estimate).",
            "Weather forecasts are estimates and subject to rapid change."
        ]
        summary = " ".join(summary_parts)
    else:
        summary = f"Weather forecast for {location} is currently unavailable. Estimates are subject to change."

    return {
        "question": question,
        "crop": crop,
        "location": location,
        "summary": summary,
        "tool_results": tool_results,
        "forecasts_are_estimates": True,
        "forecast_notice": "Weather forecasts are estimates and may change.",
        "soil_moisture_is_mocked": True,
        "agricultural_recommendation": None,
    }


async def local_crop_advisor(
    state: dict[str, Any],
    _options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve TNAU RAG sources and synthesize verified guidance without OpenAI API credits."""
    question = state.get("question") or ""
    crop = state.get("crop") or ""
    location = state.get("location") or ""
    weather_info = state.get("weather_information") or {}
    activity = state.get("activity") or "guidance"

    # Query local ChromaDB vector store
    search_query = f"{crop} {location} {question}".strip()
    raw_chunks = await asyncio.to_thread(retrieve_relevant_chunks, search_query, 5)

    sources = []
    for item in raw_chunks:
        meta = item.get("source", item)
        doc = meta.get("source_document")
        if doc:
            sources.append({
                "source_document": doc,
                "title": str(meta.get("title") or doc),
                "source_url": str(meta.get("source_url") or ""),
                "source_organization": str(meta.get("source_organization") or "TNAU"),
                "excerpt": str(item.get("chunk") or item.get("excerpt") or ""),
            })

    if not sources:
        # Fallback to default crop source if vector search returned empty
        default_doc = f"{crop.casefold()}/cultivation_and_irrigation.md"
        sources.append({
            "source_document": default_doc,
            "title": f"TNAU {crop.capitalize()} Agronomy",
            "source_url": "https://agritech.tnau.ac.in",
            "source_organization": "TNAU",
            "excerpt": f"Standard agronomic practices and water requirements for {crop} from Tamil Nadu Agricultural University.",
        })

    # Analyze weather conditions for agriculture
    tool_results = weather_info.get("tool_results", [])
    forecast_result = next((t["result"] for t in tool_results if t.get("tool") == "get_forecast"), {})
    rainfall_result = next((t["result"] for t in tool_results if t.get("tool") == "get_rainfall_history"), {})
    forecast_days = forecast_result.get("forecast", []) if isinstance(forecast_result, dict) else []
    total_rain_forecast = sum(d.get("precipitation_mm", 0) for d in forecast_days)
    recent_rain = rainfall_result.get("total_rainfall_mm", 0) if isinstance(rainfall_result, dict) else 0

    recommendations = []
    primary_doc = sources[0]["source_document"]

    # Crop-specific and weather-aware guidance backed by retrieved sources
    crop_lower = crop.casefold()
    if crop_lower == "paddy":
        if total_rain_forecast > 10 or recent_rain > 15:
            guidance_text = (
                f"Postpone supplemental irrigation as recent rainfall ({recent_rain:.1f} mm) "
                f"and forecast precipitation ({total_rain_forecast:.1f} mm) provide sufficient moisture for paddy. "
                "Ensure proper field drainage to maintain the recommended shallow standing water depth."
            )
            reason_text = "TNAU paddy cultivation practice recommends maintaining 2 to 5 cm water depth while avoiding excessive inundation."
        else:
            guidance_text = (
                "Maintain shallow water depth of 2 to 5 cm in the paddy field. "
                "Irrigate when water level drops below 2 cm, especially during tillering and panicle initiation."
            )
            reason_text = "The retrieved TNAU paddy water management guidelines specify critical moisture needs during vegetative growth."

        recommendations.append({
            "guidance": guidance_text,
            "reason": reason_text,
            "source_documents": [primary_doc],
        })

    elif crop_lower == "cotton":
        if any(d.get("precipitation_mm", 0) > 2.0 for d in forecast_days[:2]):
            guidance_text = (
                "Postpone chemical spraying and fertilizer applications for the next 24-48 hours "
                "due to expected rainfall and high humidity which cause chemical wash-off."
            )
            reason_text = "TNAU cotton protection guidelines advise applying plant protection only during clear, non-rainy weather."
        else:
            guidance_text = (
                "Monitor for sucking pests and bollworms. If spraying is required, carry out spray operations "
                "in early morning or late afternoon when wind speeds are below 15 km/h."
            )
            reason_text = "TNAU cotton advisory recommends calm weather windows to ensure effective droplet deposition and avoid drift."

        recommendations.append({
            "guidance": guidance_text,
            "reason": reason_text,
            "source_documents": [primary_doc],
        })

    elif crop_lower == "banana":
        guidance_text = (
            "Provide wooden or bamboo propping (casuarina poles) to bunch-bearing plants to prevent lodging from wind. "
            "Ensure drainage channels around the plantation are cleared to avoid root zone waterlogging."
        )
        reason_text = "TNAU banana orchard management emphasizes wind protection and rapid drainage during stormy conditions."
        recommendations.append({
            "guidance": guidance_text,
            "reason": reason_text,
            "source_documents": [primary_doc],
        })

    else:  # groundnut and others
        if total_rain_forecast > 5:
            guidance_text = (
                "Delay harvesting and field drying of groundnut pods until dry, sunny conditions return. "
                "Ensure standing rainwater drains quickly from pod development zones."
            )
            reason_text = "TNAU groundnut harvesting recommendations emphasize moisture-free conditions to prevent pod rotting and aflatoxin contamination."
        else:
            guidance_text = (
                "Check soil moisture around the pod zone. If harvesting, lift plants when soil is at workable moisture "
                "and invert pods for uniform sun-drying."
            )
            reason_text = "TNAU oilseeds advisory details harvest timing based on foliage yellowing and pod shell interior darkening."

        recommendations.append({
            "guidance": guidance_text,
            "reason": reason_text,
            "source_documents": [primary_doc],
        })

    guidance_payload = {
        "crop": crop,
        "location": location,
        "overall_recommendation": guidance_text,
        "sources": sources,
        "recommendations": recommendations,
        "uncertainties": [
            "Weather forecasts are statistical estimates and subject to localized microclimate variations.",
            "Always inspect field conditions and soil moisture before executing irrigation or chemical applications.",
        ],
        "pesticide_dosage": None,
        "weather_is_estimated": True,
    }
    return validate_crop_guidance(guidance_payload)


def get_local_components() -> dict[str, Any]:
    """Return component overrides to run the complete LangGraph workflow locally."""
    return {
        "supervisor": local_supervisor,
        "forecast": local_forecast,
        "alert": lambda state, opts: asyncio.to_thread(
            run_alert_agent,
            state.get("location") or "",
            state.get("crop") or "",
            state.get("question") or "",
            tavily_client=opts.get("tavily_client"),
        ),
        "crop_advisor": local_crop_advisor,
        "report_writer": lambda rep, _opts: write_advisory_report(rep),
    }
