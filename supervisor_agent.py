"""Extract explicit crop-request details and ask for missing required fields."""

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

from openai import APIError, AsyncOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


SYSTEM_PROMPT = """Extract information from the farmer's message. Do not answer the
agricultural question. Do not infer a crop, location, activity, or time that the
farmer did not explicitly mention. Return JSON with crop, location, question,
activity, and time_period. Use null when a value is absent. Keep question as the
farmer's own question text. Normalize crop names to lowercase. Normalize activity
to one of irrigation, spraying, planting, fertilizing, harvesting, selling,
pest_management, disease_management, or another explicitly stated activity.
Keep time_period as a short phrase from the original message, such as tomorrow
or this week."""

ACTIVITY_TERMS = {
    "irrigation": ("irrigate", "irrigation", "water the crop"),
    "spraying": ("spray", "spraying"),
    "planting": ("plant", "planting", "sow", "sowing"),
    "fertilizing": ("fertilize", "fertilizer", "fertilising", "manure"),
    "harvesting": ("harvest", "harvesting"),
    "selling": ("sell", "selling", "market price"),
    "pest_management": ("pest", "insect"),
    "disease_management": ("disease", "fungus"),
}


class SupervisorAgentError(Exception):
    """Raised when the supervisor cannot parse the farmer's request."""


def _contains_literal(text: str, phrase: str) -> bool:
    normalized_text = " ".join(text.casefold().split())
    normalized_phrase = " ".join(phrase.casefold().split())
    return bool(normalized_phrase) and normalized_phrase in normalized_text


def _explicit_activity(candidate: Any, message: str) -> str | None:
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    normalized = candidate.strip().casefold().replace(" ", "_")
    for activity, terms in ACTIVITY_TERMS.items():
        if normalized == activity and any(_contains_literal(message, term) for term in terms):
            return activity
    if _contains_literal(message, candidate):
        return candidate.strip()
    return None


def _missing_fields_question(crop: str | None, location: str | None) -> str | None:
    missing = []
    if crop is None:
        missing.append("the crop")
    if location is None:
        missing.append("the location")
    if not missing:
        return None
    if len(missing) == 2:
        return "What crop are you growing, and where is your farm located?"
    return f"What is {missing[0]}?"


def _client_or_error(client: Any | None, model: str | None) -> tuple[Any, str, bool]:
    if client is not None:
        return client, model or "test-model", False
    if not os.getenv("OPENAI_API_KEY"):
        raise SupervisorAgentError("OPENAI_API_KEY is not set in the environment.")
    configured_model = model or os.getenv("OPENAI_MODEL")
    if not configured_model:
        raise SupervisorAgentError("OPENAI_MODEL is not set in the environment.")
    base_url = os.getenv("OPENAI_BASE_URL")
    return AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI(), configured_model, True


async def extract_farmer_request(
    message: str,
    *,
    llm_client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract crop, location, question, activity, and time from a farmer message."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a non-empty farmer message.")

    client, model_name, owns_client = _client_or_error(llm_client, model)
    try:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message.strip()},
                ],
                response_format={"type": "json_object"},
            )
        except APIError as error:
            raise SupervisorAgentError(
                "The LLM request failed. Check the API key, model, and network connection."
            ) from error

        try:
            extracted = json.loads(response.choices[0].message.content or "")
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise SupervisorAgentError("The LLM returned invalid request-extraction JSON.") from error
        if not isinstance(extracted, dict):
            raise SupervisorAgentError("The LLM returned an unexpected extraction structure.")

        proposed_crop = extracted.get("crop")
        crop = (
            proposed_crop.strip().casefold()
            if isinstance(proposed_crop, str)
            and proposed_crop.strip()
            and _contains_literal(message, proposed_crop)
            else None
        )
        proposed_location = extracted.get("location")
        location = (
            proposed_location.strip()
            if isinstance(proposed_location, str)
            and proposed_location.strip()
            and _contains_literal(message, proposed_location)
            else None
        )
        proposed_question = extracted.get("question")
        question = (
            proposed_question.strip()
            if isinstance(proposed_question, str)
            and proposed_question.strip()
            and _contains_literal(message, proposed_question)
            else message.strip()
        )
        proposed_time = extracted.get("time_period")
        time_period = (
            proposed_time.strip()
            if isinstance(proposed_time, str)
            and proposed_time.strip()
            and _contains_literal(message, proposed_time)
            else None
        )
        activity = _explicit_activity(extracted.get("activity"), message)
        clarification = _missing_fields_question(crop, location)

        return {
            "crop": crop,
            "location": location,
            "question": question,
            "activity": activity,
            "time_period": time_period,
            "missing_information": [
                field
                for field, value in (("crop", crop), ("location", location))
                if value is None
            ],
            "needs_clarification": clarification is not None,
            "clarifying_question": clarification,
            "agricultural_advice": None,
        }
    finally:
        if owns_client:
            await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract details from a farmer's request.")
    parser.add_argument("message", help="The farmer's request in plain language.")
    args = parser.parse_args()
    try:
        result = asyncio.run(extract_farmer_request(args.message))
    except (SupervisorAgentError, ValueError) as error:
        parser.exit(1, f"Supervisor Agent failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()