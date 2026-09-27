"""Create source-grounded crop guidance from weather and RAG information."""

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

from forecast_agent import run_forecast_agent
from rag_knowledge_base import retrieve_relevant_chunks


SYSTEM_PROMPT = """You are a cautious, farmer-friendly crop advisor.
Use the supplied weather information and retrieved agricultural source chunks.
Base every agricultural recommendation only on those retrieved chunks. Each
recommendation must cite one or more exact source_document values from the input.
Explain the reason for each recommendation in simple language. Do not invent
facts, dates, thresholds, or conditions. Treat weather forecasts as estimates
and explain relevant uncertainty. Never output a pesticide dosage; leave
pesticide_dosage as null. If dosage is asked about and no exact pesticide dosage
is present in the retrieved excerpts, state that it is not available from the
retrieved sources. Do not mention or rely on a source that was not provided.
Return only a JSON object with these keys:
weather_summary (string), recommendations (array of objects with guidance,
reason, source_documents array), uncertainties (array of strings),
pesticide_dosage (null)."""

DOSE_QUESTION = re.compile(
    r"\b(pesticide|insecticide|fungicide|herbicide|dosage|dose|spray rate)\b",
    re.IGNORECASE,
)
DOSE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|litre|liter|ppm)\s*/\s*"
    r"(?:l|lit|ha|acre|plant|kg|hectare)\b",
    re.IGNORECASE,
)


class CropAdvisorError(Exception):
    """Raised when source-grounded crop guidance cannot safely be produced."""


def _resolve_llm(client: Any | None, model: str | None) -> tuple[Any, str, bool]:
    if client is not None:
        return client, model or "test-model", False
    if not os.getenv("OPENAI_API_KEY"):
        raise CropAdvisorError("OPENAI_API_KEY is not set in the environment.")
    configured_model = model or os.getenv("OPENAI_MODEL")
    if not configured_model:
        raise CropAdvisorError("OPENAI_MODEL is not set in the environment.")
    base_url = os.getenv("OPENAI_BASE_URL")
    return AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI(), configured_model, True


def _compact_sources(retrieved_sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources = []
    for item in retrieved_sources:
        metadata = item.get("source", item)
        document = metadata.get("source_document")
        if not isinstance(document, str) or not document.strip():
            continue
        sources.append(
            {
                "source_document": document,
                "title": str(metadata.get("title") or document),
                "source_url": str(metadata.get("source_url") or ""),
                "source_organization": str(
                    metadata.get("source_organization") or "Not specified"
                ),
                "excerpt": str(item.get("chunk") or item.get("excerpt") or ""),
            }
        )
    return sources


def _validate_model_result(
    content: str,
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        response = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise CropAdvisorError("The LLM did not return valid structured crop guidance.") from error

    if not isinstance(response, dict):
        raise CropAdvisorError("The LLM response was not a JSON object.")
    recommendations = response.get("recommendations")
    uncertainties = response.get("uncertainties")
    if not isinstance(recommendations, list) or not isinstance(uncertainties, list):
        raise CropAdvisorError("The LLM response omitted recommendations or uncertainties.")

    allowed_documents = {source["source_document"] for source in sources}
    source_text = re.sub(
        r"\s+",
        "",
        " ".join(source["excerpt"] for source in sources).casefold(),
    )
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            raise CropAdvisorError("A recommendation had an invalid structure.")
        citations = recommendation.get("source_documents")
        if not isinstance(citations, list) or not citations:
            raise CropAdvisorError("Each recommendation must cite a retrieved source document.")
        if any(citation not in allowed_documents for citation in citations):
            raise CropAdvisorError("A recommendation cited a document that was not retrieved.")
        advice_text = " ".join(
            str(recommendation.get(field, "")) for field in ("guidance", "reason")
        )
        for dosage in DOSE_PATTERN.finditer(advice_text):
            normalized_dosage = re.sub(r"\s+", "", dosage.group(0).casefold())
            if normalized_dosage not in source_text:
                raise CropAdvisorError(
                    "The LLM returned a pesticide dosage that does not appear in the retrieved sources."
                )

    response["pesticide_dosage"] = None
    return response


def _dosage_note(question: str, sources: list[dict[str, str]]) -> str:
    if not DOSE_QUESTION.search(question):
        return "No pesticide dosage was requested or recommended."
    if not any(DOSE_PATTERN.search(source["excerpt"]) for source in sources):
        return "Pesticide dosage is not available in the retrieved sources."
    return (
        "A dosage-like value appears in the retrieved text, but this agent does not "
        "interpret or recommend pesticide dosages."
    )


async def run_crop_advisor_agent(
    question: str,
    crop: str,
    location: str,
    *,
    weather_information: dict[str, Any] | None = None,
    retrieved_sources: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    forecast_llm_client: Any | None = None,
    model: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Combine forecast context and retrieved agricultural sources into advice."""
    for value, name in ((question, "question"), (crop, "crop"), (location, "location")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Please provide a non-empty {name}.")

    clean_question = question.strip()
    clean_crop = crop.strip()
    clean_location = location.strip()

    try:
        if weather_information is None:
            weather_information = await run_forecast_agent(
                clean_question,
                clean_crop,
                clean_location,
                llm_client=forecast_llm_client,
                model=model,
            )
        if not isinstance(weather_information, dict):
            raise CropAdvisorError("Weather information must be a dictionary.")

        if retrieved_sources is None:
            retrieved_sources = await asyncio.to_thread(
                retrieve_relevant_chunks,
                f"{clean_crop} {clean_location} {clean_question}",
                top_k,
            )
        sources = _compact_sources(retrieved_sources)
        if not sources:
            raise CropAdvisorError(
                "No source-attributed agricultural information was retrieved; no advice was generated."
            )

        llm, model_name, owns_client = _resolve_llm(llm_client, model)
        context = {
            "question": clean_question,
            "crop": clean_crop,
            "location": clean_location,
            "weather_information": weather_information,
            "retrieved_sources": sources,
        }
        try:
            completion = await llm.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=True)},
                ],
                response_format={"type": "json_object"},
            )
        except APIError as error:
            raise CropAdvisorError(
                "The LLM request failed. Check the API key, model, and network connection."
            ) from error
        finally:
            if owns_client:
                await llm.close()

        content = completion.choices[0].message.content or ""
        generated = _validate_model_result(content, sources)
        return {
            "question": clean_question,
            "crop": clean_crop,
            "location": clean_location,
            "weather_summary": generated.get("weather_summary", ""),
            "recommendations": generated["recommendations"],
            "uncertainties": generated["uncertainties"],
            "pesticide_dosage": None,
            "pesticide_dosage_note": _dosage_note(clean_question, sources),
            "sources": [
                {key: value for key, value in source.items() if key != "excerpt"}
                for source in sources
            ],
            "weather_is_estimated": bool(
                weather_information.get("forecasts_are_estimates", True)
            ),
        }
    except CropAdvisorError:
        raise
    except Exception as error:
        raise CropAdvisorError(
            "Could not retrieve weather or agricultural information, so advice was not generated."
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source-grounded crop guidance.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--location", required=True)
    args = parser.parse_args()
    try:
        result = asyncio.run(
            run_crop_advisor_agent(args.question, args.crop, args.location)
        )
    except (CropAdvisorError, ValueError) as error:
        parser.exit(1, f"Crop Advisor failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()