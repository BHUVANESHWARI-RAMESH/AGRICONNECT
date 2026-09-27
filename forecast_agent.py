"""Summarize weather data by calling the project's MCP tools."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters
from openai import APIError, AsyncOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


PROJECT_DIRECTORY = Path(__file__).resolve().parent
MCP_SERVER_PATH = PROJECT_DIRECTORY / "mcp_server.py"
RECENT_RAINFALL_DAYS = 7
ALLOWED_TOOLS = {
    "get_forecast",
    "get_rainfall_history",
    "get_soil_moisture",
}
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """You are a weather-data summarizer, not an agricultural advisor.
Use only the supplied MCP tool results. For each weather summary, call
get_forecast and get_rainfall_history. Decide whether get_soil_moisture is
relevant to the farmer's question. Summarize available temperature, rainfall,
humidity, wind, recent rainfall, and soil moisture when relevant. Forecasts are
estimates, not guarantees. Soil moisture is MOCKED/ESTIMATED and is not sensor data.
Never decide whether the farmer should irrigate, or give any other agricultural
recommendation. If a tool fails or data is missing, say so instead of guessing.
Keep the summary brief, factual, and tied to the supplied crop and location."""


class ForecastAgentError(Exception):
    """Raised when the forecast agent cannot complete its weather summary."""


def _create_openai_client(
    client: Any | None,
    model: str | None,
) -> tuple[Any, str, bool]:
    """Use an injected client for tests or load credentials from the environment."""
    if client is not None:
        return client, model or "test-model", False

    if not os.getenv("OPENAI_API_KEY"):
        raise ForecastAgentError("OPENAI_API_KEY is not set in the environment.")
    configured_model = model or os.getenv("OPENAI_MODEL")
    if not configured_model:
        raise ForecastAgentError("OPENAI_MODEL is not set in the environment.")

    base_url = os.getenv("OPENAI_BASE_URL")
    return AsyncOpenAI(base_url=base_url) if base_url else AsyncOpenAI(), configured_model, True


def _openai_tool_definitions(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Convert the MCP tool schemas into OpenAI function-tool definitions."""
    definitions = []
    for tool in mcp_tools:
        if tool.name not in ALLOWED_TOOLS:
            continue
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or tool.name,
                    "parameters": tool.input_schema,
                },
            }
        )

    missing_tools = ALLOWED_TOOLS - {item["function"]["name"] for item in definitions}
    if missing_tools:
        raise ForecastAgentError(
            f"Weather MCP server is missing tools: {', '.join(sorted(missing_tools))}"
        )
    return definitions


def _tool_arguments(tool_name: str, location: str) -> dict[str, Any]:
    """Use caller-provided location and a consistent recent-rainfall window."""
    arguments: dict[str, Any] = {"location": location}
    if tool_name == "get_rainfall_history":
        arguments["days"] = RECENT_RAINFALL_DAYS
    return arguments


def _mcp_result_payload(result: Any) -> dict[str, Any]:
    """Turn an MCP result into JSON-safe data for the LLM and caller."""
    if result.is_error:
        messages = [
            block.text for block in result.content if isinstance(block.text, str)
        ]
        return {"error": True, "message": " ".join(messages) or "MCP tool failed."}

    if result.structured_content is not None:
        return result.structured_content

    messages = [
        block.text for block in result.content if isinstance(block.text, str)
    ]
    return {"content": messages}


async def _ask_model(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    allow_tool_calls: bool = True,
) -> Any:
    try:
        return await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto" if allow_tool_calls else "none",
        )
    except APIError as error:
        raise ForecastAgentError(
            "The LLM request failed. Check the API key, model, and network connection."
        ) from error


async def run_forecast_agent(
    question: str,
    crop: str,
    location: str,
    *,
    llm_client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Choose MCP weather tools and return a weather-only summary for a farmer."""
    for value, name in ((question, "question"), (crop, "crop"), (location, "location")):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Please provide a non-empty {name}.")

    llm, model_name, owns_llm_client = _create_openai_client(llm_client, model)
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
    )

    try:
        async with Client(server) as mcp_client:
            listed_tools = await mcp_client.list_tools()
            tool_definitions = _openai_tool_definitions(listed_tools.tools)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Farmer question: {question.strip()}\n"
                        f"Crop: {crop.strip()}\n"
                        f"Location: {location.strip()}\n"
                        f"Recent rainfall window: the last {RECENT_RAINFALL_DAYS} completed days."
                    ),
                },
            ]
            tool_results: list[dict[str, Any]] = []
            summary = ""

            for _ in range(MAX_TOOL_ROUNDS):
                response = await _ask_model(llm, model_name, messages, tool_definitions)
                message = response.choices[0].message
                calls = message.tool_calls or []
                if not calls:
                    summary = message.content or ""
                    break

                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in calls
                        ],
                    }
                )

                for call in calls:
                    tool_name = call.function.name
                    if tool_name not in ALLOWED_TOOLS:
                        payload = {"error": True, "message": "Unknown weather tool."}
                    else:
                        try:
                            json.loads(call.function.arguments or "{}")
                            result = await mcp_client.call_tool(
                                tool_name,
                                _tool_arguments(tool_name, location.strip()),
                            )
                            payload = _mcp_result_payload(result)
                        except (ValueError, TypeError, json.JSONDecodeError) as error:
                            payload = {"error": True, "message": str(error)}
                        except Exception:
                            payload = {
                                "error": True,
                                "message": f"The {tool_name} MCP tool could not be reached.",
                            }

                    tool_results.append(
                        {"tool": tool_name, "result": payload}
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": tool_name,
                            "content": json.dumps(payload, ensure_ascii=True),
                        }
                    )

            if not summary.strip():
                response = await _ask_model(
                    llm,
                    model_name,
                    messages,
                    tool_definitions,
                    allow_tool_calls=False,
                )
                summary = response.choices[0].message.content or ""
            if not summary.strip():
                raise ForecastAgentError("The LLM returned an empty weather summary.")

            soil_result = next(
                (
                    item["result"]
                    for item in tool_results
                    if item["tool"] == "get_soil_moisture"
                    and item["result"].get("status") == "MOCKED/ESTIMATED"
                ),
                None,
            )
            return {
                "question": question.strip(),
                "crop": crop.strip(),
                "location": location.strip(),
                "summary": summary.strip(),
                "tools_used": list(dict.fromkeys(item["tool"] for item in tool_results)),
                "tool_results": tool_results,
                "forecasts_are_estimates": True,
                "forecast_notice": "Weather forecasts are estimates, not guarantees.",
                "soil_moisture_is_mocked": soil_result is not None,
                "soil_moisture_notice": (
                    "Soil moisture is MOCKED/ESTIMATED, not measured sensor data."
                    if soil_result is not None
                    else None
                ),
                "agricultural_recommendation": None,
            }
    except ForecastAgentError:
        raise
    except Exception as error:
        raise ForecastAgentError(
            "Could not connect to the weather MCP server or complete the agent request."
        ) from error
    finally:
        if owns_llm_client:
            await llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize weather for a farmer's question.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--location", required=True)
    args = parser.parse_args()

    try:
        result = asyncio.run(
            run_forecast_agent(args.question, args.crop, args.location)
        )
    except (ForecastAgentError, ValueError) as error:
        parser.exit(1, f"Forecast agent failed: {error}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()