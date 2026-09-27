"""LangGraph orchestration for the farmer crop-and-weather project."""

import argparse
import asyncio
import inspect
import json
import logging
import re
from typing import Any, Awaitable, Callable, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import END, START, StateGraph

from alert_agent import run_alert_agent
from crop_advisor_agent import run_crop_advisor_agent
from forecast_agent import run_forecast_agent
from report_writer_agent import write_advisory_report
from supervisor_agent import extract_farmer_request
from workflow_guardrails import (
    validate_alert_result,
    validate_crop_guidance,
    validate_supervisor_result,
)


logger = logging.getLogger("farmer_crop_weather.workflow")
EXTREME_WEATHER_TERMS = re.compile(
    r"\b(cyclone|typhoon|hurricane|tornado|flood|heat ?wave|storm|heavy rain|"
    r"landslide|drought|wildfire|tsunami|earthquake|weather alert|weather warning|"
    r"disaster warning|red alert|orange alert|yellow alert|evacuation warning)\b",
    re.IGNORECASE,
)


class WorkflowState(TypedDict, total=False):
    farmer_message: str
    crop: str | None
    location: str | None
    question: str
    activity: str | None
    time_period: str | None
    missing_information: list[str]
    needs_clarification: bool
    clarifying_question: str | None
    agricultural_advice: str | None
    weather_information: dict[str, Any]
    extreme_weather_signal: bool
    extreme_weather_evidence: str | None
    alert_information: dict[str, Any]
    agricultural_guidance: dict[str, Any]
    report: dict[str, Any]
    final_answer: str
    workflow_error: str
    errors: list[str]
    log: list[str]


WorkflowCallable = Callable[..., Any]


def _default_supervisor(message: str, options: dict[str, Any]) -> Awaitable[dict[str, Any]]:
    return extract_farmer_request(
        message,
        llm_client=options.get("llm_client"),
        model=options.get("model"),
    )


def _default_forecast(state: WorkflowState, options: dict[str, Any]) -> Awaitable[dict[str, Any]]:
    return run_forecast_agent(
        state["question"],
        state["crop"] or "",
        state["location"] or "",
        llm_client=options.get("llm_client"),
        model=options.get("model"),
    )


async def _default_alert(state: WorkflowState, options: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(
        run_alert_agent,
        state["location"] or "",
        state["crop"] or "",
        state["question"],
        tavily_client=options.get("tavily_client"),
    )


def _default_crop_advisor(
    state: WorkflowState,
    options: dict[str, Any],
) -> Awaitable[dict[str, Any]]:
    weather_information = dict(state.get("weather_information", {}))
    if state.get("alert_information") is not None:
        weather_information["current_alerts"] = state["alert_information"]
    return run_crop_advisor_agent(
        state["question"],
        state["crop"] or "",
        state["location"] or "",
        weather_information=weather_information,
        llm_client=options.get("llm_client"),
        forecast_llm_client=options.get("forecast_llm_client"),
        model=options.get("model"),
    )


def _default_report_writer(report_input: dict[str, Any], _options: dict[str, Any]) -> dict[str, Any]:
    return write_advisory_report(report_input)


def _has_extreme_weather_signal(state: WorkflowState) -> tuple[bool, str | None]:
    question = state.get("question", "")
    forecast = state.get("weather_information", {})
    searchable = [question]
    if isinstance(forecast, dict):
        searchable.extend([str(forecast.get("summary", "")), str(forecast.get("forecast_notice", ""))])
        for item in forecast.get("tool_results", []):
            if isinstance(item, dict):
                searchable.append(str(item.get("result", {})))
    match = EXTREME_WEATHER_TERMS.search(" ".join(searchable))
    return (match is not None, match.group(0).casefold() if match else None)


async def _call(component: WorkflowCallable, *args: Any) -> Any:
    result = component(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def build_workflow(components: dict[str, WorkflowCallable] | None = None):
    """Build the graph, allowing component injection for tests and local runs."""
    options = components or {}
    supervisor_component = options.get("supervisor", _default_supervisor)
    forecast_component = options.get("forecast", _default_forecast)
    alert_component = options.get("alert", _default_alert)
    advisor_component = options.get("crop_advisor", _default_crop_advisor)
    writer_component = options.get("report_writer", _default_report_writer)

    async def supervisor_node(state: WorkflowState) -> dict[str, Any]:
        logger.info("workflow node started: supervisor")
        try:
            extracted = await _call(supervisor_component, state["farmer_message"], options)
            extracted = validate_supervisor_result(extracted)
            updates = dict(extracted)
            updates["log"] = state.get("log", []) + ["supervisor: extracted request fields"]
            if extracted["needs_clarification"]:
                logger.info("workflow route: clarification required")
            else:
                logger.info("workflow route: request fields complete")
            return updates
        except Exception as error:
            logger.exception("workflow node failed: supervisor")
            return {
                "workflow_error": (
                    "The request could not be processed by the Supervisor Agent. "
                    "Check the configured OpenAI API key and model, then try again."
                ),
                "errors": state.get("errors", []) + [f"Supervisor failed: {error}"],
                "log": state.get("log", []) + ["supervisor: service/configuration error"],
            }

    async def workflow_failure_node(state: WorkflowState) -> dict[str, Any]:
        message = state.get("workflow_error", "The workflow could not be completed.")
        return {
            "final_answer": message,
            "report": {"error": message, "report_text": message},
            "log": state.get("log", []) + ["workflow: stopped after component error"],
        }

    async def clarification_node(state: WorkflowState) -> dict[str, Any]:
        message = state.get("clarifying_question") or "Please provide the crop and farm location."
        logger.info("workflow node: clarification")
        return {
            "final_answer": message,
            "report": {"clarification_required": True, "report_text": message},
            "log": state.get("log", []) + ["clarification: stopped before advice"],
        }

    async def forecast_node(state: WorkflowState) -> dict[str, Any]:
        logger.info("workflow node started: forecast")
        try:
            forecast = await _call(forecast_component, state, options)
            if not isinstance(forecast, dict):
                raise TypeError("Forecast Agent returned a non-dictionary result.")
            forecast.setdefault("forecasts_are_estimates", True)
            forecast.setdefault("forecast_notice", "Weather forecasts are estimates and may change.")
            return {
                "weather_information": forecast,
                "log": state.get("log", []) + ["forecast: weather data collected"],
            }
        except Exception as error:
            logger.exception("workflow node failed: forecast")
            unavailable = {
                "summary": "Weather information is unavailable; no weather values were inferred.",
                "forecasts_are_estimates": True,
                "forecast_notice": "Weather information is unavailable. Forecasts, when available, are estimates.",
                "error": True,
            }
            return {
                "weather_information": unavailable,
                "errors": state.get("errors", []) + [f"Forecast failed: {error}"],
                "log": state.get("log", []) + ["forecast: unavailable"],
            }

    async def extreme_weather_node(state: WorkflowState) -> dict[str, Any]:
        signal, evidence = _has_extreme_weather_signal(state)
        logger.info("workflow node: extreme_weather_check result=%s", signal)
        return {
            "extreme_weather_signal": signal,
            "extreme_weather_evidence": evidence,
            "log": state.get("log", []) + [
                f"extreme_weather_check: {'signal=' + evidence if signal else 'no signal'}"
            ],
        }

    async def alert_node(state: WorkflowState) -> dict[str, Any]:
        logger.info("workflow node started: alert")
        try:
            result = await _call(alert_component, state, options)
            if not isinstance(result, dict):
                raise TypeError("Alert Agent returned a non-dictionary result.")
            checked = validate_alert_result(result)
            return {
                "alert_information": checked,
                "log": state.get("log", []) + [
                    f"alert: status={checked.get('alert_status', 'unknown')}"
                ],
            }
        except Exception as error:
            logger.exception("workflow node failed: alert")
            unavailable = {
                "alert_found": None,
                "alert_status": "unavailable",
                "alert_description": "The alert search failed; no conclusion about current alerts can be made.",
                "alerts": [],
                "errors": [str(error)],
            }
            return {
                "alert_information": unavailable,
                "errors": state.get("errors", []) + [f"Alert search failed: {error}"],
                "log": state.get("log", []) + ["alert: unavailable"],
            }

    async def crop_advisor_node(state: WorkflowState) -> dict[str, Any]:
        logger.info("workflow node started: crop_advisor")
        try:
            result = await _call(advisor_component, state, options)
            if not isinstance(result, dict):
                raise TypeError("Crop Advisor returned a non-dictionary result.")
            checked = validate_crop_guidance(result)
            return {
                "agricultural_guidance": checked,
                "log": state.get("log", []) + ["crop_advisor: guidance validated against sources"],
            }
        except Exception as error:
            logger.exception("workflow node failed: crop_advisor")
            return {
                "agricultural_guidance": {
                    "overall_recommendation": "No agricultural recommendation was generated because source-grounded guidance was unavailable.",
                    "recommendations": [],
                    "sources": [],
                    "pesticide_dosage": None,
                },
                "errors": state.get("errors", []) + [f"Crop Advisor failed safely: {error}"],
                "log": state.get("log", []) + ["crop_advisor: no advice generated"],
            }

    async def report_writer_node(state: WorkflowState) -> dict[str, Any]:
        logger.info("workflow node started: report_writer")
        guidance = state.get("agricultural_guidance", {})
        report_input = {
            "farmer_question": state.get("question", state.get("farmer_message", "")),
            "crop": state.get("crop"),
            "location": state.get("location"),
            "weather_information": state.get("weather_information", {}),
            "weather_summary": guidance.get("weather_summary"),
            "weather_uncertainty": state.get("weather_information", {}).get("forecast_notice"),
            "rainfall_history": None,
            "soil_moisture": None,
            "agricultural_guidance": guidance,
            "current_alerts": state.get("alert_information"),
            "sources": guidance.get("sources", []),
            "day_by_day_advisory": guidance.get("day_by_day_advisory", []),
        }
        try:
            report = await _call(writer_component, report_input, options)
            if not isinstance(report, dict):
                raise TypeError("Report Writer returned a non-dictionary result.")
            return {
                "report": report,
                "final_answer": report.get("report_text", ""),
                "log": state.get("log", []) + ["report_writer: final report created"],
            }
        except Exception as error:
            logger.exception("workflow node failed: report_writer")
            fallback = (
                guidance.get("overall_recommendation")
                or "A final report could not be created from the available information."
            )
            return {
                "report": {"report_text": fallback, "errors": [str(error)]},
                "final_answer": fallback,
                "errors": state.get("errors", []) + [f"Report Writer failed: {error}"],
                "log": state.get("log", []) + ["report_writer: fallback response"],
            }

    def route_after_supervisor(state: WorkflowState) -> str:
        if state.get("workflow_error"):
            return "failure"
        return "clarification" if state.get("needs_clarification") else "forecast"

    def route_after_extreme_check(state: WorkflowState) -> str:
        return "alert" if state.get("extreme_weather_signal") else "crop_advisor"

    builder = StateGraph(WorkflowState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("workflow_failure", workflow_failure_node)
    builder.add_node("forecast", forecast_node)
    builder.add_node("extreme_weather_check", extreme_weather_node)
    builder.add_node("alert", alert_node)
    builder.add_node("crop_advisor", crop_advisor_node)
    builder.add_node("report_writer", report_writer_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "clarification": "clarification",
            "forecast": "forecast",
            "failure": "workflow_failure",
        },
    )
    builder.add_edge("clarification", END)
    builder.add_edge("workflow_failure", END)
    builder.add_edge("forecast", "extreme_weather_check")
    builder.add_conditional_edges(
        "extreme_weather_check",
        route_after_extreme_check,
        {"alert": "alert", "crop_advisor": "crop_advisor"},
    )
    builder.add_edge("alert", "crop_advisor")
    builder.add_edge("crop_advisor", "report_writer")
    builder.add_edge("report_writer", END)
    return builder.compile()


async def run_workflow(
    farmer_message: str,
    *,
    components: dict[str, WorkflowCallable] | None = None,
) -> WorkflowState:
    """Run the complete workflow while preserving the original farmer message."""
    if not isinstance(farmer_message, str) or not farmer_message.strip():
        raise ValueError("Please provide a non-empty farmer message.")
    graph = build_workflow(components)
    initial_state: WorkflowState = {
        "farmer_message": farmer_message.strip(),
        "errors": [],
        "log": [],
    }
    logger.info("workflow started")
    final_state = await graph.ainvoke(initial_state)
    logger.info("workflow finished status=%s", "needs_clarification" if final_state.get("needs_clarification") else "complete")
    return final_state


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the crop and weather workflow.")
    parser.add_argument("message", help="The farmer's request in plain language.")
    args = parser.parse_args()
    state = asyncio.run(run_workflow(args.message))
    print(state.get("final_answer") or json.dumps(state.get("report", {}), indent=2))


if __name__ == "__main__":
    main()