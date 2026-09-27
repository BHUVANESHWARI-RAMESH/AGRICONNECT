"""End-to-end checks for the MCP server over a local stdio subprocess."""

import anyio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.types import TextContent


async def main() -> None:
    project_directory = Path(__file__).resolve().parent
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(project_directory / "mcp_server.py")],
    )

    async with Client(server) as client:
        listed = await client.list_tools()
        tools = {tool.name: tool for tool in listed.tools}
        expected_tools = {
            "get_forecast",
            "get_rainfall_history",
            "get_soil_moisture",
        }
        if set(tools) != expected_tools:
            raise AssertionError(f"Unexpected MCP tools: {sorted(tools)}")

        expected_parameters = {
            "get_forecast": {"location"},
            "get_rainfall_history": {"location", "days"},
            "get_soil_moisture": {"location"},
        }
        for name, required in expected_parameters.items():
            actual = set(tools[name].input_schema["properties"])
            if actual != required:
                raise AssertionError(f"{name} input parameters were {actual}")
            if not tools[name].description:
                raise AssertionError(f"{name} has no description")

        forecast_result = await client.call_tool(
            "get_forecast", {"location": "Thanjavur"}
        )
        if forecast_result.is_error:
            raise AssertionError(f"Forecast tool failed: {forecast_result.content}")
        forecast = forecast_result.structured_content
        if not isinstance(forecast, dict) or len(forecast["forecast"]) != 7:
            raise AssertionError("Forecast tool did not return seven forecast days")

        rainfall_result = await client.call_tool(
            "get_rainfall_history", {"location": "Thanjavur", "days": 7}
        )
        if rainfall_result.is_error:
            raise AssertionError(f"Rainfall tool failed: {rainfall_result.content}")
        rainfall = rainfall_result.structured_content
        if not isinstance(rainfall, dict) or len(rainfall["rainfall_by_day"]) != 7:
            raise AssertionError("Rainfall tool did not return seven daily values")

        soil_result = await client.call_tool(
            "get_soil_moisture", {"location": "Thanjavur"}
        )
        if soil_result.is_error:
            raise AssertionError(f"Soil moisture tool failed: {soil_result.content}")
        soil = soil_result.structured_content
        if not isinstance(soil, dict) or soil.get("status") != "MOCKED/ESTIMATED":
            raise AssertionError("Soil moisture result is not marked as mocked")

        invalid_result = await client.call_tool(
            "get_forecast", {"location": "InvalidRandomPlace123456"}
        )
        if not invalid_result.is_error:
            raise AssertionError("Invalid location did not return an MCP tool error")
        if not any(
            isinstance(block, TextContent) and "Location not found" in block.text
            for block in invalid_result.content
        ):
            raise AssertionError("Invalid location error was not clear")

        print("PASS: all three tools are discoverable with the expected inputs")
        print("PASS: forecast and rainfall tools return structured results")
        print("PASS: soil moisture result is marked MOCKED/ESTIMATED")
        print("PASS: invalid location is returned as a clear MCP tool error")


if __name__ == "__main__":
    anyio.run(main)