"""MCP server exposing the project's existing weather functions."""

from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rainfall_history import get_rainfall_history as get_rainfall_history_data
from soil_moisture import get_soil_moisture as get_soil_moisture_data
from weather import WeatherAPIError, get_7_day_forecast


mcp = MCPServer(
    "farmer-crop-weather-advisor",
    instructions="Tools for weather forecasts, rainfall history, and mock soil moisture.",
)


@mcp.tool(title="Get 7-Day Weather Forecast")
def get_forecast(location: str) -> dict[str, Any]:
    """Get the seven-day temperature, precipitation, humidity, and wind forecast for a location."""
    try:
        return get_7_day_forecast(location)
    except (ValueError, WeatherAPIError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(title="Get Rainfall History")
def get_rainfall_history(location: str, days: int) -> dict[str, Any]:
    """Get daily historical rainfall in millimeters and its total for a location."""
    try:
        return get_rainfall_history_data(location, days)
    except (ValueError, WeatherAPIError) as error:
        raise ToolError(str(error)) from error


@mcp.tool(title="Get Estimated Soil Moisture")
def get_soil_moisture(location: str) -> dict[str, Any]:
    """Get a deterministic MOCKED/ESTIMATED soil-moisture value, not sensor data."""
    try:
        return get_soil_moisture_data(location)
    except ValueError as error:
        raise ToolError(str(error)) from error


if __name__ == "__main__":
    mcp.run()