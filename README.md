# Farmer Crop & Weather Advisor: Data and MCP Modules

Module 1 looks up a location with the Open-Meteo Geocoding API and gets a seven-day forecast. Module 2 gets daily historical rainfall from the Open-Meteo Archive API. Module 3 returns a deterministic soil-moisture mock. Module 4 exposes these existing functions as MCP tools. Module 5 retrieves source-attributed agricultural knowledge. Module 6 summarizes weather data. Modules 7 and 8 provide separate alert and crop-guidance agents. Module 9 extracts farmer-request details, and Module 10 formats supplied component outputs into a report. These remain standalone components; there is no LangGraph workflow.

Module 11 connects the existing components with LangGraph, Module 12 validates safety constraints, Module 13 provides the Streamlit farmer interface, Module 14 contains end-to-end validation scenarios, and Module 15 provides Docker deployment files.

The Module 3 soil-moisture value is explicitly estimated, not measured sensor data. Modules 1-5 do not require an LLM.

## Install

From this folder, create and activate a virtual environment (recommended), then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Run the Module 1 forecast test for Thanjavur:

```powershell
python test_weather.py
```

Run the Module 2 rainfall history test for Thanjavur:

```powershell
python test_rainfall_history.py
```

Run the Module 3 soil moisture unit tests:

```powershell
python -m unittest test_soil_moisture.py -v
```

Run the MCP server using stdio (it waits for an MCP client):

```powershell
python mcp_server.py
```

In another terminal, test the server and all three tools with the included client:

```powershell
python test_mcp_server.py
```

Build or refresh the Module 5 local vector database:

```powershell
python rag_knowledge_base.py ingest
```

Retrieve source-backed chunks for a question:

```powershell
python rag_knowledge_base.py ask "What are the irrigation considerations for paddy?"
```

Test Module 5 ingestion, retrieval, and source metadata:

```powershell
python -m unittest test_knowledge_base.py -v
```

Run the Module 6 Forecast Agent (requires `OPENAI_API_KEY` and `OPENAI_MODEL` in the environment):

```powershell
python forecast_agent.py --question "I grow paddy near Thanjavur. Should I irrigate this week?" --crop paddy --location Thanjavur
```

Test Module 6 tool selection/orchestration and failure handling. The test uses a fake LLM but launches the real MCP server and calls its weather tools:

```powershell
python -m unittest test_forecast_agent.py -v
```

Run the Module 7 Alert Agent (requires `TAVILY_API_KEY`):

```powershell
python alert_agent.py --location Thanjavur --crop paddy --question "Any cyclone or flood alerts?"
```

Run the Module 8 Crop Advisor Agent (requires `OPENAI_API_KEY` and `OPENAI_MODEL`):

```powershell
python crop_advisor_agent.py --question "I grow paddy near Thanjavur. Should I irrigate this week?" --crop paddy --location Thanjavur
```

Test Module 7 date/region handling and Module 8 source-grounded advice without API keys:

```powershell
python -m unittest test_alert_agent.py test_crop_advisor_agent.py -v
```

Run the Module 11 complete workflow and guardrail tests:

```powershell
python -m unittest test_workflow.py test_workflow_guardrails.py -v
```

Run the Module 13 farmer interface locally:

```powershell
python -m streamlit run app.py
```

Run the Module 14 scenario validation and create `test_report.md`:

```powershell
python -m unittest test_complete_application.py -v
```

Run the Module 9 Supervisor Agent (requires `OPENAI_API_KEY` and `OPENAI_MODEL`):

```powershell
python supervisor_agent.py "I grow cotton in Erode. Can I spray tomorrow?"
```

Run the Module 10 Report Writer tests:

```powershell
python -m unittest test_report_writer_agent.py -v
```

Run both Module 9 and 10 test suites:

```powershell
python -m unittest test_supervisor_agent.py test_report_writer_agent.py -v
```

Run the Module 13 farmer interface locally:

```powershell
python -m streamlit run app.py
```

Run the Module 14 12-scenario validation and generate `test_report.md`:

```powershell
python -m unittest test_complete_application.py -v
```

Run the Module 11 LangGraph workflow (requires `OPENAI_API_KEY` and `OPENAI_MODEL`; weather/alert lookups also use the configured APIs):

```powershell
python workflow.py "I grow paddy near Thanjavur. Should I irrigate this week?"
```

Run the workflow scenario and safety-guardrail tests without LLM/Tavily credentials:

```powershell
python -m unittest test_workflow.py test_workflow_guardrails.py -v
```

## Module 1: Weather Forecast

Call the seven-day forecast function from another Python file:

```python
from weather import get_7_day_forecast

forecast = get_7_day_forecast("Coimbatore")
print(forecast)
```

Use city names such as `Thanjavur`, `Erode`, or `Coimbatore`. An unknown city raises `ValueError`; connection problems or invalid API responses raise `WeatherAPIError`.

## Module 2: Rainfall History

`get_rainfall_history(location, days)` returns rainfall totals for the requested number of completed days before today. It reuses Module 1's location lookup and returns one rainfall amount per date plus the total. Rainfall is measured in millimeters; missing historical data raises `WeatherAPIError` rather than being filled in.

```python
from rainfall_history import get_rainfall_history

history = get_rainfall_history("Thanjavur", 7)
print(history)
```

The result contains `location` (name and coordinates), `number_of_days`, `rainfall_by_day` (a list of dates and amounts), and `total_rainfall_mm`.

## Module 3: Soil Moisture Mock

`get_soil_moisture(location)` returns a deterministic example percentage for a non-empty location. The result is marked `MOCKED/ESTIMATED` and says it is not real sensor data. The same location always gives the same example value, making it useful for development tests only.

```python
from soil_moisture import get_soil_moisture

estimate = get_soil_moisture("Thanjavur")
print(estimate)
```

## Module 4: MCP Server

MCP (Model Context Protocol) is a standard way for an AI application to discover and call tools exposed by a program. This server uses the official MCP Python SDK and communicates over stdio, which lets a compatible host launch it as a local process. It delegates to the existing Python functions and returns their structured dictionary results; it does not contain a separate copy of their weather logic.

The server exposes these tools:

- `get_forecast(location)`: returns the existing seven-day Open-Meteo forecast.
- `get_rainfall_history(location, days)`: returns the existing historical daily rainfall and total.
- `get_soil_moisture(location)`: returns the existing deterministic mock, marked `MOCKED/ESTIMATED`.

Tool inputs are described by Python type annotations and tool descriptions are included in the server. Expected input and API errors are returned as MCP tool errors with a readable message.

## Module 5: Agricultural Knowledge Base

RAG means Retrieval-Augmented Generation. In simple terms, the system first searches trusted documents for relevant passages, then a later AI component could use those passages. This module only prepares and retrieves information; it does not generate advice or use an AI agent.

The `knowledge/` folder has crop-specific, source-attributed notes based on TNAU material. Each note stores its original source URL. The ingestion pipeline extracts Markdown/text or selectable PDF text, splits documents into overlapping chunks, and stores local embeddings in ChromaDB. Chroma uses the local `all-MiniLM-L6-v2` embedding model by default; its model files are downloaded the first time ingestion runs. No embedding API key is needed. Retrieved results include the crop, document, title, organization, source URL, and chunk text.

Add `.md`, `.txt`, or text-based `.pdf` documents to the matching crop folder, then rerun ingestion. The local vector database is stored in `knowledge/vector_store/`. The included source notes summarize TNAU pages; consult the linked original source for complete details and current local recommendations.

## Module 6: Forecast Agent

The Forecast Agent receives the farmer question, crop, and location. It asks the LLM which of the three existing MCP tools are relevant, calls those tools, and asks the LLM to summarize only the returned weather facts. Rainfall history uses the most recent seven completed days. The response always marks forecasts as estimates, labels soil moisture as mocked when used, and leaves `agricultural_recommendation` empty. It does not use the Module 5 knowledge base.

Set `OPENAI_API_KEY` and `OPENAI_MODEL` in your shell or secret manager before running the agent. The key is read from the environment and is not stored in project files. The optional `OPENAI_BASE_URL` can configure an OpenAI-compatible endpoint. If the LLM or an MCP tool fails, the agent returns a clear error or passes the tool failure into its weather-only summary rather than inventing data.

## Module 7: Alert Agent

The Alert Agent searches recent Tavily news for cyclone, flood, heatwave, and other severe-weather warning terms for the supplied location. It only marks an alert as a current regional candidate when a recently dated result mentions the requested region, a weather hazard, and warning language. Undated, old, future-dated, or region-mismatched results are not claimed as current alerts. A search hit is a lead to check, not proof of an active official order. If the question concerns harvesting, selling, prices, or a mandi, it also searches for recent market information.

Set `TAVILY_API_KEY` in your environment; no key is stored in source. Tavily's title, URL, description, and available publication date are returned. If the search fails, the result says it is unavailable rather than claiming there are no alerts.

## Module 8: Crop Advisor Agent

The Crop Advisor Agent combines the existing Forecast Agent and RAG retriever. It produces crop-specific guidance only when source-attributed material was retrieved, and every recommendation must cite a retrieved document. It explains reasons in plain language and includes forecast uncertainty. It never returns pesticide dosage; when dosage is asked about but not present in the retrieved excerpts, it says the dosage is not available from those sources. It validates that citations refer to retrieved documents. This module does not assemble a LangGraph workflow.

## Module 9: Supervisor Agent

The Supervisor Agent extracts the crop, location, farmer's question, explicitly requested activity, and explicitly stated time period from one message. It uses the configured OpenAI model for extraction, then checks crop, location, and time against the original text so inferred values are discarded. When crop or location is missing, it returns a clarifying question and does not provide advice.

## Module 10: Report Writer Agent

The Report Writer is deterministic: it formats only the component data supplied to `write_advisory_report(report_input)`. Its report includes an overall recommendation, available day-by-day information, actions and reasons, alerts, sources, and a weather uncertainty statement. Missing information is labeled as not supplied; it is not filled in. Pesticide dosage text is included only when the exact dosage can be verified in provided source excerpts; otherwise it is omitted with a note.

```python
from report_writer_agent import write_advisory_report

report = write_advisory_report({
  "crop": "paddy",
  "location": "Thanjavur",
  "farmer_question": "I grow paddy near Thanjavur. Should I irrigate this week?",
  "weather_summary": "The forecast is an estimate.",
  "agricultural_guidance": {
    "overall_recommendation": "Use the source-backed guidance provided by the Crop Advisor."
  },
})
print(report["report_text"])
```

## Module 11: LangGraph Workflow

The workflow connects the existing Supervisor, Forecast Agent, Alert Agent, Crop Advisor, and Report Writer. It preserves the farmer's original message and extracted fields in shared graph state, records node transitions for debugging, and handles missing required fields by asking a clarification instead of continuing. A keyword-based extreme-weather check routes requests with an explicit cyclone, flood, heatwave, storm, or similar signal through the Alert Agent before crop guidance. Otherwise, the alert node is skipped. Failures are recorded and produce cautious fallback text rather than fabricated facts.

The extreme-weather router responds to explicit warning/hazard language in the farmer's question or forecast output; it does not create numeric weather thresholds. Module 12 guardrails validate supervisor fields, current-alert evidence, source citations, pesticide dosage text, and weather uncertainty at workflow boundaries. This is the first graph orchestration; it is not a frontend or deployment setup.

The workflow tests use injected component doubles for the four sample scenarios, plus an integrated run using the real MCP weather tools and local RAG retriever with fake LLM/Tavily clients. Set the required API environment variables to run against live LLM and alert search services.

## Module 13: Farmer Interface

The Streamlit page has a crop selector, farm-location input, question box, and Get Advice button. Crop and location may be left unspecified so the Supervisor can ask for them. Submitting runs the existing LangGraph backend and displays the advisory sections, weather uncertainty, alerts, and linked sources. No language selector is shown because this project currently produces English output only.

## Module 14: Complete Testing

`test_complete_application.py` exercises the twelve requested success, missing-data, and service-failure scenarios and writes a PASS/FAIL table to `test_report.md`. The scenario harness uses deterministic component doubles; `test_workflow.py` separately exercises the existing MCP weather tools and local RAG retriever together.

## Module 15: Deployment

The Docker image installs `requirements.txt`, ingests the bundled agricultural documents into a persistent Chroma volume, and starts Streamlit on port 8501. The existing workflow launches the MCP server as a subprocess using the same Python runtime in the container.

For Docker Compose:

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_MODEL`. Set `TAVILY_API_KEY` to enable current alert and market searches. `OPENAI_BASE_URL` is optional for compatible endpoints.
2. Build and start with `docker compose up --build -d`.
3. Open `http://localhost:8501` locally, or expose port 8501 through your hosting provider/reverse proxy and use its HTTPS address.
4. Inspect logs with `docker compose logs -f farmer-advisor`; stop with `docker compose down`.

For local use, create and activate the virtual environment shown above, configure the same environment variables in your shell, run `python rag_knowledge_base.py ingest` once, and start `python -m streamlit run app.py`. In hosted environments, configure API keys in the platform's secret/environment-variable settings, not in source files. Open-Meteo does not require an API key. Never commit `.env`.

## Module 13: Farmer Interface

The Streamlit page provides crop selection, farm location, and a question form. Crop may be left unspecified so the Supervisor can request it; location is also allowed to be blank for clarification. Submitting sends the farmer's message into the existing LangGraph workflow. The report page displays weather, recent rainfall, mocked soil moisture, day-by-day conditions, guidance, alerts, sources, and forecast uncertainty. There is no language selector because this project currently produces English output only.

## Module 14: Complete Testing

`test_complete_application.py` exercises all 12 requested cases using deterministic component doubles and writes a PASS/FAIL table to `test_report.md`. The other module tests exercise live APIs where available and verify the real MCP/RAG integration path. Set API keys to additionally validate live OpenAI and Tavily requests.

## Module 15: Deployment

The Docker image starts the existing RAG ingestion pipeline, then serves the Streamlit application on port 8501. The MCP server runs as a child process of the backend graph using the same Python environment inside the container. The Chroma vector store uses a named volume and is initialized from the bundled `knowledge/` documents.

For Docker Compose deployment:

1. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` and `OPENAI_MODEL`. Add `TAVILY_API_KEY` to enable current alert searches. Set `OPENAI_BASE_URL` only when using an OpenAI-compatible endpoint.
2. Build and start the application with `docker compose up --build -d`.
3. Open `http://localhost:8501`. For a remote host, expose port 8501 through your hosting provider or reverse proxy and use its HTTPS URL.
4. View logs with `docker compose logs -f farmer-advisor`; stop the service with `docker compose down`.

For a local run, create a virtual environment and install `requirements.txt`, configure the same variables in the shell, build the vector store with `python rag_knowledge_base.py ingest`, then start `python -m streamlit run app.py`. Keep `.env` and all API keys out of source control. In hosted environments, configure these values in the platform's secret/environment-variable settings instead of committing a `.env` file. Open-Meteo itself does not require a key.

## Module 1 Result Format

The function returns a dictionary shaped like this. The actual dates and weather values come from the live API and will change:

```json
{
  "location": {
    "name": "Thanjavur",
    "country": "India",
    "latitude": 10.79,
    "longitude": 79.14
  },
  "forecast": [
    {
      "date": "YYYY-MM-DD",
      "temperature_max_c": 0.0,
      "temperature_min_c": 0.0,
      "precipitation_mm": 0.0,
      "humidity_mean_percent": 0,
      "wind_speed_max_kmh": 0.0
    }
  ]
}
```

The `forecast` list contains seven daily entries. Temperatures are in Celsius, precipitation in millimeters, humidity in percent, and wind speed in kilometers per hour.