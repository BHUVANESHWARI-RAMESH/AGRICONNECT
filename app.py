"""Streamlit farmer interface for the LangGraph multi-agent crop & weather advisor."""

import asyncio
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

try:
    from dotenv import load_dotenv, set_key
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
except ImportError:
    pass

from local_advisor import get_local_components
from workflow import run_workflow


st.set_page_config(
    page_title="Farmer Crop & Weather Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, responsive appearance
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .card-metric {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #2d6a4f;
        margin-bottom: 10px;
    }
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success { background-color: #d8f3dc; color: #1b4332; }
    .badge-warning { background-color: #fff3cd; color: #856404; }
    .badge-danger { background-color: #f8d7da; color: #721c24; }
    .badge-info { background-color: #d1ecf1; color: #0c5460; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _check_openai_status(api_key: str, base_url: str | None = None) -> dict[str, Any]:
    """Test OpenAI API key format, authentication, and credit balance."""
    if not api_key:
        return {"status": "missing", "message": "No OpenAI API key configured."}
    
    url = f"{base_url.rstrip('/')}/models" if base_url else "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            # Key is authentic; now test a minimal chat completion to check credits
            chat_url = f"{base_url.rstrip('/')}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions"
            test_body = {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1,
            }
            chat_res = requests.post(chat_url, headers=headers, json=test_body, timeout=8)
            if chat_res.status_code == 200:
                return {"status": "active", "message": "API key connected & credits active!"}
            elif chat_res.status_code == 429:
                err_data = chat_res.json().get("error", {})
                code = err_data.get("code", "")
                if code == "credit_balance_exhausted" or "credit" in err_data.get("message", "").lower():
                    return {
                        "status": "exhausted",
                        "message": "Key is valid, but OpenAI credit balance is exhausted ($0.00).",
                        "details": err_data.get("message", ""),
                    }
                return {"status": "rate_limited", "message": "OpenAI rate limit reached."}
            else:
                return {
                    "status": "auth_ok_chat_error",
                    "message": f"Auth OK, but chat failed: HTTP {chat_res.status_code}",
                }
        elif response.status_code == 401:
            return {"status": "invalid", "message": "Invalid OpenAI API Key (HTTP 401)."}
        else:
            return {"status": "error", "message": f"OpenAI returned HTTP {response.status_code}."}
    except Exception as e:
        return {"status": "unreachable", "message": f"Connection error: {e}"}


def _check_tavily_status(api_key: str) -> dict[str, Any]:
    """Test Tavily API key."""
    if not api_key:
        return {"status": "missing", "message": "No Tavily API key configured."}
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        client.search(query="weather", max_results=1)
        return {"status": "active", "message": "Tavily connected & active!"}
    except Exception as e:
        return {"status": "error", "message": f"Tavily error: {e}"}


# ==========================================
# Sidebar: Configuration & API Key Inspector
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration & Keys")
    
    current_openai_key = os.getenv("OPENAI_API_KEY", "")
    current_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    current_base_url = os.getenv("OPENAI_BASE_URL", "")
    current_tavily_key = os.getenv("TAVILY_API_KEY", "")

    # Quick API Health check on load or manual test
    if "api_status" not in st.session_state:
        st.session_state["api_status"] = _check_openai_status(current_openai_key, current_base_url)

    openai_stat = st.session_state["api_status"]
    
    if openai_stat["status"] == "active":
        st.markdown('<span class="status-badge badge-success">🟢 OpenAI Live</span>', unsafe_allow_html=True)
    elif openai_stat["status"] == "exhausted":
        st.markdown('<span class="status-badge badge-warning">🟡 OpenAI Credits $0.00</span>', unsafe_allow_html=True)
        st.caption("Credit balance exhausted. Use Local Mode or add billing credits.")
    elif openai_stat["status"] == "missing":
        st.markdown('<span class="status-badge badge-info">⚪ No OpenAI Key</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-danger">🔴 OpenAI Issue</span>', unsafe_allow_html=True)

    with st.expander("🔑 Edit API Keys & Endpoints", expanded=openai_stat["status"] != "active"):
        openai_key_input = st.text_input(
            "OpenAI API Key",
            value=current_openai_key,
            type="password",
            placeholder="sk-...",
            help="OpenAI API key used for Live LLM processing.",
        )
        
        model_options = ["gpt-4o-mini", "gpt-4o", "gpt-5.6-luna", "gpt-3.5-turbo", "Other..."]
        default_idx = model_options.index(current_model) if current_model in model_options else len(model_options) - 1
        selected_model = st.selectbox("OpenAI Model", model_options, index=default_idx)
        if selected_model == "Other...":
            model_input = st.text_input("Custom Model Name", value=current_model)
        else:
            model_input = selected_model

        base_url_input = st.text_input(
            "OpenAI Base URL (Optional)",
            value=current_base_url,
            placeholder="https://api.openai.com/v1",
            help="Useful for compatible endpoints (Ollama, vLLM, DeepSeek, etc.).",
        )

        tavily_key_input = st.text_input(
            "Tavily API Key (Optional)",
            value=current_tavily_key,
            type="password",
            placeholder="tvly-...",
            help="Used for live search of recent government weather alerts.",
        )

        if st.button("Save & Validate Keys", use_container_width=True):
            os.environ["OPENAI_API_KEY"] = openai_key_input.strip()
            os.environ["OPENAI_MODEL"] = model_input.strip()
            if base_url_input.strip():
                os.environ["OPENAI_BASE_URL"] = base_url_input.strip()
            elif "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]
            os.environ["TAVILY_API_KEY"] = tavily_key_input.strip()

            # Save to .env file in workspace
            env_path = Path(__file__).resolve().parent / ".env"
            with open(env_path, "w", encoding="utf-8") as f:
                f.write(f'OPENAI_API_KEY="{openai_key_input.strip()}"\n')
                f.write(f'OPENAI_MODEL="{model_input.strip()}"\n')
                f.write(f'OPENAI_BASE_URL="{base_url_input.strip()}"\n')
                f.write(f'TAVILY_API_KEY="{tavily_key_input.strip()}"\n')

            st.session_state["api_status"] = _check_openai_status(openai_key_input.strip(), base_url_input.strip())
            st.rerun()

    # Execution Mode selector
    st.divider()
    st.subheader("🚀 Execution Mode")
    
    # If credit is exhausted or key missing, default to Local Intelligent Mode
    default_mode_index = 0 if openai_stat["status"] in {"missing", "exhausted", "invalid"} else 1
    
    execution_mode = st.radio(
        "Select Pipeline Mode:",
        [
            "⚡ Local Intelligent Mode (Free / Zero-Credits)",
            "🤖 Live OpenAI Mode",
        ],
        index=default_mode_index,
        help=(
            "Local Intelligent Mode uses real Open-Meteo weather API + real TNAU RAG vector search "
            "+ real Tavily alerts without requiring active OpenAI credits."
        ),
    )
    
    auto_fallback = st.checkbox(
        "Auto-fallback to Local Mode if OpenAI fails",
        value=True,
        help="If OpenAI returns 429 (quota exhausted) or network fails, automatically finish using Local RAG.",
    )

    st.divider()
    st.subheader("💡 Sample Scenarios")
    if st.button("🌾 Paddy Irrigation (Thanjavur)", use_container_width=True):
        st.session_state["preset_crop"] = "Paddy"
        st.session_state["preset_loc"] = "Thanjavur"
        st.session_state["preset_q"] = "Should I irrigate my paddy fields this week?"
        st.rerun()

    if st.button("🌿 Cotton Spraying (Erode)", use_container_width=True):
        st.session_state["preset_crop"] = "Cotton"
        st.session_state["preset_loc"] = "Erode"
        st.session_state["preset_q"] = "Can I spray pesticide tomorrow morning?"
        st.rerun()

    if st.button("🍌 Banana Wind Alert (Cuddalore)", use_container_width=True):
        st.session_state["preset_crop"] = "Banana"
        st.session_state["preset_loc"] = "Cuddalore"
        st.session_state["preset_q"] = "Storm warning reported. What precautions should I take for banana plants?"
        st.rerun()

    if st.button("🥜 Groundnut Harvest (Madurai)", use_container_width=True):
        st.session_state["preset_crop"] = "Groundnut"
        st.session_state["preset_loc"] = "Madurai"
        st.session_state["preset_q"] = "Is the weather suitable for harvesting groundnut this week?"
        st.rerun()


# ==========================================
# Main Interface
# ==========================================
st.markdown('<div class="main-title">🌾 Farmer Crop & Weather Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-caption">Real-time Open-Meteo forecasts, Tavily regional alerts, and TNAU agronomic guidance powered by LangGraph.</div>',
    unsafe_allow_html=True,
)

# Notification banner for credit status
if openai_stat["status"] == "exhausted":
    st.info(
        "ℹ️ **OpenAI Account Note:** The configured key authenticates successfully, but has no remaining OpenAI billing credits "
        "(HTTP 429 `credit_balance_exhausted`). The app is currently running in **Local Intelligent Mode**, querying real Open-Meteo weather "
        "and real TNAU RAG knowledge chunks for free. You can add credits at [OpenAI Billing](https://platform.openai.com/settings/organization/billing/) "
        "whenever you want to switch to Live LLM mode."
    )


def _make_farmer_message(crop: str, location: str, question: str) -> str:
    parts = []
    if crop != "Not specified":
        parts.append(f"I grow {crop.casefold()}.")
    if location.strip():
        parts.append(f"My farm is near {location.strip()}.")
    if question.strip():
        parts.append(question.strip())
    return " ".join(parts)


def _render_report(state: dict[str, Any]) -> None:
    if state.get("workflow_error"):
        st.error(state.get("final_answer", "The workflow could not be completed."))
        return
    if state.get("needs_clarification"):
        st.warning(f"⚠️ **Clarification Needed:** {state.get('final_answer', 'Please provide the missing crop or location.')}")
        return

    report = state.get("report")
    if not isinstance(report, dict):
        st.error(state.get("final_answer", "The workflow did not return a report."))
        return

    # 1. Overall Recommendation Header
    overall = report.get("overall_recommendation")
    if overall and "No overall recommendation" not in overall:
        st.success(f"### 🎯 Key Recommendation\n{overall}")
    
    # 2. Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="📍 Location",
            value=state.get("location") or "Tamil Nadu",
            delta=state.get("crop", "").capitalize() if state.get("crop") else None,
        )
    with col2:
        rain_hist = report.get("rainfall_history", "0 mm")
        st.metric(label="🌧️ Past 7-Day Rain", value=rain_hist.split(";")[0] if ";" in rain_hist else rain_hist)
    with col3:
        soil = report.get("soil_moisture", "N/A")
        st.metric(label="🌱 Soil Moisture", value=soil.split(";")[0] if ";" in soil else soil)
    with col4:
        alerts = report.get("active_alerts", [])
        alert_count = len([a for a in alerts if "no current alert" not in a.lower() and "no alert was" not in a.lower()])
        st.metric(label="⚠️ Active Alerts", value=f"{alert_count} detected")

    st.markdown("---")

    # 3. Two-Column Layout for Advisory & Weather
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("📋 Action Plan")
        
        actions = report.get("what_to_do", [])
        if actions:
            st.markdown("##### ✅ What To Do:")
            for action in actions:
                st.markdown(f"- {action}")
        else:
            st.write("No specific actions were supplied.")

        cautions = report.get("what_to_avoid", [])
        if cautions:
            st.markdown("##### ❌ What To Avoid:")
            for item in cautions:
                st.markdown(f"- {item}")

        reasons = report.get("reasons", [])
        if reasons:
            st.markdown("##### 🔍 Why This Matters (Agronomic Reasons):")
            for reason in reasons:
                st.markdown(f"- {reason}")

        st.subheader("📅 7-Day Weather & Agronomic Schedule")
        day_rows = report.get("day_by_day_advisory", [])
        if day_rows:
            for row in day_rows:
                st.markdown(f"- {row}")
        else:
            st.write("No day-by-day details were supplied.")

    with right_col:
        st.subheader("🌦️ Weather Summary")
        st.info(report.get("weather_summary", "Weather information was not supplied."))
        st.caption(f"ℹ️ {report.get('weather_uncertainty', 'Forecasts are estimates and subject to change.')}")

        st.subheader("⚠️ Regional Weather & Disaster Alerts")
        active_alerts = report.get("active_alerts", [])
        for alert in active_alerts:
            if "no alert" in alert.lower() or "no current alert" in alert.lower() or "alert status is unavailable" in alert.lower():
                st.markdown(f"🟢 *{alert}*")
            else:
                st.warning(f"⚠️ {alert}")

        st.subheader("📚 Verified TNAU Agricultural Sources")
        sources = report.get("agricultural_sources", [])
        if not sources:
            st.write("No agricultural sources were supplied.")
        for source in sources:
            title = source.get("title") or source.get("source_document") or "Source Document"
            doc = source.get("source_document")
            url = source.get("source_url")
            label = f"📄 {title}" + (f" ({doc})" if doc and doc != title else "")
            if isinstance(url, str) and url.startswith("https://"):
                st.link_button(label, url, use_container_width=True)
            else:
                st.markdown(f"- {label}")

    if state.get("log"):
        with st.expander("🛠️ Workflow Execution Audit Log"):
            for step in state["log"]:
                st.code(step, language="text")


# ==========================================
# Input Form
# ==========================================
default_crop = st.session_state.get("preset_crop", "Paddy")
default_loc = st.session_state.get("preset_loc", "Thanjavur")
default_q = st.session_state.get("preset_q", "Should I irrigate this week?")

crop_list = ["Not specified", "Paddy", "Cotton", "Banana", "Groundnut"]
crop_index = crop_list.index(default_crop) if default_crop in crop_list else 1

with st.form("farmer_request"):
    c1, c2 = st.columns([1, 1])
    with c1:
        crop = st.selectbox("Crop", crop_list, index=crop_index)
    with c2:
        location = st.text_input("Farm location (Town/District)", value=default_loc)
    
    question = st.text_area(
        "Farmer's question or planned activity",
        value=default_q,
        height=95,
        placeholder="For example: Should I irrigate this week? Can I spray tomorrow?",
    )
    
    submitted = st.form_submit_button("🌱 Get Agricultural Advice", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.warning("Please enter your question to continue.")
    else:
        farmer_message = _make_farmer_message(crop, location, question)
        use_local = "Local Intelligent Mode" in execution_mode
        
        try:
            with st.spinner("Executing LangGraph multi-agent workflow..."):
                if use_local:
                    components = get_local_components()
                    result = asyncio.run(run_workflow(farmer_message, components=components))
                else:
                    # Live OpenAI Mode
                    try:
                        result = asyncio.run(run_workflow(farmer_message))
                        # If workflow returned a supervisor failure and auto-fallback is enabled
                        if result.get("workflow_error") and auto_fallback:
                            st.warning(
                                "⚠️ Live OpenAI request failed (likely due to exhausted credits). "
                                "Automatically running Local Intelligent Mode with verified TNAU RAG & Open-Meteo weather..."
                            )
                            components = get_local_components()
                            result = asyncio.run(run_workflow(farmer_message, components=components))
                    except Exception as live_err:
                        if auto_fallback:
                            st.warning(f"⚠️ Live OpenAI failed ({live_err}). Automatically executing fallback via Local Intelligent Mode...")
                            components = get_local_components()
                            result = asyncio.run(run_workflow(farmer_message, components=components))
                        else:
                            raise live_err

                st.session_state["workflow_result"] = result

        except Exception as e:
            st.error(f"The request could not be completed: {e}")

if "workflow_result" in st.session_state:
    st.divider()
    _render_report(st.session_state["workflow_result"])