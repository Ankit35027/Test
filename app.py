import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from fleet_system import run_fleet_agent


st.set_page_config(
    page_title="Fleet AI - Maintenance Dashboard",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)


load_dotenv()


def inject_css() -> None:
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
    components.html(
        """
        <script>
        const removeSidebarToggle = () => {
          const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
          if (!sidebar) return;
          const candidates = sidebar.querySelectorAll('button');
          candidates.forEach((button) => {
            const label = ((button.getAttribute('aria-label') || '') + ' ' + (button.textContent || '')).toLowerCase();
            const isHeaderToggle =
              label.includes('collapse') ||
              label.includes('close sidebar') ||
              label.includes('hide sidebar') ||
              label.includes('toggle sidebar') ||
              (button.textContent || '').includes('«') ||
              (button.textContent || '').includes('‹');
            if (isHeaderToggle) {
              button.style.display = 'none';
              button.style.visibility = 'hidden';
              button.style.pointerEvents = 'none';
            }
          });

          const firstButtons = sidebar.querySelectorAll(':scope > div button');
          firstButtons.forEach((button) => {
            const rect = button.getBoundingClientRect();
            if (rect.top < 120) {
              button.style.display = 'none';
              button.style.visibility = 'hidden';
              button.style.pointerEvents = 'none';
            }
          });
        };
        removeSidebarToggle();
        setInterval(removeSidebarToggle, 500);
        </script>
        """,
        height=0,
        width=0,
    )


def get_secret_value(name: str, default: str = "") -> str:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, default)

@st.cache_resource
def load_fleet_system():
    return run_fleet_agent


inject_css()

with st.sidebar:
    st.markdown("## 🚛 Fleet AI")
    st.markdown("---")
    st.markdown("### 🔐 Credentials")

    groq_configured = bool(get_secret_value("GROQ_API_KEY"))
    hf_configured = bool(get_secret_value("HF_TOKEN") or get_secret_value("HUGGINGFACEHUB_API_TOKEN"))

    if groq_configured:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key missing")

    if hf_configured:
        st.success("Hugging Face token detected")
    else:
        st.warning("Hugging Face token missing")

    st.markdown("---")
    st.markdown("### 📋 Vehicle Parameters")

    vehicle_id = st.text_input("Vehicle ID", value="VH-001")
    vehicle_type = st.selectbox("Vehicle Type", ["Car", "Truck", "Bus"])

    st.markdown("**Sensor Readings**")
    usage_hours = st.slider("Usage Hours", 0, 10000, 3500)
    engine_temp = st.slider("Engine Temperature (°C)", 50, 130, 87)
    tire_pressure = st.slider("Tire Pressure (PSI)", 20, 50, 33)
    oil_quality = st.slider("Oil Quality Score", 0.0, 1.0, 0.55, step=0.01)
    battery_voltage = st.slider("Battery Voltage (V)", 10.0, 15.0, 12.6, step=0.1)
    vibration_level = st.slider("Vibration Level (g)", 0.0, 1.5, 0.35, step=0.01)
    maintenance_cost = st.number_input("Maintenance Cost ($)", 0, 50000, 1200)

    brake_condition = st.selectbox("Brake Condition", ["Good", "Fair", "Poor"])

    st.markdown("---")
    analyze_btn = st.button("🔍 Run Full Analysis", use_container_width=True)

st.markdown("# 🚛 Fleet Maintenance Intelligence")
st.markdown("*Predictive ML + Agentic AI Fleet Analytics*")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Agent Report", "ℹ️ About"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vehicle ID", vehicle_id)
    col2.metric("Type", vehicle_type)
    col3.metric("Usage Hours", f"{usage_hours:,} hrs")
    col4.metric("Engine Temp", f"{engine_temp}°C")

    st.markdown("---")

    if analyze_btn or st.session_state.get("ml_result"):
        vehicle_data = {
            "vehicle_id": vehicle_id,
            "usage_hours": usage_hours,
            "engine_temp": engine_temp,
            "tire_pressure": tire_pressure,
            "oil_quality": oil_quality,
            "battery_voltage": battery_voltage,
            "vibration_level": vibration_level,
            "maintenance_cost": maintenance_cost,
            "vehicle_type": vehicle_type,
            "brake_condition": brake_condition,
        }

        if analyze_btn:
            from fleet_system import predict_maintenance

            with st.spinner("Running ML prediction..."):
                ml_result = predict_maintenance(vehicle_data)
                st.session_state["ml_result"] = ml_result
                st.session_state["vehicle_data"] = vehicle_data

        ml_result = st.session_state.get("ml_result", {})

        if ml_result:
            risk = ml_result["risk_label"]
            score = ml_result["risk_score"]

            col_r1, col_r2 = st.columns([1, 2])
            with col_r1:
                if "High Risk" in risk:
                    st.error(f"## {risk}")
                    st.metric("Risk Probability", f"{score:.1%}")
                else:
                    st.success(f"## {risk}")
                    st.metric("Risk Probability", f"{score:.1%}")

                st.progress(score)

            with col_r2:
                st.markdown("### 🔬 Contributing Factors")
                factors = ml_result.get("contributing_factors", {})
                if factors:
                    factor_df = pd.DataFrame(
                        list(factors.items()),
                        columns=["Feature", "Assessment"],
                    )
                    st.dataframe(factor_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No major contributing factors were identified.")
    else:
        st.info("👈 Configure vehicle parameters in the sidebar and click **Run Full Analysis**")

with tab2:
    should_run_agent = analyze_btn and st.session_state.get("vehicle_data")

    if should_run_agent:
        agent_runner = load_fleet_system()
        vehicle_data = st.session_state["vehicle_data"]
        ml_result = st.session_state.get("ml_result", {})

        with st.spinner("🤖 Agent running: ML → RAG → LLM synthesis..."):
            progress = st.progress(0)
            status = st.empty()

            status.markdown("**Step 1/3** - Running ML prediction...")
            progress.progress(33)

            initial_state = {
                "vehicle_data": vehicle_data,
                "risk_score": ml_result.get("risk_score", 0.0),
                "risk_label": ml_result.get("risk_label", "Unknown"),
                "contributing_factors": ml_result.get("contributing_factors", {}),
                "retrieved_guidelines": "",
                "health_summary": "",
                "action_plan": "",
                "disclaimer": "",
            }

            status.markdown("**Step 2/3** - Retrieving maintenance guidelines (RAG)...")
            progress.progress(66)

            result = agent_runner(vehicle_data)

            status.markdown("**Step 3/3** - Generating AI report...")
            progress.progress(100)
            status.empty()
            progress.empty()

        st.session_state["agent_result"] = result

    agent_result = st.session_state.get("agent_result")

    if agent_result:
        current_vehicle = st.session_state.get("vehicle_data", {})

        st.markdown(f"### 📋 Fleet Report - Vehicle `{current_vehicle.get('vehicle_id', vehicle_id)}`")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏥 Health Summary")
            st.info(agent_result.get("health_summary", "N/A"))

        with col2:
            st.markdown("#### 📋 Action Plan")
            st.success(agent_result.get("action_plan", "N/A"))

        st.markdown("---")
        st.markdown("#### ⚠️ Operational Disclaimer")
        st.warning(
            agent_result.get(
                "disclaimer",
                "Always verify AI recommendations with a certified fleet technician.",
            )
        )

        with st.expander("📚 Retrieved Maintenance Guidelines (RAG Context)"):
            st.markdown(agent_result.get("retrieved_guidelines", "No guidelines retrieved."))
    else:
        st.info("👈 Run Full Analysis from the sidebar to generate the AI Agent report.")

with tab3:
    st.markdown(
        """
## About This System

This fleet management system combines **classical ML** (Milestone 1) with an
**agentic AI workflow** (Milestone 2) to provide predictive and prescriptive fleet analytics.

### Architecture
| Component | Technology |
|---|---|
| ML Prediction | Logistic Regression (scikit-learn) |
| Agent Framework | LangGraph |
| Knowledge Retrieval | FAISS + Sentence Transformers (RAG) |
| LLM | Groq API (Llama 3 8B) - Free Tier |
| UI | Streamlit |

### Dataset
- 40,000 vehicle records
- Logistic Regression: **86.84% accuracy**, **83.36% recall**
- Decision Tree (Tuned, max_depth=11): 82.89% accuracy

### Key Features
- **Predictive**: ML model forecasts maintenance risk from sensor telemetry
- **Prescriptive**: LangGraph agent retrieves guidelines and generates structured action plans
- **Explainable**: Contributing factors surfaced for each prediction
"""
    )
