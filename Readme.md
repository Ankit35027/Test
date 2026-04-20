# Fleet Maintenance Intelligence System

A predictive fleet maintenance system that combines classical machine learning with an agentic AI workflow. The system analyzes vehicle sensor data to forecast maintenance risk and generates structured action plans using a retrieval-augmented generation (RAG) pipeline.

---

## Overview

Traditional fleet maintenance relies on fixed schedules or reacting after something breaks. This system takes a different approach by using historical vehicle data and live sensor readings to predict which vehicles are at risk before a breakdown occurs. On top of the prediction, an AI agent retrieves relevant maintenance guidelines and produces a clear, actionable report for each vehicle.

The project is structured in two milestones:

- **Milestone 1** — Train and evaluate a classical ML model on 40,000 vehicle records to classify maintenance risk.
- **Milestone 2** — Wrap the ML prediction inside a LangGraph agentic workflow that retrieves domain knowledge via FAISS and generates a prescriptive report using a Groq-hosted LLM.

---

## Architecture

| Component | Technology |
|---|---|
| ML Prediction | Logistic Regression (scikit-learn) |
| Agent Framework | LangGraph |
| Knowledge Retrieval | FAISS + Sentence Transformers (RAG) |
| LLM | Groq API (Llama 3 8B) |
| UI | Streamlit |

---

## Dataset

The model is trained on a dataset of 40,000 vehicle records. Each record contains the following features:

**Numerical:** Usage Hours, Engine Temperature (°C), Tire Pressure (PSI), Oil Quality Score, Battery Voltage (V), Vibration Level (g), Maintenance Cost

**Categorical:** Vehicle Type (Car, Truck, Bus), Brake Condition (Good, Fair, Poor)

**Target:** Maintenance Required — binary label where `1` indicates high risk and `0` indicates safe.

During exploratory data analysis, Engine Temperature showed the highest correlation with failure risk (0.52), while Oil Quality showed a strong negative correlation (-0.35).

---

## Model Performance

Both Logistic Regression and a tuned Decision Tree were evaluated on an 8,000-sample test split.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 86.84% | 86.37% | 83.36% | 84.84% |
| Decision Tree (max_depth=11) | 82.89% | 82.73% | 77.42% | 79.99% |

Logistic Regression is the deployed model because it achieves higher recall (83.36%), which is the most important metric in this context. Missing an actual high-risk vehicle is more costly than a false alarm.

---

## Project Structure

```
.
├── app.py                      # Streamlit application entry point
├── fleet_system.py             # ML prediction and LangGraph agent logic
├── simple_fleet_model.pkl      # Trained Logistic Regression model
├── requirements.txt            # Python dependencies
├── Training_Data/              # Dataset files used for model training
├── rag/                        # RAG pipeline (FAISS index, document chunks)
├── assets/                     # Static files (CSS styling)
├── .devcontainer/              # Dev container configuration
└── PROJECT_WORKFLOW_REPORT.md  # Detailed methodology report
```

---

## Getting Started

### Prerequisites

- Python 3.9 or above
- A [Groq API key](https://console.groq.com/) (free tier works)
- A Hugging Face token (for Sentence Transformers, optional but recommended)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Ankit35027/Test.git
cd Test
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables. Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### Running the Application

```bash
streamlit run app.py
```

The app will open in your browser. Use the sidebar to enter vehicle parameters, then click **Run Full Analysis** to get the ML prediction and the AI agent report.

---

## How It Works

1. You enter vehicle sensor readings (engine temperature, tire pressure, oil quality, etc.) in the sidebar.
2. The trained Logistic Regression model predicts whether the vehicle is at high risk of requiring maintenance, along with a probability score and the contributing factors.
3. If you also request the AI agent report, a LangGraph workflow retrieves relevant maintenance guidelines from a FAISS vector store and passes everything to a Llama 3 model via Groq to generate a health summary, action plan, and operational disclaimer.
4. All results are displayed in a clean tabbed dashboard.

---

## Key Design Decisions

- **Logistic Regression over Decision Tree** — despite the Decision Tree being tuned (max_depth=11 selected to prevent overfitting), Logistic Regression consistently outperformed it on recall, which is the priority metric for safety-critical predictions.
- **RAG for domain knowledge** — instead of relying solely on the LLM's parametric knowledge, the agent retrieves grounded maintenance guidelines at inference time, making the reports more reliable and auditable.
- **Groq free tier** — the system is designed to run without paid API costs beyond the Groq free tier, making it accessible for experimentation and demos.

---


