# Fleet AI: Agentic Vehicle Maintenance System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Community_Cloud-FF4B4B)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-green)

## Project Overview
Fleet AI is an end-to-end AI system designed to transition large transportation fleets from reactive repair cycles to an intelligent, predictive, and prescriptive maintenance approach. This system progresses from classical machine learning to a fully agentic AI workflow.

### Live Application
**[Access the Live Streamlit App Here](https://g5oezmn7242npcppaw9vsm.streamlit.app/)**

---

## Key Milestones

### Milestone 1: ML-Based Maintenance Prediction
* **Dataset:** 40,000 historical vehicle telemetry records.
* **Model:** A supervised classification pipeline using **Logistic Regression**. 
* **Performance:** Achieved **86.84% accuracy** and **83.36% recall**. Logistic Regression was chosen over Decision Trees because of its superior recall, which is critical for fleet safety to minimize missed failures.

### Milestone 2: Agentic AI Fleet Management
An agentic architecture built on LangGraph that reasons autonomously over the ML outputs.
* **Node 1 (ML Prediction):** Runs Logistic Regression to compute a risk score, label, and contributing factors.
* **Node 2 (RAG Retrieval):** Queries a FAISS vector index (built from maintenance manuals) using Sentence Transformers to retrieve relevant guidelines.
* **Node 3 (LLM Synthesis):** Uses the Groq API (Llama 3 8B) to generate a structured report.
* **Outputs:** Vehicle Health Summary, Action Plan, and Operational Safety Disclaimer.

---

## Technology Stack
* **Machine Learning:** Logistic Regression (scikit-learn)
* **Agent Framework:** LangGraph
* **RAG / Vector Store:** FAISS + Sentence Transformers (Hugging Face)
* **LLM:** Groq API Llama 3 8B
* **User Interface & Deployment:** Streamlit & Streamlit Community Cloud
* **Secrets Management:** Streamlit Secrets + `.env`

---

## Project Structure
```text
├── app.py               # Streamlit UI, tab layout, sidebar, and session state
├── fleet_system.py      # ML prediction, LangGraph agent, RAG retrieval, and LLM node
├── assets/
│   └── style.css        # Custom CSS styling for the UI
└── README.md            # Setup instructions and documentation
