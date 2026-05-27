# 📡 FinRisk Radar
### AI-Powered Financial Risk Intelligence Platform

> **Detects early distress signals in public companies by combining financial risk modeling with retrieval-augmented SEC filing analysis.**
> The project turns filings into actionable risk scores and citation-grounded insights.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-purple)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-blue?logo=google)](https://ai.google.dev)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.1-red)](https://docs.ragas.io)

---

## What this project does

FinRisk Radar is a finance-focused risk analytics platform that combines quantitative scoring and retrieval-augmented reasoning over SEC filings.

It is designed to help analysts:

- identify early distress signals in public companies
- understand why a model flagged risk via SHAP explanations
- query filings with natural language and get evidence-backed answers
- explore risk trends in a dashboard UI

---

## Project structure

```
finrisk-radar/
├── api/                # FastAPI backend endpoints
├── data/               # Raw SEC filings, processed features, evaluation output
├── evaluation/         # RAGAS evaluation and validation scripts
├── frontend/           # Streamlit dashboard UI
├── ml/                 # Feature engineering, training, explainability
├── rag/                # RAG ingestion, retrieval, generation
├── notebooks/          # EDA and training notebooks
├── requirements.txt    # Backend dependencies
├── requirements_ui.txt # Frontend dependencies
├── .env.example        # Example environment variables
└── run.sh              # Pipeline helper script
```

---

## Tech stack

- Python 3.11
- FastAPI
- ChromaDB
- HuggingFace sentence-transformers
- Google Gemini / Gemini 1.5 Flash
- XGBoost
- RAGAS evaluation
- GitHub Actions

---

## Setup

```bash
git clone https://github.com/Yashas-bandargal/FinRisk-Radar-Financial-Risk-Analytics-Platform.git
cd finrisk-radar
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend setup

```bash
pip install -r requirements_ui.txt
```

### Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and provide your own API keys and production settings.

---

## Secrets & Security

- Store real API keys only in `.env` and never commit it.
- `.env` is ignored by `.gitignore`.
- Use GitHub repository secrets for CI: `GOOGLE_API_KEY`, `NEWS_API_KEY`, `PINECONE_API_KEY`.
- If any API keys were exposed before, rotate them immediately.

---

## How it works

1. **Ingestion**
   - `run.sh ingest` downloads SEC filings, financial data, and optional news.
   - Raw documents are saved under `data/raw/`.

2. **Feature engineering**
   - `ml/feature_engineering.py` computes 13 financial ratios and Altman Z-score.
   - Processed data is written to `data/processed/`.

3. **Training**
   - `ml/train.py` trains an Isolation Forest and XGBoost classifier.
   - Trained models are persisted under `ml/models/`.

4. **RAG pipeline**
   - `rag/ingestion.py` chunks and embeds SEC filings into ChromaDB.
   - `rag/retriever.py` retrieves context and reranks relevant chunks.
   - `rag/generator.py` uses Gemini to generate answers with citations.

5. **API + UI**
   - `api/main.py` exposes score, retrieval, and chat endpoints.
   - `frontend/app.py` runs the dashboard for exploration.

6. **Evaluation**
   - `evaluation/ragas_eval.py` measures RAG retrieval and answer quality.

---

## Run the project

```bash
bash run.sh ingest
bash run.sh train
bash run.sh rag
bash run.sh api
bash run.sh frontend
```

Or run everything:

```bash
bash run.sh all
```

Then open:

```bash
http://localhost:8501
```

---

## Results & metrics

| Metric | Target |
|---|---|
| Risk scoring latency | < 3s |
| ML inference | < 100ms |
| Retrieval latency | < 300ms |
| RAG faithfulness | ≥ 0.85 |

---

## Future improvements

- Add end-to-end and regression tests
- Add Pinecone support for production embeddings
- Add alert notifications via email/Slack
- Add richer out-of-scope query handling
- Expand coverage to more filings and companies

---

## CI

A GitHub Actions workflow is available at `.github/workflows/ci.yml`.
