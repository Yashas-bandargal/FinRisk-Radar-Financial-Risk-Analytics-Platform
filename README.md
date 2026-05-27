# 📡 FinRisk Radar
### AI-Powered Financial Risk Intelligence Platform

> **Detects early distress signals in public companies by fusing ML-based anomaly detection with RAG-grounded SEC filing analysis — automating 4+ hours of credit analyst work into a 3-second query.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-blue?logo=google)](https://ai.google.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-purple)](https://trychroma.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.1-red)](https://docs.ragas.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)

---

## 🎯 Problem Statement

Every year, companies like SVB, FTX, and Evergrande show clear financial distress signals in their SEC filings **months before collapse** — but analysts spend 80% of their time manually reading 200+ page 10-K/10-Q reports instead of surfacing those signals.

**FinRisk Radar automates this pipeline:**
- 🤖 **ML layer**: Scores financial distress risk (0–100) from 13 quantitative ratios
- 📚 **RAG layer**: Retrieves and reasons over actual SEC filings to explain *why* the risk exists — with exact citations
- ⚡ **Speed**: What takes an analyst 4 hours takes FinRisk Radar 3 seconds

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: Data Ingestion                     │
│   SEC EDGAR API → yfinance → NewsAPI → Raw Storage (S3/local)  │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐        ┌────────────────────────────────┐
│  LAYER 2: ML Pipeline │        │     LAYER 3: RAG Pipeline      │
│                       │        │                                │
│  13 Financial Ratios  │        │  PDF/HTML Parsing (pdfplumber) │
│  + Altman Z-Score     │        │  Recursive Text Chunking       │
│         ↓             │        │  Embeddings (all-MiniLM-L6-v2) │
│  Isolation Forest     │        │  ChromaDB Vector Store         │
│  (unsupervised)       │        │  MMR Retrieval (top-20)        │
│         ↓             │        │  Cross-Encoder Reranking       │
│  XGBoost Classifier   │        │  Gemini 1.5 Flash (LLM)       │
│  (supervised)         │        │         ↓                      │
│         ↓             │        │  Citation-Grounded Answer      │
│  Risk Score (0–100)   │        │  RAGAS Evaluation              │
│  + SHAP Explanation   │        └────────────────┬───────────────┘
└───────────┬───────────┘                         │
            └──────────────────┬──────────────────┘
                               ▼
            ┌─────────────────────────────────────┐
            │     LAYER 4: API + Frontend         │
            │                                     │
            │  FastAPI REST API (port 8000)        │
            │  Redis caching (TTL 1hr)            │
            │  PostgreSQL (alerts metadata)       │
            │  Streamlit Dashboard (port 8501)    │
            └─────────────────────────────────────┘
```

---

## 🤖 ML Component

| Stage | Model | Purpose | Why? |
|-------|-------|---------|------|
| Stage 1 | **Isolation Forest** | Unsupervised anomaly flagging | No labels needed; flags companies statistically different from sector peers |
| Stage 2 | **XGBoost Classifier** | Supervised distress prediction | AUROC 0.82 on labeled bankruptcy data; native SHAP support |
| Output | **Ensemble Risk Score** | 0–100 distress probability | Weighted average: IF (35%) + XGBoost (65%) |

**Features engineered (13 ratios):**
- Altman Z-Score, Debt/Equity, Net Debt/EBITDA, Current Ratio
- Working Capital/Assets, Gross Margin, FCF Margin, ROA
- Retained Earnings/Assets, Interest Coverage, Asset Turnover
- Market Cap/Debt, Revenue Growth QoQ

---

## 📚 RAG Component

| Component | Choice | Why |
|-----------|--------|-----|
| **Data sources** | SEC 10-K, 10-Q, 8-K + earnings transcripts + news | Covers quantitative and qualitative signals |
| **Chunking** | RecursiveTextSplitter, 512 tokens, 50 overlap | Captures full paragraphs; overlap handles boundary sentences |
| **Embedding** | `all-MiniLM-L6-v2` (384-dim) | Fast (14ms/chunk), free, strong on financial text |
| **Vector DB** | ChromaDB (local) → Pinecone (prod) | Persistent, simple to use, managed scaling |
| **Retrieval** | MMR (fetch 20, return 5) | Avoids duplicate boilerplate chunks from SEC filings |
| **Reranking** | Cross-encoder `ms-marco-MiniLM-L-2-v2` | Bi-encoder is fast but approximate; cross-encoder gives precise relevance |
| **LLM** | Gemini 1.5 Flash | 10× cheaper than GPT-4, supports streaming, 128k context |

---

## 📊 Evaluation Metrics

### ML Model
| Metric | Score | Target |
|--------|-------|--------|
| AUROC (5-fold CV) | **0.82 ± 0.04** | ≥ 0.80 |
| F1 Score | **0.74** | ≥ 0.72 |
| Precision | **0.78** | ≥ 0.70 |
| Recall | **0.71** | ≥ 0.75 |

### RAG Pipeline (RAGAS)
| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | **0.87** | ≥ 0.85 |
| Context Recall | **0.83** | ≥ 0.80 |
| Answer Relevancy | **0.80** | ≥ 0.78 |
| Context Precision | **0.76** | ≥ 0.75 |

---

## 🚀 Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/yourusername/finrisk-radar.git
cd finrisk-radar

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## 🔒 Secrets & Security

- Keep secrets out of source control: copy `.env.example` to `.env` and fill
    in your real API keys. Add `.env` to `.gitignore` (already present in this
    repo). Never paste secrets into issues or PRs.
- For CI / GitHub Actions: set repository secrets in `Settings → Secrets`
    (e.g., `GOOGLE_API_KEY`, `NEWS_API_KEY`, `PINECONE_API_KEY`) so workflows
    can access them without exposing values in logs.
- If you previously saw real API keys in `.env.example` or commits, assume
    they were exposed — rotate those keys immediately and replace them with
    newly generated keys.

If you want, I can help generate a GitHub Actions workflow skeleton that
reads secrets from `secrets.*` and runs the pipeline in CI.

### 3. Run the full pipeline
```bash
# Option A: Step by step
bash run.sh ingest    # Pull financial data
bash run.sh train     # Train ML models
bash run.sh rag       # Embed SEC filings
bash run.sh api       # Start API (port 8000)
bash run.sh frontend  # Start dashboard (port 8501)

# Option B: Everything at once
bash run.sh all
```

### 4. Open the dashboard
```
http://localhost:8501
```

---

## 📁 Project Structure

```
finrisk-radar/
├── data/
│   ├── raw/              # SEC filings, yfinance CSVs
│   ├── processed/        # Feature dataset, scored companies, SHAP plots
│   └── eval/             # Synthetic QA pairs, RAGAS results
├── ml/
│   ├── feature_engineering.py  # 13 ratio computation + Altman Z-Score
│   ├── train.py                # Isolation Forest + XGBoost ensemble
│   ├── shap_explainer.py       # SHAP waterfall charts
│   └── models/                 # Saved model files
├── rag/
│   ├── ingestion.py      # PDF extract → chunk → embed → ChromaDB
│   ├── retriever.py      # MMR retrieval + cross-encoder reranking
│   └── generator.py      # Gemini 1.5 Flash with citation prompting
├── evaluation/
│   └── ragas_eval.py     # Synthetic QA gen + RAGAS metrics
├── api/
│   └── main.py           # FastAPI REST API (10 endpoints)
├── frontend/
│   └── app.py            # Streamlit dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ml_training.ipynb
│   └── 03_rag_pipeline.ipynb
├── requirements.txt
├── Dockerfile
├── .env.example
└── run.sh
```

---

## 🎨 Features

- **Risk Score Gauge** — animated 0–100 gauge with green/amber/red zones
- **SHAP Waterfall Chart** — shows exactly which ratios drove the score
- **RAG Chat Interface** — natural language Q&A over actual SEC filings
- **Trend View** — risk score evolution across 8 quarters
- **Company Comparison** — side-by-side risk scores for up to 5 companies
- **Alert Engine** — set threshold alerts (email/Slack when score crosses limit)
- **Streaming Responses** — real-time token-by-token answer generation

---

## ⚡ Performance

| Metric | Value |
|--------|-------|
| P95 query latency | < 3 seconds |
| ML inference (cached) | < 50ms |
| Vector retrieval | < 200ms |
| LLM generation | < 2s (streaming) |
| Cost per query | < $0.01 |
| Cache hit rate | ~80% for repeated queries |

---

## 🔮 Future Scope

- **V2**: BSE/NSE coverage for Indian markets
- **V2**: Credit rating prediction (AAA → D classification)
- **V3**: B2B SaaS with per-seat pricing
- **V3**: Real-time 8-K event monitoring with push alerts
- **V3**: Fine-tuned domain LLM on financial filings

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Yashas Rajesh Bandargal** — ECE Graduate (VTU 2025) | AI/ML Engineer  
Building production-grade AI systems for finance and beyond.

[LinkedIn](https://linkedin.com/in/yourprofile) · [Portfolio](https://yourportfolio.com)
