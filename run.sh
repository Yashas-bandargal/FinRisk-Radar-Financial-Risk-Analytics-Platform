#!/bin/bash
# FinRisk Radar — Full Pipeline Runner
# Usage: bash run.sh [step]
# Steps: setup | ingest | train | rag | eval | api | frontend | all

set -e

TICKER=${TICKER:-AAPL}
echo "============================================="
echo "  FinRisk Radar Pipeline"
echo "  Ticker: $TICKER"
echo "============================================="

setup() {
    echo "[1/6] Installing dependencies..."
    pip install -r requirements.txt -q
    echo "       Done."
}

ingest() {
    echo "[2/6] Data ingestion (yfinance + SEC EDGAR)..."
    python -m data.ingestion
    echo "       Financials saved to data/raw/"
}

train() {
    echo "[3/6] Feature engineering..."
    python -m ml.feature_engineering
    echo "       Features saved to data/processed/features.csv"

    echo "[4/6] ML training (Isolation Forest + XGBoost)..."
    python -m ml.train
    echo "       Models saved to ml/models/"
}

rag_ingest() {
    echo "[5/6] RAG ingestion (embed SEC filings into ChromaDB)..."
    python -m rag.ingestion $TICKER
    echo "       ChromaDB updated at data/chroma_db/"
}

evaluate() {
    echo "       Running evaluation..."
    python -m evaluation.ragas_eval $TICKER
    echo "       Results saved to data/eval/"
}

run_api() {
    echo "Starting FastAPI backend on http://localhost:8000"
    echo "Swagger docs: http://localhost:8000/docs"
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
}

run_frontend() {
    echo "Starting Streamlit dashboard on http://localhost:8501"
    streamlit run frontend/app.py
}

case "$1" in
    setup)    setup ;;
    ingest)   ingest ;;
    train)    train ;;
    rag)      rag_ingest ;;
    eval)     evaluate ;;
    api)      run_api ;;
    frontend) run_frontend ;;
    all)
        setup
        ingest
        train
        rag_ingest
        evaluate
        echo ""
        echo "============================================="
        echo "  All steps complete!"
        echo "  Run the API:      bash run.sh api"
        echo "  Run the frontend: bash run.sh frontend"
        echo "============================================="
        ;;
    *)
        echo "Usage: bash run.sh [setup|ingest|train|rag|eval|api|frontend|all]"
        echo ""
        echo "  setup    - Install dependencies"
        echo "  ingest   - Pull financial data from yfinance + SEC EDGAR"
        echo "  train    - Train ML models (Isolation Forest + XGBoost)"
        echo "  rag      - Embed SEC filings into ChromaDB"
        echo "  eval     - Run RAGAS + ML evaluation"
        echo "  api      - Start FastAPI backend (port 8000)"
        echo "  frontend - Start Streamlit dashboard (port 8501)"
        echo "  all      - Run steps 1–5 in sequence"
        ;;
esac
