"""
FinRisk Radar — FastAPI Backend
REST API exposing ML scoring, RAG Q&A, and company search endpoints.
"""

import os
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from functools import lru_cache

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FinRisk Radar API",
    description="AI-powered financial risk intelligence platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache (replace with Redis in production)
_cache: dict = {}
CACHE_TTL_SECONDS = 3600  # 1 hour

PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    ticker: str
    question: str
    use_reranker: bool = True
    date_filter: Optional[str] = None  # e.g. "2023"
    stream: bool = False

class AlertRequest(BaseModel):
    ticker: str
    threshold: float        # Risk score threshold 0–100
    email: Optional[str]
    direction: str = "above"  # "above" or "below"

class IngestRequest(BaseModel):
    ticker: str
    form_types: List[str] = ["10-K", "10-Q"]

class CompareRequest(BaseModel):
    tickers: List[str]    # Up to 5 tickers to compare


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def cache_key(prefix: str, **kwargs) -> str:
    raw = json.dumps({"prefix": prefix, **kwargs}, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

def get_cached(key: str):
    entry = _cache.get(key)
    if entry:
        if (datetime.now() - entry["ts"]).seconds < CACHE_TTL_SECONDS:
            return entry["data"]
    return None

def set_cached(key: str, data):
    _cache[key] = {"data": data, "ts": datetime.now()}

@lru_cache(maxsize=1)
def load_scored_df():
    path = PROC_DIR / "scored_companies.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

def get_company_risk(ticker: str) -> Optional[dict]:
    df = load_scored_df()
    if df.empty:
        return None
    rows = df[df["ticker"].str.upper() == ticker.upper()]
    if rows.empty:
        return None
    latest = rows.sort_values("date").iloc[-1]
    return latest.to_dict()


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "FinRisk Radar API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/risk/{ticker}", "/query", "/compare", "/ingest", "/alerts", "/health"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": (MODEL_DIR / "xgboost_pipeline.pkl").exists(),
        "scored_companies": len(load_scored_df()),
    }


@app.get("/risk/{ticker}")
def get_risk_score(ticker: str):
    """
    Get ML risk score + SHAP explanation for a company.
    Returns cached result if available (avoids recomputing model inference).
    """
    ticker = ticker.upper()
    ck = cache_key("risk", ticker=ticker)
    cached = get_cached(ck)
    if cached:
        log.info(f"Cache hit: risk/{ticker}")
        return cached

    row = get_company_risk(ticker)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data for {ticker}. Run data ingestion first."
        )

    # Load SHAP explanation if available
    shap_path = Path(__file__).parent.parent / "data" / "processed" / "shap_plots" / f"{ticker}_shap.json"
    shap_data = None
    if shap_path.exists():
        with open(shap_path) as f:
            shap_data = json.load(f)

    result = {
        "ticker":       ticker,
        "risk_score":   float(row.get("risk_score", 0)),
        "risk_label":   str(row.get("risk_label", "Unknown")),
        "altman_z":     float(row.get("altman_z", 0)) if not pd.isna(row.get("altman_z", 0)) else None,
        "altman_zone":  str(row.get("altman_zone", "Unknown")),
        "as_of_date":   str(row.get("date", "")),
        "sector":       str(row.get("sector", "")),
        "key_ratios": {
            "debt_to_equity":    _safe_float(row.get("debt_to_equity")),
            "current_ratio":     _safe_float(row.get("current_ratio")),
            "interest_coverage": _safe_float(row.get("interest_coverage")),
            "fcf_margin":        _safe_float(row.get("fcf_margin")),
            "gross_margin":      _safe_float(row.get("gross_margin")),
        },
        "shap_explanation": shap_data,
    }

    set_cached(ck, result)
    return result


def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return round(v, 4)
    except Exception:
        return None


@app.get("/risk/{ticker}/trend")
def get_risk_trend(ticker: str, quarters: int = 8):
    """Get risk score trend across the last N quarters."""
    ticker = ticker.upper()
    df = load_scored_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="No scored data available")

    rows = df[df["ticker"].str.upper() == ticker.upper()].sort_values("date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    rows = rows.tail(quarters)
    trend = []
    for _, r in rows.iterrows():
        trend.append({
            "date":        str(r["date"]),
            "risk_score":  _safe_float(r.get("risk_score")),
            "altman_z":    _safe_float(r.get("altman_z")),
            "debt_to_equity": _safe_float(r.get("debt_to_equity")),
        })
    return {"ticker": ticker, "trend": trend}


@app.post("/query")
def rag_query(req: QueryRequest):
    """
    RAG Q&A endpoint: retrieves relevant filing chunks and generates a cited answer.
    Cached by (ticker, question) hash.
    """
    from rag.retriever import retrieve, format_context
    from rag.generator import generate, generate_structured

    ticker = req.ticker.upper()
    ck = cache_key("query", ticker=ticker, q=req.question, rerank=req.use_reranker)
    cached = get_cached(ck)
    if cached:
        log.info(f"Cache hit: query/{ticker}")
        return cached

    # Get risk score for context
    row = get_company_risk(ticker)
    risk_score = float(row.get("risk_score", 0)) if row else None

    # Retrieve
    chunks = retrieve(
        query=req.question,
        ticker=ticker,
        k=5,
        use_reranker=req.use_reranker,
        date_filter=req.date_filter,
    )

    if not chunks:
        return {
            "ticker":   ticker,
            "question": req.question,
            "answer":   "No relevant filing information found. The company may not be ingested yet.",
            "chunks":   [],
            "citations": [],
        }

    context = format_context(chunks)
    result  = generate(req.question, context, ticker, risk_score=risk_score)

    response = {
        "ticker":          ticker,
        "question":        req.question,
        "answer":          result["answer"],
        "citations_found": result.get("citations_found", []),
        "sources": [
            {
                "source":   c["source"],
                "date":     c["date"],
                "form":     c["form"],
                "excerpt":  c["text"][:200] + "...",
                "rerank_score": c.get("rerank_score"),
            }
            for c in chunks
        ],
        "model": result.get("model"),
    }

    set_cached(ck, response)
    return response


@app.post("/query/stream")
def rag_query_stream(req: QueryRequest):
    """Streaming version — yields tokens as they arrive from Gemini."""
    from rag.retriever import retrieve, format_context
    from rag.generator import generate_streaming

    ticker = req.ticker.upper()
    row    = get_company_risk(ticker)
    risk_score = float(row.get("risk_score", 0)) if row else None

    chunks  = retrieve(req.question, ticker, k=5, use_reranker=req.use_reranker)
    context = format_context(chunks)

    def token_stream():
        for token in generate_streaming(req.question, context, ticker, risk_score):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/compare")
def compare_companies(req: CompareRequest):
    """Compare risk scores across up to 5 companies."""
    if len(req.tickers) > 5:
        raise HTTPException(status_code=400, detail="Max 5 tickers per comparison")

    results = []
    for ticker in req.tickers:
        row = get_company_risk(ticker.upper())
        if row:
            results.append({
                "ticker":      ticker.upper(),
                "risk_score":  _safe_float(row.get("risk_score")),
                "risk_label":  str(row.get("risk_label", "")),
                "altman_z":    _safe_float(row.get("altman_z")),
                "sector":      str(row.get("sector", "")),
            })
        else:
            results.append({"ticker": ticker.upper(), "error": "Not found"})

    results.sort(key=lambda x: x.get("risk_score") or 0, reverse=True)
    return {"comparison": results}


@app.get("/search")
def search_companies(q: str = Query(..., min_length=1)):
    """Search for companies by ticker or name prefix."""
    df = load_scored_df()
    if df.empty:
        return {"results": []}

    q_upper = q.upper()
    matches = df[df["ticker"].str.upper().str.startswith(q_upper)].drop_duplicates("ticker")
    results = []
    for _, row in matches.head(10).iterrows():
        results.append({
            "ticker":     str(row["ticker"]),
            "risk_label": str(row.get("risk_label", "")),
            "sector":     str(row.get("sector", "")),
        })
    return {"results": results, "query": q}


@app.post("/ingest")
def trigger_ingestion(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger background ingestion of SEC filings for a ticker.
    Returns immediately; ingestion runs in background.
    """
    def run_ingestion(ticker: str, forms: list):
        from data.ingestion import ingest_sec_filings
        from rag.ingestion import ingest_ticker_all_filings
        log.info(f"Background ingestion starting: {ticker}")
        ingest_sec_filings(ticker, form_types=forms)
        ingest_ticker_all_filings(ticker)
        log.info(f"Background ingestion complete: {ticker}")

    background_tasks.add_task(run_ingestion, req.ticker.upper(), req.form_types)
    return {
        "status":  "ingestion_started",
        "ticker":  req.ticker.upper(),
        "message": "Ingestion running in background. Check /risk/{ticker} in a few minutes."
    }


_alerts: list = []  # In-memory store; use DB in production

@app.post("/alerts")
def create_alert(req: AlertRequest):
    """Create a risk score threshold alert for a company."""
    alert = {
        "id":        len(_alerts) + 1,
        "ticker":    req.ticker.upper(),
        "threshold": req.threshold,
        "direction": req.direction,
        "email":     req.email,
        "created":   datetime.now().isoformat(),
        "status":    "active",
    }
    _alerts.append(alert)
    return {"success": True, "alert": alert}

@app.get("/alerts")
def list_alerts():
    return {"alerts": _alerts}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
