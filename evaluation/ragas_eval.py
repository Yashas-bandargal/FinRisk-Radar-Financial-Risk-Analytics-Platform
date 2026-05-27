"""
FinRisk Radar — RAG Evaluation
Synthetic QA generation via Gemini + RAGAS metrics evaluation.
Tracks: Faithfulness, Context Recall, Answer Relevancy, Context Precision.
"""

import os
import json
import logging
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ─── SYNTHETIC QA GENERATION ─────────────────────────────────────────────────

SYNTH_QA_PROMPT = """You are a financial analyst creating evaluation questions for an AI system.

Given the following excerpt from a SEC financial filing, generate 3 question-answer pairs that:
1. Are directly answerable from the text (not requiring external knowledge)
2. Focus on financial risk, liquidity, debt, or business conditions
3. Have specific, factual answers grounded in the text

Return ONLY a valid JSON array, no other text:
[
  {{
    "question": "...",
    "ground_truth": "...",
    "context": "the relevant sentence or phrase from the text that answers this"
  }}
]

Filing excerpt:
{chunk_text}"""


def generate_qa_pairs(chunk_text: str, model) -> list[dict]:
    """Generate synthetic QA pairs from a filing chunk."""
    prompt = SYNTH_QA_PROMPT.format(chunk_text=chunk_text[:1500])
    try:
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=600,
                temperature=0.3,
            ),
        )
        raw = resp.text.strip()
        # Strip markdown
        raw = raw.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(raw)
        return pairs if isinstance(pairs, list) else []
    except Exception as e:
        log.warning(f"QA generation failed: {e}")
        return []


def build_eval_dataset(ticker: str, n_chunks: int = 30,
                       output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build a synthetic evaluation dataset for a ticker.
    
    Steps:
    1. Retrieve diverse chunks from ChromaDB
    2. For each chunk, generate 3 QA pairs using Gemini
    3. Save as eval dataset CSV

    This gives us ~90 QA pairs per ticker without manual labeling.
    """
    from rag.retriever import get_vectorstore

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Get diverse chunks
    vs = get_vectorstore(ticker)
    results = vs.get(limit=n_chunks, include=["documents", "metadatas"])
    chunks = results.get("documents", [])
    metas  = results.get("metadatas", [])

    if not chunks:
        log.error(f"No chunks found for {ticker}. Run rag/ingestion.py first.")
        return pd.DataFrame()

    records = []
    for i, (chunk, meta) in enumerate(zip(chunks, metas)):
        log.info(f"Generating QA for chunk {i+1}/{len(chunks)}")
        pairs = generate_qa_pairs(chunk, model)
        for pair in pairs:
            records.append({
                "ticker":       ticker,
                "question":     pair.get("question", ""),
                "ground_truth": pair.get("ground_truth", ""),
                "context":      pair.get("context", chunk[:300]),
                "chunk_source": meta.get("source", ""),
                "filing_date":  meta.get("filing_date", ""),
                "form_type":    meta.get("form_type", ""),
            })

    df = pd.DataFrame(records)
    df = df[df["question"].str.len() > 20]  # Filter bad rows

    if output_path is None:
        output_path = str(EVAL_DIR / f"{ticker}_eval_dataset.csv")
    df.to_csv(output_path, index=False)
    log.info(f"Eval dataset: {len(df)} QA pairs → {output_path}")
    return df


# ─── RAGAS EVALUATION ─────────────────────────────────────────────────────────

def run_ragas_eval(ticker: str, eval_csv: Optional[str] = None,
                   sample_size: int = 20) -> dict:
    """
    Run RAGAS evaluation on the synthetic QA dataset.

    Metrics computed:
    - faithfulness:       Does the answer contain only claims from the context?
    - answer_relevancy:  Does the answer address the question?
    - context_recall:    Does the context contain info needed to answer?
    - context_precision: How much of the context is actually relevant?

    Returns dict of metric scores.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        log.error("Install: pip install ragas datasets")
        return {}

    from rag.retriever import retrieve, format_context
    from rag.generator import generate

    if eval_csv is None:
        eval_csv = str(EVAL_DIR / f"{ticker}_eval_dataset.csv")

    if not Path(eval_csv).exists():
        log.info("Eval dataset not found — generating...")
        df = build_eval_dataset(ticker)
    else:
        df = pd.read_csv(eval_csv)

    # Sample for speed
    df_sample = df.sample(min(sample_size, len(df)), random_state=42)
    log.info(f"Running RAGAS on {len(df_sample)} QA pairs for {ticker}")

    questions, answers, contexts, ground_truths = [], [], [], []

    for _, row in df_sample.iterrows():
        q  = row["question"]
        gt = row["ground_truth"]

        # Retrieve context for this question
        chunks = retrieve(q, ticker, k=5, use_reranker=False)
        ctx    = [c["text"] for c in chunks]
        formatted = format_context(chunks)

        # Generate answer
        result = generate(q, formatted, ticker)
        ans    = result["answer"]

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx)
        ground_truths.append(gt)

    # Build HuggingFace Dataset
    dataset = Dataset.from_dict({
        "question":      questions,
        "answer":        answers,
        "contexts":      contexts,
        "ground_truth":  ground_truths,
    })

    log.info("Running RAGAS metrics...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = {
        "ticker":            ticker,
        "n_samples":         len(df_sample),
        "faithfulness":      round(float(result["faithfulness"]), 4),
        "answer_relevancy":  round(float(result["answer_relevancy"]), 4),
        "context_recall":    round(float(result["context_recall"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "timestamp":         datetime.now().isoformat(),
    }

    # Save results
    results_path = EVAL_DIR / f"{ticker}_ragas_results.json"
    with open(results_path, "w") as f:
        json.dump(scores, f, indent=2)

    log.info(f"RAGAS results saved → {results_path}")
    return scores


# ─── ML EVALUATION ────────────────────────────────────────────────────────────

def run_ml_eval(feature_csv: Optional[str] = None) -> dict:
    """
    Evaluate the ML risk model on the feature dataset.
    Reports AUROC, F1, Precision, Recall with 5-fold CV.
    """
    import joblib
    import numpy as np
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
    MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"

    FEATURE_COLS = [
        "debt_to_equity", "net_debt_ebitda", "current_ratio", "working_capital_ratio",
        "gross_margin", "fcf_margin", "roa", "retained_earnings_ratio",
        "interest_coverage", "asset_turnover", "market_cap_to_debt",
        "revenue_growth_qoq", "altman_z"
    ]

    if feature_csv is None:
        feature_csv = str(PROC_DIR / "features.csv")

    df = pd.read_csv(feature_csv)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0).clip(-10, 10).values
    y = df["distressed"].values

    pipeline = joblib.load(MODEL_DIR / "xgboost_pipeline.pkl")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (proba >= 0.5).astype(int)

    scores = {
        "auroc":     round(roc_auc_score(y, proba), 4),
        "f1":        round(f1_score(y, preds), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall":    round(recall_score(y, preds, zero_division=0), 4),
        "n_samples": int(len(y)),
        "n_distressed": int(y.sum()),
    }

    results_path = EVAL_DIR / "ml_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(scores, f, indent=2)

    return scores


# ─── COMBINED EVAL REPORT ─────────────────────────────────────────────────────

def full_evaluation_report(ticker: str):
    print("=" * 55)
    print(f"  FinRisk Radar — Evaluation Report: {ticker}")
    print("=" * 55)

    print("\n── ML Model Evaluation ──────────────────────────")
    try:
        ml_scores = run_ml_eval()
        for k, v in ml_scores.items():
            print(f"  {k:<20} {v}")
    except Exception as e:
        print(f"  ML eval failed: {e}")

    print("\n── RAG Pipeline Evaluation (RAGAS) ──────────────")
    try:
        rag_scores = run_ragas_eval(ticker, sample_size=10)
        for k, v in rag_scores.items():
            if isinstance(v, float):
                bar = "█" * int(v * 20)
                print(f"  {k:<25} {v:.4f}  {bar}")
            else:
                print(f"  {k:<25} {v}")
    except Exception as e:
        print(f"  RAG eval failed: {e}")

    print("\n── Targets ──────────────────────────────────────")
    print("  Faithfulness target:        ≥ 0.85")
    print("  Context Recall target:      ≥ 0.80")
    print("  Answer Relevancy target:    ≥ 0.78")
    print("  Context Precision target:   ≥ 0.75")
    print("  AUROC target:               ≥ 0.80")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    full_evaluation_report(ticker)
