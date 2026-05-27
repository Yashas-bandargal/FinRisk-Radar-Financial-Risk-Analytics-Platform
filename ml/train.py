"""
FinRisk Radar — ML Training
Two-stage ensemble: Isolation Forest (unsupervised) + XGBoost (supervised).
Outputs a Risk Score 0–100 per company-quarter.
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "debt_to_equity", "net_debt_ebitda", "current_ratio", "working_capital_ratio",
    "gross_margin", "fcf_margin", "roa", "retained_earnings_ratio",
    "interest_coverage", "asset_turnover", "market_cap_to_debt",
    "revenue_growth_qoq", "altman_z"
]


# ─── DATA LOADING ──────────────────────────────────────────────────────────────

def load_features(path: str = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if path is None:
        path = str(PROC_DIR / "features.csv")
    df = pd.read_csv(path)

    available = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"Using {len(available)} features: {available}")

    X = df[available].copy()
    y = df["distressed"].values if "distressed" in df.columns else None

    # Replace inf and clip extreme values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X = X.clip(-10, 10)  # Cap z-scores / ratios

    return df, X.values, y


# ─── STAGE 1: ISOLATION FOREST ─────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray, contamination: float = 0.15) -> IsolationForest:
    """
    Unsupervised anomaly detection.
    contamination = expected fraction of anomalies (distressed companies).
    15% is a reasonable estimate for a mixed company dataset.
    """
    log.info("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X)

    # Raw scores are negative; flip so higher = more anomalous
    raw_scores = iso.score_samples(X)
    anomaly_scores = -raw_scores  # Now higher = more likely anomaly

    log.info(f"IF score range: {anomaly_scores.min():.3f} – {anomaly_scores.max():.3f}")
    return iso, anomaly_scores


# ─── STAGE 2: XGBOOST CLASSIFIER ──────────────────────────────────────────────

def train_xgboost(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Supervised distress classifier with cross-validation.
    Returns trained pipeline and evaluation metrics.
    """
    log.info("Training XGBoost classifier...")

    # Class imbalance: weight distressed class more heavily
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info(f"Class ratio — Safe: {n_neg}, Distressed: {n_pos}, scale_pos_weight: {scale_pos_weight:.2f}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auroc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_scores    = cross_val_score(pipeline, X, y, cv=cv, scoring="f1", n_jobs=-1)

    log.info(f"CV AUROC: {auroc_scores.mean():.3f} ± {auroc_scores.std():.3f}")
    log.info(f"CV F1:    {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")

    # Final fit on full data
    pipeline.fit(X, y)
    xgb_proba = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "auroc_cv_mean": round(auroc_scores.mean(), 4),
        "auroc_cv_std":  round(auroc_scores.std(), 4),
        "f1_cv_mean":    round(f1_scores.mean(), 4),
        "n_samples":     len(y),
        "n_distressed":  int(n_pos),
    }

    # Full-data classification report (for reference)
    preds = (xgb_proba >= 0.5).astype(int)
    log.info("\n" + classification_report(y, preds, target_names=["Safe", "Distressed"]))

    return pipeline, xgb_proba, metrics


# ─── ENSEMBLE RISK SCORE ───────────────────────────────────────────────────────

def compute_risk_score(if_scores: np.ndarray, xgb_proba: np.ndarray,
                       if_weight: float = 0.35, xgb_weight: float = 0.65) -> np.ndarray:
    """
    Combine Isolation Forest anomaly scores and XGBoost probabilities.
    
    IF gets lower weight because it has no labeled supervision.
    XGBoost gets higher weight because it's trained on known distress events.
    
    Output: Risk Score 0–100 (100 = maximum distress risk)
    """
    # Normalize IF scores to [0, 1]
    if_min, if_max = if_scores.min(), if_scores.max()
    if if_max > if_min:
        if_norm = (if_scores - if_min) / (if_max - if_min)
    else:
        if_norm = np.zeros_like(if_scores)

    # XGBoost probabilities are already [0, 1]
    ensemble = if_weight * if_norm + xgb_weight * xgb_proba

    # Scale to 0–100
    risk_score = np.round(ensemble * 100, 1)
    return risk_score


def risk_label(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


# ─── SAVE / LOAD ───────────────────────────────────────────────────────────────

def save_models(iso_model, xgb_pipeline, metrics: dict):
    joblib.dump(iso_model,    MODEL_DIR / "isolation_forest.pkl")
    joblib.dump(xgb_pipeline, MODEL_DIR / "xgboost_pipeline.pkl")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Models saved to {MODEL_DIR}")


def load_models():
    iso = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    xgb = joblib.load(MODEL_DIR / "xgboost_pipeline.pkl")
    return iso, xgb


# ─── SCORE NEW COMPANY ─────────────────────────────────────────────────────────

def score_company(ticker: str, feature_row: dict) -> dict:
    """
    Score a single company given a dictionary of feature values.
    Returns risk_score, risk_label, and component scores.
    """
    iso, xgb = load_models()
    feat_cols = [c for c in FEATURE_COLS if c in feature_row]

    X = np.array([[feature_row.get(c, 0) for c in FEATURE_COLS]])
    X = np.clip(np.nan_to_num(X, nan=0.0), -10, 10)

    if_raw = -iso.score_samples(X)
    if_min, if_max = 0.3, 0.9  # approximate from training; ideally persist these
    if_norm = np.clip((if_raw - if_min) / (if_max - if_min), 0, 1)

    xgb_prob = xgb.predict_proba(X)[:, 1]
    risk = compute_risk_score(if_norm, xgb_prob)[0]

    return {
        "ticker": ticker,
        "risk_score": float(risk),
        "risk_label": risk_label(risk),
        "isolation_forest_score": round(float(if_norm[0]) * 100, 1),
        "xgboost_probability": round(float(xgb_prob[0]) * 100, 1),
    }


# ─── FULL TRAINING PIPELINE ────────────────────────────────────────────────────

def train(feature_csv: str = None):
    print("=" * 55)
    print("  FinRisk Radar — ML Training Pipeline")
    print("=" * 55)

    df, X, y = load_features(feature_csv)

    if y is None:
        log.error("No distress labels found. Run feature_engineering.py first.")
        return

    # Stage 1: Isolation Forest
    iso_model, if_scores = train_isolation_forest(X)

    # Stage 2: XGBoost
    xgb_pipeline, xgb_proba, metrics = train_xgboost(X, y)

    # Ensemble risk scores
    risk_scores = compute_risk_score(if_scores, xgb_proba)
    df["risk_score"] = risk_scores
    df["risk_label"] = [risk_label(s) for s in risk_scores]
    df["if_score"]   = np.round(-iso_model.score_samples(X) * 100, 1)
    df["xgb_proba"]  = np.round(xgb_proba * 100, 1)

    # Save scored dataset
    scored_path = PROC_DIR / "scored_companies.csv"
    df.to_csv(scored_path, index=False)
    log.info(f"Scored dataset saved → {scored_path}")

    # Save models
    save_models(iso_model, xgb_pipeline, metrics)

    print("\n── Results ──────────────────────────────────────")
    print(f"  AUROC (CV):     {metrics['auroc_cv_mean']:.3f} ± {metrics['auroc_cv_std']:.3f}")
    print(f"  F1   (CV):      {metrics['f1_cv_mean']:.3f}")
    print(f"  Samples:        {metrics['n_samples']}")
    print(f"  Distressed:     {metrics['n_distressed']} ({metrics['n_distressed']/metrics['n_samples']:.1%})")
    print(f"\n── Risk Score Distribution ──────────────────────")
    print(df["risk_label"].value_counts().to_string())
    print(f"\n── Top 10 Highest Risk Companies ────────────────")
    top10 = df.nlargest(10, "risk_score")[["ticker", "date", "risk_score", "risk_label", "altman_z"]]
    print(top10.to_string(index=False))


if __name__ == "__main__":
    train()
