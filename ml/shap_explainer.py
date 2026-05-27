"""
FinRisk Radar — SHAP Explainability
Generates feature-level explanations for every risk score prediction.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
PLOTS_DIR = Path(__file__).parent.parent / "data" / "processed" / "shap_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "debt_to_equity", "net_debt_ebitda", "current_ratio", "working_capital_ratio",
    "gross_margin", "fcf_margin", "roa", "retained_earnings_ratio",
    "interest_coverage", "asset_turnover", "market_cap_to_debt",
    "revenue_growth_qoq", "altman_z"
]

FEATURE_LABELS = {
    "debt_to_equity":        "Debt / Equity",
    "net_debt_ebitda":       "Net Debt / EBITDA",
    "current_ratio":         "Current Ratio",
    "working_capital_ratio": "Working Capital / Assets",
    "gross_margin":          "Gross Margin",
    "fcf_margin":            "FCF Margin",
    "roa":                   "Return on Assets",
    "retained_earnings_ratio": "Retained Earnings / Assets",
    "interest_coverage":     "Interest Coverage",
    "asset_turnover":        "Asset Turnover",
    "market_cap_to_debt":    "Market Cap / Debt",
    "revenue_growth_qoq":    "Revenue Growth QoQ",
    "altman_z":              "Altman Z-Score",
}


# ─── BUILD SHAP EXPLAINER ─────────────────────────────────────────────────────

def build_explainer(xgb_pipeline, X_background: np.ndarray):
    """
    Build a SHAP TreeExplainer for the XGBoost model inside the pipeline.
    X_background is used to compute expected values.
    """
    xgb_model = xgb_pipeline.named_steps["xgb"]
    scaler    = xgb_pipeline.named_steps["scaler"]
    X_scaled  = scaler.transform(X_background)

    explainer = shap.TreeExplainer(xgb_model, data=X_scaled, model_output="probability")
    log.info(f"SHAP explainer built. Base value: {explainer.expected_value:.4f}")
    return explainer, scaler


def get_shap_values(explainer, scaler, X_row: np.ndarray) -> np.ndarray:
    """Compute SHAP values for a single company (1 row)."""
    X_scaled = scaler.transform(X_row.reshape(1, -1))
    shap_vals = explainer.shap_values(X_scaled)
    return shap_vals[0]  # shape: (n_features,)


# ─── WATERFALL CHART ──────────────────────────────────────────────────────────

def plot_waterfall(shap_values: np.ndarray, feature_names: list,
                   feature_values: np.ndarray, base_value: float,
                   ticker: str, save: bool = True) -> plt.Figure:
    """
    Plot a SHAP waterfall chart for one company prediction.
    Shows how each ratio pushes the risk score up or down.
    """
    n = len(shap_values)
    # Sort by absolute SHAP value
    order = np.argsort(np.abs(shap_values))[::-1][:10]  # top 10 features

    vals  = shap_values[order]
    names = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    fvals = feature_values[order]

    # Build cumulative bar positions
    cumulative = base_value
    starts, widths, colors = [], [], []
    for v in vals:
        starts.append(cumulative)
        widths.append(v)
        colors.append("#e74c3c" if v > 0 else "#27ae60")  # red = increases risk
        cumulative += v

    final_score = min(max(cumulative * 100, 0), 100)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    y_pos = np.arange(len(vals))
    bars = ax.barh(y_pos, widths, left=starts, color=colors, height=0.55,
                   edgecolor="none", alpha=0.9)

    # Feature labels on left, feature values on right
    for i, (name, fval, s, w) in enumerate(zip(names, fvals, starts, widths)):
        ax.text(-0.005, i, f"{name}", va="center", ha="right",
                fontsize=9, color="#cccccc", fontfamily="monospace")
        val_str = f"= {fval:.2f}"
        ax.text(s + w + (0.002 if w >= 0 else -0.002), i, val_str,
                va="center", ha="left" if w >= 0 else "right",
                fontsize=8, color="#aaaaaa")

    # Base value line
    ax.axvline(base_value, color="#555555", linestyle="--", linewidth=1, label="Base value")

    # Formatting
    ax.set_yticks([])
    ax.set_xlabel("← Reduces Risk    SHAP Value    Increases Risk →", color="#888888", fontsize=9)
    ax.set_title(f"{ticker} — Risk Score: {final_score:.0f}/100\nSHAP Feature Contributions",
                 color="#ffffff", fontsize=13, pad=14, fontweight="bold")
    ax.tick_params(colors="#888888")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#333333")
    ax.xaxis.label.set_color("#888888")

    red_patch  = mpatches.Patch(color="#e74c3c", label="↑ Increases distress risk")
    green_patch = mpatches.Patch(color="#27ae60", label="↓ Reduces distress risk")
    ax.legend(handles=[red_patch, green_patch], loc="lower right",
              facecolor="#1a1a2e", edgecolor="#333333", labelcolor="#cccccc", fontsize=8)

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / f"{ticker}_shap_waterfall.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info(f"Saved SHAP plot → {path}")

    return fig


# ─── GLOBAL FEATURE IMPORTANCE ────────────────────────────────────────────────

def plot_global_importance(shap_values_all: np.ndarray, feature_names: list,
                           save: bool = True) -> plt.Figure:
    """Bar chart of mean |SHAP| across all companies."""
    mean_abs = np.abs(shap_values_all).mean(axis=0)
    order = np.argsort(mean_abs)
    labels = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in order]
    vals   = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(vals)))
    ax.barh(labels, vals, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Mean |SHAP value|", color="#888888")
    ax.set_title("Global Feature Importance\n(impact on distress risk prediction)",
                 color="#ffffff", fontsize=12, pad=12)
    ax.tick_params(colors="#cccccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#333333")

    plt.tight_layout()

    if save:
        path = PLOTS_DIR / "global_feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info(f"Saved global importance → {path}")

    return fig


# ─── EXPLAIN A COMPANY ────────────────────────────────────────────────────────

def explain_company(ticker: str, feature_row: dict) -> dict:
    """
    Given a ticker and its feature dict, return SHAP explanation as JSON.
    Suitable for the API endpoint.
    """
    xgb_pipeline = joblib.load(MODEL_DIR / "xgboost_pipeline.pkl")
    df_bg = pd.read_csv(PROC_DIR / "features.csv")
    available = [c for c in FEATURE_COLS if c in df_bg.columns]
    X_bg = df_bg[available].fillna(0).clip(-10, 10).values[:200]  # sample for background

    explainer, scaler = build_explainer(xgb_pipeline, X_bg)

    X_row = np.array([feature_row.get(c, 0) for c in FEATURE_COLS])
    X_row = np.clip(np.nan_to_num(X_row, nan=0.0), -10, 10)

    shap_vals = get_shap_values(explainer, scaler, X_row)

    # Build explanation dict
    contributions = []
    for feat, sv, fv in zip(FEATURE_COLS, shap_vals, X_row):
        contributions.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": round(float(fv), 4),
            "shap_value": round(float(sv), 4),
            "direction": "increases_risk" if sv > 0 else "reduces_risk",
        })

    # Sort by |SHAP|
    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Save waterfall plot
    plot_waterfall(shap_vals, FEATURE_COLS, X_row,
                   float(explainer.expected_value), ticker, save=True)

    return {
        "ticker": ticker,
        "base_value": round(float(explainer.expected_value), 4),
        "top_drivers": contributions[:5],
        "all_contributions": contributions,
        "plot_path": str(PLOTS_DIR / f"{ticker}_shap_waterfall.png"),
    }


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NKLA"

    # Load scored data to get a feature row
    scored = pd.read_csv(PROC_DIR / "scored_companies.csv")
    row = scored[scored["ticker"] == ticker].sort_values("date").iloc[-1]
    feat_row = {c: row[c] for c in FEATURE_COLS if c in scored.columns}

    print(f"\n=== SHAP Explanation for {ticker} ===")
    result = explain_company(ticker, feat_row)
    print(f"\nTop 5 risk drivers:")
    for d in result["top_drivers"]:
        arrow = "↑" if d["direction"] == "increases_risk" else "↓"
        print(f"  {arrow} {d['label']:<30} val={d['value']:>8.3f}  SHAP={d['shap_value']:>+.4f}")
    print(f"\nWaterfall plot saved to: {result['plot_path']}")
