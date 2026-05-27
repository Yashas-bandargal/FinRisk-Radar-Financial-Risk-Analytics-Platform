"""
FinRisk Radar — Feature Engineering
Computes financial ratios and the Altman Z-Score from raw balance sheet data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR   = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─── RATIO COMPUTATION ─────────────────────────────────────────────────────────

def safe_div(a, b, default=np.nan):
    """Safe division — returns default when denominator is zero or None."""
    try:
        if b is None or b == 0 or pd.isna(b):
            return default
        return a / b
    except Exception:
        return default


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 12 financial ratios from raw balance sheet columns.

    Input columns expected:
        total_assets, total_debt, current_assets, current_liabilities,
        retained_earnings, equity, working_capital, revenue, gross_profit,
        ebit, net_income, interest_expense, operating_cf, free_cf,
        capex, market_cap

    Output: same df with ratio columns appended.
    """
    out = df.copy()

    # ── Leverage ratios ──────────────────────────────────────────────────────
    # Debt-to-Equity: how much debt per rupee of equity
    out["debt_to_equity"] = out.apply(
        lambda r: safe_div(r["total_debt"], r["equity"]), axis=1
    )

    # Net Debt / EBITDA: how many years of earnings to repay debt
    # Approximate EBITDA = EBIT + D&A (we use EBIT as proxy when D&A unavailable)
    out["net_debt_ebitda"] = out.apply(
        lambda r: safe_div(
            (r["total_debt"] or 0) - max((r.get("operating_cf") or 0) * 0.1, 0),
            abs(r["ebit"]) if r["ebit"] and r["ebit"] != 0 else np.nan
        ), axis=1
    )

    # ── Liquidity ratios ─────────────────────────────────────────────────────
    # Current ratio: can the company pay short-term bills?
    out["current_ratio"] = out.apply(
        lambda r: safe_div(r["current_assets"], r["current_liabilities"]), axis=1
    )

    # Working capital to assets: short-term buffer relative to total size
    out["working_capital_ratio"] = out.apply(
        lambda r: safe_div(r["working_capital"], r["total_assets"]), axis=1
    )

    # ── Profitability ratios ──────────────────────────────────────────────────
    # Gross margin
    out["gross_margin"] = out.apply(
        lambda r: safe_div(r["gross_profit"], r["revenue"]), axis=1
    )

    # FCF margin: how much free cash is generated per dollar of revenue
    out["fcf_margin"] = out.apply(
        lambda r: safe_div(r["free_cf"], r["revenue"]), axis=1
    )

    # Return on Assets
    out["roa"] = out.apply(
        lambda r: safe_div(r["net_income"], r["total_assets"]), axis=1
    )

    # Retained earnings to assets (Altman component)
    out["retained_earnings_ratio"] = out.apply(
        lambda r: safe_div(r["retained_earnings"], r["total_assets"]), axis=1
    )

    # ── Coverage ratios ───────────────────────────────────────────────────────
    # Interest coverage: EBIT / interest expense (>3 is healthy, <1.5 is danger)
    out["interest_coverage"] = out.apply(
        lambda r: safe_div(r["ebit"], abs(r["interest_expense"]) if r["interest_expense"] else np.nan),
        axis=1
    )

    # ── Efficiency ratios ─────────────────────────────────────────────────────
    # Asset turnover: revenue per dollar of assets
    out["asset_turnover"] = out.apply(
        lambda r: safe_div(r["revenue"], r["total_assets"]), axis=1
    )

    # ── Market ratios ─────────────────────────────────────────────────────────
    # Market cap to total debt (market-based solvency)
    out["market_cap_to_debt"] = out.apply(
        lambda r: safe_div(r["market_cap"], r["total_debt"]), axis=1
    )

    # ── Revenue growth QoQ ────────────────────────────────────────────────────
    out = out.sort_values(["ticker", "date"])
    out["revenue_growth_qoq"] = out.groupby("ticker")["revenue"].pct_change()

    return out


# ─── ALTMAN Z-SCORE ────────────────────────────────────────────────────────────

def compute_altman_z(df: pd.DataFrame) -> pd.DataFrame:
    """
    Altman Z-Score for public companies (original 1968 formula).

    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

    Interpretation:
        Z > 2.99  → Safe zone
        1.81–2.99 → Grey zone
        Z < 1.81  → Distress zone
    """
    out = df.copy()

    def z_score(row):
        try:
            ta = row["total_assets"]
            if not ta or ta == 0 or pd.isna(ta):
                return np.nan

            total_liabilities = (row["total_debt"] or 0)  # proxy
            if total_liabilities == 0:
                total_liabilities = ta * 0.3  # fallback estimate

            x1 = safe_div(row["working_capital"], ta, 0)
            x2 = safe_div(row["retained_earnings"], ta, 0)
            x3 = safe_div(row["ebit"], ta, 0)
            x4 = safe_div(row["market_cap"], total_liabilities, 0)
            x5 = safe_div(row["revenue"], ta, 0)

            z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            return round(z, 4)
        except Exception:
            return np.nan

    out["altman_z"] = out.apply(z_score, axis=1)

    def z_label(z):
        if pd.isna(z):
            return "Unknown"
        if z >= 2.99:
            return "Safe"
        elif z >= 1.81:
            return "Grey"
        else:
            return "Distress"

    out["altman_zone"] = out["altman_z"].apply(z_label)
    return out


# ─── SECTOR BENCHMARKING ───────────────────────────────────────────────────────

def add_sector_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each ratio, compute how many standard deviations the company
    is from its sector median. A large deviation = anomaly signal.
    """
    ratio_cols = [
        "debt_to_equity", "current_ratio", "interest_coverage",
        "fcf_margin", "gross_margin", "roa", "altman_z",
        "asset_turnover", "net_debt_ebitda"
    ]
    out = df.copy()

    for col in ratio_cols:
        if col not in out.columns:
            continue
        sector_stats = (
            out.groupby("sector")[col]
            .agg(["median", "std"])
            .rename(columns={"median": f"{col}_sector_median", "std": f"{col}_sector_std"})
        )
        out = out.merge(sector_stats, on="sector", how="left")
        out[f"{col}_zscore"] = out.apply(
            lambda r: safe_div(
                r[col] - r[f"{col}_sector_median"],
                r[f"{col}_sector_std"]
            ), axis=1
        )
        out.drop(columns=[f"{col}_sector_median", f"{col}_sector_std"], inplace=True)

    return out


# ─── DISTRESS LABELS ──────────────────────────────────────────────────────────

def attach_distress_labels(df: pd.DataFrame, label_path: str = None) -> pd.DataFrame:
    """
    Attach binary distress labels (1 = distressed within next 4 quarters).
    
    If a label CSV is provided (ticker, distress_date), use that.
    Otherwise, use Altman Z-Score < 1.81 as a proxy label for training.
    """
    out = df.copy()

    if label_path and Path(label_path).exists():
        labels = pd.read_csv(label_path)
        # Merge on ticker; mark quarters within 1 year of distress event
        out = out.merge(labels[["ticker", "distress_date"]], on="ticker", how="left")
        out["distress_date"] = pd.to_datetime(out["distress_date"], errors="coerce")
        out["date_dt"] = pd.to_datetime(out["date"], errors="coerce")
        out["distressed"] = (
            (out["distress_date"] - out["date_dt"]).dt.days.between(0, 365)
        ).astype(int)
        out.drop(columns=["distress_date", "date_dt"], inplace=True)
    else:
        # Proxy: Altman Z < 1.81 → distressed (good enough for training demo)
        log.warning("No label file provided — using Altman Z < 1.81 as proxy label")
        out["distressed"] = (out["altman_z"] < 1.81).astype(int)

    log.info(f"Label distribution:\n{out['distressed'].value_counts()}")
    return out


# ─── FULL PIPELINE ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "debt_to_equity", "net_debt_ebitda", "current_ratio", "working_capital_ratio",
    "gross_margin", "fcf_margin", "roa", "retained_earnings_ratio",
    "interest_coverage", "asset_turnover", "market_cap_to_debt",
    "revenue_growth_qoq", "altman_z"
]


def build_feature_dataset(raw_csv: str = None, label_csv: str = None) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.
    Reads raw financials CSV → computes ratios → Z-score → sector benchmarks → labels.
    """
    if raw_csv is None:
        raw_csv = str(RAW_DIR / "all_financials.csv")

    if not Path(raw_csv).exists():
        log.error(f"Raw CSV not found: {raw_csv}")
        log.info("Run data/ingestion.py first to pull data.")
        return pd.DataFrame()

    log.info(f"Loading raw data from {raw_csv}")
    df = pd.read_csv(raw_csv)
    log.info(f"Raw shape: {df.shape}")

    # Fill numeric NaNs with column median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    log.info("Computing financial ratios...")
    df = compute_ratios(df)

    log.info("Computing Altman Z-Score...")
    df = compute_altman_z(df)

    log.info("Adding sector benchmarks...")
    if "sector" in df.columns:
        df = add_sector_benchmarks(df)

    log.info("Attaching distress labels...")
    df = attach_distress_labels(df, label_path=label_csv)

    # Drop rows missing too many features
    df_clean = df.dropna(subset=FEATURE_COLS, thresh=8)
    log.info(f"Clean dataset shape: {df_clean.shape}")

    out_path = PROC_DIR / "features.csv"
    df_clean.to_csv(out_path, index=False)
    log.info(f"Saved feature dataset → {out_path}")

    return df_clean


if __name__ == "__main__":
    print("=== FinRisk Radar: Feature Engineering ===\n")
    df = build_feature_dataset()
    if not df.empty:
        print(f"\nFeature dataset: {df.shape}")
        print(f"Distress rate: {df['distressed'].mean():.1%}")
        print(f"\nSample (5 rows):\n{df[['ticker','date','altman_z','altman_zone','debt_to_equity','distressed']].head()}")
