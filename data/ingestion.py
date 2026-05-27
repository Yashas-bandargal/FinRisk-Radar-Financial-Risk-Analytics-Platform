"""
FinRisk Radar — Data Ingestion
Pulls financial data from SEC EDGAR and yfinance.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

EDGAR_HEADERS = {"User-Agent": "FinRiskRadar research@finriskradar.com"}

# ─── SEC EDGAR ────────────────────────────────────────────────────────────────

def get_cik(ticker: str) -> Optional[str]:
    """Convert stock ticker to SEC CIK number."""
    url = "https://efts.sec.gov/LATEST/search-index?q=%22{}%22&dateRange=custom&startdt=2020-01-01&forms=10-K".format(ticker)
    try:
        r = requests.get(
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={}&type=10-K&dateb=&owner=include&count=5&search_text=&output=atom".format(ticker),
            headers=EDGAR_HEADERS, timeout=10
        )
        # Use the company_tickers JSON (most reliable)
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=EDGAR_HEADERS, timeout=10)
        data = resp.json()
        for entry in data.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                log.info(f"CIK for {ticker}: {cik}")
                return cik
    except Exception as e:
        log.error(f"CIK lookup failed for {ticker}: {e}")
    return None


def get_filings_list(cik: str, form_type: str = "10-K", count: int = 4) -> list:
    """Get list of recent filings for a CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        data = resp.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        results = []
        for f, a, d in zip(forms, accessions, dates):
            if f == form_type and len(results) < count:
                results.append({"form": f, "accession": a, "date": d, "cik": cik})
        log.info(f"Found {len(results)} {form_type} filings for CIK {cik}")
        return results
    except Exception as e:
        log.error(f"Filings list error: {e}")
        return []


def download_filing_text(cik: str, accession: str, save_dir: Path) -> Optional[Path]:
    """Download the primary document of a filing as text."""
    acc_clean = accession.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"
    try:
        resp = requests.get(index_url, headers=EDGAR_HEADERS, timeout=15)
        # Try to find the main .htm document
        lines = resp.text.split("\n")
        doc_url = None
        for line in lines:
            if ".htm" in line.lower() and "10-k" in line.lower():
                start = line.find("href=")
                if start != -1:
                    end = line.find('"', start + 6)
                    doc_url = "https://www.sec.gov" + line[start + 6:end]
                    break
        if not doc_url:
            # Fallback: get the full submission text file
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}.txt"

        doc_resp = requests.get(doc_url, headers=EDGAR_HEADERS, timeout=30)
        save_path = save_dir / f"{cik}_{accession}.txt"
        save_path.write_text(doc_resp.text, encoding="utf-8", errors="replace")
        log.info(f"Saved filing: {save_path}")
        return save_path
    except Exception as e:
        log.error(f"Filing download error ({accession}): {e}")
        return None


def ingest_sec_filings(ticker: str, form_types: list = ["10-K", "10-Q"]) -> list:
    """Full pipeline: ticker → CIK → filings list → download text."""
    cik = get_cik(ticker)
    if not cik:
        log.warning(f"Could not find CIK for {ticker}")
        return []

    save_dir = RAW_DIR / ticker.upper()
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for form in form_types:
        filings = get_filings_list(cik, form_type=form, count=4)
        for filing in filings:
            path = download_filing_text(cik, filing["accession"], save_dir)
            if path:
                saved.append({
                    "ticker": ticker,
                    "cik": cik,
                    "form": form,
                    "date": filing["date"],
                    "path": str(path)
                })
            time.sleep(0.5)  # EDGAR rate limit: max 10 req/sec

    # Save metadata
    meta_path = save_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(saved, f, indent=2)
    log.info(f"Ingested {len(saved)} filings for {ticker}")
    return saved


# ─── YFINANCE FINANCIAL RATIOS ─────────────────────────────────────────────────

RATIO_METRICS = [
    "totalDebt", "totalAssets", "totalCurrentAssets", "totalCurrentLiabilities",
    "totalRevenue", "grossProfit", "ebitda", "netIncome", "operatingCashflow",
    "freeCashflow", "interestExpense", "retainedEarnings", "marketCap",
    "totalStockholdersEquity", "workingCapital"
]


def fetch_financials(ticker: str) -> pd.DataFrame:
    """Fetch multi-quarter financial data from yfinance."""
    stock = yf.Ticker(ticker)
    records = []

    try:
        bs = stock.quarterly_balance_sheet
        inc = stock.quarterly_income_stmt
        cf = stock.quarterly_cashflow
        info = stock.info

        dates = bs.columns.tolist()[:8]  # Last 8 quarters

        for date in dates:
            rec = {"ticker": ticker.upper(), "date": str(date)[:10]}

            def safe_get(df, row):
                try:
                    return float(df.loc[row, date]) if row in df.index else None
                except Exception:
                    return None

            # Balance sheet
            rec["total_assets"]       = safe_get(bs, "Total Assets")
            rec["total_debt"]         = safe_get(bs, "Total Debt")
            rec["current_assets"]     = safe_get(bs, "Current Assets")
            rec["current_liabilities"]= safe_get(bs, "Current Liabilities")
            rec["retained_earnings"]  = safe_get(bs, "Retained Earnings")
            rec["equity"]             = safe_get(bs, "Stockholders Equity")
            rec["working_capital"]    = safe_get(bs, "Working Capital")

            # Income statement
            rec["revenue"]            = safe_get(inc, "Total Revenue")
            rec["gross_profit"]       = safe_get(inc, "Gross Profit")
            rec["ebit"]               = safe_get(inc, "EBIT")
            rec["net_income"]         = safe_get(inc, "Net Income")
            rec["interest_expense"]   = safe_get(inc, "Interest Expense")

            # Cash flow
            rec["operating_cf"]       = safe_get(cf, "Operating Cash Flow")
            rec["free_cf"]            = safe_get(cf, "Free Cash Flow")
            rec["capex"]              = safe_get(cf, "Capital Expenditure")

            # Market data from info
            rec["market_cap"]         = info.get("marketCap")
            rec["sector"]             = info.get("sector", "Unknown")
            rec["industry"]           = info.get("industry", "Unknown")

            records.append(rec)

    except Exception as e:
        log.error(f"yfinance error for {ticker}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        save_path = RAW_DIR / f"{ticker.upper()}_financials.csv"
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} quarters for {ticker} → {save_path}")
    return df


def ingest_multiple_tickers(tickers: list) -> pd.DataFrame:
    """Batch ingest financials for a list of tickers."""
    all_dfs = []
    for ticker in tickers:
        log.info(f"Fetching financials: {ticker}")
        df = fetch_financials(ticker)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out = RAW_DIR / "all_financials.csv"
        combined.to_csv(out, index=False)
        log.info(f"Combined dataset: {len(combined)} rows → {out}")
        return combined
    return pd.DataFrame()


# ─── SAMPLE TICKER LIST ────────────────────────────────────────────────────────

SAMPLE_TICKERS = [
    # Large cap (low risk baseline)
    "AAPL", "MSFT", "GOOGL", "JPM", "BAC",
    # Mid cap
    "SBUX", "NFLX", "LYFT", "SNAP", "PINS",
    # Distressed / volatile (higher risk signals)
    "BBBYQ", "RIDE", "NKLA", "SPCE", "CLOV",
]


if __name__ == "__main__":
    print("=== FinRisk Radar: Data Ingestion ===\n")

    # 1. Pull financial ratios
    print("Step 1: Fetching financial data from yfinance...")
    df = ingest_multiple_tickers(SAMPLE_TICKERS)
    print(f"  → {len(df)} rows across {df['ticker'].nunique() if not df.empty else 0} companies\n")

    # 2. Pull SEC filings for a few tickers
    print("Step 2: Downloading SEC filings from EDGAR...")
    for ticker in ["AAPL", "NKLA"]:
        print(f"  → {ticker}")
        ingest_sec_filings(ticker, form_types=["10-K"])

    print("\nIngestion complete. Check data/raw/")
