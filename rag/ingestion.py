"""
FinRisk Radar — RAG Ingestion
Downloads SEC filings, extracts clean text, chunks into passages,
embeds with sentence-transformers, and stores in ChromaDB.
"""

import os
import re
import json
import time
import logging
import requests
import pdfplumber
from pathlib import Path
from typing import Optional
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EDGAR_HEADERS = {"User-Agent": "FinRiskRadar research@finriskradar.com"}

# Sections of SEC filings we care about most
TARGET_SECTIONS = [
    "management",
    "risk factor",
    "liquidity",
    "capital resource",
    "result of operation",
    "going concern",
    "debt",
    "obligation",
    "cash flow",
    "interest expense",
    "credit facilit",
]


# ─── TEXT EXTRACTION ──────────────────────────────────────────────────────────

def extract_text_from_html(html: str) -> str:
    """Strip HTML tags and boilerplate from SEC filing HTML."""
    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Clean whitespace
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove XBRL / numeric boilerplate lines
    text = "\n".join(
        line for line in text.splitlines()
        if len(line.strip()) > 30 and not re.match(r"^[\d\s\.\-\$,]+$", line.strip())
    )
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF filing using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    except Exception as e:
        log.error(f"PDF extraction error: {e}")
    return "\n\n".join(text_parts)


def filter_relevant_sections(text: str) -> str:
    """
    Keep only paragraphs that contain financially relevant keywords.
    Reduces noise from boilerplate exhibits, cover pages, etc.
    """
    paragraphs = text.split("\n\n")
    kept = []
    for para in paragraphs:
        lower = para.lower()
        if any(kw in lower for kw in TARGET_SECTIONS):
            kept.append(para)
        elif len(para.split()) > 40:  # keep any long paragraph (likely narrative)
            kept.append(para)
    return "\n\n".join(kept)


# ─── CHUNKING ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, ticker: str, filing_date: str,
               form_type: str, chunk_size: int = 512,
               chunk_overlap: int = 50) -> list[dict]:
    """
    Split filing text into overlapping chunks with metadata.
    Returns list of {text, metadata} dicts ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 80:  # Skip tiny fragments
            continue
        docs.append({
            "text": chunk.strip(),
            "metadata": {
                "ticker":      ticker,
                "filing_date": filing_date,
                "form_type":   form_type,
                "chunk_id":    i,
                "source":      f"{ticker}_{form_type}_{filing_date}_chunk{i}",
                "year":        filing_date[:4] if filing_date else "unknown",
            }
        })

    log.info(f"Created {len(docs)} chunks for {ticker} {form_type} {filing_date}")
    return docs


# ─── EMBEDDINGS + VECTOR STORE ────────────────────────────────────────────────

def get_embedding_model():
    """
    Load sentence-transformers embedding model.
    all-MiniLM-L6-v2: fast (14ms/chunk), 384-dim, good for finance text.
    """
    log.info("Loading embedding model: all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(ticker: str = None, embedding_model=None):
    """
    Get or create a ChromaDB collection.
    One collection per ticker for efficient per-company retrieval.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    collection_name = f"finrisk_{ticker.lower()}" if ticker else "finrisk_global"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore


def ingest_chunks_to_chroma(docs: list[dict], ticker: str, embedding_model=None):
    """Embed chunks and upsert into ChromaDB."""
    if not docs:
        log.warning("No documents to ingest.")
        return

    if embedding_model is None:
        embedding_model = get_embedding_model()

    vs = get_vectorstore(ticker, embedding_model)

    texts    = [d["text"]     for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids      = [d["metadata"]["source"] for d in docs]

    # Batch embed in groups of 100 to avoid memory issues
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_meta  = metadatas[i:i+batch_size]
        batch_ids   = ids[i:i+batch_size]
        vs.add_texts(texts=batch_texts, metadatas=batch_meta, ids=batch_ids)
        log.info(f"  Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    log.info(f"Ingested {len(texts)} chunks for {ticker} into ChromaDB")
    return vs


# ─── FULL INGESTION PIPELINE ──────────────────────────────────────────────────

def ingest_filing(ticker: str, filing_path: str, filing_date: str,
                  form_type: str, embedding_model=None) -> int:
    """
    Full pipeline for one filing:
    file path → extract text → filter → chunk → embed → store
    Returns number of chunks ingested.
    """
    path = Path(filing_path)
    if not path.exists():
        log.error(f"Filing not found: {filing_path}")
        return 0

    log.info(f"Processing {ticker} {form_type} {filing_date}")

    # Extract text
    if path.suffix.lower() == ".pdf":
        raw_text = extract_text_from_pdf(str(path))
    else:
        raw_text = extract_text_from_html(path.read_text(encoding="utf-8", errors="replace"))

    if not raw_text or len(raw_text) < 500:
        log.warning(f"Extraction yielded very little text ({len(raw_text)} chars)")
        return 0

    # Filter to relevant sections
    filtered = filter_relevant_sections(raw_text)
    log.info(f"Text: {len(raw_text):,} chars → filtered: {len(filtered):,} chars")

    # Chunk
    docs = chunk_text(filtered, ticker, filing_date, form_type)

    # Embed + store
    ingest_chunks_to_chroma(docs, ticker, embedding_model)

    return len(docs)


def ingest_ticker_all_filings(ticker: str):
    """
    Ingest all downloaded filings for a ticker.
    Reads metadata.json from the raw/TICKER/ directory.
    """
    meta_path = RAW_DIR / ticker.upper() / "metadata.json"
    if not meta_path.exists():
        log.error(f"No metadata found for {ticker}. Run data/ingestion.py first.")
        return

    with open(meta_path) as f:
        filings = json.load(f)

    embedding_model = get_embedding_model()
    total = 0
    for filing in filings:
        n = ingest_filing(
            ticker=filing["ticker"],
            filing_path=filing["path"],
            filing_date=filing["date"],
            form_type=filing["form"],
            embedding_model=embedding_model,
        )
        total += n
        time.sleep(0.2)

    log.info(f"Total chunks ingested for {ticker}: {total}")
    return total


def ingest_news_articles(ticker: str, articles: list[dict], embedding_model=None):
    """
    Ingest news article snippets into the same vectorstore.
    articles: list of {title, description, publishedAt, url}
    """
    docs = []
    for i, art in enumerate(articles):
        text = f"{art.get('title', '')}\n\n{art.get('description', '')}".strip()
        if len(text) < 50:
            continue
        docs.append({
            "text": text,
            "metadata": {
                "ticker":      ticker,
                "filing_date": art.get("publishedAt", "")[:10],
                "form_type":   "NEWS",
                "chunk_id":    i,
                "source":      f"{ticker}_news_{i}",
                "url":         art.get("url", ""),
                "year":        art.get("publishedAt", "")[:4],
            }
        })
    if docs:
        ingest_chunks_to_chroma(docs, ticker, embedding_model)
        log.info(f"Ingested {len(docs)} news articles for {ticker}")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"=== RAG Ingestion: {ticker} ===")
    ingest_ticker_all_filings(ticker)
    print("Done. Chunks stored in ChromaDB.")
