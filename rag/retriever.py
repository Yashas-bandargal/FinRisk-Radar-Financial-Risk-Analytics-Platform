"""
FinRisk Radar — RAG Retriever
MMR retrieval + cross-encoder reranking over ChromaDB vector store.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

# Cross-encoder for reranking (tiny model, runs on CPU fine)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"


# ─── SETUP ───────────────────────────────────────────────────────────────────

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(ticker: str, embedding_model=None):
    if embedding_model is None:
        embedding_model = get_embedding_model()
    collection_name = f"finrisk_{ticker.lower()}"
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


# ─── RETRIEVAL STRATEGIES ────────────────────────────────────────────────────

def retrieve_mmr(query: str, ticker: str, k: int = 5,
                 fetch_k: int = 20, lambda_mult: float = 0.7,
                 date_filter: Optional[str] = None,
                 embedding_model=None) -> list[dict]:
    """
    Maximum Marginal Relevance retrieval.

    fetch_k:    candidates fetched from vector store before MMR
    k:          final number of chunks to return
    lambda_mult: diversity factor (0=max diversity, 1=pure similarity)
    date_filter: "2023" filters to filings from that year onward

    Why MMR?
    SEC filings repeat boilerplate — MMR penalizes redundant chunks
    so we get diverse, information-rich context instead of 5 near-copies.
    """
    vs = get_vectorstore(ticker, embedding_model)

    search_kwargs = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}

    # Date filtering via ChromaDB metadata
    if date_filter:
        search_kwargs["filter"] = {"year": {"$gte": date_filter}}

    try:
        docs = vs.max_marginal_relevance_search(
            query=query,
            **search_kwargs
        )
    except Exception as e:
        log.warning(f"MMR search error (falling back to similarity): {e}")
        docs = vs.similarity_search(query, k=k)

    results = []
    for doc in docs:
        results.append({
            "text":     doc.page_content,
            "metadata": doc.metadata,
            "source":   doc.metadata.get("source", "unknown"),
            "date":     doc.metadata.get("filing_date", ""),
            "form":     doc.metadata.get("form_type", ""),
        })

    log.info(f"MMR retrieved {len(results)} chunks for '{query[:60]}...'")
    return results


def retrieve_similarity(query: str, ticker: str, k: int = 5,
                        embedding_model=None) -> list[dict]:
    """Standard cosine similarity retrieval (fallback / comparison)."""
    vs = get_vectorstore(ticker, embedding_model)
    docs = vs.similarity_search_with_score(query, k=k)
    results = []
    for doc, score in docs:
        results.append({
            "text":     doc.page_content,
            "metadata": doc.metadata,
            "source":   doc.metadata.get("source", "unknown"),
            "date":     doc.metadata.get("filing_date", ""),
            "form":     doc.metadata.get("form_type", ""),
            "score":    round(float(score), 4),
        })
    return results


# ─── CROSS-ENCODER RERANKING ─────────────────────────────────────────────────

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        log.info(f"Loading cross-encoder: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Cross-encoder reranking on top of MMR candidates.

    Why rerank?
    The bi-encoder (MiniLM) is fast but approximate — it encodes query
    and document independently. The cross-encoder sees query+document
    together and gives a much more accurate relevance score.
    Pipeline: MMR(20 candidates) → cross-encoder → top 5.
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]

    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        log.warning(f"Reranker failed: {e}. Returning MMR order.")
        return chunks[:top_k]

    # Attach scores and sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
    log.info(f"Reranked {len(chunks)} → top {top_k}. Best score: {reranked[0]['rerank_score']:.3f}")
    return reranked[:top_k]


# ─── FULL RETRIEVAL PIPELINE ─────────────────────────────────────────────────

def retrieve(query: str, ticker: str, k: int = 5,
             use_reranker: bool = True,
             date_filter: Optional[str] = None,
             embedding_model=None) -> list[dict]:
    """
    Full retrieval pipeline: MMR(20) → optional cross-encoder rerank → top k.

    Args:
        query:        Natural language question from user
        ticker:       Company ticker symbol
        k:            Number of final chunks to return
        use_reranker: Enable cross-encoder reranking (slower but better)
        date_filter:  Year string e.g. "2022" to filter older filings

    Returns:
        List of chunk dicts with text, metadata, source, scores
    """
    fetch_k = max(k * 4, 20)  # Fetch 4× more for reranker to work on

    chunks = retrieve_mmr(
        query=query,
        ticker=ticker,
        k=fetch_k if use_reranker else k,
        fetch_k=min(fetch_k * 2, 50),
        date_filter=date_filter,
        embedding_model=embedding_model,
    )

    if use_reranker and len(chunks) > k:
        chunks = rerank(query, chunks, top_k=k)
    else:
        chunks = chunks[:k]

    return chunks


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.
    Each chunk includes its source metadata for citation.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", f"source_{i}")
        date   = chunk.get("date", "")
        form   = chunk.get("form", "")
        header = f"[Source {i}: {form} filed {date} | {source}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    query  = sys.argv[2] if len(sys.argv) > 2 else "What are the main liquidity risks?"

    print(f"\n=== Retrieval Test: {ticker} ===")
    print(f"Query: {query}\n")

    chunks = retrieve(query, ticker, k=3, use_reranker=True)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c['source']} (rerank: {c.get('rerank_score', 'N/A')})")
        print(f"    {c['text'][:200]}...\n")
