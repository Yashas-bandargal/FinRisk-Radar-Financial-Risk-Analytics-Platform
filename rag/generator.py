"""
FinRisk Radar — RAG Generator
Gemini 1.5 Flash generation with citation-grounded prompting.
Uses google.genai SDK (new style).
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Generator, Optional
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── SETUP ───────────────────────────────────────────────────────────────────

def setup_gemini():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.\n"
            "Get your key at: https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are FinRisk Radar, an AI financial risk analyst assistant.

Your task: Answer the user's question about a company's financial health using ONLY the retrieved filing excerpts provided below. 

STRICT RULES:
1. ONLY use information from the provided context. Do NOT use your prior training knowledge about the company.
2. If the context does not contain the answer, say exactly: "The retrieved filings do not contain sufficient information to answer this question."
3. Cite your sources using [Source N] notation where N matches the source number in the context.
4. Structure your answer as:
   - A direct answer (2-3 sentences)
   - Supporting evidence from the filings (with citations)
   - Risk implication (what this means for financial health)
5. Keep the answer concise — under 250 words.
6. Use professional financial analyst language.
7. Never fabricate numbers, dates, or facts not present in the context.

Remember: Your credibility depends on grounding every claim in the provided context."""


# ─── PROMPT BUILDER ──────────────────────────────────────────────────────────

def build_rag_prompt(query: str, context: str, ticker: str,
                     risk_score: Optional[float] = None) -> str:
    """
    Build the full prompt sent to Gemini.
    Includes the ML risk score as additional context when available.
    """
    risk_context = ""
    if risk_score is not None:
        label = "High Risk" if risk_score >= 70 else "Medium Risk" if risk_score >= 40 else "Low Risk"
        risk_context = f"\nML Risk Score Context: {ticker} currently has a Risk Score of {risk_score:.0f}/100 ({label}) based on quantitative financial ratio analysis.\n"

    prompt = f"""{SYSTEM_PROMPT}

Company: {ticker}
{risk_context}
=== RETRIEVED FILING CONTEXT ===
{context}
=== END OF CONTEXT ===

User Question: {query}

Answer:"""
    return prompt


# ─── GENERATION ──────────────────────────────────────────────────────────────

def generate(query: str, context: str, ticker: str,
             risk_score: Optional[float] = None,
             max_tokens: int = 512,
             temperature: float = 0.1) -> dict:
    """
    Generate a cited answer using Gemini 1.5 Flash.

    Low temperature (0.1) → consistent, factual, conservative responses.
    This is intentional: financial analysis should not be "creative."

    Returns:
        {answer, citations_found, token_usage, model}
    """
    model = setup_gemini()
    prompt = build_rag_prompt(query, context, ticker, risk_score)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            ),
        )
        answer = response.text.strip()

        # Extract citation numbers from response
        citations = re.findall(r"\[Source (\d+)\]", answer)
        citations = sorted(set(int(c) for c in citations))

        return {
            "answer":          answer,
            "citations_found": citations,
            "model":           "gemini-1.5-flash",
            "prompt_chars":    len(prompt),
        }

    except Exception as e:
        log.error(f"Gemini generation error: {e}")
        return {
            "answer":          f"Generation error: {str(e)}",
            "citations_found": [],
            "model":           "gemini-1.5-flash",
            "error":           str(e),
        }


def generate_streaming(query: str, context: str, ticker: str,
                       risk_score: Optional[float] = None) -> Generator[str, None, None]:
    """
    Streaming generation — yields tokens as they arrive.
    Used by the Streamlit frontend for real-time display.
    """
    model = setup_gemini()
    prompt = build_rag_prompt(query, context, ticker, risk_score)

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=512,
                temperature=0.1,
            ),
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n[Error: {str(e)}]"


# ─── STRUCTURED OUTPUT ────────────────────────────────────────────────────────

STRUCTURED_PROMPT_SUFFIX = """

Respond ONLY with a valid JSON object (no markdown, no explanation) in this exact format:
{
  "summary": "2-3 sentence direct answer",
  "risk_factors": ["factor 1", "factor 2", "factor 3"],
  "supporting_evidence": [
    {"claim": "specific claim", "citation": "Source N", "filing_date": "YYYY-MM-DD"}
  ],
  "risk_implication": "what this means for the company's financial health",
  "confidence": "high|medium|low"
}"""


def generate_structured(query: str, context: str, ticker: str,
                         risk_score: Optional[float] = None) -> dict:
    """
    Generate structured JSON output for API consumption.
    Used by the FastAPI backend to return machine-readable analysis.
    """
    model = setup_gemini()
    prompt = build_rag_prompt(query, context, ticker, risk_score) + STRUCTURED_PROMPT_SUFFIX

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800,
                temperature=0.05,
            ),
        )
        raw = response.text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        parsed = json.loads(raw)
        return {"success": True, "data": parsed}

    except json.JSONDecodeError as e:
        log.error(f"JSON parse error: {e}\nRaw: {raw[:200]}")
        return {"success": False, "error": "JSON parse failed", "raw": raw}
    except Exception as e:
        log.error(f"Structured generation error: {e}")
        return {"success": False, "error": str(e)}


# ─── QUERY SUGGESTIONS ────────────────────────────────────────────────────────

DEFAULT_QUERIES = [
    "What are the main liquidity risks mentioned in the latest filing?",
    "What does the filing say about the company's debt obligations and refinancing plans?",
    "Are there any going concern warnings or audit qualifications?",
    "What is the management outlook for revenue and profitability?",
    "What are the key risk factors listed by management?",
    "How has free cash flow changed compared to prior periods?",
    "What credit facilities does the company have and what are the covenants?",
    "Is there any mention of regulatory investigations or legal proceedings?",
]


if __name__ == "__main__":
    import sys
    from rag.retriever import retrieve, format_context

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    query  = sys.argv[2] if len(sys.argv) > 2 else "What are the main liquidity risks?"

    print(f"\n=== RAG Query: {ticker} ===")
    print(f"Q: {query}\n")

    chunks = retrieve(query, ticker, k=5, use_reranker=True)
    context = format_context(chunks)

    if not chunks:
        print("No chunks retrieved. Run rag/ingestion.py first.")
    else:
        result = generate(query, context, ticker, risk_score=45.0)
        print("Answer:")
        print(result["answer"])
        print(f"\nCitations used: {result['citations_found']}")
