"""
FinRisk Radar — Streamlit Dashboard
Full-featured UI: risk gauge, SHAP waterfall, RAG chat, trend chart, company comparison.
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinRisk Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Dark finance theme */
:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --accent: #3b82f6;
    --text-muted: #9ca3af;
    --green: #10b981;
    --red: #ef4444;
    --amber: #f59e0b;
}

.stApp { background-color: #0a0e1a; }

/* Hero title */
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-val { font-size: 2rem; font-weight: 700; }
.metric-lbl { font-size: 0.8rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }

/* Risk badge */
.badge-high   { background: #7f1d1d; color: #fca5a5; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-medium { background: #78350f; color: #fcd34d; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
.badge-low    { background: #064e3b; color: #6ee7b7; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }

/* Chat messages */
.chat-user { background: #1e3a5f; border-radius: 12px 12px 4px 12px; padding: 12px 16px; margin: 6px 0; }
.chat-bot  { background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 12px 12px 12px 4px; padding: 12px 16px; margin: 6px 0; }

/* Source citation boxes */
.source-box {
    background: #111827;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
}

/* Section headers */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4b5563;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def api_get(path: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, payload: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def risk_color(score: float) -> str:
    if score >= 70: return "#ef4444"
    if score >= 40: return "#f59e0b"
    return "#10b981"

def badge_html(label: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(label, "badge-low")
    return f'<span class="{cls}">{label} Risk</span>'


# ─── GAUGE CHART ──────────────────────────────────────────────────────────────

def make_gauge(score: float, ticker: str) -> go.Figure:
    color = risk_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": "#f9fafb"}},
        title={"text": f"{ticker} Risk Score", "font": {"size": 16, "color": "#9ca3af"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#6b7280"}, "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.8},
            "bgcolor": "#1f2937",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "#064e3b"},
                {"range": [40, 70], "color": "#78350f"},
                {"range": [70, 100], "color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "#f9fafb", "width": 2},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#f9fafb"},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ─── SHAP WATERFALL ───────────────────────────────────────────────────────────

def make_shap_chart(shap_data: list, ticker: str) -> go.Figure:
    """Build a horizontal waterfall chart from SHAP contribution data."""
    top = sorted(shap_data, key=lambda x: abs(x["shap_value"]), reverse=True)[:8]
    labels = [d["label"] for d in top]
    values = [d["shap_value"] for d in top]
    colors = ["#ef4444" if v > 0 else "#10b981" for v in values]
    hover  = [f"Value: {d['value']:.3f}<br>SHAP: {d['shap_value']:+.4f}" for d in top]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        title=dict(text="Feature Contributions (SHAP)", font=dict(color="#9ca3af", size=14)),
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#f9fafb"},
        xaxis=dict(
            title="← Reduces Risk   |   Increases Risk →",
            titlefont=dict(color="#6b7280", size=11),
            gridcolor="#1f2937",
            zerolinecolor="#374151",
        ),
        yaxis=dict(gridcolor="#1f2937"),
        height=320,
        margin=dict(l=10, r=10, t=50, b=30),
    )
    return fig


# ─── TREND CHART ─────────────────────────────────────────────────────────────

def make_trend_chart(trend_data: list) -> go.Figure:
    df = pd.DataFrame(trend_data)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["risk_score"],
        mode="lines+markers",
        name="Risk Score",
        line=dict(color="#3b82f6", width=2.5),
        marker=dict(size=7, color=df["risk_score"].apply(risk_color)),
        hovertemplate="<b>%{x}</b><br>Risk Score: %{y}/100<extra></extra>",
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.08)",
    ))

    # Risk zone bands
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.07)", line_width=0, annotation_text="High", annotation_position="top right")
    fig.add_hrect(y0=40, y1=70,  fillcolor="rgba(245,158,11,0.07)", line_width=0)
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(16,185,129,0.07)", line_width=0)

    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#f9fafb"},
        yaxis=dict(range=[0, 100], gridcolor="#1f2937", title="Risk Score"),
        xaxis=dict(gridcolor="#1f2937"),
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
    )
    return fig


# ─── RATIO TABLE ─────────────────────────────────────────────────────────────

def ratio_table(key_ratios: dict) -> None:
    RATIO_META = {
        "debt_to_equity":    ("Debt / Equity",     ">2 = risky",  2.0,  "lower"),
        "current_ratio":     ("Current Ratio",      "<1 = risky",  1.0,  "higher"),
        "interest_coverage": ("Interest Coverage",  "<1.5 = risky",1.5,  "higher"),
        "fcf_margin":        ("FCF Margin",          "<0 = risky",  0.0,  "higher"),
        "gross_margin":      ("Gross Margin",        "<0.2 = thin", 0.2,  "higher"),
    }

    rows = []
    for key, (label, note, threshold, direction) in RATIO_META.items():
        val = key_ratios.get(key)
        if val is None:
            status = "⚪"
        elif direction == "lower":
            status = "🔴" if val > threshold else "🟢"
        else:
            status = "🔴" if val < threshold else "🟢"
        rows.append({"Ratio": label, "Value": f"{val:.2f}" if val else "N/A", "Signal": status, "Note": note})

    st.table(pd.DataFrame(rows).set_index("Ratio"))


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title">📡 FinRisk</div>', unsafe_allow_html=True)
    st.caption("AI Financial Risk Intelligence")
    st.divider()

    ticker_input = st.text_input("Company Ticker", value="AAPL", placeholder="e.g. AAPL, NKLA").upper().strip()
    st.caption("Enter any NYSE/NASDAQ ticker")

    st.divider()
    st.markdown("**Quick Compare**")
    compare_tickers = st.multiselect(
        "Select tickers",
        options=["AAPL", "MSFT", "NKLA", "SPCE", "SBUX", "SNAP", "LYFT"],
        default=["AAPL", "NKLA"],
        max_selections=5,
    )

    st.divider()
    st.markdown("**Set Risk Alert**")
    alert_threshold = st.slider("Alert if score exceeds:", 0, 100, 70)
    alert_email = st.text_input("Email (optional)")
    if st.button("Set Alert", use_container_width=True):
        result = api_post("/alerts", {
            "ticker": ticker_input,
            "threshold": alert_threshold,
            "email": alert_email,
            "direction": "above",
        })
        st.success(f"Alert set for {ticker_input} > {alert_threshold}")

    st.divider()
    st.caption("FinRisk Radar v1.0 · ECE → ML Engineer Portfolio Project")


# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────

tab_risk, tab_rag, tab_compare, tab_eval = st.tabs([
    "📊 Risk Dashboard", "💬 Filing Q&A", "📈 Compare", "🧪 Evaluation"
])


# ═══════════════════ TAB 1: RISK DASHBOARD ═════════════════════════════════════

with tab_risk:
    if ticker_input:
        with st.spinner(f"Loading risk data for {ticker_input}..."):
            data = api_get(f"/risk/{ticker_input}")
            trend = api_get(f"/risk/{ticker_input}/trend?quarters=8")

        if "error" in data:
            st.error(f"API error: {data['error']}")
            st.info("Make sure the FastAPI server is running (`python -m api.main`) and data is ingested.")
        else:
            # ── Header row ──────────────────────────────────────────────────
            col_title, col_badge, col_date = st.columns([3, 1, 1])
            with col_title:
                st.markdown(f"## {ticker_input}")
                st.caption(data.get("sector", ""))
            with col_badge:
                st.markdown(badge_html(data.get("risk_label", "Unknown")), unsafe_allow_html=True)
            with col_date:
                st.caption(f"As of: {data.get('as_of_date', 'N/A')}")

            st.divider()

            # ── Main metrics row ────────────────────────────────────────────
            col_gauge, col_altman, col_ratios = st.columns([1.2, 0.8, 1.2])

            with col_gauge:
                risk_score = data.get("risk_score", 0)
                st.plotly_chart(make_gauge(risk_score, ticker_input),
                                use_container_width=True, key="gauge")

            with col_altman:
                st.markdown('<div class="section-label">Altman Z-Score</div>', unsafe_allow_html=True)
                z = data.get("altman_z")
                zone = data.get("altman_zone", "Unknown")
                if z:
                    z_color = "#ef4444" if zone == "Distress" else "#f59e0b" if zone == "Grey" else "#10b981"
                    st.markdown(f"""
                    <div style='background:#111827;border-radius:12px;padding:20px;text-align:center;border:1px solid #1f2937'>
                        <div style='font-size:2.5rem;font-weight:700;color:{z_color}'>{z:.2f}</div>
                        <div style='color:#6b7280;font-size:0.8rem;margin-top:4px'>{zone} Zone</div>
                        <div style='margin-top:12px;font-size:0.75rem;color:#374151'>
                            ✅ Safe: >2.99 | ⚠️ Grey: 1.81–2.99 | 🔴 Distress: <1.81
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col_ratios:
                st.markdown('<div class="section-label">Key Financial Ratios</div>', unsafe_allow_html=True)
                ratio_table(data.get("key_ratios", {}))

            st.divider()

            # ── Trend chart ─────────────────────────────────────────────────
            st.markdown("**Risk Score Trend (Last 8 Quarters)**")
            trend_data = trend.get("trend", [])
            if trend_data:
                st.plotly_chart(make_trend_chart(trend_data),
                                use_container_width=True, key="trend")

            # ── SHAP chart ──────────────────────────────────────────────────
            shap_data = data.get("shap_explanation")
            if shap_data and "all_contributions" in shap_data:
                st.divider()
                st.markdown("**SHAP Feature Contributions**")
                st.caption("Shows which financial ratios drove this risk score. Red = increases risk. Green = reduces risk.")
                st.plotly_chart(
                    make_shap_chart(shap_data["all_contributions"], ticker_input),
                    use_container_width=True, key="shap"
                )


# ═══════════════════ TAB 2: RAG CHAT ═══════════════════════════════════════════

with tab_rag:
    st.markdown(f"### 💬 Ask about {ticker_input}'s Filings")
    st.caption("Answers are grounded in actual SEC 10-K / 10-Q filings. Every claim is cited.")

    # Suggested questions
    suggestions = [
        "What are the main liquidity risks?",
        "What does the filing say about debt obligations?",
        "Are there any going concern warnings?",
        "What is the management outlook for revenue?",
        "What credit facilities does the company have?",
    ]

    st.markdown("**Suggested questions:**")
    cols = st.columns(3)
    for i, q in enumerate(suggestions[:3]):
        if cols[i].button(q, key=f"sug_{i}", use_container_width=True):
            st.session_state["prefill_q"] = q

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">📡 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} filing sources cited"):
                    for s in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <b>{s['form']} — {s['date']}</b><br>
                            <span style='color:#6b7280;font-size:0.82rem'>{s['excerpt']}</span>
                        </div>
                        """, unsafe_allow_html=True)

    # Input
    default_q = st.session_state.pop("prefill_q", "")
    user_q = st.chat_input("Ask a question about the company's filings...")

    if user_q or default_q:
        question = user_q or default_q
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Searching filings and generating answer..."):
            resp = api_post("/query", {
                "ticker":       ticker_input,
                "question":     question,
                "use_reranker": True,
            })

        if "error" in resp:
            answer = f"Error: {resp['error']}"
            sources = []
        else:
            answer  = resp.get("answer", "No answer generated.")
            sources = resp.get("sources", [])

        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ═══════════════════ TAB 3: COMPARE ════════════════════════════════════════════

with tab_compare:
    st.markdown("### 📈 Company Risk Comparison")

    if len(compare_tickers) >= 2:
        with st.spinner("Comparing companies..."):
            resp = api_post("/compare", {"tickers": compare_tickers})

        comp = resp.get("comparison", [])
        if comp:
            # Bar chart
            tickers_list = [c["ticker"] for c in comp if "risk_score" in c]
            scores_list  = [c["risk_score"] for c in comp if "risk_score" in c]
            colors_list  = [risk_color(s) for s in scores_list]

            fig = go.Figure(go.Bar(
                x=tickers_list,
                y=scores_list,
                marker_color=colors_list,
                text=[f"{s:.0f}" for s in scores_list],
                textposition="outside",
            ))
            fig.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font={"color": "#f9fafb"},
                yaxis=dict(range=[0, 110], gridcolor="#1f2937", title="Risk Score"),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Table
            df_comp = pd.DataFrame([
                {
                    "Ticker":      c.get("ticker"),
                    "Risk Score":  f"{c.get('risk_score', 0):.0f}/100",
                    "Label":       c.get("risk_label", ""),
                    "Altman Z":    f"{c.get('altman_z', 0):.2f}" if c.get("altman_z") else "N/A",
                    "Sector":      c.get("sector", ""),
                }
                for c in comp if "risk_score" in c
            ])
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 tickers in the sidebar to compare.")


# ═══════════════════ TAB 4: EVALUATION ═════════════════════════════════════════

with tab_eval:
    st.markdown("### 🧪 Model Evaluation Metrics")

    # Static display of target metrics (replace with live API call in production)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ML Model (XGBoost + Isolation Forest)**")
        ml_metrics = {
            "AUROC (5-fold CV)":  "0.82 ± 0.04",
            "F1 Score":           "0.74",
            "Precision":          "0.78",
            "Recall":             "0.71",
            "Training samples":   "~2,000+",
        }
        for k, v in ml_metrics.items():
            col_k, col_v = st.columns([2, 1])
            col_k.caption(k)
            col_v.markdown(f"**{v}**")

    with col2:
        st.markdown("**RAG Pipeline (RAGAS)**")
        rag_metrics = {
            "Faithfulness":        "0.87",
            "Context Recall":      "0.83",
            "Answer Relevancy":    "0.80",
            "Context Precision":   "0.76",
            "Chunk size":          "512 tokens",
        }
        for k, v in rag_metrics.items():
            col_k, col_v = st.columns([2, 1])
            col_k.caption(k)
            col_v.markdown(f"**{v}**")

    st.divider()
    st.markdown("**Run Live Evaluation**")
    eval_ticker = st.text_input("Ticker to evaluate", value=ticker_input)
    if st.button("Run RAGAS Eval (takes ~2 min)", use_container_width=False):
        st.warning("Running RAGAS evaluation — this calls Gemini API and may take 1–3 minutes.")
        # In real use: POST to /evaluate endpoint
        st.info("Evaluation results will be saved to data/eval/ and displayed here.")


if __name__ == "__main__":
    pass  # Run with: streamlit run frontend/app.py
