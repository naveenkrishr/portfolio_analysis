"""
Streamlit Web UI for Portfolio Analysis — Render.com deployment.

Uses Groq API (Llama 3.3 70B) instead of local MLX model.
Enter holdings manually → run analysis pipeline → see report in browser.

Holdings are persisted to holdings.json between sessions.
Mobile-friendly layout.

Usage (local testing):
    GROQ_API_KEY=xxx streamlit run render_app.py

Render.com:
    Set GROQ_API_KEY and FINNHUB_API_KEY as environment variables.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📊",
    layout="centered",
)

# ── Load Groq LLM (cached — survives Streamlit reruns) ──────────────────────

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq


@st.cache_resource
def load_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=4096,
    )


llm = load_model()

# ── Holdings persistence ─────────────────────────────────────────────────────

HOLDINGS_FILE = Path("holdings.json")


def _load_saved_holdings() -> list[dict]:
    if HOLDINGS_FILE.exists():
        try:
            data = json.loads(HOLDINGS_FILE.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return []


def _save_holdings(rows: list[dict]):
    HOLDINGS_FILE.write_text(json.dumps(rows, indent=2))


if "holdings" not in st.session_state:
    st.session_state.holdings = _load_saved_holdings()

# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("📊 Portfolio Analysis")
st.caption("Groq — Llama 3.3 70B")

# ── Add holding form (stacked for mobile) ────────────────────────────────────

with st.expander("Add Holding", expanded=len(st.session_state.holdings) == 0):
    new_ticker = st.text_input("Ticker", placeholder="e.g. AAPL", key="new_ticker")
    col_qty, col_avg = st.columns(2)
    with col_qty:
        new_qty = st.number_input("Quantity", min_value=0.0, step=1.0, key="new_qty")
    with col_avg:
        new_avg = st.number_input("Avg Price ($)", min_value=0.0, step=0.01, key="new_avg")

    if st.button("Add", use_container_width=True):
        ticker = new_ticker.strip().upper()
        if ticker and new_qty > 0 and new_avg > 0:
            st.session_state.holdings.append({
                "ticker": ticker,
                "quantity": new_qty,
                "avg_price": new_avg,
            })
            _save_holdings(st.session_state.holdings)
            st.rerun()
        else:
            st.toast("Please fill in all fields (ticker, quantity > 0, avg price > 0)")

# ── Current holdings table ───────────────────────────────────────────────────

if st.session_state.holdings:
    st.subheader(f"Holdings ({len(st.session_state.holdings)})")

    # Build a DataFrame for responsive display
    df = pd.DataFrame(st.session_state.holdings)
    df["ticker"] = df["ticker"].str.upper()
    df["est_value"] = df["quantity"] * df["avg_price"]
    display_df = df.rename(columns={
        "ticker": "Ticker",
        "quantity": "Qty",
        "avg_price": "Avg Price",
        "est_value": "Est. Value",
    })
    display_df["Avg Price"] = display_df["Avg Price"].map("${:,.2f}".format)
    display_df["Est. Value"] = display_df["Est. Value"].map("${:,.0f}".format)
    display_df["Qty"] = display_df["Qty"].map("{:,.2f}".format)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # Delete individual rows
    col_del, col_del_btn = st.columns([3, 1])
    with col_del:
        del_idx = st.selectbox(
            "Remove row",
            options=range(len(st.session_state.holdings)),
            format_func=lambda i: f"{st.session_state.holdings[i]['ticker']} — {st.session_state.holdings[i]['quantity']} shares",
            label_visibility="collapsed",
        )
    with col_del_btn:
        if st.button("Remove", use_container_width=True):
            st.session_state.holdings.pop(del_idx)
            _save_holdings(st.session_state.holdings)
            st.rerun()

    if st.button("Clear All", type="secondary"):
        st.session_state.holdings = []
        _save_holdings([])
        st.rerun()

    st.divider()

    # ── Run analysis ─────────────────────────────────────────────────────────

    if st.button("Run Analysis", type="primary", use_container_width=True):
        from render_runner import build_holdings, run_analysis

        status_container = st.status("Running analysis...", expanded=True)
        synthesis_placeholder = st.empty()
        synthesis_tokens: list[str] = []

        def progress_callback(msg: str):
            status_container.write(msg)

        def token_callback(token: str):
            synthesis_tokens.append(token)
            if len(synthesis_tokens) % 3 == 0 or token.endswith("\n"):
                synthesis_placeholder.markdown(
                    "**LLM Synthesis (streaming):**\n\n" + "".join(synthesis_tokens)
                )

        t0 = time.time()
        try:
            holdings = build_holdings(
                st.session_state.holdings,
                progress_callback=progress_callback,
            )

            html = run_analysis(
                holdings=holdings,
                llm=llm,
                progress_callback=progress_callback,
                token_callback=token_callback,
            )
            elapsed = time.time() - t0

            if synthesis_tokens:
                synthesis_placeholder.markdown(
                    "**LLM Synthesis (complete):**\n\n" + "".join(synthesis_tokens)
                )

            status_container.update(
                label=f"Analysis complete ({elapsed:.0f}s)",
                state="complete",
                expanded=False,
            )

            st.subheader("Report")
            components.html(html, height=2000, scrolling=True)
            st.download_button(
                "Download HTML Report",
                data=html,
                file_name="portfolio_report.html",
                mime="text/html",
                use_container_width=True,
            )

        except Exception as e:
            status_container.update(label="Analysis failed", state="error")
            st.error(f"Error: {e}")
            raise

else:
    st.info("Add holdings above to get started.")
