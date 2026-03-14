"""
Streamlit Web UI for Portfolio Analysis.

Enter holdings manually (ticker, quantity, avg_price) → run the full analysis
pipeline (agents 3-9) → see the styled HTML report in the browser.

Holdings are persisted to holdings.json between sessions.

Usage:
    streamlit run web_app.py
    streamlit run web_app.py -- --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# ── Page config (must be first Streamlit call) ───────────────────────────────

st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📊",
    layout="wide",
)

# ── Parse CLI args passed after "--" ─────────────────────────────────────────

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-14B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=4096)
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known

args = _parse_args()

# ── Load MLX model (cached — survives Streamlit reruns) ──────────────────────

from dotenv import load_dotenv
load_dotenv()

from mlx_wrapper import MLXChatModel


@st.cache_resource(show_spinner="Loading MLX model (this takes ~30s on first run)...")
def load_model(model_path: str, max_tokens: int):
    llm = MLXChatModel(
        model_path=model_path,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    llm._load()
    return llm


llm = load_model(args.model, args.max_tokens)

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


# Initialize session state from saved file
if "holdings" not in st.session_state:
    st.session_state.holdings = _load_saved_holdings()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: `{args.model}`")
    mem = llm.memory_stats()
    st.caption(f"Memory: {mem['active_gb']} GB active / {mem['peak_gb']} GB peak")
    st.divider()
    st.markdown(
        "**How to use:**\n"
        "1. Add holdings using the form below\n"
        "2. Click **Run Analysis**\n"
        "3. Wait 2-5 minutes for the full analysis\n"
        "4. View or download the report\n"
    )

# ── Main UI ──────────────────────────────────────────────────────────────────

st.title("Portfolio Analysis")

# ── Add holding form ─────────────────────────────────────────────────────────

st.subheader("Add Holding")
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    new_ticker = st.text_input("Ticker", placeholder="e.g. AAPL", key="new_ticker")
with col2:
    new_qty = st.number_input("Quantity", min_value=0.0, step=1.0, key="new_qty")
with col3:
    new_avg = st.number_input("Avg Price ($)", min_value=0.0, step=0.01, key="new_avg")
with col4:
    st.write("")  # spacing
    st.write("")  # align button with inputs
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
    st.subheader(f"Holdings ({len(st.session_state.holdings)} rows)")

    # Display as a table with delete buttons
    header_cols = st.columns([2, 1.5, 1.5, 1.5, 0.5])
    header_cols[0].markdown("**Ticker**")
    header_cols[1].markdown("**Quantity**")
    header_cols[2].markdown("**Avg Price**")
    header_cols[3].markdown("**Est. Value**")
    header_cols[4].markdown("")

    to_delete = None
    for i, row in enumerate(st.session_state.holdings):
        cols = st.columns([2, 1.5, 1.5, 1.5, 0.5])
        cols[0].write(row["ticker"])
        cols[1].write(f"{row['quantity']:,.2f}")
        cols[2].write(f"${row['avg_price']:,.2f}")
        cols[3].write(f"${row['quantity'] * row['avg_price']:,.0f}")
        if cols[4].button("X", key=f"del_{i}"):
            to_delete = i

    if to_delete is not None:
        st.session_state.holdings.pop(to_delete)
        _save_holdings(st.session_state.holdings)
        st.rerun()

    # Clear all button
    col_clear, col_spacer = st.columns([1, 4])
    with col_clear:
        if st.button("Clear All"):
            st.session_state.holdings = []
            _save_holdings([])
            st.rerun()

    st.divider()

    # ── Run analysis ─────────────────────────────────────────────────────────

    if st.button("Run Analysis", type="primary", use_container_width=True):
        from web_runner import build_holdings, run_analysis

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
            # Build holdings (fetch current prices)
            holdings = build_holdings(
                st.session_state.holdings,
                progress_callback=progress_callback,
            )

            # Run the pipeline
            html = run_analysis(
                holdings=holdings,
                llm=llm,
                progress_callback=progress_callback,
                token_callback=token_callback,
            )
            elapsed = time.time() - t0

            # Final token flush
            if synthesis_tokens:
                synthesis_placeholder.markdown(
                    "**LLM Synthesis (complete):**\n\n" + "".join(synthesis_tokens)
                )

            status_container.update(
                label=f"Analysis complete ({elapsed:.0f}s)",
                state="complete",
                expanded=False,
            )

            # Display report
            st.subheader("Report")
            components.html(html, height=2000, scrolling=True)
            st.download_button(
                "Download HTML Report",
                data=html,
                file_name="portfolio_report.html",
                mime="text/html",
            )

        except Exception as e:
            status_container.update(label="Analysis failed", state="error")
            st.error(f"Error: {e}")
            raise

else:
    st.info("Add holdings above to get started.")
