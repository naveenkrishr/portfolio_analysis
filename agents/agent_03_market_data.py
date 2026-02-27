"""
Agent 3 — Market Data

Fetches 1yr daily OHLCV via yfinance (single batch call) and computes
technical indicators (SMA 50/200, RSI, MACD, Bollinger Bands) for each
equity ticker in the portfolio.

Produces: state["market_data"] = dict[ticker, TechnicalSnapshot]

Design:
  - fetch() is the core logic — testable standalone
  - run()   is the LangGraph node wrapper (not wired in until Step 8)
  - Cash tickers are skipped (yfinance has no data for SPAXX/FCASH)
"""
from __future__ import annotations

import time

from analysis.technicals import TechnicalSnapshot, compute
from tools.yfinance_client import fetch_price_history


def fetch(tickers: list[str]) -> dict[str, TechnicalSnapshot]:
    """
    Fetch price history + compute technicals for a list of equity tickers.

    Args:
        tickers: equity ticker symbols (cash tickers are silently skipped
                 because yfinance won't have data for them)

    Returns:
        { ticker: TechnicalSnapshot }  — only tickers yfinance returned data for
    """
    print(f"\n[Agent 3] Fetching market data for: {tickers}")
    t0 = time.time()

    history = fetch_price_history(tickers)
    print(f"[Agent 3] Downloaded {len(history)}/{len(tickers)} tickers "
          f"({time.time() - t0:.1f}s)")

    snapshots = compute(history)
    print(f"[Agent 3] Technicals computed for {len(snapshots)} tickers")

    return snapshots


def run(state: dict) -> dict:
    """
    LangGraph node — not wired into the graph yet (Step 8).
    Reads equity tickers from state, writes market_data back.
    """
    from state.models import Holding

    holdings: list[Holding] = state["holdings"]
    equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]

    market_data = fetch(equity_tickers)
    return {**state, "market_data": market_data}
