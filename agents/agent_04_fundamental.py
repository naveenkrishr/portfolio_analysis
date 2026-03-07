"""
Agent 4 — Fundamental Analysis

Fetches fundamental data for all equity positions using yfinance .info.
Cash symbols (SPAXX, FCASH, etc.) are automatically excluded.

Data pipeline:
  tools/yfinance_fundamentals.py  →  fetch_raw_info()
  analysis/fundamentals.py        →  compute()   →  FundamentalSnapshot per ticker
  state["fundamentals"]           →  dict[ticker, FundamentalSnapshot]

Source waterfall (future):
  PRIMARY:   yfinance .info  (no key, ~2000/hr)
  SECONDARY: Finnhub         (needs FINNHUB_API_KEY, 60/min)
  TERTIARY:  FMP             (needs FMP_API_KEY, 250/day)
"""
from __future__ import annotations

import time

from analysis.fundamentals import FundamentalSnapshot, compute
from tools.yfinance_fundamentals import fetch_raw_info
from cache.cache_manager import cache
from state.models import Holding


# ── Standalone entry point (testable without LangGraph) ───────────────────────

def fetch(tickers: list[str]) -> dict[str, FundamentalSnapshot]:
    """
    Fetch and compute fundamentals for the given equity tickers.
    Uses cache to skip tickers with fresh data (weekly TTL).

    Args:
        tickers: list of equity ticker symbols (no cash symbols)

    Returns:
        { ticker: FundamentalSnapshot }
    """
    cached, stale = cache.partition("fundamentals", tickers)
    if cached:
        print(f"  Cache hit: {list(cached.keys())}")

    if stale:
        raw = fetch_raw_info(stale)
        fresh = compute(stale, raw)
        for ticker, snap in fresh.items():
            cache.set("fundamentals", ticker, snap)
    else:
        fresh = {}
        print("  All tickers served from cache")

    return {**cached, **fresh}


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node — fetches fundamentals for all equity positions.
    Adds state["fundamentals"] = dict[ticker, FundamentalSnapshot].
    Cash positions are excluded (asset_type == "cash").
    """
    holdings: list[Holding] = state["holdings"]
    equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]

    print("\n" + "="*70)
    print("AGENT 4 — Fundamental Analysis")
    print("="*70)
    print(f"Fetching fundamentals for: {', '.join(equity_tickers)}")

    t0 = time.time()
    fundamentals = fetch(equity_tickers)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s — {len(fundamentals)}/{len(equity_tickers)} tickers retrieved")

    return {**state, "fundamentals": fundamentals}
