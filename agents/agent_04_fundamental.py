"""
Agent 4 — Fundamental Analysis

Fetches fundamental data for equity positions using yfinance .info.
Cash symbols (SPAXX, FCASH, etc.) are automatically excluded.
ETFs get ETF-specific data (expense ratio, AUM, holdings) instead of stock fundamentals.

Data pipeline:
  Stocks: tools/yfinance_fundamentals.py → analysis/fundamentals.py → FundamentalSnapshot
  ETFs:   tools/yfinance_etf.py          → analysis/etf_info.py     → ETFSnapshot

  state["fundamentals"] → dict[ticker, FundamentalSnapshot]  (stocks only)
  state["etf_data"]     → dict[ticker, ETFSnapshot]          (ETFs only)
"""
from __future__ import annotations

import time

from analysis.fundamentals import FundamentalSnapshot, compute
from analysis.etf_info import ETFSnapshot, compute as compute_etf
from tools.yfinance_fundamentals import fetch_raw_info
from tools.yfinance_etf import fetch_etf_info
from cache.cache_manager import cache
from state.models import Holding


# ── Standalone entry points ───────────────────────────────────────────────────

def fetch(tickers: list[str]) -> dict[str, FundamentalSnapshot]:
    """
    Fetch and compute fundamentals for the given stock tickers.
    Uses cache to skip tickers with fresh data (weekly TTL).
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
        print("  All stock tickers served from cache")

    return {**cached, **fresh}


def fetch_etf(tickers: list[str]) -> dict[str, ETFSnapshot]:
    """
    Fetch and compute ETF-specific data for the given ETF tickers.
    Uses cache to skip tickers with fresh data (weekly TTL).
    """
    cached, stale = cache.partition("etf_info", tickers)
    if cached:
        print(f"  ETF cache hit: {list(cached.keys())}")

    if stale:
        raw_info, top_holdings = fetch_etf_info(stale)
        fresh = compute_etf(stale, raw_info, top_holdings)
        for ticker, snap in fresh.items():
            cache.set("etf_info", ticker, snap)
    else:
        fresh = {}
        print("  All ETF tickers served from cache")

    return {**cached, **fresh}


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node — fetches fundamentals for stocks and ETF data for ETFs.
    Adds state["fundamentals"] and state["etf_data"].
    Cash positions are excluded (asset_type == "cash").
    """
    holdings: list[Holding] = state["holdings"]
    stock_tickers = [h.ticker for h in holdings if h.asset_type == "stock"]
    etf_tickers = [h.ticker for h in holdings if h.asset_type == "etf"]

    print("\n" + "="*70)
    print("AGENT 4 — Fundamental Analysis")
    print("="*70)

    t0 = time.time()

    # Fetch stock fundamentals
    fundamentals = {}
    if stock_tickers:
        print(f"Fetching stock fundamentals for: {', '.join(stock_tickers)}")
        fundamentals = fetch(stock_tickers)
        print(f"  Stocks: {len(fundamentals)}/{len(stock_tickers)} retrieved")

    # Fetch ETF data
    etf_data = {}
    if etf_tickers:
        print(f"Fetching ETF data for: {', '.join(etf_tickers)}")
        etf_data = fetch_etf(etf_tickers)
        print(f"  ETFs: {len(etf_data)}/{len(etf_tickers)} retrieved")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    return {**state, "fundamentals": fundamentals, "etf_data": etf_data}
