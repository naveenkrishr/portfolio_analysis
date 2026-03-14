"""
Agent 6 — Earnings & Events

Fetches upcoming earnings dates via Finnhub calendar endpoint
(single batch API call for all tickers) and produces EarningsSnapshot
per ticker for LLM context.

Data pipeline:
  tools/finnhub_earnings.py   →  fetch_earnings()  (1 API call)
  analysis/earnings.py        →  compute_all()      →  EarningsSnapshot per ticker
  state["earnings_data"]      →  dict[ticker, EarningsSnapshot]

No LLM required — pure data fetch + transformation.
"""
from __future__ import annotations

import time

from analysis.earnings import EarningsSnapshot, compute_all
from tools.finnhub_earnings import fetch_earnings
from cache.cache_manager import cache
from state.models import Holding


# ── Standalone entry point (testable without LangGraph) ───────────────────────

def fetch(tickers: list[str]) -> dict[str, EarningsSnapshot]:
    """
    Fetch upcoming earnings for the given equity tickers.
    Uses batch cache (daily TTL) since the Finnhub calendar is one API call.

    Args:
        tickers: list of equity ticker symbols (no cash symbols)

    Returns:
        { ticker: EarningsSnapshot }
    """
    # Batch cache — one key for the whole ticker set
    cached = cache.get_batch("earnings_calendar", tickers)
    if cached is not None:
        print("  Earnings served from cache")
        return cached

    raw = fetch_earnings(tickers)
    result = compute_all(tickers, raw)
    cache.set_batch("earnings_calendar", tickers, result)
    return result


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node — fetches earnings calendar for all equity positions.
    Adds state["earnings_data"] = dict[ticker, EarningsSnapshot].
    Cash positions are excluded (asset_type == "cash").
    """
    holdings: list[Holding] = state["holdings"]
    # Only fetch earnings for stocks — ETFs don't have earnings dates
    equity_tickers = [h.ticker for h in holdings if h.asset_type == "stock"]

    print("\n" + "="*70)
    print("AGENT 6 — Earnings & Events")
    print("="*70)
    skipped_etfs = [h.ticker for h in holdings if h.asset_type == "etf"]
    if skipped_etfs:
        print(f"Skipping ETFs (no earnings): {', '.join(skipped_etfs)}")
    print(f"Fetching earnings for: {', '.join(equity_tickers) or '(none)'}")

    t0 = time.time()
    earnings_data = fetch(equity_tickers)
    elapsed = time.time() - t0

    with_dates = sum(1 for s in earnings_data.values() if s.next_earnings_date)
    print(f"Done in {elapsed:.1f}s — {with_dates}/{len(earnings_data)} tickers with upcoming earnings")
    for ticker, snap in earnings_data.items():
        print(f"  {snap.summary()}")

    return {**state, "earnings_data": earnings_data}
